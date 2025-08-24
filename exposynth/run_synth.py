from __future__ import annotations

import argparse
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set
import sys

import numpy as np
import pandas as pd
import yaml

# Allow running as a script: python exposynth/run_synth.py ...
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from exposynth.constraints import Constraints


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _anchors_from_history(hist: List[int], items_df: pd.DataFrame, k: int = 4) -> List[Tuple[str, float]]:
    items_idx = items_df.set_index("item_id")
    pg = items_idx["genres"].astype(str).map(lambda s: s.split("|")[0] if isinstance(s, str) else "")
    genres = [str(pg.get(int(i), "")) for i in hist]
    cnt = Counter([g for g in genres if g])
    if not cnt:
        return []
    total = sum(cnt.values())
    top = cnt.most_common(k)
    return [(g, c / total) for g, c in top]


def _build_tail_sampling(items_df: pd.DataFrame) -> pd.Series:
    # target: amplify tail bins (pop_bin>=3) to be comparable to head
    pop = items_df.set_index("item_id")["pop_bin"].astype(int)
    w = np.where(pop >= 3, 1.0, 0.5)  # simple upweight tail
    s = pd.Series(w, index=pop.index, dtype=float)
    return s / s.sum()


def synthesize(cfg: Dict, *, steps_per_user: int = 8, anchors_k: int = 4, resume_users: Set[int] | None = None) -> pd.DataFrame:
    proc_dir = cfg["paths"]["proc"]
    inter_path = os.path.join(proc_dir, "ml1m_interactions.parquet")
    items_path = os.path.join(proc_dir, "ml1m_items.parquet")
    interactions = pd.read_parquet(inter_path)
    items = pd.read_parquet(items_path)

    # constraints (genre transition + item similarity)
    cons = Constraints(interactions, items, genre_prob_thresh=0.02, sim_thresh=0.05, topk_neighbors=50)

    # per-user history
    inter_sorted = interactions.sort_values(["user_id", "ts"])  # oldest->newest
    user_hist = inter_sorted.groupby("user_id")["item_id"].apply(list)

    # sampling distributions
    tail_prob = _build_tail_sampling(items)
    items_idx = items.set_index("item_id")
    item_ids = items_idx.index.to_numpy()

    # genre index for anchors
    item_genre = items_idx["genres"].astype(str).map(lambda s: s.split("|")[0] if isinstance(s, str) else "")
    genre_to_items: Dict[str, np.ndarray] = defaultdict(lambda: np.array([], dtype=np.int64))
    for g, grp in items.assign(primary=item_genre.values).groupby("primary"):
        genre_to_items[str(g)] = grp["item_id"].astype(int).to_numpy()

    rng = np.random.default_rng(42)
    out_rows: List[Dict] = []
    accepted = 0
    rejected = 0

    for uid, hist in user_hist.items():
        if resume_users and int(uid) in resume_users:
            continue
        anchors = _anchors_from_history(hist, items, k=anchors_k)
        if not anchors:
            continue
        # build a per-user candidate sampler: mixture of tail sampling and anchor genres
        for step in range(int(steps_per_user)):
            prev = int(hist[-1]) if len(hist) > 0 else None
            # choose an anchor genre proportional to weight
            genres, probs = zip(*anchors)
            probs = np.array(probs, dtype=float)
            probs = probs / probs.sum()
            g_pick = rng.choice(genres, p=probs)

            # candidate pool: items of this genre, with tail upweighting
            cand_ids = genre_to_items.get(str(g_pick), item_ids)
            if cand_ids.size == 0:
                cand_ids = item_ids

            # build sampling weights combining tail_prob and slight recency diversity
            weights = tail_prob.reindex(cand_ids).fillna(0.0).to_numpy()
            if np.all(weights <= 0):
                weights = np.ones_like(cand_ids, dtype=float)
            weights = weights / weights.sum()

            # try up to 10 times to satisfy constraints
            ok = False
            for _ in range(10):
                pick = int(rng.choice(cand_ids, p=weights))
                if cons.is_valid_transition(prev, pick):
                    ok = True
                    break
            if not ok:
                rejected += 1
                continue

            accepted += 1
            out_rows.append({
                "user_id": int(uid),
                "item_id": int(pick),
                "ts": int(inter_sorted[inter_sorted["user_id"] == uid]["ts"].max()) + step + 1,
                "source": "synth",
                "reason": "anchor-based",
            })

    df_out = pd.DataFrame(out_rows)
    logging.info("Synthesized: %d accepted, %d rejected (accept_ratio=%.2f)", accepted, rejected, (accepted/(accepted+rejected) if (accepted+rejected)>0 else 0.0))

    # tail ratio
    if not df_out.empty:
        pop = items.set_index("item_id")["pop_bin"].astype(int)
        tail_ratio = float((pop.reindex(df_out["item_id"]).fillna(0).astype(int) >= 3).mean())
        logging.info("Tail (pop_bin>=3) ratio in synth=%.3f", tail_ratio)

    return df_out


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Exposure-aware synthetic interaction generator")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume synthesis by skipping users already present in output file")
    parser.add_argument("--log_file", type=str, default=None, help="Optional log file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    steps = int(cfg.get("exposynth", {}).get("steps_per_user", 8))
    anchors_k = int(cfg.get("exposynth", {}).get("anchors_k", 4))

    # Optional file logging
    if args.log_file:
        ensure_dir(os.path.dirname(args.log_file))
        fh = logging.FileHandler(args.log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(fh)

    # Resume users set
    resume_users: Set[int] | None = None
    if args.resume and os.path.exists(args.out):
        try:
            prev = pd.read_parquet(args.out)
            if {"user_id"}.issubset(set(prev.columns)):
                resume_users = set(prev["user_id"].astype(int).unique().tolist())
                logging.info("Resume enabled: will skip %d users already in %s", len(resume_users), args.out)
        except Exception as e:
            logging.warning("Failed to load existing output for resume: %s", e)

    df_out = synthesize(cfg, steps_per_user=steps, anchors_k=anchors_k, resume_users=resume_users)
    ensure_dir(os.path.dirname(args.out))
    # Merge with existing output if present
    if os.path.exists(args.out):
        try:
            prev = pd.read_parquet(args.out)
            merged = pd.concat([prev, df_out], ignore_index=True)
            merged = merged.drop_duplicates(["user_id", "item_id", "ts"], keep="last")
            merged.to_parquet(args.out, index=False)
            logging.info("Appended and saved synth interactions to %s (now %d rows)", args.out, len(merged))
        except Exception:
            df_out.to_parquet(args.out, index=False)
            logging.info("Saved synth interactions to %s (%d rows)", args.out, len(df_out))
    else:
        df_out.to_parquet(args.out, index=False)
        logging.info("Saved synth interactions to %s (%d rows)", args.out, len(df_out))


if __name__ == "__main__":
    main()

