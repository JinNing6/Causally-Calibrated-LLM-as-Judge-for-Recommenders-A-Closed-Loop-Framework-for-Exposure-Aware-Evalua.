import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _read_parquets(interactions_path: str, items_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(interactions_path):
        raise FileNotFoundError(interactions_path)
    if not os.path.exists(items_path):
        raise FileNotFoundError(items_path)
    interactions = pd.read_parquet(interactions_path)
    items = pd.read_parquet(items_path)
    return interactions, items


def sample_pairs(
    interactions_path: str,
    items_path: str,
    L: int,
    pairs_per_user: int,
    match_cols: List[str],
    out_path: str,
) -> None:
    interactions, items = _read_parquets(interactions_path, items_path)

    items_idx = items.set_index("item_id")
    all_item_ids = items_idx.index.values

    df = interactions.sort_values(["user_id", "ts"])  # oldest -> newest
    histories = df.groupby("user_id").tail(L)

    # build full interaction set per user for candidate exclusion
    user_to_items: Dict[int, set] = defaultdict(set)
    for row in interactions.itertuples(index=False):
        user_to_items[int(row.user_id)].add(int(row.item_id))

    # history list per user (most recent first)
    hist_lists: Dict[int, List[int]] = {}
    for uid, g in histories.groupby("user_id", sort=False):
        gg = g.sort_values("ts", ascending=False)
        hist_lists[int(uid)] = [int(x) for x in gg["item_id"].tolist()]

    outputs: List[Dict] = []
    rng = random.Random(2025)

    for uid, interacted in user_to_items.items():
        hist_items = hist_lists.get(uid, [])
        if len(hist_items) == 0:
            continue

        # candidate pool: items user never interacted with
        cand_ids = np.setdiff1d(all_item_ids, np.fromiter(interacted, dtype=np.int64), assume_unique=False)
        if len(cand_ids) < 2:
            continue

        cand_df = items_idx.loc[cand_ids, match_cols].reset_index()

        # groups with at least 2 items
        valid_groups = []
        for key, grp in cand_df.groupby(match_cols, dropna=False):
            ids = grp["item_id"].tolist()
            if len(ids) >= 2:
                valid_groups.append((key if isinstance(key, tuple) else (key,), ids))

        if not valid_groups:
            continue

        need = pairs_per_user
        attempts = 0
        max_attempts = pairs_per_user * 10
        while need > 0 and attempts < max_attempts:
            attempts += 1
            key, ids = rng.choice(valid_groups)
            i, j = rng.sample(ids, 2)

            if i == j:
                continue
            if i in interacted or j in interacted:
                continue

            match_bins = {col: int(val) for col, val in zip(match_cols, key)}
            outputs.append(
                {
                    "user_id": int(uid),
                    "hist_items": [int(x) for x in hist_items],
                    "cand_i": int(i),
                    "cand_j": int(j),
                    "match_bins": match_bins,
                }
            )
            need -= 1

    if not outputs:
        logging.warning("No pairs were sampled. Check inputs and match_cols.")
        return

    out_df = pd.DataFrame(outputs)

    # Try to save dict/list to parquet. Fallback to JSON for match_bins if necessary.
    ensure_dir(os.path.dirname(out_path))
    try:
        out_df.to_parquet(out_path, index=False)
    except Exception:
        out_df = out_df.copy()
        out_df["match_bins_json"] = out_df["match_bins"].apply(json.dumps)
        out_df.to_parquet(out_path, index=False)

    logging.info("Saved pairs to %s (%d rows)", out_path, len(out_df))

    # print examples for 5 random users
    sample_users = out_df["user_id"].drop_duplicates().sample(n=min(5, out_df["user_id"].nunique()), random_state=42)
    for uid in sample_users.tolist():
        sub = out_df[out_df["user_id"] == uid].head(1)
        row = sub.iloc[0]
        logging.info(
            "Example user %s: hist_items=%s, cand_i=%s, cand_j=%s, match_bins=%s",
            uid,
            row["hist_items"],
            row["cand_i"],
            row["cand_j"],
            row["match_bins"],
        )


def _cli() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Sample candidate item pairs per user for LLM judging")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    proc_dir = cfg["paths"]["proc"]
    props_dir = cfg["paths"].get("props", os.path.join("data", "props"))
    ensure_dir(props_dir)

    interactions_path = os.path.join(proc_dir, "ml1m_interactions.parquet")
    items_path = os.path.join(proc_dir, "ml1m_items.parquet")
    L = int(cfg.get("recjudge", {}).get("L_history", 10))
    pairs_per_user = int(cfg.get("recjudge", {}).get("pairs_per_user", 60))
    match_cols = list(cfg.get("recjudge", {}).get("match_cols", []))
    out_path = args.out or os.path.join(props_dir, "pairs.parquet")

    sample_pairs(
        interactions_path=interactions_path,
        items_path=items_path,
        L=L,
        pairs_per_user=pairs_per_user,
        match_cols=match_cols,
        out_path=out_path,
    )


if __name__ == "__main__":
    _cli()


