from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

import pandas as pd
import yaml

# Allow running as a script: python recjudge/run_judge.py ...
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from recjudge.prompts import build_history_block, build_item_block, render_pair_prompt
from recjudge.llm import call_llm


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _normalize_json_text(text: str) -> str:
    # Try to find the first JSON object in text
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else text


def _parse_json_with_tolerance(text: str) -> Dict:
    cleaned = _normalize_json_text(text)
    try:
        return json.loads(cleaned)
    except Exception:
        # try to relax quotes for keys by adding quotes via regex (simple heuristic)
        t = re.sub(r"(\w+)\s*:\s*", r'"\\1": ', cleaned)
        try:
            return json.loads(t)
        except Exception:
            return {}


def _validate_evidence_is_substring(response_obj: Dict, prompt_text: str) -> Tuple[int, int]:
    rationals = response_obj.get("rationales") or []
    total = 0
    valid = 0
    for r in rationals:
        ev = str(r.get("evidence", ""))
        if not ev:
            continue
        total += 1
        if ev in prompt_text:
            valid += 1
    return valid, total


async def _judge_one(row, items_df: pd.DataFrame, semaphore: asyncio.Semaphore, llm_cfg: Dict, *, retries: int = 3, backoff: float = 1.0) -> Dict:
    async with semaphore:
        history_block = build_history_block(row.hist_items, items_df)
        i_block = build_item_block(int(row.cand_i), items_df)
        j_block = build_item_block(int(row.cand_j), items_df)
        prompt = render_pair_prompt(history_block, i_block, j_block)

        # LLM call (sync wrapper inside thread to avoid blocking loop) with retries
        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None
        reply_text: str = ""
        for attempt in range(int(retries) + 1):
            try:
                reply_text = await loop.run_in_executor(None, call_llm, prompt)
                last_exc = None
                break
            except Exception as e:  # network/API robustness
                last_exc = e
                if attempt < int(retries):
                    await asyncio.sleep(float(backoff) * (2 ** attempt))
                else:
                    reply_text = "{}"  # degrade to empty JSON

        obj = _parse_json_with_tolerance(reply_text)
        winner = (obj.get("winner") or "").lower().strip()
        p = obj.get("p")
        if isinstance(p, str):
            try:
                p = float(p)
            except Exception:
                p = None
        if winner not in {"i", "j", "tie"}:
            winner = "tie"

        valid, total = _validate_evidence_is_substring(obj, prompt)
        valid_ratio = (valid / total) if total > 0 else 0.0

        result = {
            "user_id": int(row.user_id),
            "i": int(row.cand_i),
            "j": int(row.cand_j),
            "winner": winner,
            "p": float(p) if p is not None else None,
            "valid_ratio": float(valid_ratio),
            "raw": obj,
        }
        if last_exc is not None:
            result["error"] = type(last_exc).__name__
        return result


async def main_async(
    cfg_path: str,
    pairs_path: str,
    out_raw_path: str,
    out_clean_path: str,
    *,
    limit: int | None = None,
    concurrency: int = 8,
    retries: int = 3,
    backoff: float = 1.0,
    resume_keys: Set[str] | None = None,
) -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    items_path = os.path.join(cfg["paths"]["proc"], "ml1m_items.parquet")
    items_df = pd.read_parquet(items_path)

    pairs_df = pd.read_parquet(pairs_path)
    if isinstance(limit, int) and limit > 0 and len(pairs_df) > limit:
        pairs_df = pairs_df.sample(n=limit, random_state=42).reset_index(drop=True)

    # Resume: skip already judged pairs if provided
    if resume_keys:
        def _mk_key(df: pd.DataFrame) -> pd.Series:
            return df["user_id"].astype(str) + "-" + df["cand_i"].astype(str) + "-" + df["cand_j"].astype(str)

        if {"user_id", "cand_i", "cand_j"}.issubset(set(pairs_df.columns)):
            pairs_df = pairs_df.assign(_k=_mk_key(pairs_df))
            before = len(pairs_df)
            pairs_df = pairs_df[~pairs_df["._k" if "._k" in pairs_df.columns else "_k"].isin(resume_keys)].drop(columns=[c for c in ["_k", "._k"] if c in pairs_df.columns])
            skipped = before - len(pairs_df)
            logging.info("Resume enabled: skipped %d already-judged pairs; remaining %d", skipped, len(pairs_df))

    ensure_dir(os.path.dirname(out_raw_path))
    ensure_dir(os.path.dirname(out_clean_path))

    sem = asyncio.Semaphore(int(concurrency))
    tasks = [
        _judge_one(row, items_df, sem, cfg.get("recjudge", {}).get("llm", {}), retries=int(retries), backoff=float(backoff))
        for row in pairs_df.itertuples(index=False)
    ]

    results = await asyncio.gather(*tasks)

    # Save raw JSONL (append if resume)
    mode = "a" if os.path.exists(out_raw_path) else "w"
    with open(out_raw_path, mode, encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r["raw"], ensure_ascii=False) + "\n")

    clean_rows = [
        {k: r[k] for k in ["user_id", "i", "j", "p", "winner", "valid_ratio"]}
        for r in results
    ]
    clean_df = pd.DataFrame(clean_rows)
    # If resume and existing clean parquet exists, merge and deduplicate by (user_id,i,j)
    if os.path.exists(out_clean_path):
        try:
            prev = pd.read_parquet(out_clean_path)
            merged = pd.concat([prev, clean_df], ignore_index=True)
            merged = merged.drop_duplicates(["user_id", "i", "j"], keep="last")
            merged.to_parquet(out_clean_path, index=False)
            clean_df = merged
        except Exception:
            # Fallback: write current block only
            clean_df.to_parquet(out_clean_path, index=False)
    else:
        clean_df.to_parquet(out_clean_path, index=False)

    ok_ratio = (clean_df["valid_ratio"] >= 0.5).mean() if not clean_df.empty else 0.0
    parse_ok = (~clean_df["winner"].isna()).mean() if not clean_df.empty else 0.0
    logging.info("Parsed=%d; evidence_valid>=0.5 ratio=%.2f", len(clean_df), ok_ratio)

    if ok_ratio < 0.8:
        # print top-3 failures
        bad = clean_df.sort_values("valid_ratio").head(3)
        logging.warning("Top-3 low-valid examples:\n%s", bad)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run LLM judging for item pairs")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pairs", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Max number of pairs to judge")
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Max concurrent LLM requests"
    )
    parser.add_argument("--retries", type=int, default=3, help="LLM call retry times on timeout/failure")
    parser.add_argument("--backoff", type=float, default=1.0, help="Exponential backoff base seconds")
    parser.add_argument("--resume", action="store_true", help="Resume judging by skipping already-judged pairs and appending outputs")
    parser.add_argument("--log_file", type=str, default=None, help="Optional path to write logs to file")
    args = parser.parse_args()

    # Optional file logging
    if args.log_file:
        ensure_dir(os.path.dirname(args.log_file))
        fh = logging.FileHandler(args.log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(fh)

    out_raw_path = args.out
    out_clean_path = os.path.splitext(args.out)[0] + ".parquet"

    # Build resume key set if requested and existing clean parquet found
    resume_keys: Set[str] | None = None
    if args.resume and os.path.exists(out_clean_path):
        try:
            prev = pd.read_parquet(out_clean_path)
            if {"user_id", "i", "j"}.issubset(set(prev.columns)):
                resume_keys = set(prev["user_id"].astype(str) + "-" + prev["i"].astype(str) + "-" + prev["j"].astype(str))
                logging.info("Loaded %d previously judged pairs for resume", len(resume_keys))
        except Exception as e:
            logging.warning("Failed to load previous clean parquet for resume: %s", e)

    asyncio.run(
        main_async(
            args.config,
            args.pairs,
            out_raw_path,
            out_clean_path,
            limit=args.limit,
            concurrency=args.concurrency,
            retries=args.retries,
            backoff=args.backoff,
            resume_keys=resume_keys,
        )
    )


if __name__ == "__main__":
    main()


