from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _safe_inv_prob(p: float) -> float:
    clipped = max(float(p), 1e-6)
    return 1.0 / clipped


def _join_propensity(judge_df: pd.DataFrame, prop_df: pd.DataFrame) -> pd.DataFrame:
    # left join for i and j sides
    left = judge_df.merge(
        prop_df.rename(columns={"item_id": "i"}), how="left", on=["user_id", "i"], suffixes=("", "")
    )
    left = left.rename(columns={"propensity": "pi_ui"})

    left = left.merge(
        prop_df.rename(columns={"item_id": "j"}), how="left", on=["user_id", "j"], suffixes=("", "")
    )
    left = left.rename(columns={"propensity": "pi_uj"})

    # fill missing (unseen user-item) with small prob to avoid zero-div
    left["pi_ui"] = left["pi_ui"].fillna(1e-6).astype(float)
    left["pi_uj"] = left["pi_uj"].fillna(1e-6).astype(float)
    return left


def _pair_weight(row: pd.Series) -> float:
    return _safe_inv_prob(row["pi_ui"]) + _safe_inv_prob(row["pi_uj"])


def _pair_match(row: pd.Series) -> float:
    # p>=0.5 => predict i>j; else j>i. Ties in winner count as 0 match.
    pred_i = bool(float(row.get("p", 0.5)) >= 0.5)
    winner = str(row.get("winner", "")).strip().lower()
    if winner not in {"i", "j"}:
        return 0.0
    return 1.0 if (pred_i and winner == "i") or ((not pred_i) and winner == "j") else 0.0


def pair_auc_ips(judge_df: pd.DataFrame, prop_df: pd.DataFrame) -> float:
    """Compute IPS-weighted pair agreement (PairAUC_IPS).

    For each (u,i,j): w = 1/max(pi_ui,1e-6) + 1/max(pi_uj,1e-6).
    Agreement if (p>=0.5 and winner=='i') or (p<0.5 and winner=='j').
    Returns weighted agreement rate.
    """
    df = _join_propensity(judge_df, prop_df)
    if df.empty:
        return float("nan")
    df["w"] = df.apply(_pair_weight, axis=1)
    df["match"] = df.apply(_pair_match, axis=1)
    num = float((df["w"] * df["match"]).sum())
    den = float(df["w"].sum())
    return num / den if den > 0 else float("nan")


def _sample_user_pairs(df: pd.DataFrame, M: int, rng: random.Random) -> pd.DataFrame:
    """Sample up to M rows per user uniformly from their available pairs."""
    out_parts: list[pd.DataFrame] = []
    for user_id, g in df.groupby("user_id", sort=False):
        if len(g) <= M:
            out_parts.append(g)
        else:
            idx = rng.sample(range(len(g)), M)
            out_parts.append(g.iloc[idx])
    if not out_parts:
        return df.head(0)
    return pd.concat(out_parts, axis=0, ignore_index=True)


def rjs_tau(judge_df: pd.DataFrame, prop_df: pd.DataFrame, M: int = 200) -> float:
    """Compute Kendall-like tau_w via IPS-weighted agreement per user.

    Steps:
      - For each user, sample up to M pairs
      - Compute IPS weights and agreement for those pairs
      - User agreement a_u = sum(w*match)/sum(w)
      - Map to tau_u = 2*a_u - 1 ∈ [-1, 1]
      - Return weighted average over users with weights sum_w_u
    """
    rng = random.Random(42)
    df_all = _join_propensity(judge_df, prop_df)
    if df_all.empty:
        return float("nan")

    df_s = _sample_user_pairs(df_all, M, rng)
    if df_s.empty:
        return float("nan")

    df_s["w"] = df_s.apply(_pair_weight, axis=1)
    df_s["match"] = df_s.apply(_pair_match, axis=1)

    per_user = (
        df_s.groupby("user_id").apply(lambda g: pd.Series({
            "sum_w": float(g["w"].sum()),
            "a": float(((g["w"] * g["match"]).sum()) / max(g["w"].sum(), 1e-12)),
        }))
    )
    if per_user.empty:
        return float("nan")

    per_user["tau_u"] = 2.0 * per_user["a"] - 1.0
    num = float((per_user["sum_w"] * per_user["tau_u"]).sum())
    den = float(per_user["sum_w"].sum())
    return num / den if den > 0 else float("nan")


def _attach_pair_pop_bin(judge_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    items_idx = items_df.set_index("item_id")["pop_bin"].astype(int)
    pop_i = judge_df["i"].map(items_idx)
    pop_j = judge_df["j"].map(items_idx)
    # Pair bucket: max of the two items' pop_bin
    pair_bin = np.maximum(pop_i.fillna(-1).astype(int).values, pop_j.fillna(-1).astype(int).values)
    out = judge_df.copy()
    out["pair_pop_bin"] = pair_bin
    return out


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Compute RJS metrics (PairAUC_IPS and Kendall-like tau_w)")
    parser.add_argument("--judge", type=str, default=os.path.join("data", "judge", "judge_raw.parquet"))
    parser.add_argument("--props", type=str, default=os.path.join("data", "props", "propensity.parquet"))
    parser.add_argument(
        "--items", type=str, default=os.path.join("data", "proc", "ml1m_items.parquet"), help="Items parquet for pop_bin"
    )
    parser.add_argument("--out_csv", type=str, default=os.path.join("reports", "tables", "rjs_ml1m.csv"))
    parser.add_argument("--M", type=int, default=200)
    args = parser.parse_args()

    # judge file may be named judge_clean.parquet per instruction
    judge_path = args.judge
    if not os.path.exists(judge_path):
        alt = os.path.join("data", "judge", "judge_clean.parquet")
        if os.path.exists(alt):
            judge_path = alt

    if not os.path.exists(judge_path):
        raise FileNotFoundError(judge_path)
    if not os.path.exists(args.props):
        raise FileNotFoundError(args.props)

    judge_df = pd.read_parquet(judge_path)
    prop_df = pd.read_parquet(args.props)

    # Robust column normalization
    rename_map: Dict[str, str] = {}
    for col in list(judge_df.columns):
        lc = col.lower()
        if lc == "user_id":
            rename_map[col] = "user_id"
        elif lc in {"i", "item_i", "cand_i"}:
            rename_map[col] = "i"
        elif lc in {"j", "item_j", "cand_j"}:
            rename_map[col] = "j"
        elif lc == "winner":
            rename_map[col] = "winner"
        elif lc == "p":
            rename_map[col] = "p"
    if rename_map:
        judge_df = judge_df.rename(columns=rename_map)

    # Ensure dtypes
    judge_df["user_id"] = judge_df["user_id"].astype(int)
    judge_df["i"] = judge_df["i"].astype(int)
    judge_df["j"] = judge_df["j"].astype(int)

    # Metrics (overall)
    pair_auc = pair_auc_ips(judge_df, prop_df)
    tau_w = rjs_tau(judge_df, prop_df, M=int(args.M))

    logging.info("PairAUC_IPS (all) = %.4f", pair_auc)
    logging.info("RJS tau_w (all)   = %.4f", tau_w)

    # Per-pop_bin RJS
    rows: list[Dict] = []
    rows.append({"bucket": "ALL", "pair_auc_ips": float(pair_auc), "rjs_tau": float(tau_w)})

    try:
        if os.path.exists(args.items):
            items_df = pd.read_parquet(args.items)[["item_id", "pop_bin"]]
            judge_with_bin = _attach_pair_pop_bin(judge_df, items_df)

            for b, g in judge_with_bin.groupby("pair_pop_bin", sort=True):
                if int(b) < 0:
                    continue
                pair_auc_b = pair_auc_ips(g, prop_df)
                tau_b = rjs_tau(g, prop_df, M=int(args.M))
                logging.info("RJS by pop_bin=%s: tau_w=%.4f | PairAUC_IPS=%.4f", b, tau_b, pair_auc_b)
                rows.append({"bucket": f"pop_bin={int(b)}", "pair_auc_ips": float(pair_auc_b), "rjs_tau": float(tau_b)})
        else:
            logging.warning("Items file not found: %s. Skipping per-pop_bin RJS.", args.items)
    except Exception as e:
        logging.warning("Failed per-pop_bin RJS: %s", e)

    # Save CSV
    out_csv = args.out_csv
    ensure_dir(os.path.dirname(out_csv))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logging.info("Saved metrics table to %s", out_csv)


if __name__ == "__main__":
    main()


