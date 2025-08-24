from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from scipy.stats import kendalltau, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_inv_prob(p: float, *, clip: float = 1e-6) -> float:
    return 1.0 / max(float(p), float(clip))


def _overlap_diagnostics(prop_df: pd.DataFrame) -> Dict[str, float]:
    p = prop_df["propensity"].astype(float).values
    if len(p) == 0:
        return {"min": float("nan"), "p1": float("nan"), "p5": float("nan"), "mean": float("nan"), "pct_<1e-4": float("nan")}
    q = np.quantile(p, [0.01, 0.05])
    out = {
        "min": float(np.min(p)),
        "p1": float(q[0]),
        "p5": float(q[1]),
        "mean": float(np.mean(p)),
        "pct_<1e-4": float(np.mean(p < 1e-4)),
    }
    logging.info("Propensity overlap: min=%.2e, p1=%.2e, p5=%.2e, mean=%.2e, pct<1e-4=%.2f%%", out["min"], out["p1"], out["p5"], out["mean"], 100*out["pct_<1e-4"])
    return out


def _join_propensity_pairwise(judge_df: pd.DataFrame, prop_df: pd.DataFrame) -> pd.DataFrame:
    left = judge_df.merge(prop_df.rename(columns={"item_id": "i"}), on=["user_id", "i"], how="left")
    left = left.rename(columns={"propensity": "pi_ui"})
    left = left.merge(prop_df.rename(columns={"item_id": "j"}), on=["user_id", "j"], how="left")
    left = left.rename(columns={"propensity": "pi_uj"})
    left["pi_ui"] = left["pi_ui"].fillna(1e-6).astype(float)
    left["pi_uj"] = left["pi_uj"].fillna(1e-6).astype(float)
    return left


def _pair_weight(row: pd.Series) -> float:
    return _safe_inv_prob(row["pi_ui"]) + _safe_inv_prob(row["pi_uj"])


def _pair_label(row: pd.Series) -> int:
    # +1 if i wins, -1 if j wins, 0 for tie/invalid
    w = str(row.get("winner", "")).strip().lower()
    if w == "i":
        return +1
    if w == "j":
        return -1
    return 0


def _build_relevance_from_judge(judge_df: pd.DataFrame, p_thresh: float = 0.5) -> pd.DataFrame:
    """Derive per-user relevant items from judge outcomes.

    We treat the winner with confidence p>=p_thresh as a relevant item for that user.
    """
    rows: List[Dict[str, int]] = []
    for r in judge_df.itertuples(index=False):
        winner = str(getattr(r, "winner")).lower().strip()
        p = getattr(r, "p", None)
        if isinstance(p, str):
            try:
                p = float(p)
            except Exception:
                p = None
        if p is None or float(p) < p_thresh:
            continue
        if winner == "i":
            rows.append({"user_id": int(r.user_id), "item_id": int(r.i)})
        elif winner == "j":
            rows.append({"user_id": int(r.user_id), "item_id": int(r.j)})
    rel = pd.DataFrame(rows)
    if rel.empty:
        return rel
    return rel.drop_duplicates(["user_id", "item_id"]).reset_index(drop=True)


def _read_items(proc_dir: str) -> pd.DataFrame:
    items_path = os.path.join(proc_dir, "ml1m_items.parquet")
    if not os.path.exists(items_path):
        raise FileNotFoundError(items_path)
    return pd.read_parquet(items_path)


def _load_model_scores_from_files(files: List[str]) -> Dict[str, pd.DataFrame]:
    """Load model score files. Expect columns: user_id, item_id, score."""
    out: Dict[str, pd.DataFrame] = {}
    for f in files:
        name = Path(f).stem
        if f.lower().endswith(".parquet"):
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f)
        # normalize column names
        cols_lower = {c.lower(): c for c in df.columns}
        df = df.rename(
            columns={
                cols_lower.get("user_id", "user_id"): "user_id",
                cols_lower.get("item_id", "item_id"): "item_id",
                cols_lower.get("score", "score"): "score",
            }
        )
        out[name] = df[["user_id", "item_id", "score"]].copy()
        logging.info("Loaded model scores: %s (%d rows)", name, len(df))
    return out


def _simulate_model_scores(judge_df: pd.DataFrame, items_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Simulate 3 models on the candidate universe (items appearing in judge pairs)."""
    cand_rows: List[Tuple[int, int]] = []
    for r in judge_df.itertuples(index=False):
        cand_rows.append((int(r.user_id), int(r.i)))
        cand_rows.append((int(r.user_id), int(r.j)))
    cand_df = pd.DataFrame(cand_rows, columns=["user_id", "item_id"]).drop_duplicates()

    items_idx = items_df.set_index("item_id")[
        ["pop_bin", "age_bin", "genres"]
    ].copy()
    items_idx["pop_bin"] = items_idx["pop_bin"].fillna(0).astype(int)
    items_idx["age_bin"] = items_idx["age_bin"].fillna(0).astype(int)
    items_idx["genres"] = items_idx["genres"].astype(str)

    tmp = cand_df.join(items_idx, on="item_id")
    user_mode_genre = tmp.groupby("user_id")["genres"].agg(
        lambda s: s.value_counts().index[0] if len(s) else ""
    )

    def score_model_popularity(df: pd.DataFrame, noise: float = 0.3, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        feat = df.join(items_idx, on="item_id")
        s = feat["pop_bin"].astype(float).values + noise * rng.standard_normal(len(feat))
        return pd.DataFrame({"user_id": df["user_id"].values, "item_id": df["item_id"].values, "score": s})

    def score_model_content(df: pd.DataFrame, noise: float = 0.5, seed: int = 1) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        feat = df.join(items_idx, on="item_id")
        age = feat["age_bin"].astype(float).values
        genres = feat["genres"].astype(str).values
        uids = feat["user_id"].values
        match = np.array(
            [
                1.0 if genres[k].split("|")[0] == str(user_mode_genre.get(uids[k], "")) else 0.0
                for k in range(len(genres))
            ]
        )
        s = (0.6 * match + 0.4 * (4 - age)) + noise * rng.standard_normal(len(feat))
        return pd.DataFrame({"user_id": df["user_id"].values, "item_id": df["item_id"].values, "score": s})

    def score_model_random(df: pd.DataFrame, seed: int = 2) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        s = rng.standard_normal(len(df))
        return pd.DataFrame({"user_id": df["user_id"].values, "item_id": df["item_id"].values, "score": s})

    models = {
        "model_pop": score_model_popularity(cand_df, noise=0.3, seed=0),
        "model_content": score_model_content(cand_df, noise=0.5, seed=1),
        "model_random": score_model_random(cand_df, seed=2),
    }
    for k, v in models.items():
        logging.info("Simulated model: %s (%d rows)", k, len(v))
    return models


def _scores_to_topk(df_scores: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    if df_scores.empty:
        return df_scores.head(0)
    df_scores = df_scores.sort_values(["user_id", "score"], ascending=[True, False])
    df_scores["rank"] = df_scores.groupby("user_id").cumcount() + 1
    return df_scores[df_scores["rank"] <= k][["user_id", "item_id", "score"]]


def ips_recall_at_k(
    df_scores: pd.DataFrame,
    relevance: pd.DataFrame,
    prop_df: pd.DataFrame,
    k: int = 20,
    *,
    clip: float = 1e-6,
    snips: bool = False,
    trim_threshold: float | None = None,
) -> float:
    if relevance.empty or df_scores.empty:
        return float("nan")
    topk = _scores_to_topk(df_scores, k=k)
    rel_w = relevance.merge(prop_df, on=["user_id", "item_id"], how="left")
    rel_w["propensity"] = rel_w["propensity"].astype(float).fillna(clip)
    if trim_threshold is not None:
        rel_w = rel_w[rel_w["propensity"] >= float(trim_threshold)]
        if rel_w.empty:
            return float("nan")
    rel_w["w"] = rel_w["propensity"].map(lambda p: _safe_inv_prob(p, clip=clip))
    hit = rel_w.merge(topk[["user_id", "item_id"]], on=["user_id", "item_id"], how="inner")
    if snips:
        w_all = float(rel_w["w"].sum())
        if w_all <= 0:
            return float("nan")
        num = float(hit["w"].sum()) / w_all
        den = float(rel_w["w"].sum()) / w_all
        return num / den if den > 0 else float("nan")
    else:
        num = float(hit["w"].sum())
        den = float(rel_w["w"].sum())
        return num / den if den > 0 else float("nan")


def ips_pair_auc_for_model(
    judge_df: pd.DataFrame,
    prop_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    *,
    clip: float = 1e-6,
    snips: bool = False,
    trim_threshold: float | None = None,
) -> float:
    df = _join_propensity_pairwise(judge_df, prop_df)
    if df.empty:
        return float("nan")
    if trim_threshold is not None:
        df = df[(df["pi_ui"] >= float(trim_threshold)) & (df["pi_uj"] >= float(trim_threshold))]
        if df.empty:
            return float("nan")
    s = scores_df.copy()
    s_i = s.rename(columns={"item_id": "i", "score": "s_i"})[["user_id", "i", "s_i"]]
    s_j = s.rename(columns={"item_id": "j", "score": "s_j"})[["user_id", "j", "s_j"]]
    df = df.merge(s_i, on=["user_id", "i"], how="left").merge(s_j, on=["user_id", "j"], how="left")
    df["s_i"] = df["s_i"].fillna(0.0)
    df["s_j"] = df["s_j"].fillna(0.0)
    df["w"] = df.apply(lambda r: _safe_inv_prob(r["pi_ui"], clip=clip) + _safe_inv_prob(r["pi_uj"], clip=clip), axis=1)
    df["pred_i"] = (df["s_i"] >= df["s_j"]).astype(int)
    lab = df.apply(_pair_label, axis=1)
    df["match"] = np.where(
        lab == 0,
        0.0,
        np.where(((df["pred_i"] == 1) & (lab == +1)) | ((df["pred_i"] == 0) & (lab == -1)), 1.0, 0.0),
    )
    if snips:
        w_all = float(df["w"].sum())
        if w_all <= 0:
            return float("nan")
        num = float((df["w"] * df["match"]).sum()) / w_all
        den = float(df["w"].sum()) / w_all
        return num / den if den > 0 else float("nan")
    else:
        num = float((df["w"] * df["match"]).sum())
        den = float(df["w"].sum())
        return num / den if den > 0 else float("nan")


def dr_recall_at_k(
    df_scores: pd.DataFrame,
    relevance: pd.DataFrame,
    prop_df: pd.DataFrame,
    k: int = 20,
    *,
    clip: float = 1e-6,
    trim_threshold: float | None = None,
) -> float:
    """Doubly-robust recall@k on relevant set with regression adjustment m(score).

    We fit a logistic regression predicting hit (in top-k) from the model score,
    then compute DR: mean(m) + sum(w*(y-m))/sum(w), where w=1/max(pi,clip).
    """
    if relevance.empty or df_scores.empty:
        return float("nan")
    topk = _scores_to_topk(df_scores, k=k)
    rel = relevance.merge(prop_df, on=["user_id", "item_id"], how="left")
    rel["propensity"] = rel["propensity"].astype(float).fillna(clip)
    if trim_threshold is not None:
        rel = rel[rel["propensity"] >= float(trim_threshold)]
        if rel.empty:
            return float("nan")
    # Join scores
    s = df_scores.rename(columns={"score": "s"})[["user_id", "item_id", "s"]]
    rel = rel.merge(s, on=["user_id", "item_id"], how="left")
    rel["s"] = rel["s"].fillna(0.0)
    # Outcome y
    rel = rel.merge(topk[["user_id", "item_id"]].assign(y=1), on=["user_id", "item_id"], how="left")
    rel["y"] = rel["y"].fillna(0).astype(int)
    X = rel[["s"]].astype(float).values
    y = rel["y"].astype(int).values
    try:
        lr = LogisticRegression(max_iter=200, solver="lbfgs")
        lr.fit(X, y)
        m = lr.predict_proba(X)[:, 1]
    except Exception:
        # fallback: sigmoid of standardized score
        svec = rel["s"].astype(float).values
        sstd = (svec - np.mean(svec)) / (np.std(svec) + 1e-6)
        m = 1.0 / (1.0 + np.exp(-sstd))
    w = rel["propensity"].map(lambda p: _safe_inv_prob(p, clip=clip)).astype(float).values
    den = float(np.sum(w))
    if den <= 0:
        return float("nan")
    dr = float(np.mean(m)) + float(np.sum(w * (y - m)) / den)
    return dr


def rjs_tau_for_model(judge_df: pd.DataFrame, prop_df: pd.DataFrame, scores_df: pd.DataFrame) -> float:
    df = _join_propensity_pairwise(judge_df, prop_df)
    if df.empty:
        return float("nan")
    s = scores_df.copy()
    s_i = s.rename(columns={"item_id": "i", "score": "s_i"})[["user_id", "i", "s_i"]]
    s_j = s.rename(columns={"item_id": "j", "score": "s_j"})[["user_id", "j", "s_j"]]
    df = df.merge(s_i, on=["user_id", "i"], how="left").merge(s_j, on=["user_id", "j"], how="left")
    df["s_i"] = df["s_i"].fillna(0.0)
    df["s_j"] = df["s_j"].fillna(0.0)
    df["w"] = df.apply(_pair_weight, axis=1)
    df["pred_i"] = (df["s_i"] >= df["s_j"]).astype(int)
    lab = df.apply(_pair_label, axis=1)
    df["match"] = np.where(
        lab == 0,
        0.0,
        np.where(((df["pred_i"] == 1) & (lab == +1)) | ((df["pred_i"] == 0) & (lab == -1)), 1.0, 0.0),
    )
    per_user = df.groupby("user_id").apply(
        lambda g: pd.Series({
            "sum_w": float(g["w"].sum()),
            "a": float(((g["w"] * g["match"]).sum()) / max(g["w"].sum(), 1e-12)),
        })
    )
    if per_user.empty:
        return float("nan")
    per_user["tau_u"] = 2.0 * per_user["a"] - 1.0
    num = float((per_user["sum_w"] * per_user["tau_u"]).sum())
    den = float(per_user["sum_w"].sum())
    return num / den if den > 0 else float("nan")


def evaluate_models(
    judge_df: pd.DataFrame,
    prop_df: pd.DataFrame,
    items_df: pd.DataFrame,
    model_files: Optional[List[str]] = None,
    *,
    clip: float = 1e-6,
    clip_alt: float = 1e-4,
    trim_threshold: float | None = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    if model_files:
        scores_map = _load_model_scores_from_files(model_files)
    else:
        scores_map = _simulate_model_scores(judge_df, items_df)

    relevance = _build_relevance_from_judge(judge_df, p_thresh=0.5)
    _overlap_diagnostics(prop_df)

    rows: List[Dict[str, float]] = []
    for name, df_scores in scores_map.items():
        ips_auc = ips_pair_auc_for_model(judge_df, prop_df, df_scores, clip=clip, snips=False, trim_threshold=trim_threshold)
        ips_auc_snips = ips_pair_auc_for_model(judge_df, prop_df, df_scores, clip=clip, snips=True, trim_threshold=trim_threshold)
        ips_auc_clip_alt = ips_pair_auc_for_model(judge_df, prop_df, df_scores, clip=clip_alt, snips=False, trim_threshold=trim_threshold)

        ips_rec20 = ips_recall_at_k(df_scores, relevance, prop_df, k=20, clip=clip, snips=False, trim_threshold=trim_threshold)
        ips_rec20_snips = ips_recall_at_k(df_scores, relevance, prop_df, k=20, clip=clip, snips=True, trim_threshold=trim_threshold)
        ips_rec20_clip_alt = ips_recall_at_k(df_scores, relevance, prop_df, k=20, clip=clip_alt, snips=False, trim_threshold=trim_threshold)
        ips_rec20_dr = dr_recall_at_k(df_scores, relevance, prop_df, k=20, clip=clip, trim_threshold=trim_threshold)

        rjs_tau = rjs_tau_for_model(judge_df, prop_df, df_scores)
        rows.append({
            "model": name,
            "ips_pair_auc": ips_auc,
            "ips_pair_auc_snips": ips_auc_snips,
            "ips_pair_auc_clip1e-4": ips_auc_clip_alt,
            "ips_recall@20": ips_rec20,
            "ips_recall@20_snips": ips_rec20_snips,
            "ips_recall@20_clip1e-4": ips_rec20_clip_alt,
            "ips_recall@20_dr": ips_rec20_dr,
            "rjs_tau": rjs_tau,
        })
        logging.info(
            "Model=%s | PairAUC_IPS=%.4f (snips=%.4f, clip1e-4=%.4f) | IPS-Recall@20=%.4f (snips=%.4f, clip1e-4=%.4f, DR=%.4f) | RJS=%.4f",
            name, ips_auc, ips_auc_snips, ips_auc_clip_alt, ips_rec20, ips_rec20_snips, ips_rec20_clip_alt, ips_rec20_dr, rjs_tau,
        )

    metrics = pd.DataFrame(rows)
    return metrics, scores_map


def correlate_and_calibrate(metrics: pd.DataFrame, fig_path: str, *, y_col: str, k_label: int) -> Dict[str, float]:
    x = metrics["rjs_tau"].values
    if y_col not in metrics.columns:
        logging.warning("y_col=%s not in metrics; falling back to 'ips_recall@20'", y_col)
    col = y_col if y_col in metrics.columns else "ips_recall@20"
    y = metrics[col].values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        logging.warning("Not enough points for correlation/calibration.")
        return {"spearman": float("nan"), "kendall": float("nan"), "r2": float("nan")}

    rho, _ = spearmanr(x, y)
    tau, _ = kendalltau(x, y)

    iso = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0, out_of_bounds="clip")
    y_pred = iso.fit_transform(x, y)
    r2 = r2_score(y, y_pred)

    ensure_dir(os.path.dirname(fig_path))
    xs = np.linspace(min(x), max(x), 200)
    ys = iso.predict(xs)
    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, c="tab:blue", label="models")
    plt.plot(xs, ys, c="tab:red", label="isotonic g(RJS)")
    plt.xlabel("RJS (tau_w)")
    plt.ylabel(f"IPS-Recall@{k_label}")
    plt.title("Calibration: RJS -> IPS-Recall@20")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    logging.info("Correlation: Spearman=%.4f | Kendall=%.4f | R^2=%.4f", rho, tau, r2)
    return {"spearman": float(rho), "kendall": float(tau), "r2": float(r2)}


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Correlation and calibration between RJS and IPS metrics")
    parser.add_argument("--config", type=str, required=True, help="Path to configs/movielens1m.yaml")
    parser.add_argument(
        "--judge",
        type=str,
        default=os.path.join("data", "judge", "judge_clean.parquet"),
        help="Path to cleaned judge parquet (user_id,i,j,winner,p,valid_ratio)",
    )
    parser.add_argument(
        "--props",
        type=str,
        default=os.path.join("data", "props", "propensity.parquet"),
        help="Path to user-item propensity parquet (user_id,item_id,propensity)",
    )
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Optional list of model score files")
    parser.add_argument("--k", type=int, default=20, help="top-k for recall evaluation")
    parser.add_argument("--clip", type=float, default=1e-6, help="smallest propensity for IPS")
    parser.add_argument("--clip_alt", type=float, default=1e-4, help="alternative clip for sensitivity")
    parser.add_argument("--trim", type=float, default=1e-4, help="propensity trimming threshold (None to disable)")
    parser.add_argument(
        "--out_csv", type=str, default=os.path.join("reports", "tables", "corr_ml1m.csv"), help="Output CSV path"
    )
    parser.add_argument(
        "--out_fig", type=str, default=os.path.join("reports", "figs", "calib_isotonic.png"), help="Output figure path"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    proc_dir = cfg["paths"]["proc"]

    if not os.path.exists(args.judge):
        raise FileNotFoundError(args.judge)
    if not os.path.exists(args.props):
        raise FileNotFoundError(args.props)

    judge_df = pd.read_parquet(args.judge)
    # Normalize columns from potential variants
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

    judge_df["user_id"] = judge_df["user_id"].astype(int)
    judge_df["i"] = judge_df["i"].astype(int)
    judge_df["j"] = judge_df["j"].astype(int)

    prop_df = pd.read_parquet(args.props)[["user_id", "item_id", "propensity"]]
    items_df = _read_items(proc_dir)

    metrics, _ = evaluate_models(
        judge_df,
        prop_df,
        items_df,
        model_files=args.models,
        clip=float(args.clip),
        clip_alt=float(args.clip_alt),
        trim_threshold=float(args.trim) if args.trim is not None else None,
    )

    # Rename y columns to selected k
    k = int(args.k)
    if "ips_recall@20" in metrics.columns and k != 20:
        metrics[f"ips_recall@{k}"] = metrics["ips_recall@20"]
    y_col = f"ips_recall@{k}" if f"ips_recall@{k}" in metrics.columns else "ips_recall@20"
    # auto-rename output files with k
    out_fig = args.out_fig
    out_csv = args.out_csv
    if k != 20:
        root_f, ext_f = os.path.splitext(out_fig)
        root_c, ext_c = os.path.splitext(out_csv)
        out_fig = f"{root_f}_k{k}{ext_f}"
        out_csv = f"{root_c}_k{k}{ext_c}"

    stats = correlate_and_calibrate(metrics, fig_path=out_fig, y_col=y_col, k_label=k)

    ensure_dir(os.path.dirname(args.out_csv))
    out = metrics.copy()
    out["spearman_rjs_vs_ips_recall20"] = stats["spearman"]
    out["kendall_rjs_vs_ips_recall20"] = stats["kendall"]
    out["r2_isotonic_rjs_to_ips_recall20"] = stats["r2"]
    out.to_csv(out_csv, index=False)
    logging.info("Saved correlation table to %s", out_csv)


if __name__ == "__main__":
    main()


