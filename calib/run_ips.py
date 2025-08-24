import argparse
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_inputs(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    proc_dir = cfg["paths"]["proc"]
    props_dir = cfg["paths"].get("props", os.path.join("data", "props"))
    inter_path = os.path.join(proc_dir, "ml1m_interactions.parquet")
    items_path = os.path.join(proc_dir, "ml1m_items.parquet")

    if not os.path.exists(inter_path) or not os.path.exists(items_path):
        raise FileNotFoundError("Expected parquet files not found. Please run scripts/prep_movielens.py first.")

    interactions = pd.read_parquet(inter_path)
    items = pd.read_parquet(items_path)
    return interactions, items, props_dir


def compute_user_entropy(items_df: pd.DataFrame, interactions_df: pd.DataFrame, window: int = 50) -> pd.Series:
    # Use primary genre (first token) for entropy calculation
    item_to_genre = (
        items_df.assign(primary_genre=items_df["genres"].astype(str).str.split("|").str[0].fillna("Unknown"))
        .set_index("item_id")["primary_genre"]
    )

    df = interactions_df.copy()
    df = df.sort_values(["user_id", "ts"])  # oldest to newest
    df["primary_genre"] = df["item_id"].map(item_to_genre)

    user_entropy: Dict[int, float] = {}
    for user_id, g in df.groupby("user_id", sort=False):
        tail = g.tail(window)
        counts = Counter(tail["primary_genre"].dropna().tolist())
        total = sum(counts.values())
        if total == 0:
            H = 0.0
        else:
            probs = np.array([c / total for c in counts.values()], dtype=float)
            H = float(-(probs * np.log(probs + 1e-12)).sum())
        user_entropy[user_id] = H

    return pd.Series(user_entropy, name="user_entropy")


def build_dataset(interactions: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    # item popularity and earliest timestamp
    item_freq = interactions.groupby("item_id").size().rename("item_pop")
    interactions = interactions.join(item_freq, on="item_id")

    # hour and weekday from timestamp (seconds)
    dt = pd.to_datetime(interactions["ts"], unit="s")
    interactions["hour"] = dt.dt.hour.astype(np.int64)
    interactions["weekday"] = dt.dt.weekday.astype(np.int64)

    # user frequency
    user_freq = interactions.groupby("user_id").size().rename("user_freq")
    interactions = interactions.join(user_freq, on="user_id")

    # map age_bin and pop_bin from items
    items_idx = items.set_index("item_id")
    interactions["item_age"] = items_idx["age_bin"].reindex(interactions["item_id"]).astype(np.int64).values
    interactions["pop_bin"] = items_idx["pop_bin"].reindex(interactions["item_id"]).astype(np.int64).values

    # user entropy (global last L records per user)
    ue = compute_user_entropy(items, interactions)
    interactions["user_entropy"] = interactions["user_id"].map(ue).fillna(0.0)

    # weak label: high-popularity items as positive (exposed), others negative
    label = (interactions["pop_bin"] >= 2).astype(int)
    interactions["label"] = label

    features = ["user_freq", "item_pop", "item_age", "hour", "weekday", "user_entropy"]
    return interactions[["user_id", "item_id", *features, "label"]]


def fit_propensity_model(df: pd.DataFrame, random_state: int = 42) -> Tuple[LogisticRegression, float, pd.Series]:
    X = df[["user_freq", "item_pop", "item_age", "hour", "weekday", "user_entropy"]].astype(float)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=200, solver="lbfgs")
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, y_pred)
    except Exception:
        auc = float("nan")

    coefs = pd.Series(model.coef_[0], index=X.columns)
    return model, auc, coefs


def save_outputs(df: pd.DataFrame, model: LogisticRegression, props_dir: str) -> None:
    ensure_dir(props_dir)

    X_all = df[["user_freq", "item_pop", "item_age", "hour", "weekday", "user_entropy"]].astype(float)
    propensity = model.predict_proba(X_all)[:, 1]

    out_pairs = df[["user_id", "item_id"]].copy()
    out_pairs["propensity"] = propensity.astype(float)

    pair_path = os.path.join(props_dir, "propensity.parquet")
    out_pairs.to_parquet(pair_path, index=False)

    item_marginal = out_pairs.groupby("item_id")["propensity"].mean().rename("pop_propensity").reset_index()
    item_path = os.path.join(props_dir, "props_item.parquet")
    item_marginal.to_parquet(item_path, index=False)

    logging.info("Saved: %s (%d rows)", pair_path, len(out_pairs))
    logging.info("Saved: %s (%d rows)", item_path, len(item_marginal))


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Estimate exposure propensity via weakly-supervised logistic regression")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    interactions, items, props_dir = read_inputs(cfg)

    logging.info(
        "Loaded interactions: %d rows; items: %d rows", len(interactions), len(items)
    )

    df = build_dataset(interactions, items)
    logging.info("Built training dataset: %d rows", len(df))

    model, auc, coefs = fit_propensity_model(df)
    if np.isnan(auc) or auc < 0.6:
        logging.warning("AUC=%.4f (<=0.6). This weak-supervision is simplistic; consider better exposure proxies.", auc)
    else:
        logging.info("AUC=%.4f", auc)

    # print feature importances (coefficients)
    coef_str = ", ".join(f"{k}={v:.4f}" for k, v in coefs.items())
    logging.info("Coefficients: %s", coef_str)

    save_outputs(df, model, props_dir)


if __name__ == "__main__":
    main()


