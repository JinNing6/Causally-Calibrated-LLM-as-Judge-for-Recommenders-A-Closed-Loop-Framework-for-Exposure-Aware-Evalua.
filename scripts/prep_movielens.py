import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directory_exists(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_parquet_engine() -> str:
    try:
        import pyarrow  # noqa: F401

        return "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401

            return "fastparquet"
        except Exception as exc:
            raise RuntimeError(
                "Parquet engine not found. Please install 'pyarrow' or 'fastparquet'."
            ) from exc


def read_movielens_ratings(raw_dir: str) -> pd.DataFrame:
    ratings_path = os.path.join(raw_dir, "ratings.dat")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"ratings.dat not found at: {ratings_path}")

    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "item_id", "rating", "ts"],
        dtype={"user_id": np.int64, "item_id": np.int64, "rating": np.float64, "ts": np.int64},
    )
    # Keep only required columns
    df = df[["user_id", "item_id", "ts"]]
    return df


def read_movielens_movies(raw_dir: str) -> pd.DataFrame:
    movies_path = os.path.join(raw_dir, "movies.dat")
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"movies.dat not found at: {movies_path}")

    # MovieLens 1M titles may contain non-UTF8 bytes; ISO-8859-1 works in practice
    df = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["item_id", "title", "genres"],
        dtype={"item_id": np.int64, "title": str, "genres": str},
        encoding="ISO-8859-1",
    )
    return df


def assign_quantile_bins_by_rank(values: pd.Series, num_bins: int) -> pd.Series:
    if len(values) == 0:
        return pd.Series(dtype=np.int64)

    rank_pct = values.rank(method="average", pct=True)
    # Convert (0, 1] to integer bins in [0, num_bins-1]
    bins = np.floor(rank_pct * num_bins).astype(int)
    bins = bins.clip(lower=0, upper=num_bins - 1)
    return pd.Series(bins, index=values.index, dtype=np.int64)


def compute_item_bins(interactions: pd.DataFrame, num_pop_bins: int, num_age_bins: int) -> Tuple[pd.Series, pd.Series]:
    # Popularity: frequency per item
    item_freq = interactions.groupby("item_id").size().rename("freq")
    pop_bin = assign_quantile_bins_by_rank(item_freq, num_pop_bins)

    # Age: earliest timestamp per item (smaller ts => older)
    item_first_ts = interactions.groupby("item_id")["ts"].min().rename("first_ts")
    age_bin = assign_quantile_bins_by_rank(item_first_ts, num_age_bins)

    return pop_bin, age_bin


def log_basic_stats(interactions: pd.DataFrame, pop_bin: pd.Series, age_bin: pd.Series) -> None:
    num_interactions = len(interactions)
    num_users = interactions["user_id"].nunique()
    num_items = interactions["item_id"].nunique()
    ts_min = int(interactions["ts"].min()) if num_interactions > 0 else None
    ts_max = int(interactions["ts"].max()) if num_interactions > 0 else None

    logging.info(
        "Interactions: %d | Users: %d | Items: %d | ts[min,max]=[%s,%s]",
        num_interactions,
        num_users,
        num_items,
        ts_min,
        ts_max,
    )

    pop_counts = pop_bin.value_counts().sort_index().to_dict()
    age_counts = age_bin.value_counts().sort_index().to_dict()
    logging.info("pop_bin distribution: %s", pop_counts)
    logging.info("age_bin distribution: %s", age_counts)


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Prepare MovieLens-1M dataset: interactions and item metadata with bins")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config, e.g., configs/movielens1m.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = cfg["paths"]["raw"]
    proc_dir = cfg["paths"]["proc"]
    num_pop_bins = int(cfg.get("prep", {}).get("pop_bins", 5))
    num_age_bins = int(cfg.get("prep", {}).get("age_bins", 5))

    ensure_directory_exists(proc_dir)

    logging.info("Reading raw ratings and movies from %s", raw_dir)
    ratings = read_movielens_ratings(raw_dir)
    movies = read_movielens_movies(raw_dir)

    # Interactions parquet
    interactions = ratings[["user_id", "item_id", "ts"]].copy()
    interactions.sort_values(["user_id", "ts", "item_id"], inplace=True)

    # Compute bins
    pop_bin, age_bin = compute_item_bins(interactions, num_pop_bins, num_age_bins)
    log_basic_stats(interactions, pop_bin, age_bin)

    # Items parquet: merge title/genres and attach bins, plus price_bin=0 placeholder
    items = movies.set_index("item_id").copy()
    items["pop_bin"] = pop_bin
    items["age_bin"] = age_bin
    items["price_bin"] = 0
    items.reset_index(inplace=True)

    # Ensure dtypes
    items["item_id"] = items["item_id"].astype(np.int64)
    items["title"] = items["title"].astype(str)
    items["genres"] = items["genres"].astype(str)
    items["pop_bin"] = items["pop_bin"].fillna(0).astype(np.int64)
    items["age_bin"] = items["age_bin"].fillna(0).astype(np.int64)
    items["price_bin"] = items["price_bin"].astype(np.int64)

    engine = get_parquet_engine()

    interactions_out = os.path.join(proc_dir, "ml1m_interactions.parquet")
    items_out = os.path.join(proc_dir, "ml1m_items.parquet")

    logging.info("Writing interactions to %s", interactions_out)
    interactions.to_parquet(interactions_out, index=False, engine=engine)

    logging.info("Writing items to %s", items_out)
    items[["item_id", "title", "genres", "pop_bin", "age_bin", "price_bin"]].to_parquet(
        items_out, index=False, engine=engine
    )

    logging.info("Done. Interactions: %d rows; Items: %d rows", len(interactions), len(items))


if __name__ == "__main__":
    main()


