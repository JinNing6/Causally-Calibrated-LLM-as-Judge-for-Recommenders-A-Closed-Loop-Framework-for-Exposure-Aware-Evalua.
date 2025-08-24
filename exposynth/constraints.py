from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


def _primary_genre(genre_str: str) -> str:
    if not isinstance(genre_str, str) or len(genre_str) == 0:
        return ""
    return genre_str.split("|")[0].strip()


def compute_genre_transition_matrix(
    interactions: pd.DataFrame, items: pd.DataFrame
) -> pd.DataFrame:
    """Estimate genre transition probabilities from user sequences.

    Returns a DataFrame with index=src_genre, columns=dst_genre, values=P(dst|src).
    """
    items_idx = items.set_index("item_id").copy()
    items_idx["primary_genre"] = items_idx["genres"].astype(str).map(_primary_genre)

    df = interactions.sort_values(["user_id", "ts"]).copy()
    df["primary_genre"] = df["item_id"].map(items_idx["primary_genre"])  # type: ignore[index]

    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    prev_by_user: Dict[int, str] = {}
    for r in df.itertuples(index=False):
        uid = int(getattr(r, "user_id"))
        g = str(getattr(r, "primary_genre", ""))
        if not g:
            continue
        if uid in prev_by_user:
            counts[(prev_by_user[uid], g)] += 1
        prev_by_user[uid] = g

    if not counts:
        return pd.DataFrame(dtype=float)

    # Build matrix
    src_genres = sorted({k[0] for k in counts.keys()})
    dst_genres = sorted({k[1] for k in counts.keys()})
    mat = pd.DataFrame(0, index=src_genres, columns=dst_genres, dtype=float)
    for (ga, gb), c in counts.items():
        mat.loc[ga, gb] += float(c)
    # row-normalize
    rs = mat.sum(axis=1)
    rs = rs.replace(0, np.nan)
    mat = mat.div(rs, axis=0).fillna(0.0)
    return mat


def compute_item_similarity_topk(
    interactions: pd.DataFrame, *, topk: int = 50
) -> Dict[int, Dict[int, float]]:
    """Cosine similarity over user-item incidence; return top-k neighbors per item.

    Returns: {item_id: {neighbor_item_id: sim, ...}}
    """
    # Build item-user sparse matrix
    users = interactions["user_id"].astype(int).to_numpy()
    items = interactions["item_id"].astype(int).to_numpy()
    uniq_users, user_idx = np.unique(users, return_inverse=True)
    uniq_items, item_idx = np.unique(items, return_inverse=True)

    nnz = len(users)
    data = np.ones(nnz, dtype=np.float32)
    iu = sparse.coo_matrix((data, (item_idx, user_idx)), shape=(len(uniq_items), len(uniq_users)))
    iu = iu.tocsr()

    # Compute cosine similarity in batches to limit memory
    sim_map: Dict[int, Dict[int, float]] = {}
    batch = 512
    for start in range(0, iu.shape[0], batch):
        end = min(start + batch, iu.shape[0])
        sims = cosine_similarity(iu[start:end], iu)
        for local_i in range(end - start):
            i = start + local_i
            row = sims[local_i]
            # Exclude self and get topk
            row[i] = -1.0
            top_idx = np.argpartition(row, -topk)[-topk:]
            top_idx = top_idx[np.argsort(row[top_idx])][::-1]
            ii = int(uniq_items[i])
            sim_map[ii] = {int(uniq_items[j]): float(row[j]) for j in top_idx if row[j] > 0.0}
    return sim_map


class Constraints:
    def __init__(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame,
        *,
        genre_prob_thresh: float = 0.02,
        sim_thresh: float = 0.05,
        topk_neighbors: int = 50,
    ) -> None:
        self.items = items.copy()
        self.items.set_index("item_id", inplace=True)
        self.items["primary_genre"] = self.items["genres"].astype(str).map(_primary_genre)

        self.T = compute_genre_transition_matrix(interactions, items)
        self.sim_map = compute_item_similarity_topk(interactions, topk=topk_neighbors)
        self.genre_prob_thresh = float(genre_prob_thresh)
        self.sim_thresh = float(sim_thresh)

    def _genre(self, item_id: int) -> str:
        try:
            return str(self.items.loc[int(item_id), "primary_genre"])  # type: ignore[index]
        except Exception:
            return ""

    def _genre_prob(self, ga: str, gb: str) -> float:
        if ga in self.T.index and gb in self.T.columns:
            return float(self.T.loc[ga, gb])
        return 0.0

    def _sim(self, i: int, j: int) -> float:
        row = self.sim_map.get(int(i))
        if not row:
            return 0.0
        return float(row.get(int(j), 0.0))

    def is_valid_transition(self, prev_item: int | None, next_item: int) -> bool:
        if prev_item is None:
            return True
        if int(prev_item) == int(next_item):
            return False
        ga = self._genre(prev_item)
        gb = self._genre(next_item)
        prob = self._genre_prob(ga, gb) if ga and gb else 0.0
        sim = self._sim(prev_item, next_item)
        # Accept if either genre transition is plausible or items are sufficiently similar
        return (prob >= self.genre_prob_thresh) or (sim >= self.sim_thresh)


__all__ = [
    "Constraints",
    "compute_genre_transition_matrix",
    "compute_item_similarity_topk",
]


