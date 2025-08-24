from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


def build_history_block(hist_ids: list[int] | Iterable[int] | None, items_df: pd.DataFrame) -> str:
    """Build a compact history text from item ids (reverse order, newest first).

    Includes a placeholder for time difference since exact timestamps are not provided here.
    """
    if hist_ids is None:
        return "(无历史)"

    # Normalize to Python list
    if isinstance(hist_ids, (pd.Series, np.ndarray)):
        ids_list: List[int] = [int(x) for x in list(hist_ids)]
    else:
        ids_list = [int(x) for x in list(hist_ids)]
    if len(ids_list) == 0:
        return "(无历史)"

    items_indexed = items_df.set_index("item_id")
    lines: list[str] = []
    for rank, iid in enumerate(reversed(ids_list), start=1):
        if iid not in items_indexed.index:
            continue
        row = items_indexed.loc[iid]
        title = str(row.get("title", ""))
        genres = str(row.get("genres", ""))
        # 时间差占位（未知 ts，用占位提示）
        lines.append(f"H{rank}: {title} | {genres} [Δt≈?]")
    return "\n".join(reversed(lines))  # 倒序：最新在最上方


def build_item_block(item_id: int, items_df: pd.DataFrame) -> str:
    items_indexed = items_df.set_index("item_id")
    if item_id not in items_indexed.index:
        raise KeyError(f"item_id={item_id} 不存在于 items_df")
    row = items_indexed.loc[item_id]
    title = str(row.get("title", ""))
    genres = str(row.get("genres", ""))
    pop_bin = int(row.get("pop_bin", 0))
    age_bin = int(row.get("age_bin", 0))
    return f"Title: {title}\nCategory: {genres}\nPopBin: {pop_bin}\nAgeBin: {age_bin}"


PAIR_TEMPLATE = (
    """
你将作为偏好评审员，对比两个候选物品 i 与 j，基于提供的用户历史与物品信息进行判别。

严格输出 JSON（不要任何多余文字、解释或 Markdown）：
{{"winner":"i|j|tie","p":0.0~1.0,
  "rationales":[{{"aspect":"...", "evidence":"...", "source":"history|i|j"}}]}}

必须遵守：
- 证据必须为输入子串；不允许编造，禁止虚构标题或类别。
- "winner" 只能取 "i"、"j" 或 "tie"；"p" 表示对 winner 的置信度（0~1）。
- "rationales" 至少给出两条，分别明确 aspect 与来自何处的 evidence，并标注 source。

[Inputs]
<User History>
{history_block}

<Candidate i>
{i_block}

<Candidate j>
{j_block}
    """
    .strip()
)


def render_pair_prompt(history_block: str, i_block: str, j_block: str) -> str:
    return PAIR_TEMPLATE.format(history_block=history_block, i_block=i_block, j_block=j_block)


def demo_render() -> str:
    # 构造一个最小可运行示例
    items_df = pd.DataFrame(
        [
            {"item_id": 1, "title": "Toy Story (1995)", "genres": "Animation|Children's|Comedy", "pop_bin": 4, "age_bin": 4},
            {"item_id": 2, "title": "Jumanji (1995)", "genres": "Adventure|Children's|Fantasy", "pop_bin": 3, "age_bin": 4},
            {"item_id": 3, "title": "Grumpier Old Men (1995)", "genres": "Comedy|Romance", "pop_bin": 2, "age_bin": 4},
            {"item_id": 4, "title": "Waiting to Exhale (1995)", "genres": "Comedy|Drama", "pop_bin": 2, "age_bin": 4},
        ]
    )

    hist_ids = [1, 3, 2]
    cand_i, cand_j = 4, 2

    history_block = build_history_block(hist_ids=hist_ids, items_df=items_df)
    i_block = build_item_block(cand_i, items_df)
    j_block = build_item_block(cand_j, items_df)
    return render_pair_prompt(history_block, i_block, j_block)


if __name__ == "__main__":
    print(demo_render())


