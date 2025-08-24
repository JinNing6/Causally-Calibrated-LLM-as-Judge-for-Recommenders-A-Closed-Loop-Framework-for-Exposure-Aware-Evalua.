## LURE — LLM-based Unbiasing & Rationale-grounded Evaluation/Data Generation for Recsys

LURE 提供一套用于推荐系统的偏置消除与基于理由的评测/数据生成流水线，包含三个模块：

- **RecJudge-X（暴露感知 LLM 评测）**: 通过引入曝光建模，利用 LLM 对推荐结果进行更公平的成对评测与解释抽取。
- **ExpoSynth（曝光可控合成）**: 基于约束的合成数据生成，使不同曝光/人群条件下的数据可控并可复现。
- **Calib-Link（单调回归对齐线上指标）**: 使用单调回归将离线指标与线上指标进行校准，提升离线评测与线上效果的一致性。

### 快速开始（7 条命令流水线）

请先确保数据已按下文路径放置，并已安装依赖（见“环境与安装”）。按顺序执行以下命令：

```bash
python scripts/prep_movielens.py
python calib/run_ips.py
python -c "from recjudge.pair_sampler import sample_pairs; sample_pairs('data/proc/ml1m_interactions.parquet','data/proc/ml1m_items.parquet',10,60,['pop_bin','age_bin'],'data/judge/pairs.parquet')"
python recjudge/run_judge.py --pairs data/judge/pairs.parquet --out data/judge/judge_raw.jsonl --limit 200
python recjudge/rjs_metric.py
python exposynth/run_synth.py --config configs/movielens1m.yaml
python calib/run_correlation.py
```

### 数据放置路径说明

- 原始数据请放置于 `data/raw/ml-1m/...`，例如：
  - `data/raw/ml-1m/ratings.dat`
  - `data/raw/ml-1m/movies.dat`
  - 运行 `scripts/prep_movielens.py` 会在 `data/proc/` 下生成处理后的交互与物品属性文件。

### 结果产物清单

- 表格类结果：`reports/tables/*.csv`
- 图形类结果：`reports/figs/*.png`

### 论文

- LaTeX 初稿位于 `paper/` 目录，可用 `pdflatex` 或 `xelatex` 编译。

### 环境与安装

- **环境**: Python >= 3.10
- **安装**:

```bash
pip install -r requirements.txt
```

安装完成后，按“快速开始”执行整套流程即可复现核心实验与产物。
