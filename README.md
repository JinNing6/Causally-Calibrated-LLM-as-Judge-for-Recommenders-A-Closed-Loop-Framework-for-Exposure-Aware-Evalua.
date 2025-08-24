## Causally Calibrated LLM-as-Judge for Recommenders: A Closed-Loop Framework for Exposure-Aware Evaluation and Synthetic Interaction Generation

本仓库实现一个面向推荐系统的闭环框架：以 LLM 作为评测器并进行因果校准，贯穿“曝光感知评测 → 曝光可控合成 → 离线-线上指标单调校准”的全流程。框架核心由三大模块构成：

- **RecJudge-X（暴露感知 LLM 评测）**: 在成对比较中显式建模曝光与人群条件，利用 LLM 进行解释抽取与更公平的偏置感知评测。
- **ExpoSynth（曝光可控合成）**: 基于约束条件的合成器，按曝光、人群或属性约束生成可复现的交互数据，支持针对性尾部增强。
- **Calib-Link（单调回归对齐线上指标）**: 以单调回归将离线指标与线上指标进行因果一致性校准，提升离线评测与线上效果的一致性。

### 快速开始（7 条命令流水线）

按顺序执行以下命令即可跑通从数据准备到评测、合成与指标校准的完整链路（先完成“环境与安装”与“数据放置”）：

```bash
python scripts/prep_movielens.py
python calib/run_ips.py
python -c "from recjudge.pair_sampler import sample_pairs; sample_pairs('data/proc/ml1m_interactions.parquet','data/proc/ml1m_items.parquet',10,60,['pop_bin','age_bin'],'data/judge/pairs.parquet')"
python recjudge/run_judge.py --pairs data/judge/pairs.parquet --out data/judge/judge_raw.jsonl --limit 200
python recjudge/rjs_metric.py
python exposynth/run_synth.py --config configs/movielens1m.yaml
python calib/run_correlation.py
```

### 数据放置

- 将 MovieLens-1M 原始文件放入 `data/raw/ml-1m/`：
  - `data/raw/ml-1m/ratings.dat`
  - `data/raw/ml-1m/movies.dat`
- 运行 `scripts/prep_movielens.py` 后，会在 `data/proc/` 生成处理后的交互与物品属性文件。

### 结果产物

- 表格结果输出在 `reports/tables/*.csv`
- 图形结果输出在 `reports/figs/*.png`

### 环境与安装

- **环境**: Python >= 3.10
- **安装**:

```bash
pip install -r requirements.txt
```

安装完成后，直接执行“快速开始”部分的 7 条命令即可复现主要实验流程与产物。

### 目录结构（节选）

```
configs/                # 运行配置（示例：MovieLens-1M）
calib/                  # IPS、单调回归关联及相关分析脚本
recjudge/               # LLM 评测器、成对采样与指标计算
exposynth/              # 曝光可控合成与约束定义
data/                   # 原始与处理后的数据（已在 .gitignore 中忽略）
reports/                # 评测结果表与图（已在 .gitignore 中忽略）
```

### 许可证与引用

- **License**: Apache-2.0（详见 `LICENSE`）
- **Citation**: 请参考仓库根目录的 `CITATION.cff`
