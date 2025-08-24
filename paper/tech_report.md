# Causally Calibrated LLM-as-Judge for Recommenders：闭环式曝光感知评测与合成交互技术方案

作者：匿名（Anonymous）  
目标会议：SIGIR（acmart/sigconf 模板）

## 执行摘要（Executive Summary）
本方案提出一个“评测→合成→再评测”的闭环，用因果校正与可审计证据让 LLM 评委在推荐系统评估中更公平、更可靠：
- 曝光感知的 LLM 成对评审（RecJudge‑X）：同难度配对、强制证据子串、并发+重试+宽容解析、证据有效率（valid_ratio）。
- 倾向估计与因果指标：使用 IPS/SNIPS/DR 估计器，提供重叠诊断与裁剪/修剪敏感性；给出全局一致率（IPS‑PairAUC）与用户公平一致率（RJS, τ_w）。
- 单调校准（Calib‑Link）：把评委分（RJS）映射到“近似线上”的 IPS‑Recall@K（或 DR/SNIPS），提供 Spearman/Kendall/R² 与置信区间。
- 受约束的曝光可控合成（ExpoSynth）：基于体裁迁移矩阵与物品相似度，在控制尾部暴露的前提下补充“像真的一样”的交互。

代码已实现可复现 CLI、断点续跑与日志落盘（judge/synth），并给出健康检查与配置模板。

---

## 1. 问题定义与目标
### 1.1 挑战
- 线下指标（HR/NDCG/AUC）受“曝光分布”“负采样协议”影响，与线上真实影响（CTR/GMV/留存等）对齐差。
- 现有 LLM‑as‑Judge 常忽视曝光偏置、证据不可审计，难以建立信任，也缺少外部效度验证。
- 长尾稀疏导致模型在尾部学习不足，评测对尾部不敏感。

### 1.2 目标
- 构建“曝光感知+证据可审计”的 LLM 评审范式；
- 用 IPS/SNIPS/DR 做因果校正，报告重叠诊断与稳定性；
- 将评委分数单调校准到“近似线上”指标；
- 通过受约束的合成交互补齐尾部，实现评测–合成闭环的可重复改进。

---

## 2. 方法总览
### 2.1 曝光感知 LLM 成对评审（RecJudge‑X）
- 同难度配对：对每个用户，从“未看过”的物品里按 `pop_bin/age_bin`（可扩展 `price_bin/category`）分组，在同组内随机抽两件形成一对，减少热度/年代混淆。
- 提示词与证据：强制严格 JSON 输出，`rationales[].evidence` 必须是“输入子串”；解析失败做宽容修复；逐条核对得到 `valid_ratio`。
- 工程特性：异步并发、指数退避重试；支持 `--resume` 跳过已评审对并追加写；`--log_file` 落盘日志。
- 产物：
  - 原始 JSONL：`data/judge/judge_raw.jsonl`
  - 结构化结果：`data/judge/judge_clean.parquet`（`user_id,i,j,winner,p,valid_ratio`）

相关代码：`recjudge/pair_sampler.py`, `recjudge/prompts.py`, `recjudge/run_judge.py`。

### 2.2 倾向估计（Exposure Propensity）
用弱监督逻辑回归近似曝光概率 $\hat{\pi}(i\mid u)$：
- 特征：`user_freq, item_pop, item_age, hour, weekday, user_entropy`；
- 近似标签：将较热门（`pop_bin \ge 2`）视作“更易被曝光”的正类；
- 输出：`data/props/propensity.parquet`（`user_id,item_id,propensity`）。

重叠诊断：报告 $\min\pi$、分位数、`pct(π<阈值)`，并在评估中做 `clip/trim` 敏感性。

相关代码：`calib/run_ips.py`。

### 2.3 因果加权指标（全局与用户公平）
设每对样本 $(u,i,j)$ 权重 $w=1/\max(\pi(u,i),\epsilon)+1/\max(\pi(u,j),\epsilon)$。
- 全局一致率（IPS‑PairAUC）：
$$
\mathrm{PairAUC}_{\mathrm{IPS}} = \frac{\sum w \cdot \mathbf{1}[\text{模型偏好与 LLM 一致}] }{\sum w} \in [0,1].
$$
- 用户公平一致率（RJS, $\tau_w$）：先得用户内加权一致率 $a_u$，再映射 $\tau_u=2a_u-1$，最后按用户权重汇总：
$$
\tau_w = \frac{\sum_u (\sum w)_u\,\tau_u}{\sum_u (\sum w)_u} \in [-1,1].
$$
在“同批样本、同加权”前提下有 $\tau_w \approx 2\,\mathrm{PairAUC}_{\mathrm{IPS}}-1$。

相关代码：`recjudge/rjs_metric.py`, `calib/run_correlation.py`。

### 2.4 Calib‑Link：评委分→“线上代理指标”的单调校准
1) 用评委结果构造相关集合 $\mathcal{R}(u)$：赢家且 $p\ge p_0$（默认 0.5），可先用 `valid_ratio` 过滤；
2) 给定模型分数文件（`user_id,item_id,score`），对每个用户取 Top‑K；
3) 计算 IPS‑Recall@K / SNIPS / DR：
$$
\mathrm{Recall}^{\mathrm{IPS}}@K \,=\, \frac{\sum_{u}\sum_{i\in \mathcal{R}(u)\cap \mathrm{Top}K(u)} \tfrac{1}{\max(\hat{\pi}(i\mid u),\,\epsilon)}}{\sum_{u}\sum_{i\in \mathcal{R}(u)} \tfrac{1}{\max(\hat{\pi}(i\mid u),\,\epsilon)}}.
$$
DR 采用“模型期望 + IPS 残差”的双稳健形式，SNIPS 为自归一化，均显著降低方差。

4) 单调回归（等距回归） $g$：用 $(\mathrm{RJS}_m,\; \mathrm{IPS\text{-}Recall@K}_m)$ 拟合 $y\approx g(x)$。报告 Spearman/Kendall/R² 与 bootstrap CI。若拿到真实线上 KPI，只需将 $y$ 换为线上指标重复拟合。

相关代码：`calib/run_correlation.py`。

### 2.5 受约束的曝光可控合成（ExpoSynth）
- 体裁迁移矩阵 $T[g\to g']$：由相邻交互统计得到 $P(g'|g)$，刻画序列合理性；
- 物品相似度图 $S(i,j)$：item–user 稀疏矩阵的余弦相似，保留 top‑k 邻居；
- 锚点体裁：从用户近期历史提取 top‑k 主流体裁与权重；
- 尾部上权重抽样：`pop_bin>=3` 赋更高采样概率；
- 约束接受：满足“$T$ 达阈值”或“$S$ 达阈值”之一即接受，记录 `user_id,item_id,ts,source,reason`。

工程增强：`--resume` 基于已生成用户跳过、合并去重写回；`--log_file` 落盘日志；报告接受率与尾部占比。

相关代码：`exposynth/constraints.py`, `exposynth/run_synth.py`。

---

## 3. 实现与复现
### 3.1 代码结构（关键文件）
- 倾向与校准：`calib/run_ips.py`, `calib/run_correlation.py`
- 评委：`recjudge/run_judge.py`, `recjudge/prompts.py`, `recjudge/pair_sampler.py`, `recjudge/rjs_metric.py`
- 合成：`exposynth/run_synth.py`, `exposynth/constraints.py`
- 工具：`scripts/health_check.py`, `scripts/llm_ping.py`
- 配置：`configs/movielens1m.yaml`

### 3.2 一键复现流程（可断点续跑）
```bash
# 1) 数据准备与倾向
python scripts/prep_movielens.py --config configs/movielens1m.yaml
python calib/run_ips.py --config configs/movielens1m.yaml

# 2) 采样成对（同难度对比）
python -c "from recjudge.pair_sampler import sample_pairs; \
  sample_pairs('data/proc/ml1m_interactions.parquet','data/proc/ml1m_items.parquet',10,60,['pop_bin','age_bin'],'data/judge/pairs.parquet')"

# 3) 评委（支持断点与日志落盘）
python recjudge/run_judge.py --config configs/movielens1m.yaml \
  --pairs data/judge/pairs.parquet --out data/judge/judge_raw.jsonl \
  --concurrency 8 --retries 3 --resume --log_file reports/judge_run.log

# 4) 公平性指标（PairAUC_IPS / RJS）
python recjudge/rjs_metric.py

# 5) 合成交互（支持断点与日志落盘）
python exposynth/run_synth.py --config configs/movielens1m.yaml \
  --out data/synth/ml1m_tail_aug.parquet --resume --log_file reports/exposynth_run.log

# 6) 校准与相关（可带你的模型分数文件）
python calib/run_correlation.py --config configs/movielens1m.yaml \
  --models your_model_scores.parquet --k 20 --clip 1e-6 --clip_alt 1e-4 --trim 1e-4
```

### 3.3 配置建议（`configs/movielens1m.yaml`）
- `recjudge.L_history=10, pairs_per_user=60, match_cols=[pop_bin,age_bin]`
- `recjudge.llm.temperature=0.2, max_tokens=256`（稳定优先）
- `exposynth.steps_per_user=8, anchors_k=4`（按算力与需要调节）
- 倾向重叠：优先 `clip=1e-6/1e-4`，必要时 `trim=1e-4` 做稳健性对比

### 3.4 LLM 后端与健康检查
- `recjudge/llm.py` 支持 DashScope 兼容 / Tongyi 原生 / OpenAI；
- 环境连通：`python scripts/llm_ping.py`；
- 项目体检：`python scripts/health_check.py --pretty`（会输出到 `reports/progress.*`）。

---

## 4. 实验设计与评估协议
### 4.1 数据与模型
- 数据：MovieLens‑1M（可拓展更多域）；
- 评委规模：建议 ≥10k 成对；
- 模型：≥3–5 个（真实或脚本模拟），分数文件列为 `user_id,item_id,score`。

### 4.2 评估协议
- 过滤：`valid_ratio ≥ 0.5` 为主结果；≥0.8 作为敏感性；
- 指标：报告 IPS‑PairAUC 与 RJS（分桶：`pop_bin/age_bin`），并给出 `clip/trim` 组的稳定性；
- 校准：RJS→IPS‑Recall@K（K=10/20/50），报告 Spearman/Kendall/R² 与 bootstrap CI；
- 倾向重叠诊断：`min/p1/p5/mean/pct(π<1e-4)`。

### 4.3 消融与鲁棒性
- 无/有 IPS 权重；无/有证据对齐检查；单评委 vs 多评委；
- ExpoSynth 约束开/关；`steps_per_user/anchors_k` 扫描；
- LLM 温度/提示词敏感性；
- 采样上限 M 对用户（RJS 的 per‑user 采样）对偏差与方差影响。

---

## 5. 预期结果与风险
### 5.1 预期
- 体现“同难度+IPS”的公平性改进：尾部分桶中 RJS/PairAUC_IPS 的可解释差异；
- RJS 与 IPS‑Recall@K（或 DR/SNIPS）呈正相关，等距回归得到稳定单调曲线；
- 合成后的再训练在尾部 Recall/一致性上有可量化增益。

### 5.2 风险与缓解
- 倾向错置与重叠不足：做 `clip/trim` 与重叠诊断，必要时改进特征与模型（如引入位置/会话上下文）。
- LLM 方差与成本：多评委聚合、并发与重试策略；
- 证据“伪子串”：对 evidence 做更严格校验（位置/字段来源），并抽样人工复核；
- 合成偏移：监控 T 与 S 的分布拟合度（KL/JS），控制合成占比。

---

## 6. 扩展与未来工作
- 多评委一致性与仲裁：Cohen’s κ、加权投票、裁剪低一致对儿；
- 更强因果估计：交叉拟合、DML、CI 报告；
- 多域多模态：引入文本/图像证据核验；
- 线上小流量/人审金标：对外部效度的更直接验证。

---

## 7. 伦理与合规
- 保护隐私与人群公平；
- 合成数据不得视为真实行为；
- 证据对齐降低但不消除幻觉风险；
- 透明记录：日志落盘、断点可追踪、报告复现清单。

---

## 8. 里程碑与时间线（示例）
- W1–W2：完成 ≥10k 成对评委；倾向估计与诊断；
- W3–W4：多模型评分与 RJS/PairAUC_IPS 评估；校准与 CI；
- W5–W6：ExpoSynth 生成与再训练；回到评测闭环；
- W7：消融与鲁棒性；
- W8：撰写与整理提交物（代码、数据、附录）。

---

## 9. 复现性清单（Checklist）
- 提供完整代码与配置（已在仓库）；
- 固定随机种子与版本（`pyproject.toml/requirements.txt`）；
- 脚本命令、输出目录与日志落盘；
- 报告所有阈值（`p_thresh/clip/trim/valid_ratio`）、重叠诊断与 CI；
- 发布 `reports/tables/*.csv` 与 `reports/figs/*.png`。

---

## 10. 参考实现路径
- 评委：`recjudge/run_judge.py`，`recjudge/prompts.py`，`recjudge/pair_sampler.py`
- 倾向：`calib/run_ips.py`
- 指标与校准：`recjudge/rjs_metric.py`，`calib/run_correlation.py`
- 合成：`exposynth/run_synth.py`，`exposynth/constraints.py`
- 健康检查：`scripts/health_check.py`，`scripts/llm_ping.py`

---

## 附录 A：关键公式汇总
1) 成对权重：$w(u,i,j)=1/\max(\pi(u,i),\epsilon)+1/\max(\pi(u,j),\epsilon)$。

2) 全局一致率（IPS‑PairAUC）：$\frac{\sum w\,\mathrm{match}}{\sum w}$。

3) 用户公平一致率（RJS）：$\tau_w = \frac{\sum_u (\sum w)_u (2a_u-1)}{\sum_u (\sum w)_u},\; a_u=\frac{\sum w\,\mathrm{match}}{\sum w}$。

4) IPS‑Recall@K：见 §2.4；SNIPS 为自归一化，DR 为“模型期望+加权残差”。

5) 单调回归（等距回归）：$y\approx g(x)$，$g$ 非降且截断到 $[0,1]$。

---

（完）


