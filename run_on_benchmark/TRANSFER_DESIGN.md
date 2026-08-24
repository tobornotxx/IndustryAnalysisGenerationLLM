# 迁移测试设计:从 InsightBench 到 InsightEval(主)+ DataGovBench(次)

> 状态:设计稿(待确认决策后实施)。本版取代 v1 的 DiscoveryBench 导向——经 2026-07 深度调研,DiscoveryBench(NeurIPS 2024,venue 早于 InsightBench)降为可选支线,主线改为 InsightEval + DataGovBench。
> 日期:2026-07-29
> 置信度标注:[核实] = 已直接抓 arXiv HTML 全文确认;[调研] = 经子代理检索但未逐字核实全文。

---

## 0. 一句话讲清"到底咋回事"

我们在 **InsightBench**(arXiv 2407.06423,**ICLR 2025** accepted —— 之前误记为"非顶会",已更正)上的 prompt finetuning 已到顶,继续抠是针对小切片过拟合。导师方向:把现有多 agent 数据探索 pipeline(planner + SQL/Python executor → NL insights + summary)当成一个 **"skill"**,换到**更新的、同脉络的** benchmark 上做**迁移/泛化测试**。

调研结论:InsightBench 有一条活跃的 2025–2026 后续脉络,且这些后续 benchmark 的**评分栈几乎全部继承自 InsightBench**(LLM-judge best-match + recall 为核)。所以我们不需要换尺子,只需要换数据 + 补几个新指标。

**两个靶子,两条互补的泛化轴**:
- **InsightEval(arXiv 2511.22884,2025-11)**[核实] —— 主靶。InsightBench 的直接继任者(同模态 CSV、同业务工单域、同评分哲学),严格更新,100 实例。测的是**"同分布、更难、更严谨 curate"**下的泛化(尤其它的 precision/F1 会直接罚我们 pipeline 的过量生成)。
- **DataGovBench /「Data Analysis in the Wild」(arXiv 2607.06482,2026-07)**[核实] —— 次靶。真实政府开放数据(房产/医疗/环境/海洋/人口),**真正的跨域分布迁移**,多表 + 外部数据字典。但其 Table Insight 子集只有 **6 个 case**,统计意义弱,只能当**定性硬样本**。

**核心原则:pipeline 本身不动(它是被测 skill),只换"数据入口 + 评分出口 + 必要的外部知识摄入"。**

---

## 1. 背景与动机

### 1.1 为什么停掉 InsightBench finetuning
- v7→v8 已验证 prompt 修复(goal-sufficiency 四维分解 + schema 趋势分解准则)对 flag-4 GT4(0.1→1.0)、flag-6 GT2(0.3→1.0)是**判官无关的真实覆盖修复**。
- 残余盲区(flag-5 的 `closed_by` 字段始终不被当分析轴)说明:继续抠是**过拟合 benchmark 特定字段**,不是提升通用能力。
- InsightBench 自身已被同行点名有缺陷(见下 InsightEval 的 5 条指控)。

### 1.2 关键事实更正(影响靶子选择)
- **InsightBench = ICLR 2025 accepted**(arXiv 2407.06423)。已 WebFetch 核实。
- 后果:按"比 InsightBench 更早的不考虑",DiscoveryBench(NeurIPS 2024 D&B,2024-12,venue 更早)**被排除出主线**。它 arXiv 投稿同为 2024-07,领域不同(社会/生物/经济),仍是合法的"跨域科学发现"支线,但不是首选。

### 1.3 迁移要测的"skill"是什么(5 个可迁移单元)
不是权重,是 5 个 pipeline 行为单元。判断一个新 benchmark 有没有迁移价值,看它**把哪几个单元既用上又打分**:
1. **Planner 探索决策**(树形生成"接下来查什么")
2. **Goal-sufficiency 覆盖自评**(四维分解找漏 + 反过早停)
3. **Executor 计算**(写 SQL + Python 出结果)
4. **Schema 画像 + 趋势按列分解**(把趋势/失衡按分类列拆开)
5. **Insight articulation**(把发现写成 NL + summary)

InsightBench 考 1~5 全套,打分打在 NL insight 质量/召回。**靶子必须同样考 1~5 并打分在"发现质量"**,才有迁移意义;只考 3(算具体答案)、打分打在精确匹配的(如 DSBench/TableQA)不算 skill 迁移。

---

## 2. benchmark 脉络全景(2024–2026)

### 2.1 InsightBench 正统后续(开放式 NL insight,LLM-judge 评质量)[核心脉络]

| benchmark | arXiv | 日期 | 任务 | 评分 | 与我们的关系 |
|---|---|---|---|---|---|
| **InsightBench**(锚) | 2407.06423 | ICLR 2025 | CSV+goal→NL insights+summary | G-Eval judge,纯 recall best-match | 我们的训练集 |
| **InsightEval** ⭐ | 2511.22884 | 2025-11 | 同上(更严谨 curate) | recall+**precision**+**F1**+novelty,多 LLM | **主靶** |
| MedInsightBench | 2512.13297 | 2025-12 | 医学多模态病理图→insight | 同 InsightEval 评分栈 | ❌ 多模态,我们进不去 |
| DataGovBench-Insight | 2607.06482 | 2026-07 | 政府数据自由探索→发现 | LLaMA-3-Eval 4 维 rubric | **次靶**(6 case) |
| AgentAda | 2504.07421 | 2025-04 | skill-adaptive 分析 | 人类偏好+LLM-judge | InsightBench 原班人马的 agent,对标 |
| DataSage / DataSTORM | 2511.14299 / 2604.06474 | 2025-26 | 方法,**在 InsightBench 上评** | 复用 InsightBench 评分 | 说明 InsightBench 仍是活跃基座 |

### 2.2 DiscoveryBench 支线(NL 假设 / 科学发现)[可选]

| benchmark | arXiv | 日期 | 任务 | 评分 |
|---|---|---|---|---|
| DiscoveryBench | 2407.01725 | NeurIPS 2024 | dataset+query→NL 假设 | faceted:变量/关系匹配,recall×accuracy |
| D3-Gym | 2604.27977 | 2026-04 | NL 指令+可执行环境→发现 | 可验证执行环境(DiscoveryBench 原班人马升级) |
| ScienceAgentBench | 2410.05080 | ICLR 2025 | 写 Python 程序 | 执行正确性(~34%) |

### 2.3 闭式数据科学任务(能力不同,划清界限,不迁)
DSBench(ICLR 2025,选择题+Kaggle 建模,精确匹配)、InfiAgent-DABench(ICML 2024,`@mean[值]`正则)、TableBench(TableQA)、AutoKaggle/DS-Agent/Agent K/AIDE/AutoMind(Kaggle 竞赛,代码树搜索)。这些只激活 executor 底层(单元 3),不考探索+insight(单元 1/2/4/5),不用 LLM-judge 评 insight 质量。**迁移价值薄,不纳入。**

### 2.4 这条脉络的共性(为什么迁移不用换尺子)
1. GT 都是**标注好的"发现列表"**,不是标准答案。
2. 匹配都用 **best-match**(每条 GT 找最像的 pred),不做字符串精确匹配。
3. 打分核心是 **recall**;新一代(InsightEval/MedInsightBench)**加 precision + F1** 罚过量生成,加 **novelty** 评"GT 之外的新发现"。
4. judge 都是 **LLM**(G-Eval / LLaMA-3-Eval),新一代用**多个 LLM 平均**去偏。
5. 评分维度趋同:相关性 / 叙事对齐 / 定性细节 / **定量细节**。

> **关键:我们现有 `unified_scorer.py` 就是这套(G-Eval logprobs + 纯 recall best-match),InsightEval 的 recall 公式和我们一字不差。** 迁移不用换尺子,只需在 recall 之上**加 precision / F1 / novelty**。

---

## 3. 选靶子:为什么 InsightEval 主 + DataGovBench 次

### 3.1 InsightEval —— 主靶(同分布、更难、更严谨)

**是什么**[核实]:直接点名批判 InsightBench 的论文,用专家重新 curate 了一套,并提出新指标。

**数据**:100 个 CSV 实例,1000 条 insight(10 条/实例),6 种 insight 类型 × 8 个业务类别 × 4 个难度等级。**仍是 ServiceNow 风格工单数据**(例子就是 incident/caller/manager)。

**任务**:和 InsightBench 完全同构——数据 + goal → 探索 → NL insights。

**具体例子**[核实]:
- goal:"分析 caller_id/opened_at,评估随时间的报障频率,找出相对同侪持续上升的 caller。"
- GT insight:"IT 部门每个 manager 的下属数量异常偏高。"
- agent 输出:"IT 部门 manager 的管理跨度约为组织平均的 4 倍。" → judge 判语义命中。

**评分(关键差异 —— 在我们 recall 基础上加 precision/F1)**[核实]:
- recall = `E_gt~Unif(GT)[ max_i∈I S(gt,i) ]` ← **我们现在的公式,一字不差**
- precision = `E_i~Unif(I)[ max_gt∈GT S(i,gt) ]` ← 每条 pred 取最佳 GT,**罚多余 pred**
- F1(主打新指标)= `2·recall·precision/(recall+precision)`
- novelty = `(M + δ·Σ_i 1(Σ_{j=1}^3 LLM_j(i)≥2)) / (N+M)`,M=正确 insight 数,N=错误数,δ∈{0,1},3 个 LLM 里 ≥2 票认"新"才算
- judge = ROUGE-1 + G-Eval(平均 GPT-3.5-Turbo 与 Gemini 2.5 Pro);novelty 用 3 个独立 LLM;30 样本人工验证显示 **F1 比纯 recall 更贴近人类打分**

**它对 InsightBench 的 5 条指控**[核实]:E1 目标模糊、E2 数据类型未定义、E3 问题引用了不存在的列、E4 数据存在却说"数据不足"、E5 insight 重复。要求 R1 明确目标(含度量/维度/量化标准)、R2 高质量多视角问题、R3 多 LLM 评 + 评"发现 GT 之外新 insight 的能力"。

**为什么选它**:
- 同模态(CSV)、同数据风格(工单)、同评分哲学,**严格更新**(2025-11 > ICLR 2025)。
- 评分栈基本现成,只需加 precision/F1/novelty。
- **precision/F1 直接检验我们 pipeline 的老毛病**:我们一次吐一大堆 insight,纯 recall 不罚,但上了 precision/F1,过量生成会掉分。**这正是该测的泛化信号。**

**诚实权衡(不藏)**:InsightEval 仍是 ServiceNow 工单域,和 InsightBench **同分布**。所以它测的是**"同分布、更难、更严谨 curate"下的鲁棒性**,不是"跨域迁移"。跨域那根轴由 DataGovBench 补。

### 3.2 DataGovBench —— 次靶(真正的跨域硬测,但样本小)

**数据**[核实]:178 个数据集,**平均 21 万行 × 18 列**,36% 是**多表**(最常见 5 张表),57% 附**外部知识**(PDF/XLSX 数据字典)。领域:房产/医疗/环境/海洋生物/人口。

**两个任务**:
- **Table QA**:211 题(95 简单 + 116 可分解 = 414 子问题)。输出文本或可视化。文本 = **Exact Match**;可视化 = 4 个 MLLM 多数投票。**❌ 不适合我们**(考精确回答,不考探索+insight,我们的 skill 不被激活也不被评)。**不迁。**
- **Table Insight**:✅ **适合我们**。指令:"**无显式用户问句,自由探索数据,直接产实质性 insight**",评分 LLaMA-3-Eval(GPT-4o),Summary-level + Insight-level + 4 维 rubric(主题相关/叙事对齐/定性细节/定量细节,各 1–5)。baseline 是 InsightBench 自己的 **AgentPoirot**。

**Table Insight 例子**[核实]:
- goal:分析加拿大氡浓度分布 → GT finding:"New Brunswick 氡浓度超 200 Bq/m³ 住户,原始比例 24.8%,人口加权比例 20.6%"
- goal:化学生物富集 → GT finding:"PCBs 随年龄明显累积,40–79 岁组平均量显著更高"

**"只有 6 个数据集"是什么意思**[核实]:论文原话——"a curated subset of **six datasets**, each accompanied by a report from domain experts"。**Table Insight 总共就 6 个评估点**(6 个数据集,每个 = 1 个 case,自由探索产多条 insight,整体评一次)。6 个数算不出稳定均值、做不了分领域分解,**统计意义弱,只能当定性硬样本**,不能当主靶。

**为什么仍纳入**:它是唯一**真正的跨域**(政府数据,非工单)、**真实大规模**(21 万行)、**多表 + 外部知识**的 insight 评测。即便只有 6 个 case,也能定性回答"我们的探索 skill 换到全新领域还灵不灵"。和 InsightEval(同分布、大样本、严谨)互补,形成两轴。

### 3.3 两条泛化轴(总故事)

| 轴 | 靶子 | 测什么 | 样本 | 信号类型 |
|---|---|---|---|---|
| **同分布、更难** | InsightEval | skill 在更严谨 curate + precision/F1 罚过量下还灵不灵 | 100 | 定量、统计可靠 |
| **跨域、真实世界** | DataGovBench-Insight | skill 换到政府数据/多表/外部知识还灵不灵 | 6 | 定性、硬样本 |

两轴合起来比单靶强:InsightEval 给统计置信度,DataGovBench 给跨域真实性。

### 3.4 被排除的
- **MedInsightBench**:核心输入是病理切片图,靠视觉模型抽证据。我们 executor 是 SQL/Python,**无视觉能力**,进不去。[核实评分虽同]
- **DiscoveryBench / D3-Gym**:venue 早于 InsightBench,降为可选支线。若导师坚持要"科学发现"维度再加。
- **Table QA / 闭式 benchmark**:能力不对口。

---

## 4. 现状架构(Before:InsightBench-only)

```
┌─────────────────────────────────────────────────────────────────┐
│  run_benchmark.py  (入口)                                        │
│  - 加载 InsightBench flag-N.json (dataset_csv_path + metadata.goal)│
│  - 并行 ProcessPoolExecutor / 断点续跑 / 立即落盘                 │
└───────────────┬─────────────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  DataStormAdapter.get_insights(csv, user_csv, goal, desc,        │
│      return_summary=True) → (pred_insights, pred_summary)        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  MyDataStorm pipeline (被测 skill)                          │ │
│  │  PlannerAgent (tree) + GoalSufficiencyModule (四维+反早停)  │ │
│  │  ExecutorAgent (SQL + execute_python_from_sql)              │ │
│  │  CsvDatabaseBridge (CSV→SQLite, schema 画像 + 趋势分解准则) │ │
│  │  InsightBank → FinalReport → 提取 insights + summary        │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────┬─────────────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  unified_scorer.py  (G-Eval, deepseek, logprobs)                 │
│  - score_insights: 纯 recall (每 GT 取最佳 pred)                 │
│  - score_summary:  单次 judge                                    │
└─────────────────────────────────────────────────────────────────┘
```

**现状局限**:
- 数据入口绑死 InsightBench(`flag-N.json` 字段)。
- 评分只有 recall,无 precision/F1/novelty。
- `CsvDatabaseBridge` 只支持主表 + 1 张 user_csv(≤2 表),无 metadata 推断表间关系。
- 不摄入外部文档(PDF 数据字典)。

---

## 5. 新架构(After:两靶 + 双评分 + 字典阅读器)

```
   ┌────────────────────────────┐      ┌────────────────────────────┐
   │ InsightEval (主靶)          │      │ DataGovBench-Insight (次靶) │
   │ run_insighteval.py ★新      │      │ run_datagov.py ★新          │
   │ InsightEvalLoader ★新       │      │ DataGovLoader ★新           │
   │ (CSV+goal → 输入)           │      │ (多表+字典 → 输入)          │
   └─────────────┬──────────────┘      └─────────────┬──────────────┘
                 │   共用 (被测 skill, 零改动)          │
                 ▼                                     ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ DataStormAdapter.get_insights(csv, user_csvs[], goal, desc,   │ ★小改
   │     return_summary=True) → (pred_insights, pred_summary)      │  (多表)
   │ ┌──────────────────────────────────────────────────────────┐ │
   │ │ MyDataStorm pipeline (planner/executor/goal_sufficiency/  │ │
   │ │ insight_bank/report) —— 零改动                            │ │
   │ │ CsvDatabaseBridge ★小改: N 表 + metadata 注入 schema      │ │
   │ └──────────────────────────────────────────────────────────┘ │
   └─────────────┬────────────────────────────────────┬───────────┘
                 ▼                                     ▼
   ┌────────────────────────────┐      ┌────────────────────────────┐
   │ 字典阅读器 ★新模块           │      │ 评分栈                       │
   │ DictionaryReader:           │      │ unified_scorer ★扩展:        │
   │  PDF/XLSX → deepseek-v4-    │ ───→ │  + score_precision / F1      │
   │  flash(本身多模态) → 文本    │ 注入  │  + score_novelty (可选)      │
   │ (无需额外 key, 仅次靶用)     │ desc │ judge 统一用 deepseek        │
   └────────────────────────────┘      └────────────────────────────┘
```

**设计要点**:
1. **pipeline 零改动**:planner/executor/goal_sufficiency/insight_bank/report 全不动。动了就没法归因。
2. **两个新入口 + loader**,与 `run_benchmark.py` 平行,职责是"把新 benchmark 的数据喂给同一个 adapter"。
3. **bridge 小改**:支持 N 表 + 把 dataset metadata 注入 schema context(让 planner 知道表间关系)。
4. **字典阅读器**:独立模块,主要服务 DataGovBench(57% 数据集需要),用 **deepseek-v4-flash 本身的多模态能力**读 PDF/XLSX,**无需额外 key、无需换模型**,不污染主流程。
5. **scorer 扩展**:在现有 recall 上加 precision/F1/novelty。**judge 统一用 deepseek**(与 InsightBench 同尺子),不接 benchmark 自家的 GPT-4o/Gemini 协议。

---

## 6. InsightEval 迁移细节(主靶)

### 6.1 数据对接
InsightEval 数据形态与 InsightBench 几乎一致(CSV + goal + GT insight 列表),loader 工作量小:
- 找到 InsightEval 数据发布(HF/GitHub),解析出每实例的 `{csv_path, goal, gt_insights[]}`。
- 直接喂 `DataStormAdapter.get_insights(csv_path=..., user_csv_path=None, goal=..., dataset_description=..., return_summary=True)`。
- bridge 无需多表(InsightEval 是单表工单数据)。

### 6.2 评分扩展(已实施)

**核心洞察:recall / precision / F1 共享同一个 pairwise 矩阵,新增 precision 与 F1 不产生任何额外 API 调用。**

历史实现只用了矩阵的一半信息(按行取 max = recall)。重构后一次矩阵三指标全出:

```python
def score_insight_matrix(pred_insights, gt_insights) -> dict:
    # matrix[i][j] = S(pred_j, gt_i)，外层 GT、内层 pred
    recall    = arr.max(axis=1).mean()   # 每行(GT)取 max —— 覆盖度
    precision = arr.max(axis=0).mean()   # 每列(pred)取 max —— 罚冗余
    f1        = 2*R*P/(R+P)
    return {"recall","precision","f1","matrix","n_pairs"}

score_insights(...)  # 保留原签名 = matrix["recall"]，InsightBench 历史结果可比
```

**已完成的实现**(`unified_scorer.py` + `run_benchmark.py`):
- ✅ `score_insight_matrix()` —— 一次矩阵产出三指标,已通过 mock 单元测试验证取向正确
- ✅ `score_insights()` 保留原签名与语义(纯 recall),历史结果可比
- ✅ `result.json` 新增 `insights_recall` / `insights_precision` / `insights_f1`
- ✅ 汇总输出三指标均值
- ⏸️ `score_novelty` —— **已决定暂不做**(唯一真三倍成本项,加分项非主指标,主结果出来后再定)

### 6.3 Prompt 缓存优化(已实施,待稳定性论证)

**实测起因**(v8 flag-4 真实日志,418 次调用):
- scoring 占 **210/418 = 50%** 的调用,输入约 206K token,但每次只输出约 10 token —— **极度输入重、输出轻**,是缓存优化的理想目标。

**问题**:原 prompt 把变化的 `{answer}` 放在最前、固定的 instructions 放在后面,导致稳定前缀只有约 33%。

**改法**:新增 `gt_first` 模板,把「开头 + Ground Truth + 全部固定 instructions」前置,唯一变化的 `{answer}` 移到末尾。配合 `score_insight_matrix` 的「外层 GT、内层 pred」循环,同一 GT 的连续 N 次调用中有 N-1 次可命中前缀缓存。

**实测前缀增益**:

| 模板 | 稳定前缀占比 |
|---|---|
| `answer_first`(v8 及之前基线) | **32.9%** |
| `gt_first`(新默认) | **91.6%** |

已验证两模板**字符多重集完全相同**——内容一致,仅顺序不同。

> ⚠️ **待论证的风险**:LLM 对输入顺序敏感,调换 answer/gt 顺序**可能改变打分**,严格说影响与 v8 基线的可比性。
> **已留 A/B 钩子**:环境变量 `GEVAL_PROMPT_ORDER=answer_first|gt_first` 可切换,同一份代码即可跑对照实验。
> **验证方式**:同一批 case 两种顺序各跑一次,比较 per-GT 分数漂移;若漂移可忽略则采用 gt_first,否则回退或重跑基线。**此项由用户负责论证。**

### 6.4 Token / 缓存用量核算(已实施)

日志里原本**完全没有** token 与缓存字段,导致"缓存命中多少"无法回答。已加:
- `_record_usage()` 累计 `prompt_tokens` / `completion_tokens` / `cache_hit_tokens` / `cache_miss_tokens`
- 兼容 DeepSeek(`prompt_cache_hit_tokens`)与 OpenAI(`prompt_tokens_details.cached_tokens`)两种字段
- `get_usage_stats()` 输出含 `cache_hit_rate`,落进 `result.json` 的 `scorer_usage`
- 记账失败不影响打分主流程(try/except 兜住)

**这样才能把"缓存命中率"从猜变成测。**

**双评分**:
- (a) **我们的 G-Eval(deepseek)**:同 InsightBench 尺子,算 recall+precision+F1。这是最干净的泛化信号——尺子不变,只变数据(更难、更严谨 curate)。**✅ 已定:只做这条。**
- (b) ~~InsightEval 自家协议(ROUGE-1 + G-Eval 平均 GPT-3.5/Gemini,3-LLM novelty)~~:**❌ 已定不做。** judge 模型是我们自选的,他们用 GPT-3.5/Gemini 只是他们的选择,不是约束;唯一意义是与其 leaderboard 数字逐位可比。我们用 deepseek 反而更好——和 InsightBench 同一把尺子,跨 benchmark 才可比。将来若要投稿对标 leaderboard 再补。

### 6.3 预期(诚实)
- **recall 可能持平或略降**(题目更难、目标更明确,我们若探索不够覆盖会掉)。
- **precision/F1 是真正的考题**:我们 pipeline 一次吐大量 insight,纯 recall 下不罚;上 precision 后,无关/冗余 insight 会拉低 F1。**如果 F1 明显低于 recall,说明我们的 skill 偏"狂喷"而非"精准发现"**——这是最有价值的负结果。
- novelty:我们能否发现 GT 之外的合理新发现(测创造力,次要)。

---

## 7. DataGovBench 迁移细节(次靶)

### 7.1 多表 bridge 难度
**现状**:`CsvDatabaseBridge` 主表 + 1 张 user_csv,无 metadata 推断。
**要改**:
- 加载 N 张表进 SQLite(工程不难,bridge 本就是多表)。
- 把 dataset metadata 注入 `get_schema_context()`(哪张表覆盖哪一年、可能的 key 列)。
- **真正的难点不是"加表",而是"无显式 key 时让 agent 自己判断怎么连/选哪张表"**——论文错误分析专门有 "Wrong Choice of Tables (4.3%)"。这是 agent 推理能力问题(InsightBench 工单都是单表,我们从没训过多表 join 推理),不纯是工程。**这本身就是 transfer 要暴露的 skill 边界,有诊断价值。**

### 7.2 外部数据字典接入
**57% 数据集带外部知识(PDF/XLSX 数据字典),不接这些题等于缺信息直接掉分。** 详见 §8 字典阅读器模块。

### 7.3 评分
- **我们的 G-Eval(deepseek)**:同尺子新分布,算 recall(+precision/F1)。**✅ 已定:只做这条。**
- ~~LLaMA-3-Eval(GPT-4o)4 维 rubric~~:**❌ 已定不做。** GPT-4o 只是论文作者的选择,非约束。但**保留他们的 4 维 rubric 维度**(主题相关/叙事对齐/定性细节/定量细节)作为诊断视角,用 deepseek 评——尤其看我们是否也栽在"定量细节"上(他们所有 agent 在该维度无一 ≥3 分)。

### 7.4 样本量警告
Table Insight 只有 **6 个 case**。报告时**只做定性分析 + 个案复盘**,不报"均值排名"(6 个数没统计意义)。诚实标注。

---

## 8. 字典阅读器模块(跨靶通用,主服务 DataGovBench)

### 8.1 "数据字典"是什么
数据字典 = **解释数据集每列含义的说明书**。政府数据列名常是缩写/代码,光看列名不知道是啥,字典是翻译。例(DataGovBench 氡数据风格):
```
RADONCONC  氡浓度,单位 Bq/m³。>200 为超标。
PROV       省份代码。NB=New Brunswick, SK=Saskatchewan, MB=Manitoba...
WTYPE      住宅类型。1=独立屋, 2=半独立, 3=公寓
WEIGHT     人口加权系数,按人口比例汇总时用(非原始计数)
```
GT finding 里"原始比例 24.8%、**人口加权**比例 20.6%"——"人口加权"这个操作就藏在 `WEIGHT` 列的字典说明里。没字典 agent 半瞎。

### 8.2 为什么直接用 deepseek-v4-flash 读(✅ 已定)
**我们的 `default.model_name = deepseek-v4-flash` 本身就是多模态模型**,可直接读 PDF/图片。所以:
- **不需要额外的 Claude/GPT-4o key,不需要混用模型,主流程零改动。**
- 相比本地文本工具(pdfplumber/PyMuPDF)的优势:不挑 PDF 类型——**扫描件/图片型 PDF 纯文本工具直接废**(提取为空或乱码),多模态靠视觉兜底;复杂版面(多栏/嵌套表格)文本抽取易乱序,视觉理解更准。
- 代价:长字典的 token 消耗;必要时分页读。数据字典通常远小于数据本身(讲列含义,不是讲行),一般压得住。
- 降级路径:若某些 PDF 读取异常,再用 pdfplumber 兜底。

### 8.3 模块设计
```
DictionaryReader(dict_path: str)   # 复用现有 LLMConfig,无需额外 key
  -> read() -> str                 # 返回结构化列描述文本
```
- 输入:PDF/XLSX 字典文件路径。
- 实现:PDF(或页面渲染图)→ deepseek-v4-flash → 抽"列名:含义:单位:取值码"→ 文本。
- 输出注入 adapter 的 `dataset_description=`,进 bridge 的 schema context。
- **InsightEval 基本不需要**(工单数据列干净);DataGovBench 必需。

### 8.4 工程量
~半天(读取 + 调用 + 注入)。因无需新 key/新 SDK,比原方案更省。

---

## 8.5 实测成本账单(v8 flag-4 真实日志,非估算)

来源:`results/test_run_v8/flag-4/run.log`(6 MB),按 httpx 响应行精确计数,418 次调用全部 200 OK 无重试。

### 单 case 开销

| 阶段 | 调用数 | 输入 token(实测) | 模型分布 |
|---|---|---|---|
| Pipeline(探索+报告+summary) | **208** | ~676K | 187 flash + 22 pro |
| Scoring(G-Eval) | **210** | ~206K | 210 flash |
| **合计** | **418** | **~882K 输入** | |

输出 token 原日志未记(已在 §6.4 补上记录)。粗估 ~60-100K,**单 case 总量约 95 万-100 万 token**。

### Scoring 数字完全闭合
flag-4 有 **8 条 GT × 26 条 pred = 208 次 pairwise**,+1 summary +1 logprobs 探测 = **210**,与日志一字不差。

### Pipeline prompt 分布
中位数 **2810 token**,p90 **5112**,最大 **12588**。长 prompt 主因是 executor 的 ReAct 循环中 `action_history` 逐轮累积。
- executor 是 ReAct 式 agentic loop,`max_turns=5`,每问约 2-6 次调用(含 1 次 answer summary)
- 每 case 共回答 **10 个问题**(L1:2 + L2:4 + L3:4)

### 规模推算(InsightEval 100 实例)

| 场景 | 调用数 | token |
|---|---|---|
| 主臂单次 | ~41,800 | ~9500 万 |
| + novelty(已决定不做) | +~4,500 | — |
| ablation 7 切片 × 100 实例 | ×7 | **不可接受** |

→ **这就是文档中 ablation 抽样 20-30 实例决定的量化依据**(§8 ablation 成本控制)。

### 已实施的优化与其效果

| 优化 | 效果 |
|---|---|
| recall/precision/F1 共享矩阵 | 新增两个指标 **零额外调用** |
| `gt_first` prompt 顺序 | scoring 稳定前缀 32.9% → **91.6%** |
| usage 记录 | 缓存命中率从"不可知"变为可测 |

### 尚未采纳的优化(记录备查)

**批量打分**:把 pairwise 矩阵改为一次 prompt 内塞多条 pred 对一条 GT,可能砍掉约 80% 的 scoring 调用。
> ❌ **暂不采用**:会改变判分口径(单对独立打分 vs 批内相对比较),破坏与 v8 基线及 InsightEval 论文口径的可比性。缓存优化已能拿到大部分收益,风险收益比更好。

---

## 9. 评分方案总览(统一 deepseek judge)
**✅ 已定:两个靶子都只用我们的 deepseek G-Eval,不接 benchmark 自家的 GPT-4o/Gemini 协议。**

| 靶子 | 评分 | 维度 |
|---|---|---|
| InsightEval | deepseek G-Eval | recall + precision + F1(+ novelty 可选) |
| DataGovBench | deepseek G-Eval | recall(+ precision/F1),按其 4 维 rubric 做诊断视角 |

**理由**:judge 模型是我们自选的,论文作者用 GPT-4o/Gemini 只是他们的选择,不构成约束。用 deepseek 的好处是**和 InsightBench 同一把尺子**——尺子不变、只变数据分布,这才是最干净的泛化信号。代价是不能与其 leaderboard 数字逐位对比;若将来投稿需要对标,再补跑他们的协议即可(纯增量工作,不影响主结论)。

**报告**:
- InsightEval:100 实例的 recall/precision/F1 均值 + 按难度/类别分解 + 失败 case 复盘(类似 v8 per-GT 分析:是探索盲区还是过量生成)。
- DataGovBench:6 case 定性 + 个案复盘,标注多表/外部知识是否成为障碍;按 4 维 rubric 看是否也栽在"定量细节"。
- 关键诊断:**F1 vs recall 的差距** = 我们 skill"过量生成"的程度。

---

## 10. 预期结果与假设

### 10.1 主假设
在 InsightBench 上训出的通用探索 skill(四维分解、趋势按列分解、时间画像、goal-sufficiency 覆盖自评)是**数据无关的通用 EDA 行为**(bridge 按"任意分类列"分解,不认字段名),应当能:
- **InsightEval**:在同分布更难题上 recall 不塌、F1 暴露过量生成问题。
- **DataGovBench**:在跨域真实数据上仍能产出方向性发现,但多表 join 推理和外部知识利用是暴露的边界。

### 10.2 预期会暴露的问题(有价值的负结果)
- **过量生成**:precision/F1 < recall,说明 skill 偏"狂喷"。
- **多表推理盲区**:InsightBench 单表训练,DataGovBench 多表 join 可能选错表。
- **外部知识利用**:字典注入后 agent 是否真的"读懂"列含义、是否据此做对分析(如人口加权)。
- **精确量级**:DataGovBench GT 含具体数值(24.8%),方向对、量级偏。

### 10.3 成功标准
- 两个靶子端到端跑通、产出合规输出(工程成功)。
- InsightEval:recall 不显著低于 InsightBench(skill 可迁移),F1 给出过量生成的量化诊断。
- DataGovBench:6 case 中至少部分产出方向性正确发现;失败集中在"多表/外部知识"而非"探索盲区"(说明通用探索 skill 迁移了,边界在工程/领域知识)。

---

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| InsightEval 数据未公开/格式变动 | loader 容错 + 记录跳过;必要时按其描述自行 curate 等价子集 |
| DataGovBench 只有 6 个 case,统计弱 | 明确只做定性,不报均值排名 |
| 多表 bridge 改动影响 InsightBench 兼容 | bridge 加表是向后兼容的(单表仍可用);InsightBench 入口回归测试 |
| 字典阅读器成本 | 用现有 deepseek-v4-flash(本身多模态),无额外 key;长字典分页读;异常时 pdfplumber 兜底 |
| precision/F1 让分数变难看 | 这正是要测的;不粉饰,如实报告并诊断 |
| 与 InsightBench 不可比 | judge 全线统一 deepseek G-Eval,同尺子,跨 benchmark 可比 |
| 不与 leaderboard 逐位可比 | 已知取舍(§9)。主结论靠"同尺子跨分布"成立;投稿需要时再补跑其协议,纯增量 |

---

## 12. 实施计划(分阶段)

1. **阶段 0(半天)**:InsightEval loader,在样本上验证 CSV+goal 能读出、bridge 能加载、`get_schema_context()` 正常。
2. **阶段 1(半天)**:`run_insighteval.py`(仿 `run_benchmark.py`,并行/续跑),用现有 recall 评分跑通 train 子集。
3. **阶段 2(半天)**:`unified_scorer` 扩展 precision/F1(+novelty),出第一份 InsightEval 分数,校准预期。**重点看 F1 vs recall 差距。**
4. **阶段 3(半天)**:bridge 扩多表 + metadata 注入;`DictionaryReader` 多模态模块。
5. **阶段 4(半天)**:`run_datagov.py` + `DataGovLoader`,在 6 个 Table Insight case 上跑通,接 LLaMA-3-Eval(或先用我们 G-Eval)。
6. **阶段 5**:完整报告 + 失败 case 复盘 + F1 诊断。

**总计 ~2.5 天**,核心 pipeline 零改动。

---

## 13. 已确认决策(2026-07-30 拍板)

| # | 决策 | 结论 |
|---|---|---|
| 1 | InsightEval 自家评分协议 | **❌ 不做**。只用 deepseek G-Eval(同 InsightBench 尺子)。judge 模型我们自选,GPT-3.5/Gemini 非约束。投稿需对标 leaderboard 时再补。 |
| 2 | DataGovBench GPT-4o LLaMA-3-Eval | **❌ 不做**。同上,用 deepseek。但保留其 4 维 rubric 作诊断视角。 |
| 3 | 字典阅读器实现 | **✅ 直接用 deepseek-v4-flash**(本身多模态)。无需额外 key、无需混用模型。异常时 pdfplumber 兜底。 |
| 4 | DiscoveryBench/D3-Gym 支线 | **❌ 先不做**。只做 InsightEval + DataGovBench。若导师要"科学发现"维度再加。 |

**总原则:judge 全线统一 deepseek,模型全线统一 deepseek-v4-flash/pro,不引入任何额外 API key。**

---

## 附录 A:为什么不是过拟合(设计自检)
- pipeline 零改动 → 不可能针对新 benchmark 特化。
- bridge 多表/metadata 注入是**通用能力**(任意多表数据都适用),非 DataGovBench 特化。
- bridge 的时间画像/趋势分解准则是**数据无关**的(引用"上方列出的分类列",不命名具体值)。
- 字典阅读器是**通用文档理解**,非特化。
- 双评分里 G-Eval 是 InsightBench 同尺子 → 跨 benchmark 可比。
- precision/F1/novelty 是 InsightEval **论文定义的通用指标**,非我们为刷分自造。

## 附录 B:benchmark 全景速查

**适合迁移(开放式 NL insight,LLM-judge 评质量)**:
- InsightEval(2511.22884)[核实] —— 主靶
- DataGovBench-Insight(2607.06482)[核实] —— 次靶(6 case)
- MedInsightBench(2512.13297)[核实] —— ❌ 多模态进不去

**支线(NL 假设/科学发现)**:
- DiscoveryBench(2407.01725,NeurIPS 2024)、D3-Gym(2604.27977)、ScienceAgentBench(2410.05080,ICLR 2025)

**不适合(闭式/执行评分,只激活 executor 底层)**:
- DSBench(2409.07703,ICLR 2025)、InfiAgent-DABench(2401.05507,ICML 2024)、TableBench、AutoKaggle/DS-Agent/Agent K/AIDE/AutoMind

**最新 insight AGENT(对标,非 benchmark)**:
- AIDA(2605.07202,RL+DSL,反 workflow)、DeepAnalyze(2510.16872,训练式 8B agentic LLM)、NAACL2025 Insight Generation(2503.11664,Hypothesis→NL2SQL→Summarize,和我们架构几乎一样)、Data-to-Dashboard(2505.23695)
