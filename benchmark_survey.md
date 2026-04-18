# 表格数据深度分析 Benchmark 调研报告

## 概述

本文调研了当前主要的"基于表格数据进行深度分析并得出结论"的 Benchmark。这些 Benchmark 的核心任务是：给定一些结构化表格数据和分析目标，要求 LLM 或 Agent 对数据进行多步骤分析，产出有意义的洞察/结论/建议。

---

## 1. InsightBench（ICLR 2025）

**论文**: InsightBench: Evaluating Business Analytics Agents Through Multi-Step Insight Generation  
**链接**: https://arxiv.org/abs/2407.06423  
**代码**: https://github.com/ServiceNow/insight-bench

### 1.1 任务定义

InsightBench 评估的是 LLM-based Agent 进行**端到端数据分析**的能力。与传统 benchmark 只要求回答一个具体查询不同，InsightBench 要求 Agent 完成整个分析流程：**确定分析目标 → 自主提出问题 → 编写代码执行分析 → 解读结果生成 Insight → 汇总并提出行动建议**。

### 1.2 数据集组成

每个数据样本由以下部分组成：

| 组成部分 | 详细说明 |
|---------|---------|
| **CSV 数据表** | 500 行合成企业数据。数据结构来自 ServiceNow 平台的真实系统表，但数据内容为合成生成（随机采样 + GPT-4 生成文本字段）。每个数据集都在"可控列"中人为植入了特定的趋势/异常（如线性增长趋势、分布偏斜等）。 |
| **SMART 目标 (Goal)** | 每个数据集附带一个 Specific（具体）、Measurable（可衡量）、Attainable（可达成）、Relevant（相关）、Timely（有时限）的分析目标。例如："分析事件分配在各类别中的差异和不平衡，旨在识别硬件故障的主要原因。" |
| **Ground-Truth Jupyter Notebook** | 专家标注的完整分析 Notebook，包含：① 数据集概述 ② 按顺序递进的 3-5 个分析问题 ③ 每个问题对应的 Python 代码块 ④ 代码生成的图表 ⑤ 每个图表的 JSON 元数据（描述 Insight）⑥ 每条 Insight 的类型标签 |
| **Insight 分类标签** | 每条 Insight 被分为四类：Descriptive（发生了什么）、Diagnostic（为什么发生）、Predictive（未来可能发生什么）、Prescriptive（应该采取什么行动） |
| **Insight 总结 + 行动建议** | Notebook 最后一节，汇总所有发现并提出可执行建议（如"打开工单进一步调查"） |
| **难度级别** | 1-5 级：Easy (1-2) 30 个数据集，涉及直接数据检索和基础分析；Medium (3) 36 个，需要整合多数据源或中等数据变换；Hard (4-5) 34 个，需要计算多个中间量或显著的数据转换 |

**整体规模**: 100 个数据集，共 475 条 Insight。

**五大主题领域及数据表来源**:

| 主题 | 数据集数量 | 数据表来源 | 说明 |
|------|----------|-----------|------|
| Incident Management（事件管理） | 20 | ServiceNow incidents 表 | 字段：事件编号、打开时间、关闭时间、分类、分配组、描述等 |
| Asset Management（资产管理） | 20 | alm_hardware 表 | 字段：资产标签、用户、状态、位置、购买日期、保修到期等 |
| User Management（用户管理） | 15 | sys_user 表 | 字段：用户 ID、姓名、部门、角色、最后登录时间等 |
| Finance Management（财务管理） | 15 | fm_expense_list 表 | 字段：支出金额、日期、关联用户、部门、类别、处理状态等 |
| Goal Management（目标管理） | 10 | sn_gf_goal 表 | 字段：目标描述、开始/结束日期、状态、负责人、优先级、完成百分比等 |
| 组合主题 | 20 | 多表组合 | 10 个 Asset & User 组合 + 10 个 Finance & User 组合 |

**数据构造四步流程**:
1. **选择数据表结构** — 从 ServiceNow 系统表中选取相关列，定义 Schema
2. **设计趋势/异常** — 用数学模型（如线性回归）在"可控列"中植入趋势。例如：`TTR = 1 + slope × (incident_open_date − data_start_date)`，slope 控制增长速率
3. **合成数据填充** — 可控列按趋势公式生成，非可控列用随机采样或 GPT-4 生成（如事件描述文本）
4. **编写 Ground-Truth Notebook** — 专家手动编写包含问题-代码-图表-Insight 的完整分析流程

### 1.3 具体 Case 示例

#### Case: Incident Resolution Time Dataset（数据集 #2，难度 3）

**数据表**: incidents 表（500 行），包含字段：

| 字段名 | 类型 | 示例值 |
|--------|------|--------|
| `number` | string | INC0001234 |
| `opened_at` | datetime | 2023-01-15 08:30:00 |
| `closed_at` | datetime | 2023-01-20 14:22:00 |
| `category` | categorical | Hardware / Software / Network / Database / Service Desk |
| `assignment_group` | string | Network / Hardware / Software |
| `caller_id` | string | Don Goodliffe |
| `short_description` | text | "Printer on 3rd floor is not responding to print jobs"（GPT-4 生成） |

**人为植入的趋势**: 事件解决时间（TTR = closed_at − opened_at）随时间线性增长，slope 参数控制增长速率。

**SMART 目标**: "Analyze the discrepancy and imbalance in the distribution of incidents assigned across categories, aimed at identifying the primary causes of hardware failures in an organization."

**Ground-Truth Notebook 的完整分析步骤**:

**Step 1**（Descriptive）:  
- 问题: "What is the distribution of incident categories over the past year?"  
- 代码: 生成 incidents 各类别的柱状图  
- 图表数据（JSON 元数据）: `{"Hardware": 314, "Software": 56, "Network": 48, "Database": 43, "Service Desk": 39}`  
- Insight: "Hardware 类别事件数量最多（314 件），远超其他类别，占总事件的 62.8%。"

**Step 2**（Diagnostic）:  
- 问题: "What is the trend in time to resolution (TTR) of incidents across categories?"  
- 代码: 生成各类别的 TTR 时间序列折线图  
- Insight: "TTR 从 2023 年 1 月的平均 113.03 小时**线性增长**到 2024 年 6 月的 3150.86 小时，Hardware 类别的 TTR 增长最为显著。"

**Step 3**（Diagnostic）:  
- 问题: "What is the wordcloud of the most common words in incident descriptions?"  
- 代码: 生成 Hardware 事件描述的词云  
- Insight: "'printer'、'server'、'malfunction' 是最高频词汇，指向打印设备和服务器为主要故障源。"

**Step 4**（Predictive）:  
- 问题: "What is the forecast for future incident volume based on current trends?"  
- 代码: 基于线性回归预测未来 6 个月趋势  
- Insight: "按当前增速，2024 年底 Hardware 类别 TTR 预计突破 5000 小时。"

**Step 5**（Prescriptive）:  
- 问题: "What strategies can mitigate this projected increase?"  
- Insight: "建议：① 增加 Hardware 团队人力 ② 建立打印机和服务器的预防性维护机制 ③ 开启专项工单进行根因分析。"

**最终 Summary**: "事件解决时间呈现明显的线性增长趋势，尤其集中在 Hardware 类别。根因分析显示打印设备和服务器故障是主要驱动因素。建议立即增加技术支持资源并开启根因调查工单。"

#### 各主题的 Insight 示例对照表

| 主题 | 问题示例 | Insight 示例 |
|------|---------|-------------|
| Incident Management | "各类别事件的 TTR 趋势如何？" | "TTR 在特定时间段内对 Hardware 事件突然开始线性增长" |
| User Management | "各部门经理的直接汇报人数分布如何？" | "IT 部门平均每位经理管理 50.5 人，远超其他部门（客服 8.8、财务 11.6、HR 12.8、销售 13.0）" |
| Asset Management | "资产购买日期与保修期的关系？" | "购买日期越晚的资产，保修期越长，存在显著正相关" |
| Finance Management | "不同费用区间的处理时间分布如何？" | "反直觉地，低费用区间的处理时间反而显著更长" |
| Goal Management | "'成本缩减'类目标的达成时长趋势如何？" | "随时间推移，'成本缩减'类目标的达成时间在增加" |

### 1.4 LLM 的输入与输出

**输入（喂给 Agent 的内容）**:

| 输入项 | 详细内容 |
|--------|---------|
| **数据集 Schema** | 通过自动提取获得：每列的名称、数据类型、唯一值数量、NA 值数量。数值列还提取 min/max/mean/std；日期列提取 min/max 日期；分类列提取 top 5 唯一值 |
| **SMART Goal** | 一句话的分析目标，如 "Analyze the discrepancy and imbalance in the distribution of incidents assigned across categories" |
| **CSV 数据文件** | 作为 DataFrame 提供给 Agent 的代码执行环境 |

**输出（Agent 应当生成的内容）**:

| 输出项 | 详细内容 |
|--------|---------|
| **高层问题** | Agent 自主生成 k=3 个高层问题（如"各类别事件分布如何？"） |
| **Python 分析代码** | 每个问题对应的可执行 Python 代码，用于生成图表和提取数据 |
| **Insight 描述** | 对每个图表/代码输出的文字解读 |
| **Follow-up 问题** | 对每个高层问题生成 n=4 个追问（覆盖 Descriptive/Diagnostic/Predictive/Prescriptive 四种类型），每个追问同样有代码和 Insight |
| **Insight 总结** | 汇总所有 (n+1)×k = 15 条 Insight 的报告 + 行动建议 |

**AgentPoirot 的具体工作流**:
1. 输入 (Dataset, Goal, Schema) → "Question Generation Prompt" → 生成 3 个高层问题
2. 每个高层问题 → "Code Generation Prompt" → 生成 Python 代码并执行
3. 代码输出 → "Insight Extraction Prompt" → 生成 insight + justification
4. insight → "Follow-Up Question Prompt" → 生成 4 个多样化追问（descriptive/diagnostic/predictive/prescriptive）
5. 选择最佳追问 → 重复步骤 2-3
6. 所有 insight → "Summary Prompt" → 生成最终报告

### 1.5 评分标准

#### 评估框架：两层评估机制

**1. Insight 级别评分（One-to-Many Matching）**:

对每条 Ground-Truth Insight $gt \in GT$，在 Agent 生成的所有 Insight $A$ 中找最佳匹配，然后取平均：

$$score_{insight} = \frac{\sum_{gt \in GT} \max_{a \in A} \mathcal{M}(gt, a)}{|GT|}$$

其中 $\mathcal{M}$ 是 LLaMA-3-Eval 评分函数。

**2. Summary 级别评分**:

直接将 Agent 生成的总结文本与 Ground-Truth 总结文本做 LLaMA-3-Eval 比较，得到一个 0-1 分数。

**最终分数** = 100 个数据集上 Insight 分和 Summary 分的平均值。

#### 评分器详情

| 评分器 | 说明 |
|--------|------|
| **LLaMA-3-Eval（主指标）** | 使用 LLaMA-3-70b 替代 GPT-4 的 G-Eval 技术。优势：① 开源，避免 API 成本 ② 模型权重固定，评分稳定可复现。温度设为 0。 |
| **G-Eval（对比验证）** | 使用 GPT-4 评估，与 LLaMA-3-Eval 结果高度一致，证明 LLaMA-3-Eval 可作为有效替代 |
| **ROUGE-1（辅助指标）** | n-gram 重合度，作为补充参考 |

#### 评分示例

| 质量 | Ground-Truth Insight | Agent 生成的 Insight | LLaMA-3-Eval 分数 | 评判理由 |
|------|---------------------|---------------------|------------------|---------|
| 好 | "HR 部门的资产成本显著更高" | "HR 部门的平均资产成本为 $4874.25，显著高于其他部门的 $1967.26" | **0.81** | 准确传达了核心信息，有具体数值支持 |
| 差 | "用户数量与 HR 部门电脑高成本之间存在弱相关" | "HR 部门的资产分布包括 19 台电脑、5 台 Web 服务器和 4 台服务器" | **0.18** | 未涉及用户数量与成本的相关性，描述了无关的资产分布 |
| 差 | "培训时长增加与员工绩效提升之间存在强正相关" | "培训项目持续了 6 个月，涉及 6 名员工" | **0.11** | 完全未涉及培训与绩效的关系 |

### 1.6 关键实验结果

| Agent | Backbone | ROUGE-1 | Insight LLaMA-3-Eval | Summary LLaMA-3-Eval |
|-------|----------|---------|----------------------|----------------------|
| Pandas Agent | gpt-4o | 0.35 | 0.54 | 0.40 |
| **AgentPoirot** | **gpt-4o** | **0.32** | **0.60** | **0.44** |
| AgentPoirot | gpt-4-turbo | 0.30 | 0.56 | 0.35 |
| AgentPoirot | gpt-3.5-turbo | 0.34 | 0.50 | 0.31 |
| AgentPoirot | llama-3-70b | 0.33 | 0.52 | 0.33 |
| AgentPoirot (generic goal) | gpt-4o | 0.30 | 0.40 | 0.33 |

**关键发现**:
- AgentPoirot (gpt-4o) 取得最佳表现，但 Insight 级别仅 0.60/1.0，Summary 级别仅 0.44/1.0，**提升空间巨大**
- 使用精心设计的 SMART Goal 比使用通用目标（"I want to find interesting insights"）高出 **20 个百分点**
- 按 Insight 类型排序的难度：Descriptive (0.52-0.62) > Diagnostic > Prescriptive > Predictive（预测性 Insight 最难）
- 多样化 Follow-up 问题（覆盖四种类型）优于单一类型的追问

### 1.7 质量保证

20 名具备基础数据科学技能的志愿者通过专门的 Gradio 界面评估了数据集质量，覆盖三个维度：① 数据集描述和目标的清晰度 ② 问题和 Insight 的相关性和趣味性 ③ 总结的准确性和完整性。平均评分 4/5。

---

## 2. DACO（NeurIPS 2024 Dataset & Benchmark Track）

**论文**: DACO: Towards Application-Driven and Comprehensive Data Analysis via Code Generation  
**链接**: https://arxiv.org/abs/2403.02528  
**代码**: https://github.com/shirley-wu/daco

### 2.1 数据集组成

每个数据样本包含以下部分：

| 组成部分 | 说明 |
|---------|------|
| **关系型数据库** | 真实场景的多表关系型数据库（来自 Spider 和 Kaggle），每个数据库平均 2.3 个表 |
| **应用驱动的查询 (Query)** | 模拟真实利益相关者角色提出的分析查询，如"作为广告主管，我想选择投放渠道" |
| **中间代码步骤** | 多轮 Python 代码生成与执行过程（平均 3.3 轮，1.9k 行代码） |
| **最终分析答案** | 包含 Findings（发现）和 Suggestions（建议）两个列表，平均约 10 个 bullet point |

**整体规模**:
- 440 个数据库
- 1,942 个 query-answer 对
- Train: 1,558 queries / Dev: 100 / Test A (自动标注): 284 / Test H (人工精标): 100
- 覆盖 10+ 个领域主题（商业、体育、医疗、天气、教育等）

### 2.2 具体 Case 示例

**数据库**: 某咖啡店会员数据库，包含 `member` 表（会员信息，含年龄）和 `happy_hour_member` 表（欢乐时光活动参与记录）

**Query**: "作为咖啡店经理，我想调查是否存在年龄歧视问题。"

**LLM 的多轮分析过程**:

**第 1 轮（代码）**: 
```python
import pandas as pd
member = pd.read_csv('member.csv')
print(member['age'].describe())
print(member.groupby(pd.cut(member['age'], bins=[18,30,40,50,60,70])).size())
```
→ 观察：会员年龄分布信息

**第 2 轮（代码）**: 
```python
happy_members = pd.merge(happy_hour_member, member, on='member_id')
print(happy_members['age'].mean())
print(member['age'].mean())
```
→ 观察：参加欢乐时光的会员与全体会员的平均年龄对比

**第 3 轮（代码）**: 更深入的统计检验

**最终输出**:
- **Finding 1**: 会员年龄分布较为均匀，各年龄段均有覆盖（数学推理 + 分析推理）
- **Finding 2**: 欢乐时光参与者的平均年龄与总体会员无显著差异（数学推理）
- **Suggestion 1**: 数据不支持年龄歧视的假说（逻辑推理）
- **Suggestion 2**: 建议进一步调查其他可能的差异化因素如消费金额、到店频次（策略推理）

### 2.3 LLM 的输入与输出

| | 内容 |
|---|------|
| **输入** | ① 数据库（多个 DataFrame/表）② 应用驱动的自然语言查询（含角色设定） |
| **输出** | ① 多轮 Python 代码（action）② 每轮代码的执行结果（observation）③ 最终答案：Findings 列表 + Suggestions 列表（约 10 个 bullet point） |

### 2.4 评分标准

**主指标 — Helpfulness（有用性）**，通过**成对比较 (Pairwise Comparison)** 评估：

- 将两个系统的分析结果并列呈现给评判者（人类或 LLM），评判者选出更有帮助的一个
- Helpfulness 定义三个维度：① 与查询的相关性 ② 有效且深入的数据解读 ③ 分析视角的多样性
- 使用 GPT-4o mini + Claude 3.5 Sonnet + Llama 3 8B 三个评判器取平均分

**辅助指标**:
- BLEU（n-gram 重合度）
- Entailment score（NLI 模型评估生成内容是否被 Ground-Truth 蕴含）
- Point-wise helpfulness（人类逐 bullet point 打分：0=不有用, 1=边界有用, 2=非常有用）

**关键结果**: GPT-4 with code generation 的 helpfulness 为 41.88%（与人类精标对比的胜率），说明即使最强模型也远未达到人类水平。

---

## 3. TableBench（AAAI 2025）

**论文**: TableBench: A Comprehensive and Complex Benchmark for Table Question Answering  
**链接**: https://arxiv.org/abs/2408.09174  
**代码**: https://github.com/TableBench/TableBench

### 3.1 数据集组成

每个数据样本包含以下部分：

| 组成部分 | 说明 |
|---------|------|
| **数据表** | 来自 WTQ、SQA、TabFact、FeTaQA、FinQA、AIT-QA 等已有数据集的表格，每表至少 8 行 5 列，平均 16.71 行 × 6.68 列 |
| **问题** | 人工构造的复杂问题，覆盖 4 大类 18 子类 |
| **答案** | 通过 3 个 LLM Agent（分别用 TCoT/SCoT/PoT 方法）投票 + 人工审核后的标准答案 |
| **推理步骤** | 每个答案附带完整的推理步骤（平均 6.26 步） |

**四大问题类别**:

| 大类 | 子类 | 说明 |
|------|------|------|
| **Fact Checking** | Match-Based / Multi-hop | 基于表格的事实核验 |
| **Numerical Reasoning** | Arithmetic / Comparison / Aggregation / Ranking / Counting / Time-based / Multi-hop / Domain-Specific | 数值计算和推理 |
| **Data Analysis** | Descriptive / Anomaly Detection / Statistical / Correlation / Causal / Trend Forecasting / Impact Analysis | 数据分析 |
| **Visualization** | Chart Generation | 图表生成 |

**整体规模**: 886 个测试样本 + 19,661 个训练样本（TableInstruct），覆盖 20 个主题领域（金融、竞赛、体育、科学等）。

### 3.2 具体 Case 示例

**Data Analysis — Correlation Analysis 类型**:

**数据表**: 一个金融报表，包含公司各季度收入、营销支出、研发投入等字段

| Quarter | Revenue | Marketing_Spend | R&D_Spend | Employee_Count | Profit |
|---------|---------|----------------|-----------|----------------|--------|
| Q1 2022 | 450M | 35M | 80M | 12000 | 85M |
| Q2 2022 | 480M | 42M | 82M | 12500 | 90M |
| ... | ... | ... | ... | ... | ... |

**问题**: "分析营销支出与收入之间的相关性，并判断是否存在显著的正相关关系。需要给出相关系数和统计检验结果。"

**期望答案**: "营销支出与收入之间的 Pearson 相关系数为 0.87（p < 0.01），存在显著正相关。每增加 1M 营销支出，预期收入增加约 6.4M。"

**推理步骤（约 6-8 步）**: ① 识别相关列 → ② 计算描述性统计 → ③ 计算 Pearson 相关系数 → ④ 执行 t 检验 → ⑤ 判断显著性 → ⑥ 计算回归系数 → ⑦ 综合结论

### 3.3 LLM 的输入与输出

| | 内容 |
|---|------|
| **输入** | ① 表格数据（Markdown/HTML 格式）② 自然语言问题 ③ 任务指令（指定推理方法 TCoT/SCoT/PoT）④ Few-shot 示例 |
| **输出（TCoT）** | 文本形式的逐步推理过程 + 最终答案 |
| **输出（SCoT）** | 交替的"分析-Python指令-模拟结果"步骤 + 最终答案 |
| **输出（PoT）** | 完整的可执行 Python 代码 → 执行得到最终答案 |

### 3.4 评分标准

- **主指标**: ROUGE-L（答案与 Ground-Truth 的 n-gram 重合度）
- **可视化任务**: pass@1（解析执行代码，检查图表 y 轴字段是否准确）
- **一致性验证**: 同时使用 GPT-4 评估和人类评估，计算 Pearson Correlation Coefficient (PCC)，三者高度一致（PCC > 0.98）

**关键结果**: 人类整体 85.91 分，GPT-4-Turbo 最高 51.32 分（TCoT），差距显著。Data Analysis 类别上 GPT-4 仅 41.03 分（人类 82.1），Visualization 类别 GPT-4 为 62.00（人类 86.3）。

---

## 4. KAHAN（EMNLP 2025 Findings）

**论文**: KAHAN: Knowledge-Augmented Hierarchical Analysis and Narration for Financial Data Narration  
**链接**: https://arxiv.org/abs/2509.17037  
**代码**: https://github.com/yajingyang/kahan

### 4.1 数据集组成

基于 **DataTales** 金融报告 Benchmark，数据样本包含：

| 组成部分 | 说明 |
|---------|------|
| **金融数据表** | 公司财报中的结构化数值表格（收入、利润、各业务线数据等） |
| **分析层级** | 四级层次化分析：Entity（单实体）→ Pairwise（成对比较）→ Group（分组）→ System（整体） |
| **知识增强** | LLM 作为领域专家提供的背景知识（行业上下文、市场环境等） |
| **叙述文本** | 基于数据分析生成的金融叙述报告 |

### 4.2 具体 Case 示例

**数据表**: 某科技公司 2023 年季度财报

| Segment | Q1 Revenue | Q2 Revenue | Q3 Revenue | Q4 Revenue | YoY Growth |
|---------|-----------|-----------|-----------|-----------|------------|
| Cloud Services | 12.5B | 13.2B | 14.1B | 15.0B | +22% |
| Hardware | 8.3B | 7.9B | 8.1B | 9.5B | +5% |
| Advertising | 5.1B | 5.5B | 5.8B | 6.2B | +12% |

**层级化分析输出**:

- **Entity 级别**: "Cloud Services 是增长最快的业务线，Q4 收入达到 15.0B，同比增长 22%。"
- **Pairwise 级别**: "Cloud Services 的增速是 Hardware 的 4.4 倍，两者差距在持续扩大。"
- **Group 级别**: "数字服务（Cloud + Advertising）占总收入的比重从 Q1 的 68% 上升到 Q4 的 69%。"
- **System 级别**: "整体来看，公司正在经历从硬件向云服务的转型，这一趋势与行业大势一致。"

### 4.3 LLM 的输入与输出

| | 内容 |
|---|------|
| **输入** | ① 金融数据表 ② 领域知识（由 LLM 作为领域专家生成的背景信息）③ 分析层级指示 |
| **输出** | 层次化的金融叙述报告（Entity → Pairwise → Group → System 四个层面的分析文本） |

### 4.4 评分标准

- **叙述质量**: GPT-4o 评估（KAHAN 超出基线方法 20%+）
- **事实准确性**: 98.2% factuality（人类评估）
- **人类评估**: 整体实用性评价

---

## 5. QRData（ACL Findings 2024）

**论文**: Are LLMs Capable of Data-Based Statistical and Causal Reasoning? Benchmarking Advanced Quantitative Reasoning with Data  
**链接**: https://aclanthology.org/2024.findings-acl.548/

### 5.1 数据集组成

| 组成部分 | 说明 |
|---------|------|
| **数据表** | 真实世界的统计数据表 |
| **统计/因果推理问题** | 需要进行统计检验或因果推理的问题 |
| **标准答案** | 包含统计方法选择、计算过程和结论 |

### 5.2 具体 Case 示例

**数据表**: 某研究的实验数据（处理组 vs 对照组的结果指标）

**问题**: "根据提供的数据，变量 X 和变量 Y 之间是否存在因果关系？请选择合适的统计方法进行验证。"

**期望回答**: ① 识别需要使用的统计方法（如回归分析/卡方检验/t-检验）② 执行计算得到统计量 ③ 判断显著性 ④ 给出因果推断的结论及限制条件

### 5.3 LLM 的输入与输出

| | 内容 |
|---|------|
| **输入** | ① 数据表 ② 统计/因果推理问题 |
| **输出** | 统计方法选择 + 推理过程 + 量化结论 |

### 5.4 评分标准

评估 LLM 在以下维度的表现：
- 统计方法选择的正确性
- 计算结果的准确性
- 因果推断结论的合理性

**关键发现**: 最好的闭源 LLM 在高级定量推理上仍表现不佳，与简单的 Table QA 存在巨大差距。

---

## 6. DSEval（2024）

**论文**: Benchmarking Data Science Agents  
**链接**: https://arxiv.org/abs/2402.17168

### 6.1 数据集组成

| 组成部分 | 说明 |
|---------|------|
| **数据科学任务** | 覆盖数据科学全生命周期：数据探索、数据清洗、特征工程、建模、评估 |
| **数据集** | 多种类型的真实数据集 |
| **任务描述** | 自然语言描述的数据科学任务 |
| **评估标准** | 通过 bootstrapped annotation 方法生成的标准答案 |

### 6.2 具体 Case 示例

**任务**: "对给定数据集进行探索性分析，找出缺失值模式并提出处理策略。"

**输入**: 一个包含缺失值的 CSV 数据集 + 任务描述

**期望输出**: ① 缺失值统计报告 ② 缺失模式识别（随机/非随机）③ 推荐的填充策略 + 理由

### 6.3 评分标准

- 评估 Agent 在数据科学各环节的行为正确性
- 侧重整体行为评估而非单纯的代码生成质量

---

## 7. MMTU（NeurIPS 2025）

**论文**: MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark  
**链接**: https://arxiv.org/abs/2506.05587  
**代码**: https://github.com/MMTU-Benchmark/MMTU  
**数据**: https://huggingface.co/datasets/MMTU-benchmark/MMTU

### 7.1 任务定义

MMTU 旨在全面评估 LLM 在**专家级表格任务**上的能力——包括表格理解、推理和操作。这些任务来自数十年的数据管理 (SIGMOD/VLDB)、编程语言 (PLDI/POPL)、Web 数据 (WWW/WSDM) 等领域的计算机科学研究，反映了数据工程师、数据分析师、数据库管理员在真实工作中面临的挑战。

与现有 benchmark（主要聚焦 NL-to-SQL 和 Table-QA）不同，MMTU 涵盖了 25 种不同的表格任务，其中 19 种是**首次用于评估基础模型**的任务类型。

### 7.2 数据集组成

#### 整体统计

| 统计项 | 数值 |
|--------|------|
| **总问题数** | 28,136 |
| **任务类型数 / 数据集数** | 25 / 52 |
| **总表格数** | 61,763 |
| **编码类问题** | 7,331 (26.1%)：其中 SQL 2,489 / Python Pandas 1,329 / 电子表格公式 3,513 |
| **非编码类问题** | 20,805 (73.9%) |
| **单表问题** | 20,048 (71.3%) |
| **双表问题** | 5,285 (18.8%) |
| **三表及以上问题** | 2,803 (10.0%) |
| **平均表格行数** | 2,659 |
| **平均表格列数** | 11 |
| **平均表格单元格数** | 33,251 |
| **表格来源** | Web 表格 74.9% / 电子表格 7.4% / 关系型表格 17.7% |

#### 每条数据的标准三元组格式

MMTU 的每个问题都统一为 **`<Instruction, Input-Table(s), Ground-truth answer>`** 三元组格式：

| 组成部分 | 说明 |
|---------|------|
| **Instruction** | 任务指令，描述需要完成的具体操作和期望的输出格式（通常要求 JSON 格式输出） |
| **Input-Table(s)** | 一个或多个输入表格（Markdown/CSV/JSON/HTML 格式序列化） |
| **Ground-truth answer** | 人工标注或经过验证的标准答案 |

### 7.3 完整的 25 种任务类型

MMTU 的 25 种任务分为 **10 个大类**：

#### 大类 1：Table Transform（表格变换）

| 任务 | 缩写 | 说明 | 评估指标 | 问题数 |
|------|------|------|---------|--------|
| Table Transform by Relationalization | TTBR | 将半结构化表格关系化（如展开嵌套数据） | Accuracy | 230 |
| Table Transform by Output Schema | TTBS | 根据目标 Schema 合成变换程序 | Accuracy | 685 |
| Table Transform by Output Table | TTBT | 根据输入/输出表格对推断变换程序（SQL 或 Pandas） | Accuracy | 86 |

**Case 示例（TTBT）**:
- **Instruction**: "Please examine the input table, and a desired output table below. Generate a SQL script that can run on the input to produce the output table, in JSON `{\"sql\": <SQL>}`"
- **Input**: 一个包含 Product 和 Sales 列的明细表 + 一个按 Product 汇总 Sales 的目标表
- **Ground-truth**: `{"sql": "SELECT Product, SUM(Sales) from table GROUP BY Product"}`

#### 大类 2：Table Matching（表格匹配）

| 任务 | 缩写 | 说明 | 评估指标 | 问题数 |
|------|------|------|---------|--------|
| Entity Matching | EM | 判断两个表中的行是否指向同一实体 | Accuracy | 4,228 |
| Schema Matching | SM | 匹配两个表中语义相同的列 | F1 | 669 |
| Head Value Matching | HVM | 将无标题表格的列与打乱的列标题匹配 | Accuracy | 900 |

**Case 示例（SM）**:
- **Instruction**: "Please inspect the two related tables below, and identify pairs of columns that correspond to the same semantic concept, in JSON `{\"match\": <pairs>}`"
- **Input**: 表 A 包含 `Product, Sales` 列；表 B 包含 `Item, Revenue` 列
- **Ground-truth**: `{"match": [("tab-A.Product", "tab-B.Item"), ("tab-A.Sales", "tab-B.Revenue")]}`

#### 大类 3：Data Cleaning（数据清洗）

| 任务 | 缩写 | 说明 | 评估指标 | 问题数 |
|------|------|------|---------|--------|
| Data Imputation | DI | 预测表格中的缺失值 | Accuracy | 1,971 |
| Error Detection | ED | 检测表格中的错误单元格 | F1 | 1,891 |
| List to Table | L2T | 将无分隔符的列表数据拆分为规范表格 | Accuracy | 957 |

**Case 示例（DI）**:
- **Instruction**: "Please review the table below, and predict the value in the `<MISSING>` cell in the table, in JSON `{\"missing\": <value>}`"
- **Input**: 一个含有姓名、国籍等字段的表格，某个国籍单元格标记为 `<MISSING>`
- **Ground-truth**: `{"missing": "Europe"}`
- **典型错误**: 模型可能填入 "D. H. McFadden"（缩写）而非正确的 "David Henry McFadden"（全名），因为忽视了同列其他值的格式一致性

#### 大类 4：Table Join（表格连接）

| 任务 | 缩写 | 说明 | 评估指标 | 问题数 |
|------|------|------|---------|--------|
| Semantic Join | SJ | 基于语义关系（如国家-首都）连接两表 | Accuracy | 130 |
| Equi-Join Detect | EJ | 识别多表之间的等值连接关系 | F1 | 494 |

**Case 示例（EJ）**:
- **Instruction**: "Given a database with multiple tables below, identify all key/foreign key join relationships between tables, using JSON `{\"join\": <columns>}`"
- **Input**: 三个关联表（员工表、部门表、项目表）
- **Ground-truth**: `{"join": [("tab-A.Department", "tab-B.D_id"), ("tab-A.Emp_id", "tab-C.E_id")]}`

#### 大类 5：Column Transform（列变换）

| 任务 | 缩写 | 说明 | 评估指标 | 问题数 |
|------|------|------|---------|--------|
| Program Transform by Example | PTBE | 根据输入/输出示例合成列变换程序 | Accuracy | 558 |
| Formula by Context | FBC | 根据电子表格上下文预测公式 | Accuracy | 3,513 |
| Semantic Transform by Example | STBE | 预测语义变换（如公司→股票代码） | Accuracy | 131 |

**Case 示例（PTBE）**:
- **Instruction**: "Please examine the two example output values in 'Output Column' below, and generate Python Pandas code that can produce the desired output"
- **Input**: 含 First/Last/Middle 列的表格 + 2 个输出示例如 "Smith, John A."
- **Ground-truth**: `{"python": "df['Output'] = df['Last'] + ', ' + df['First'] + df['Middle'].apply(lambda...)"}`

#### 大类 6：Column Relationship（列关系发现）

| 任务 | 缩写 | 说明 | 评估指标 | 问题数 |
|------|------|------|---------|--------|
| Arithmetic Relationship | AR | 发现列之间的算术关系 | F1 | 818 |
| Functional Relationship | FR | 发现列之间的函数依赖关系 | F1 | 267 |
| String Relationship | SR | 发现列之间的字符串变换关系 | F1 | 765 |

**Case 示例（AR）**:
- **Instruction**: "Please review the table below, and identify arithmetic relationships that exist between columns"
- **Input**: 含 Sales/Cost/Profit/Margin 列的表格
- **Ground-truth**: `{"relationship": ["Profit = Sales - Cost", "Margin = Profit / Sales"]}`

#### 大类 7：Table Understanding（表格理解）

| 任务 | 缩写 | 说明 | 评估指标 | 问题数 |
|------|------|------|---------|--------|
| Needle-in-a-Haystack Table | NIHT | 在大表中检索特定单元格的值 | Accuracy | 999 |
| Needle-in-a-Haystack Index | NIHI | 根据单元格值检索其行列索引 | Accuracy | 1,000 |

这是 MMTU 设计的**最简单的基准测试**——单纯的值检索，不需要额外推理。但实验表明，即使是这种简单任务，前沿模型在大表上仍表现不佳（特别是列数 > 25 时准确率跌破 0.5）。

#### 大类 8：NL-2-Code

| 任务 | 缩写 | 说明 | 评估指标 | 问题数 |
|------|------|------|---------|--------|
| NL-2-SQL | NS | 将自然语言翻译为 SQL 查询 | Accuracy | 2,489 |

数据来源于 Spider、WikiSQL、Bird、KaggleDBQA、Archer 等经典 NL-to-SQL 数据集。

#### 大类 9：Table QA（表格问答）

| 任务 | 缩写 | 说明 | 评估指标 | 问题数 |
|------|------|------|---------|--------|
| Table Question Answering | TQA | 基于表格回答问题 | Accuracy | 1,793 |
| Fact Verification | FV | 验证陈述是否被表格事实支持 | Accuracy | 916 |

数据来源于 WikiQA、FinQA、TableBench、TabFact 等。

#### 大类 10：KB Mapping（知识库映射）

| 任务 | 缩写 | 说明 | 评估指标 | 问题数 |
|------|------|------|---------|--------|
| Column Type Annotation | CTA | 将表格列映射到知识库实体类型 | Accuracy | 881 |
| Column Property Annotation | CPA | 预测列对之间的知识库属性关系 | Accuracy | 873 |
| Cell Entity Annotation | CEA | 将单元格映射到知识库实体 ID | Accuracy | 892 |

数据来源于 SemTab2019。

### 7.4 LLM 的输入与输出

| | 内容 |
|---|------|
| **输入** | ① 任务指令（Instruction）— 描述任务目标和期望输出格式 ② 一个或多个输入表格（序列化为 Markdown/CSV/JSON/HTML 格式）③ 无 Few-shot 示例（zero-shot 评估） |
| **输出** | 结构化 JSON 格式的答案。编码类任务输出 SQL/Python/公式代码（通过执行验证正确性）；非编码类任务输出预测值/匹配对/关系描述等 |

### 7.5 评分标准

#### 评估框架设计

与使用多选题（A/B/C/D）的 MMLU 等 benchmark 不同，MMTU 采用**结构化开放式答案格式**，支持多种评估方式：

| 评估类型 | 适用任务 | 方法 |
|---------|---------|------|
| **执行验证 (Execution-based)** | SQL 生成、Python 代码生成 | 执行生成的代码，比较执行结果与标准答案 |
| **结构化输出比较** | 匹配、关系发现、数据清洗 | 比较 JSON 结构（支持无序列表比较等） |
| **精确匹配** | 表格理解、事实验证 | 直接字符串匹配 |

#### 每个任务的评估指标

- **Accuracy**: 大多数任务使用准确率（预测完全正确的比例），适用于有唯一答案的任务（如 Data Imputation、Entity Matching、NL-to-SQL 等）
- **F1**: 用于需要精确率和召回率平衡的任务（如 Error Detection、Schema Matching、Arithmetic Relationship 等，因为可能存在多个正确答案）
- **pass@1**: 用于代码生成类任务（执行一次，检查结果是否正确）

#### 质量保证流程

1. **LLM 质量检查**: 使用 o4-mini 检查每个问题是否有歧义或 ground-truth 是否正确，约 8% 的问题被剔除
2. **隐私安全筛查**: 使用 GPT-4o 排除可能涉及隐私或安全风险的内容
3. **数据集容量上限**: 单个数据集最多取 1000 条，保证来源多样性
4. **专家验证**: 每个任务抽样 20 题，由多年经验的领域专家验证数据整合正确性、ground-truth 准确性、指令一致性和评分脚本正确性

### 7.6 关键实验结果

#### 总体表现（Top 模型）

| 模型类型 | 模型 | MMTU 分数 | 每题成本 (US$) |
|---------|------|----------|--------------|
| **Reasoning** | GPT-5 | **0.696** | 0.01727 |
| Reasoning | o3 | 0.691 | 0.01539 |
| Reasoning | GPT-5-mini | 0.667 | 0.00276 |
| Reasoning | Gemini-2.5-pro | 0.665 | 0.00790 |
| Reasoning | DeepSeek-R1 | 0.579 | 0.00167 |
| **Chat** | GPT-5-Chat | 0.577 | 0.00534 |
| Chat | DeepSeek-V3 | 0.555 | 0.00095 |
| Chat | GPT-4o | 0.507 | 0.01019 |
| Chat | Llama-3.3-70B | 0.454 | 0.00150 |
| Chat | Qwen2.5-7B-Instruct | 0.310 | 0.00010 |
| Chat | Llama-3.1-8B | 0.268 | 0.00003 |

#### 各任务类别表现对比（四个前沿模型）

| 任务类别 | GPT-5-Chat | DeepSeek-V3 | GPT-5 (Reasoning) | DeepSeek-R1 (Reasoning) |
|---------|-----------|-----------|-------------------|------------------------|
| Entity Matching | 0.955 | 0.948 | **0.982** | 0.940 |
| Schema Matching | 0.927 | 0.935 | **0.943** | 0.947 |
| Fact Verification | 0.852 | 0.838 | **0.940** | 0.927 |
| Semantic Join | 0.848 | 0.824 | **0.890** | 0.884 |
| Needle-in-Haystack (NIHT) | 0.546 | 0.446 | **0.920** | 0.703 |
| Data Imputation | 0.599 | 0.555 | **0.742** | 0.582 |
| NL-to-SQL | 0.571 | 0.517 | **0.574** | 0.551 |
| Table QA | 0.460 | 0.449 | **0.635** | 0.589 |
| Column Transform (PTBE) | 0.501 | 0.449 | **0.694** | 0.625 |
| Formula Prediction | 0.081 | 0.069 | **0.170** | 0.134 |
| Error Detection | 0.119 | 0.168 | **0.170** | 0.133 |

#### 错误分析

基于人工抽样分析（每任务 10 题）+ LLM 自动错误分类（11K 条错误预测）：

| 错误类别 | 占比 | 说明 |
|---------|------|------|
| **Reasoning & Coding Errors** | 40.3% | 代码逻辑接近正确但缺少关键细节，如忽略列内数据格式一致性 |
| **Table Understanding Errors** | 25.6% | 行/列索引错位（最常见子类，28.3%），长上下文理解困难，多表理解错误 |
| **Knowledge Errors** | 16.4% | 知识幻觉（如 CEA 中编造不存在的知识库引用）、不准确的事实 |
| **Other Errors** | 11.7% | 格式错误、超时、ground-truth 歧义等 |

#### 关键发现

1. **Reasoning 模型明显优于 Chat 模型**: 前沿 Reasoning 模型比 Chat 模型高出 10+ 百分点，因为 MMTU 任务通常需要编码（SQL/Pandas）和任务分解能力
2. **长表格上下文是主要瓶颈**: 随着表格 token 数增加，所有模型准确率持续下降。传统 NLP 的 Needle-in-a-Haystack 测试已被解决（100% 准确），但**表格版 NIHT 仅 ~70%**（GPT-5），且列数 > 25 时跌破 50%
3. **列方向阅读是核心难点**: 50 行 × 5 列的表格准确率为 0.81，但 5 行 × 50 列的表格准确率仅 0.43——说明 LLM 在预训练中主要处理一维文本，对二维表格的"纵向阅读"能力很弱
4. **表格置换敏感性**: 行/列打乱后性能下降（尽管语义不变），列打乱的影响大于行打乱
5. **格式不敏感**: 前沿模型对 Markdown/CSV/JSON/HTML 四种格式已不太敏感（除 HTML 略差外）
6. **公式预测和错误检测最难**: Formula Prediction 最高仅 17%，Error Detection 最高仅 17%，是当前模型最薄弱的领域

---

## 各 Benchmark 对比总结

| 维度 | InsightBench | DACO | TableBench | KAHAN | QRData | DSEval | MMTU |
|------|-------------|------|-----------|-------|--------|-------|------|
| **会议** | ICLR 2025 | NeurIPS 2024 | AAAI 2025 | EMNLP 2025 | ACL 2024 | 2024 | NeurIPS 2025 |
| **核心任务** | 端到端业务洞察生成 | 应用驱动的数据分析 | 综合表格问答 | 金融数据叙述 | 统计/因果推理 | 数据科学全流程 | 多任务表格理解 |
| **数据规模** | 100 数据集 | 1,942 queries | 886 样本 | DataTales 基础 | - | 多任务 | 28K+ 问题 |
| **是否要求代码** | ✅ 需要生成代码 | ✅ 多轮代码生成 | ✅ 支持多种推理 | ❌ 文本输出 | 部分 | ✅ | 部分 |
| **分析深度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **开放程度** | 开放式 Insight | 开放式分析 | 半开放 | 开放式叙述 | 有标准答案 | 有标准答案 | 有标准答案 |
| **评估方式** | LLM-as-Judge | Pairwise 比较 | ROUGE-L + 人工 | GPT-4o + 人工 | 准确率 | 行为正确性 | 准确率 |
| **与你的场景匹配度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |

### 推荐

1. **InsightBench** — 最匹配"表格 + 分析方向 + 分析结论"的需求，数据集包含完整的 Goal → Questions → Insights → Summary 链条
2. **DACO** — 最匹配"应用驱动的深度分析"需求，每个样本有多轮代码分析过程和最终的 Findings + Suggestions
3. **KAHAN** — 适合金融领域的层次化分析场景
4. **TableBench** — 覆盖面最广但部分任务偏简单 QA，Data Analysis 子类别可参考
