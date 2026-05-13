# Benchmark 评测报告：InsightBench 推理链路分析与系统适配评估

> **文档版本**: v2.0  
> **评测日期**: 2026-04-23  
> **评测范围**: InsightBench 5 个 case（flag-1、flag-10、flag-11、flag-12、flag-100）  
> **评测总分**: overall = 0.7233（insight_score = 0.6467, summary_score = 0.8）

---

## 目录

1. [Benchmark 数据集介绍与论文基准分数](#1-benchmark-数据集介绍与论文基准分数)
2. [五个 Case 的 Ground Truth 原始数据](#2-五个-case-的-ground-truth-原始数据)
3. [Agent 生成结果与 GT 逐步对比](#3-agent-生成结果与-gt-逐步对比)
4. [每步推理的因果分析与深层变量发掘评估](#4-每步推理的因果分析与深层变量发掘评估)
5. [独立质量评估：不对照 GT，Agent 结果是否是好的分析？](#5-独立质量评估不对照-gtagent-结果是否是好的分析)
6. [Benchmark 评分是否合理？](#6-benchmark-评分是否合理)
7. [适配层代码差异分析：adapter_insightbench vs 项目主体 data_analysis](#7-适配层代码差异分析adapter_insightbench-vs-项目主体-data_analysis)
8. [评估机制说明](#8-评估机制说明)
9. [总结与改进建议](#9-总结与改进建议)

---

## 1. Benchmark 数据集介绍与论文基准分数

### 1.1 InsightBench

**论文**: [InsightBench: Evaluating Business Analytics Agents Through Multi-Step Insight Generation](https://arxiv.org/abs/2407.06423)（ICLR 2025, ServiceNow Research）

**定位**: 评估端到端数据分析 Agent 的多步洞察生成能力。它不只看代码是否正确，而是评估 Agent 能否从原始数据出发，通过多步分析生成有商业价值的 insight。

**数据集构成**:
- 共 **100 个数据集**（flag-1 至 flag-100），模拟 ServiceNow 企业工作流场景
- 覆盖 5 大主题：Incident Management（20 个）、Asset Management（20 个）、User Management（15 个）、Finance Management（15 个）、Goal Management（10 个）等
- 每个数据集包含：
  - **CSV 数据文件**: 500 条记录，字段包括 `category`、`state`、`opened_at`、`closed_at`、`assigned_to`、`short_description`、`priority`、`location` 等
  - **元数据（metadata）**: 包含分析目标（goal）、角色（role）、数据集描述（dataset_description）
  - **Ground Truth insight 列表（insight_list）**: 人工标注的 insight 序列，每条包含问题（question）、洞察（insight）、数据类型（data_type）、可操作建议（actionable_insight）、分析代码（code）、图表定义（plot）
  - **摘要（summary）**: 人工撰写的全局分析摘要
  - **精简 insight 列表（insights）**: 纯文本的 insight 列表，用于评估匹配
- 难度分 3 档：Easy（30 个，Level 1-2）、Medium（36 个，Level 3）、Hard（34 个，Level 4-5）

**数据设计方法**: 作者通过数学模型（如线性回归控制 TTR 增长斜率）在数据中**人为植入异常和趋势**（称为"flag"），然后由专家标注 Ground Truth 分析路径。flag-100 作为"无异常"基线。

**评估方式**: 双层评估——Insight-Level（逐条 GT insight 找最佳匹配预测，取均分）+ Summary-Level（GT summary vs 预测 summary 语义匹配）。原始方案使用 LLaMA-3-Eval（LLaMA-3-70b 做 judge），我们使用 Kimi-K2.5 做 LLM Judge。

### 1.2 论文报告的基准分数（Table 1）

以下为论文中各 Agent/模型在**全部 100 个数据集**上的表现（5 seeds 平均，LLaMA-3-Eval 评分）：

| Agent | ROUGE-1 | Insight Score | Summary Score | 整体表现定位 |
|-------|---------|--------------|---------------|-------------|
| **AgentPoirot (gpt-4o)** | 0.32±0.02 | **0.60±0.03** | **0.44±0.03** | 🥇 最佳 |
| AgentPoirot (gpt-4-turbo) | 0.30±0.02 | 0.56±0.02 | 0.35±0.04 | 第二梯队 |
| AgentPoirot (llama-3-70b) | 0.33±0.02 | 0.52±0.04 | 0.33±0.01 | 第三梯队 |
| AgentPoirot (gpt-3.5-turbo) | 0.34±0.01 | 0.50±0.02 | 0.31±0.06 | 第四梯队 |
| Pandas Agent (gpt-4o) | 0.35±0.03 | 0.54±0.01 | 0.40±0.04 | 对标基线 |
| AgentPoirot (gpt-4o) w/ generic goal | 0.30±0.03 | 0.40±0.03 | 0.33±0.12 | 退化版本 |

**论文关键发现**:
- **最好的 Agent（AgentPoirot + gpt-4o）在 100 个数据集上 Insight Score 也仅为 0.60**，说明该 benchmark 有相当难度
- 使用泛化 goal（"I want to find interesting insights"）后分数从 0.60 降到 0.40，说明精确的分析目标对性能至关重要
- 按 Insight 类型看，Descriptive（0.52-0.62）> Diagnostic > Prescriptive > Predictive，越复杂的分析类型得分越低
- 按难度看，Easy > Medium > Hard，Hard 数据集得分显著下降

**我们的 Agent 对比定位**:

| 指标 | 我们的 Agent（5 case） | AgentPoirot+gpt-4o（100 case） | Pandas Agent+gpt-4o（100 case） |
|------|----------------------|-------------------------------|-------------------------------|
| Insight Score | **0.6467** | 0.60 | 0.54 |
| Summary Score | **0.80** | 0.44 | 0.40 |

> **注意**: 我们仅测了 5 个 case，样本量太小不具备统计意义，但初步表明 Agent 能力不弱于论文 SOTA 水平。Summary Score 显著偏高（0.80 vs 0.44），可能因为使用不同的 LLM Judge（Kimi-K2.5 vs LLaMA-3-70b），也可能因为我们的 `_generate_summary()` 的 prompt 引导生成了结构更完整的 summary。

### 1.3 DACO

**论文**: [DACO: Towards Application-Driven and Comprehensive Data Analysis via Code Generation](https://arxiv.org/abs/2403.02528)（arXiv 2024）

**定位**: 面向应用驱动的、需要数学和逻辑推理链的数据分析任务。不同于 InsightBench 的"开放探索"模式，DACO 以对话式代码生成为核心。

**数据集构成**:
- **440 个数据库**: 从真实世界场景（Kaggle 等）收集的表格数据
- **约 2000 条输入查询与答案标注**: 分为 `train`、`validation`、`test` 和 `test_h`（人工精标测试集）四个 split
- 每条数据包含：
  - `table_id`: 数据库标识
  - `data_id`: 数据条目唯一 ID
  - `messages`: 对话格式消息列表（user/assistant 交替），assistant 的消息中包含 Python 代码块
- 数据库以 `.pkl` 文件存储，包含 `title`（数据库标题）、`database`（表名→表内容的字典）、`name`（唯一名称）

**DACO 与 InsightBench 的核心差异**:

| 维度 | InsightBench | DACO |
|------|-------------|------|
| 任务形式 | 开放式多步探索 | 对话式逐步代码生成 |
| 输入 | CSV + 一句话目标 | 数据库 + 角色化查询（"As a ..., I want to ..."） |
| 输出 | Insight 列表 + Summary | Findings + Suggestions + Code Trajectory |
| GT 标注 | 人工设计的问题链 + 代码 + 图表 + insight | 人工标注的多轮对话（含代码和输出） |
| 评估重点 | 是否发现了关键 insight | 分析结果对用户是否有帮助 |
| 代码角色 | 分析手段（中间产物） | 核心评估对象（代码质量也被评分） |

两者互补：DACO 测"能不能写对代码"，InsightBench 测"能不能找到关键发现"。

---

## 2. 五个 Case 的 Ground Truth 原始数据

### 2.1 Case 1: flag-1 — 硬件工单分布不均（难度 Level 4）

**元数据**:
- **数据集标题**: Hardware Incident Dataset (Flag 1)
- **CSV 路径**: `data/notebooks/csvs/flag-1.csv`
- **分析目标（Goal）**: "Find the discrepancy and imbalance in distribution of incidents assigned across categories"（找出工单在各分类间分配的差异和不平衡）
- **角色**: L2 Support Agent（二级技术支持人员）
- **分类**: Incidents Management（事件管理）
- **数据集描述**: 该数据集包含 500 条模拟 ServiceNow 事件表的记录，涵盖分类（category）、状态（state）、开启/关闭日期、相关人员、事件详情（位置、描述、优先级等）属性。字段包括 `opened_at`、`closed_at`、`assigned_to`、`short_description`、`priority`，反映了不同地点和分类下事件的运营处理和紧急程度。

**Ground Truth Insight 列表（6 条）**:

#### GT Insight 1（描述性）
- **问题**: 所有类别中事件的分布如何？（What is the distribution of incidents across all categories?）
- **发现**: Hardware 事件数量（336件）显著高于其他类别
- **具体数据**: Hardware=336, Network=51, Software=41, Database=40, Inquiry/Help=32
- **可操作建议**: 由于 Hardware 类别事件数最多，可以考虑为处理该类别的团队分配更多资源或提供额外培训
- **可视化**: 水平柱状图，展示各分类的事件计数

#### GT Insight 2（诊断性）
- **问题**: 为什么大部分事件被分配到 Hardware 类别？（Is there a specific reason why a majority of incidents are being assigned to the hardware category?）
- **发现**: 特定的硬件问题——打印机故障——在事件描述中被大量提及
- **具体数据**: Hardware 类别的工单描述中高频词包括 "printer"、"Issue"、"working properly"、"malfunctioning"、"Australia"
- **分析方法**: 对每个类别的 `short_description` 字段做词云分析
- **可操作建议**: "printer" 的频繁出现说明打印机存在反复性问题，需进一步分析确定具体的故障设备

#### GT Insight 3（诊断性）
- **问题**: 事件描述中 "Printer" 一词的出现分布如何？（What is the occurrence distribution of the word Printer in the incidents?）
- **发现**: 大部分 Hardware 事件与打印机问题相关
- **具体数据**: "Printer" 在事件描述中出现了 225 次
- **可操作建议**: 打印机问题频繁出现，建议针对打印机问题进行专项调查，可能需要联系打印机制造商或服务提供商

#### GT Insight 4（描述性）
- **问题**: 硬件事件是否集中在特定地点？（Are the hardware incidents concentrated in a specific location?）
- **发现**: 大部分硬件事件发生在 Australia 地点
- **具体数据**: Australia=241, USA=25, UK=25, India=25, Canada=20
- **可操作建议**: 鉴于 Australia 是硬件事件的主要来源，应向该地点调配更多资源或支持

#### GT Insight 5（描述性）
- **问题**: 事件分布随时间是否存在趋势？（Is there a pattern or trend over time in the distribution of incidents across categories?）
- **发现**: Hardware 事件没有显著的增长趋势，但始终保持在较高水平且高于其他类别
- **分析方法**: 按月份对 `opened_at` 重采样，按类别统计事件数，绘制折线图

#### GT Insight 6（诊断性）
- **问题**: 哪个具体的 Printer ID 引起最多问题？（What is the printer ID causing the most issues?）
- **发现**: Printer546 是引起最多事件的打印机
- **具体数据**: Printer546=158 次（从 `short_description` 中用正则提取 `Printer\d+`）
- **可操作建议**: 对 Printer546 进行彻底检查（物理检查、软件/固件问题排查、联系制造商），如确认故障则更换或维修

**Ground Truth Summary**:
1. **事件分布严重偏斜**: Hardware 类别占全部事件的 67%，远高于其他类别
2. **根因是打印机硬件故障**: 大部分 Hardware 事件归因于打印机故障，导致硬件相关工单激增
3. **地理集中在 Australia**: 大量硬件事件集中在 Australia，建议向该地点调配更多资源

**GT 推理逻辑链**:  
看分布（发现 Hardware 占 67%）→ 文本分析找根因（词云发现 printer）→ 量化根因（Printer 出现 225 次）→ 地理维度（Australia 占 241 件）→ 时间维度（无增长趋势）→ 锁定具体设备（Printer546 = 158 次）

---

### 2.2 Case 2: flag-10 — 工单解决时间趋势分析（难度 Level 3）

**元数据**:
- **数据集标题**: Incident Resolution Time Trends Analysis (Flag 10)
- **CSV 路径**: `data/notebooks/csvs/flag-10.csv`
- **分析目标（Goal）**: "Identify trends and underlying factors or correlations contributing to the increase in TTR."（识别导致 TTR 增长的趋势和潜在因素或相关性）
- **角色**: Incidents Manager（事件经理）
- **分类**: Incident Management（事件管理）
- **数据集描述**: 与 flag-1 类似的 500 条 ServiceNow 事件记录，但重点关注解决时间相关的趋势。

**Ground Truth Insight 列表（4 条）**:

#### GT Insight 1（诊断性）
- **问题**: TTR（解决时间）随时间的趋势是什么？（What is the trend of time to resolution (TTR) over time?）
- **发现**: 事件解决时间随时间**持续增长**
- **分析方法**: 将 `opened_at` 和 `closed_at` 转为 datetime，计算 `resolution_time = (closed_at - opened_at).dt.total_seconds() / 86400`（以天为单位），绘制散点趋势线
- **可操作建议**: TTR 增长可能由事件量增加、事件复杂度上升或资源限制等因素导致

#### GT Insight 2（诊断性）
- **问题**: 事件量与 TTR 之间是否有相关性？（Is there a correlation between the volume of incidents and the TTR?）
- **发现**: **正相关**——事件量增加时 TTR 也增加
- **分析方法**: 按 `opened_at` 日期分组，同时统计事件数和平均 TTR，绘制双轴折线图
- **可操作建议**: 需要评估容量规划和流程效率，以应对高工单量

#### GT Insight 3（时间序列）
- **问题**: TTR 增长在所有类别中是否均匀？（Is the increase in TTR uniform across all categories of incidents or is it more pronounced in a specific category?）
- **发现**: TTR 增长在所有类别中**均匀**，不是某个类别的特定问题
- **分析方法**: 按类别和日期分组计算平均 TTR，多线绘图
- **可操作建议**: 均匀增长说明问题是系统性的，而非类别特有

#### GT Insight 4（描述性）
- **问题**: Agent 的生产力是否均匀？（Are there any trends in the productivity of the human agents over time?）
- **发现**: 各 agent 的生产力（处理事件数）**大致均匀**
- **分析方法**: 按 `assigned_to` 分组统计事件数，绘制柱状图
- **可操作建议**: 工作负载均匀分配是好的信号，但应持续监控

**Ground Truth Summary**:
1. **TTR 线性增长趋势**: 解决时间随时间线性增长
2. **与事件量正相关**: 事件量增加时 TTR 也增加，可能因为资源限制或处理效率不足
3. **其他因素**: 需考虑 agent 是否具备有效处理事件的技能和能力

**GT 推理逻辑链**:  
直接从 opened_at/closed_at 推导 TTR → 看 TTR 时间趋势（发现线性增长）→ 找相关因素（事件量正相关）→ 排除类别差异（各类别均匀）→ 排除 agent 生产力问题（均匀）

---

### 2.3 Case 3: flag-11 — 分类别解决时间趋势（难度 Level 4）

**元数据**:
- **数据集标题**: Category based Incident Trends Analysis (Flag 11)
- **CSV 路径**: `data/notebooks/csvs/flag-11.csv`
- **分析目标（Goal）**: "Analyze the incident data to identify trends and underlying causes for the increasing resolution time in certain category."（分析事件数据以识别特定类别解决时间增加的趋势和根本原因）
- **角色**: L2 Engineering Manager（二级工程经理）
- **分类**: Incident Management（事件管理）
- **数据集描述**: 与 flag-1 类似的 500 条 ServiceNow 事件记录，涵盖各种属性，用于分析特定类别的解决时间趋势。

**Ground Truth Insight 列表（6 条）**:

#### GT Insight 1（描述性）
- **问题**: Hardware 的 TTR 趋势如何，特别是在异常期间？（What is the trend in the TTR for Hardware incidents, especially during the identified anomaly periods?）
- **发现**: Hardware 的 TTR 从 2023-07 开始呈线性增长
- **分析方法**: 按类别绘制 TTR 折线图，用 `(closed_at - opened_at).dt.total_seconds() / 86400` 计算 TTR
- **可操作建议**: 解决这些时期 TTR 增加的根本原因可以提高整体服务效率和客户满意度

#### GT Insight 2（描述性）
- **问题**: 各类别事件量随时间如何分布？（How are incidents distributed across different categories over time?）
- **发现**: Hardware 在 2023-06 到 2023-08 期间事件量激增，达到平常的 4-5 倍
- **分析方法**: 按月和类别分组统计事件数，绘制计数图
- **可操作建议**: 识别高事件率的特定时间有助于提前分配资源和应对准备

#### GT Insight 3（描述性）
- **问题**: 哪些时间窗口观察到 Hardware 事件激增？（During which periods do we observe spikes in incident reports, particularly in the Hardware category?）
- **发现**: 具体时间窗口——2023-07（47 件）和 2023-08（43 件），而月平均仅约 6 件
- **分析方法**: 筛选 Hardware 类别，按月分组统计，绘制柱状图并标注数值
- **可操作建议**: 聚焦这些高活跃期可以指导有针对性的故障排除和预防措施

#### GT Insight 4（描述性）
- **问题**: Hardware 事件激增是否有地理模式？（Are there geographical patterns associated with the spikes in Hardware incidents?）
- **发现**: Hardware 事件在激增期间主要集中在 Australia
- **分析方法**: 按月和 location 分组统计事件数

#### GT Insight 5（描述性）
- **问题**: 异常期间 Hardware 的 TTR 趋势如何？（What is the trend in the TTR for Hardware incidents during the identified anomaly periods?）
- **发现**: Hardware 在事件频率升高的时期 TTR 呈增长趋势
- **分析方法**: 过滤 2023-06-01 至 2023-08-31 的 Hardware 数据，绘制 TTR 折线图
- **可操作建议**: 解决异常期间 TTR 增加的根本原因可以增强服务效率

#### GT Insight 6（描述性）
- **问题**: 能否识别在异常期间最有问题的硬件类型？（Can we identify specific sub-categories or types of hardware that are most problematic during these anomaly periods?）
- **发现**: 特定的系统宕机类型被识别为问题根源——Email 服务器宕机
- **分析方法**: 对各类别的 `short_description` 进行词云分析，高频词包括 email、outage、system
- **可操作建议**: 针对宕机频发的特定硬件类型进行维护或升级可以缓解高事件率

**Ground Truth Summary**:
1. **Hardware TTR 线性增长**: Hardware 类别的 TTR 从 2023-07-01 开始呈线性增长趋势
2. **与事件量激增吻合**: TTR 线性增长与 Hardware 事件量激增同期发生，2023 年 7-8 月间 Hardware 工单是平时的 4-5 倍，可能与新硬件部署、软件更新或外部因素有关
3. **地理集中**: 激增主要集中在 Australia
4. **根因是 email 服务器宕机**: Hardware 事件频率增长和解决时间延长主要归因于 email 服务器宕机问题

**GT 推理逻辑链**:  
看 TTR 趋势（发现 Hardware 从 2023-07 异常增长）→ 看各类别事件量趋势（发现 Hardware 在 7-8 月激增 4-5 倍）→ 量化激增窗口（2023-07 = 47 件，2023-08 = 43 件，月均仅 6 件）→ 地理维度（集中在 Australia）→ 异常期 TTR 验证（确认上升）→ 文本分析找根因（词云显示 email server outage）

---

### 2.4 Case 4: flag-12 — 简单版硬件工单分布（难度 Level 1，最简单）

**元数据**:
- **数据集标题**: Hardware Incident Easy Dataset (Flag 12)
- **CSV 路径**: `data/notebooks/csvs/flag-12.csv`
- **分析目标（Goal）**: "Find the discrepancy and imbalance in incidents assigned"（找出工单分配中的差异和不平衡）
- **角色**: L1 Agent（一级支持人员）
- **分类**: Incidents Management（事件管理）
- **数据集描述**: 与 flag-1 类似的 500 条记录，但没有 location 字段，难度为 Level 1。

**Ground Truth Insight 列表（4 条）**:

#### GT Insight 1（描述性）
- **问题**: 各类别事件分布如何？（What is the distribution of incidents across all categories?）
- **发现**: Hardware 事件数量显著高于其他类别
- **具体数据**: Hardware=406, Software=33, Network=22, Inquiry/Help=20, Database=19
- **可操作建议**: Hardware 事件最多，应为处理该类别的团队分配更多资源

#### GT Insight 2（描述性）
- **问题**: 为什么大部分事件被分配到 Hardware 类别？（Is there a specific reason why a majority of incidents are being assigned to the hardware category?）
- **发现**: 大部分 Hardware 事件与打印机问题相关
- **具体数据**: "Printer" 在 Hardware 工单描述中出现 166 次
- **可操作建议**: 建议对打印机问题进行专项调查

#### GT Insight 3（描述性）
- **问题**: 硬件事件是否集中在某个地点？（Are the hardware incidents concentrated in a specific location?）
- **发现**: **数据集中没有 location 字段**，也没有在 short_description 中提及地理信息
- **可操作建议**: 鉴于地理位置信息缺失，需要投入时间和资源来确认事件最频繁发生的地点

#### GT Insight 4（描述性）
- **问题**: 事件分布随时间是否有趋势？（Is there a pattern or trend over time in the distribution of incidents across categories?）
- **发现**: Hardware 没有显著增长趋势，相对稳定但持续高于其他类别
- **分析方法**: 按月重采样，按类别统计事件数，绘制折线图

**Ground Truth Summary**:
1. **事件分布严重偏斜**: Hardware 类别占全部事件的 71%（406/500），远高于其他类别
2. **根因不完全明确**: 打印机相关问题突出，但需进一步调查其他原因

**GT 推理逻辑链**:  
分布（Hardware 占 81.2%）→ 文本根因（printer 166 次）→ 地理（字段缺失，需进一步调查）→ 时间趋势（稳定）

---

### 2.5 Case 5: flag-100 — 均衡分布基线（对照实验）

**元数据**:
- **数据集标题**: Hardware Incident Dataset (Flag 100)
- **CSV 路径**: `data/notebooks/csvs/flag-100.csv`
- **分析目标（Goal）**: "Find the discrepancy and imbalance in distribution of incidents assigned across categories"（与 flag-1 目标完全相同）
- **角色**: L2 Support Agent
- **分类**: Incidents Management
- **数据集描述**: 同 flag-1

**Ground Truth Insight 列表（6 条）**:

#### GT Insight 1（描述性）
- **问题**: 所有类别中事件的分布如何？
- **发现**: 事件**完全均匀分布**，每个类别恰好 100 件
- **具体数据**: Software=100, Network=100, Inquiry/Help=100, Hardware=100, Database=100
- **可操作建议**: 工作量均衡，无需针对特定类别额外关注

#### GT Insight 2（诊断性）
- **问题**: 为什么大部分事件被分配到 Hardware 类别？
- **发现**: 事件描述中**没有特定问题突出**
- **分析方法**: 词云分析
- **可操作建议**: 基于词云结果没有可采取的行动

#### GT Insight 3（诊断性）
- **问题**: 事件描述中 "Printer" 出现多少次？
- **发现**: Printer 出现次数为 **0 次**（与 flag-1 形成对比，此数据集没有打印机问题）
- **可操作建议**: 无需采取行动

#### GT Insight 4（描述性）
- **问题**: 硬件事件是否集中在特定地点？
- **发现**: **不集中**，各地点分布均匀（Australia=22, USA=21, UK=20, India=19, Canada=18）
- **可操作建议**: 无需针对特定地点采取行动

#### GT Insight 5（描述性）
- **问题**: 事件分布随时间是否有趋势？
- **发现**: Hardware 和其他类别都**没有显著增长趋势**

#### GT Insight 6（其他）
- **问题**: 不同类型事件最有效的解决方法是什么？
- **发现**: **Restart service** 是最常见的解决方法
- **分析方法**: 对 `resolution_method` 字段做 value_counts

**Ground Truth Summary**:
1. **最常见解决方法**: Restart service 是最常见的解决方法，其次是更新软件和用户培训
2. **优先级分布**: 事件在不同优先级间分布均匀
3. **类别分布**: 最常见的事件类别包括 Hardware、Software、Network、Database，以及相当数量的 Inquiry/Help

**GT 推理逻辑链**:  
与 flag-1 使用相同问题模板 → 逐一确认"没有异常"（分布均匀、无打印机问题、不集中、无趋势）→ 额外分析 resolution_method

**flag-100 的设计意图**: 作为 flag-1 的"对照实验"——相同的分析目标和问题模板，但数据中没有植入异常，期望 Agent 能正确识别"一切正常，没有不平衡"。

---

## 3. Agent 生成结果与 GT 逐步对比

### 3.1 Case 1: flag-1 对比（得分: insight=0.80, summary=1.0）

#### GT 推理链路（6 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | descriptive | 各类别的事件分布如何？ | Hardware=336 件占 67%，远高于其他类别 |
| 2 | diagnostic | 为什么 Hardware 最多？（词云分析） | short_description 中 "printer"、"malfunctioning" 频繁出现 |
| 3 | diagnostic | Printer 出现多少次？ | 出现 225 次 |
| 4 | descriptive | 硬件事件是否集中在某地点？ | Australia=241 件，远超其他地点 |
| 5 | descriptive | 事件分布随时间有无趋势？ | Hardware 无增长趋势但始终最高 |
| 6 | diagnostic | 哪个 Printer ID 最多？ | Printer546=158 次 |

#### Agent 推理链路（5 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | descriptive | category 与 assignment_group 的分布对比 | Hardware 336-341 件占 67-68%；发现 32 件 Inquiry/Help 无分配组、25 件 Service Desk 无分类（11.4% 路由失败） |
| 2 | predictive | 各 category 月度分布与环比增长率 | Hardware 月均 25.8 件稳定；小类别波动剧烈（Database/Software 有 300-700% 月度尖峰） |
| 3 | descriptive | short_description 按 category 的高频关键词 | Hardware 词频 908（是 Network 6.1 倍）；"printer" 占 98.2%（275/280），"australia" 占 93.2% |
| 4 | descriptive | assigned_to × category 交叉（含解决时长） | 5 人均 60-70% 工作量在 Hardware；Inquiry/Help 平均 204.2h 最慢；Charlie W. 最重（115 件） |
| 5 | descriptive | priority × location × category 交叉 | Hardware 在 Australia 集中 241 件（71.7%）；92.3% 高/Critical 优先级；三重集中 |

#### 逐步对比分析

**步骤 1 — 类别分布（两方均有）**:
- GT 和 Agent 都以类别分布为起点，得到了一致的结论（Hardware 占比最高）
- Agent 额外分析了 assignment_group 与 category 的对比，发现了 11.4% 的路由失败
- **一致性**: ✅ 高度一致

**步骤 2 — 深入原因分析（方向分歧）**:
- GT 选择对 `short_description` 做词云分析，找文本层面的根因
- Agent 选择分析月度时间趋势和环比增长率
- **差异原因**: Agent 的 LLM 根据 schema 列名自动规划，优先选择了时间维度分析
- **评价**: GT 的方向更直接触及问题根因（为什么 Hardware 多→因为打印机坏了），Agent 的方向回答了时间维度的问题

**步骤 3 — 文本分析（Agent 也做了）**:
- GT 量化 "Printer" 出现次数（225 次）
- Agent 进行了按类别的关键词频率分析，发现 "printer" 占 98.2%（275/280），"australia" 占 93.2%
- **评价**: Agent 最终也做了文本分析，找到了 printer 和 australia 的极端集中度。量化程度更高。

**步骤 4 — 地理维度 vs 解决时间**:
- GT 分析 Hardware 事件的地理分布（Australia 占大头）
- Agent 分析了 assigned_to × category 的人员负载矩阵和解决时长
- **差异**: 完全不同的分析维度。GT 继续沿着"Hardware 为什么多"深挖，Agent 转向了效率分析
- **评价**: 两者都有价值，但 GT 的路径更紧扣 goal（找分配不平衡）

**步骤 5 — 多维交叉（Agent 覆盖了地理）**:
- GT 做了时间趋势分析
- Agent 做了 priority × location × category 的三维交叉，**发现了 Australia 的 241 件集中**
- **评价**: Agent 虽然在第 4 步没做地理分析，但在第 5 步通过多维交叉补上了

**GT 独有但 Agent 缺失的分析**:
- **具体 Printer ID**（GT 步骤 6）: Agent 未锁定到 Printer546 这个具体设备 ID

**Agent 独有但 GT 缺失的分析**:
- **11.4% 路由失败**: category-assignment_group 之间的脱耦，GT 未涉及
- **人员负载矩阵**: assigned_to × category 的交叉分析，GT 未涉及
- **priority 分布偏斜**: 92.3% 高/Critical 优先级的发现，GT 未涉及

---

### 3.2 Case 2: flag-10 对比（得分: insight=0.75, summary=1.0）

#### GT 推理链路（4 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | diagnostic | TTR 随时间的趋势？ | TTR 线性增长（从 opened_at/closed_at 计算） |
| 2 | diagnostic | 事件量与 TTR 的相关性？ | 正相关 |
| 3 | time_series | TTR 增长是否各类别均匀？ | 是，均匀增长 |
| 4 | descriptive | Agent 生产力是否均匀？ | 是，各 agent 处理量大致相同 |

#### Agent 推理链路（5 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | descriptive | TTR 描述统计（均值/中位数/分位） | 中位 614h，均值 594h，84 件未解决（16.8%） |
| 2 | predictive | 月度 TTR 趋势 + 与事件量相关性 | TTR 从 1 月 3.32 天增到 12 月 35.74 天（11 倍）；**r=0.94** |
| 3 | prescriptive | short_description 关键词 × TTR 关联 | 各类别 TTR 差异仅 1.27 天（software 最长 25.60，hardware 最短 24.33） |
| 4 | diagnostic | priority/group/assigned_to × TTR | Priority 2 (High) 反而最慢（602.80h）；Hardware/Software 组比 Network 慢 80h |
| 5 | diagnostic | TTR 异常值（95th percentile）根因 | Network 组占极端异常 73.7%；Beth Anglin 独占 42.1% |

#### 逐步对比分析

**步骤 1 — TTR 基础统计 vs TTR 趋势**:
- GT 直接计算 TTR 并看时间趋势
- Agent 先计算 TTR 的整体分布统计（平均、最小、最大、分区间分布）
- **评价**: Agent 的起步更扎实——先看整体分布再看趋势是更严谨的分析方法。但 GT 更直接地切入了核心问题

**步骤 2 — 事件量相关性（两方覆盖）**:
- GT 分析事件量和 TTR 的相关性（发现正相关）
- Agent 在月度趋势分析中**直接计算了相关系数 r=0.94**，同时发现 11 倍增长
- **评价**: Agent 这次覆盖了 GT 的核心发现，且量化更精确

**步骤 3 — 类别均匀性 vs 文本 × TTR**:
- GT 验证 TTR 增长在所有类别中均匀（排除类别特异性）
- Agent 做了 short_description 关键词与 TTR 的关联分析，间接发现类别间差异仅 1.27 天
- **评价**: Agent 的发现间接说明了均匀性，但没有明确按类别拆分 TTR 趋势做验证

**步骤 4 — Agent 生产力 vs 多因素交叉**:
- GT 验证各 agent 生产力均匀
- Agent 分析了 priority、assignment_group、assigned_to 与 TTR 的关联，**发现 Priority 2 (High) 反而最慢**
- **评价**: Agent 的分析方向更广，且发现了优先级系统失效这个重要管理问题。同时个人差异 77h 的发现与 GT"均匀"结论一致

**步骤 5 — Agent 独有的异常值分析**:
- GT 没有做异常值分析
- Agent 识别了 95th percentile 异常值，发现 Beth Anglin 独占 42%
- **评价**: 有价值的额外分析，精准定位了人因瓶颈

**关键发现对比**:
- GT 发现了**事件量与 TTR 的正相关**（Agent ✅ 覆盖，r=0.94）
- GT 验证了**各类别均匀增长**（Agent ✅ 间接覆盖，类别间差异仅 1.27 天）
- Agent 发现了**优先级系统无效**（GT 未关注优先级）
- Agent 量化了**月度增长斜率**（GT 只说"线性增长"）

---

### 3.3 Case 3: flag-11 对比（得分: insight=0.85, summary=1.0）

#### GT 推理链路（6 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | descriptive | Hardware 的 TTR 趋势？ | 从 2023-07 开始线性增长 |
| 2 | descriptive | 各类别事件量随时间波动？ | Hardware 在 2023-06~08 激增 4-5 倍 |
| 3 | descriptive | 哪些时间窗口 Hardware 激增？ | 2023-07（47 件）、2023-08（43 件），月均仅 6 件 |
| 4 | descriptive | 激增有地理模式吗？ | 集中在 Australia |
| 5 | descriptive | 异常期间 TTR 趋势？ | Hardware TTR 在异常期显著上升 |
| 6 | descriptive | 哪种硬件最有问题？ | Email 服务器宕机 |

#### Agent 推理链路（5 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | descriptive | 各 category 平均解决时间 | Hardware 平均 1237.45h（是 Inquiry/Help 166.83h 的 **7.42 倍**）；标准差 816h |
| 2 | predictive | 月度 TTR 趋势（按 category） | Hardware 从 149h(1月) 涨到 2723h(次年1月)，**821% 增长**；7 月突增 525.5% |
| 3 | diagnostic | short_description 文本分析 | 82.5% 工单超 72h；"email" 出现在 98.2% 超时工单；最严重案例 2803h（117 天） |
| 4 | diagnostic | priority × group × location × assigned_to 交叉 | **Australia Service Desk 是瓶颈**（929.43h），其他地区仅 315-381h |
| 5 | diagnostic | 极端异常值（>95th percentile） | **100% 极端案例均为 Hardware**（33 件），69.7% Critical |

#### 逐步对比分析

**步骤 1 — TTR 趋势 vs TTR 描述统计**:
- GT 直接看 Hardware 的 TTR 时间趋势
- Agent 先做了各类别的 TTR 描述统计，发现 Hardware 是 Inquiry/Help 的 7.42 倍
- **评价**: Agent 先建立全局视图再深入，方法论严谨

**步骤 2 — 事件量激增 vs 月度 TTR 趋势**:
- GT 步骤 2-3 专门分析 Hardware 事件量的激增窗口（2023-07=47 件、08=43 件，月均 6 件）
- Agent 分析了月度 TTR 趋势，发现 821% 增长和 7 月 525.5% 突增
- **差异**: Agent 分析的是 **TTR** 的月度变化，GT 分析的是**事件量**的月度变化。这是两个不同维度。Agent **没有单独量化事件数量的激增窗口**
- **评价**: 这是 Agent 最主要的遗漏——GT 步骤 2-3 的核心是"事件量激增"，Agent 只看了 TTR 没看 volume

**步骤 3 — 文本分析（两方都有）**:
- GT 用词云分析发现 email server outage
- Agent 发现 "email" 出现在 98.2% 的超时工单中，"outage" 出现率 29.7%
- **一致性**: ✅ 结论高度一致，Agent 的量化更精确

**步骤 4 — 地理模式（两方都有）**:
- GT 发现激增集中在 Australia
- Agent 发现 Australia Service Desk 是瓶颈（929.43h vs 其他 315-381h）
- **一致性**: ✅ 结论一致，Agent 额外定位到了组织层面（Service Desk）

**步骤 5 — Agent 独有的异常值分析**:
- GT 没有做极端异常值分析
- Agent 发现 100% 的极端案例均为 Hardware（33 件），69.7% Critical
- **评价**: 提供了决定性证据，证明 Hardware 是唯一的系统性问题来源

**Agent 独有发现**: Australia Service Desk 929.43h 是最差组合；100% 极端异常均为 Hardware；人员粒度分析（Fred Luddy 在澳洲高优先级工单 1303.39h）。

---

### 3.4 Case 4: flag-12 对比（得分: insight=0.50, summary=0.50）

#### GT 推理链路（4 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | descriptive | 各类别事件分布？ | Hardware=406 占 81.2% |
| 2 | descriptive | 为什么 Hardware 最多？ | Printer 出现 166 次 |
| 3 | descriptive | 硬件事件集中在某地点？ | 数据集没有 location 字段 |
| 4 | descriptive | 有时间趋势吗？ | 无增长，稳定且高 |

#### Agent 推理链路（5 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | descriptive | 各 assigned_to 的事件总数和负载比 | Luke Wilson 最重（116 件, 23.2%），Beth Anglin 最轻（85 件）；比值 1.36 |
| 2 | predictive | 按周的各人员事件创建量波动 | 大量零工单周与突发峰值交替（Beth Anglin 15 周零件） |
| 3 | descriptive | 各人员平均解决时长 | Fred Luddy 最快（165.03h），Luke Wilson 最慢（195.55h） |
| 4 | descriptive | assigned_to × priority 交叉 | Wilson 承担最多高优先级工单（96 件） |
| 5 | diagnostic | short_description 关键词分析 | 各人员关键词高度同质化（printer 占 40-50%），专业化指数极低 |

#### 逐步对比分析

**步骤 1 — 类别分布 vs 人员分布（核心分歧）**:
- GT 第一步就分析了 category 分布，发现 Hardware=406 占 81.2%
- Agent 第一步分析了 assigned_to 的分布——**完全没有看 category**
- **评价**: 这是 Agent 在 flag-12 上的**根本性错误**。Goal 是 "Find the discrepancy and imbalance in incidents assigned"，Agent 将 "assigned" 理解为"分配给人"，GT 理解为"分配到类别"

**步骤 2-4 — Agent 全部围绕人员展开**:
- Agent 的 5 步分析全部以 `assigned_to` 为核心变量，从未分析 `category`
- GT 在步骤 2 就做了文本分析找到 printer 根因
- **评价**: 方向完全错位，导致所有后续分析都偏离了 GT 的核心发现

**步骤 5 — 文本分析（部分覆盖）**:
- Agent 的关键词分析发现 printer 占 40-50%，但这是在人员维度下的分析，没有与 category 关联
- GT 的文本分析是在确认 Hardware 占 81% 后专门分析 Hardware 类别的描述
- **评价**: Agent 间接发现了 printer 问题，但缺乏 category 上下文

**低分根因**: Agent 将 goal 中的 "incidents assigned" 理解为"人员分配"而非"类别分配"，导致 5 步分析完全围绕 `assigned_to` 展开，遗漏了数据中最显著的特征（Hardware 占 81.2%）。对 Level 1（最简单）数据集来说，这种基础信息的遗漏尤为不可接受。

---

### 3.5 Case 5: flag-100 对比（得分: insight=0.3333, summary=0.50）

#### GT 推理链路（6 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | descriptive | 各类别事件分布？ | 完全均匀，每类 100 件 |
| 2 | diagnostic | 描述中有特定问题吗？ | 无，词云无突出模式 |
| 3 | diagnostic | Printer 出现多少次？ | 0 次 |
| 4 | descriptive | 硬件事件集中在某地点吗？ | 不集中（22/21/20/19/18） |
| 5 | descriptive | 有时间趋势吗？ | 无 |
| 6 | 其他 | 最有效的解决方法？ | Restart service 最常见 |

#### Agent 推理链路（5 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | descriptive | category 频数分布和不平衡度 | 完全均匀（每类 100 件），不平衡比 1.00 |
| 2 | descriptive | 月度 × category 时间趋势 | Inquiry/Help 波动最大（CV=0.44），10 月异常 52 件 |
| 3 | descriptive | assignment_group × assigned_to 交叉 | Hardware 组内不均最大（CV=0.29） |
| 4 | descriptive | 各 category 平均成本与受影响用户 | Software 成本最高（$31,276） |
| 5 | diagnostic | rca_category × priority 交叉（含熵值） | 无结构性偏差（HHI 0.169-0.175） |

#### 逐步对比分析

**步骤 1 — 类别分布（两方一致）**:
- 都发现了完全均匀分布（每类 100 件）
- **一致性**: ✅ 完全一致

**步骤 2 — 词云分析 vs 时间趋势**:
- GT 做词云确认无特定问题
- Agent 分析了月度时间趋势，发现 Inquiry/Help 波动最大
- **差异**: Agent 利用了不同的分析路径。GT 意在确认"没有文本异常"，Agent 意在探索时间维度

**步骤 3 — Printer 频率 vs 人员分配**:
- GT 量化 Printer 出现次数（0 次），确认无打印机问题
- Agent 分析了 assignment_group × assigned_to 的内部分配
- **差异**: GT 刻意验证"flag-1 的问题在 flag-100 中不存在"，Agent 做了独立的组织分析

**步骤 4 — 地理分布 vs 成本与影响**:
- GT 确认地理分布均匀
- Agent 分析了 `estimated_cost` 和 `users_affected` 字段
- **差异**: Agent 使用了更多字段（GT 未涉及成本），但没有验证地理维度

**步骤 5 — 时间趋势 vs 根因分类×优先级**:
- GT 确认无时间趋势
- Agent 分析 `rca_category` 与 `priority` 的交叉分布
- **差异**: Agent 发现了一个 GT 完全没有涉及的维度（根因分类的熵值分析）

**步骤 6 — GT 独有: resolution_method 分析**:
- GT 分析了最有效的解决方法（Restart service）
- Agent 完全没有分析 `resolution_method` 字段

**关键差异总结**:

| 维度 | GT | Agent |
|------|-----|-------|
| 核心结论 | 一切均匀，无异常 | 一切均匀，但发现成本/满意度/根因的细粒度差异 |
| 分析 short_description | ✅ 词云 | ❌ 未涉及 |
| 分析 resolution_method | ✅ | ❌ |
| 分析 user_satisfaction_score | ❌ | ✅ |
| 分析 estimated_cost / users_affected | ❌ | ✅ |
| 分析 rca_category | ❌ | ✅ |
| 正确识别"无不平衡" | ✅ | ✅ |

**低分原因分析**: 尽管 Agent 做了更多维度的分析，但由于分析方向与 GT 的问题模板差异较大，LLM Judge 在语义匹配时很难将 Agent 的发现映射到 GT 的 insight 上。特别是 GT 的核心逻辑是"逐一确认没有异常"，而 Agent 的逻辑是"在均匀中寻找细粒度差异"，两者的叙事框架不同。

---

## 4. 每步推理的因果分析与深层变量发掘评估

### 4.1 评估框架

对每步推理按以下三个维度评分（1-5 分）：

| 维度 | 定义 |
|------|------|
| **因果分析深度** | 是否建立了变量间的因果/相关关系，而非只做描述性统计 |
| **变量发掘广度** | 是否利用了数据中的多个字段，发现隐藏变量或推导新指标 |
| **分析合理性** | 推理逻辑是否自洽，结论是否有数据支撑 |

### 4.2 flag-1 各步评估

| 步骤 | 因果深度 | 变量广度 | 合理性 | 综评 |
|------|---------|---------|--------|------|
| 1. 类别 × 分配组分布 | 2/5（纯描述统计） | 3/5（category + assignment_group） | 5/5（数据准确） | 基础扎实，发现路由失败有附加价值 |
| 2. 月度趋势与环比增长 | 2/5（描述性趋势） | 2/5（opened_at + category） | 5/5（数据准确） | 有效的时间维度验证 |
| 3. 文本关键词分析 | 4/5（从描述中提取根因线索） | 3/5（short_description 文本挖掘） | 4/5（找到 printer 和 australia） | 直接触及了问题根因 |
| 4. 人员负载矩阵 | 3/5（发现负载不均和效率悖论） | 4/5（assigned_to + category + 推导解决时长） | 4/5（逻辑自洽） | 从组织管理角度做了诊断 |
| 5. 三维交叉分析 | 3/5（发现三重集中模式） | 5/5（priority + location + category） | 4/5（结论有说服力） | 多维度交叉验证 |

**总体评价**: Agent 在 flag-1 上的推理质量较高，特别是步骤 3 的文本分析和步骤 5 的三维交叉展现了多变量分析能力。步骤 4 发现的"简单问题反而解决慢"悖论是 GT 没有提到的有价值洞察。主要不足是没有锁定到 Printer546 这个具体设备 ID。

### 4.3 flag-10 各步评估

| 步骤 | 因果深度 | 变量广度 | 合理性 | 综评 |
|------|---------|---------|--------|------|
| 1. TTR 分布统计 | 2/5（描述统计） | 3/5（推导了 TTR） | 5/5（数据准确） | 正确推导了 TTR，基础扎实 |
| 2. 月度趋势 + 相关性 | 4/5（计算了 r=0.94） | 3/5（时间 + 事件量 + TTR） | 5/5（相关系数极具说服力） | **关键步骤**：覆盖 GT 核心发现 |
| 3. 文本 × TTR 关联 | 4/5（关联分析） | 4/5（short_description + TTR） | 4/5（发现高复杂问题类型） | GT 未做此分析，有附加价值 |
| 4. 多因素 × TTR | 4/5（发现优先级系统失效） | 4/5（priority + group + assigned_to） | 5/5（结论有力） | **关键发现**：优先级形同虚设 |
| 5. 异常值根因 | 4/5（精准定位人因瓶颈） | 3/5（TTR + assigned_to） | 4/5（42% 集中有说服力） | 精准定位了瓶颈人员 |

**总体评价**: Agent 在 flag-10 上表现出色。步骤 2 直接计算了 r=0.94 的相关系数，覆盖了 GT 的核心发现。步骤 4 发现的优先级悖论是重要的管理洞察。步骤 5 的异常值分析超出了 GT 覆盖范围。

### 4.4 flag-11 各步评估

| 步骤 | 因果深度 | 变量广度 | 合理性 | 综评 |
|------|---------|---------|--------|------|
| 1. 各类别 TTR 统计 | 3/5（发现 7.42 倍差异） | 3/5（category + TTR） | 5/5（数据准确） | 有效定位 Hardware 为异常源 |
| 2. 月度 TTR 趋势 | 3/5（量化了 821% 增长） | 2/5（时间 + TTR） | 5/5（7月突增 525.5% 有冲击力） | 量化精确，但缺事件量维度 |
| 3. 文本 × 超时关联 | 4/5（email 98.2% 命中根因） | 4/5（short_description + TTR 阈值） | 5/5（2803h 案例有冲击力） | **关键步骤**：直接命中根因 |
| 4. 四维交叉分析 | 4/5（发现地理×组织瓶颈） | 5/5（priority + group + location + assigned_to） | 4/5（929h vs 315-381h 有说服力） | 多维度定位到 Australia Service Desk |
| 5. 极端异常值分析 | 4/5（100% Hardware 提供决定性证据） | 3/5（TTR + category + priority） | 5/5（统计上极具说服力） | 为结论提供了最终验证 |

**总体评价**: Agent 在 flag-11 上展现了极佳的分析深度。从多个独立维度（时间、地理、文本、异常值）交叉验证了同一个结论（Hardware email 服务器在 Australia 出了系统性问题），形成了令人信服的证据链。主要不足是未单独量化事件数量的激增窗口。

### 4.5 flag-12 各步评估

| 步骤 | 因果深度 | 变量广度 | 合理性 | 综评 |
|------|---------|---------|--------|------|
| 1. 人员负载统计 | 2/5（纯描述） | 2/5（仅 assigned_to） | 4/5（方法正确） | 方向偏离——未分析 category |
| 2. 周级人员波动 | 2/5（描述性） | 2/5（时间 + assigned_to） | 4/5（数据准确） | 有价值但偏离目标 |
| 3. 人员效率比较 | 3/5（发现效率悖论） | 3/5（assigned_to + TTR） | 4/5（Wilson 负载最重+最慢） | 管理洞察有价值 |
| 4. 人员 × 优先级 | 2/5（描述统计） | 3/5（assigned_to + priority） | 4/5（逻辑合理） | 未触及核心 |
| 5. 关键词同质化 | 3/5（发现路由缺陷） | 3/5（short_description 文本） | 4/5（专业化指数有意义） | 间接发现 printer 问题 |

**总体评价**: 方法论本身没有问题，但因为**完全误解了分析目标**（"incidents assigned" 理解为人员分配而非类别分配），导致 5 步分析全部偏离了数据中最显著的特征（Hardware 占 81.2%）。这说明 LLM 的 goal 理解能力对最终结果有决定性影响。

### 4.6 flag-100 各步评估

| 步骤 | 因果深度 | 变量广度 | 合理性 | 综评 |
|------|---------|---------|--------|------|
| 1. 类别分布 | 2/5（纯描述） | 2/5（仅 category） | 5/5（正确识别均匀） | 准确但基础 |
| 2. 月度异常检测 | 3/5（统计方法检测异常） | 2/5（opened_at + category） | 4/5（CV=0.44 方法合理） | 标准差方法严谨 |
| 3. 组内分配分析 | 2/5（描述统计） | 3/5（assignment_group + assigned_to） | 4/5（CV=0.29 有意义） | 有价值的内部视角 |
| 4. 成本与影响 | 3/5（发现 Software 高成本） | 4/5（estimated_cost + users_affected） | 3/5（负值 TTR 未标记为异常） | 多维度但漏掉数据质量问题 |
| 5. 根因 × 优先级 | 4/5（熵值分析有深度） | 4/5（rca_category + priority） | 4/5（HHI 指标严谨） | **最佳步骤**：深层因果关联 |

**总体评价**: Agent 展示了丰富的分析维度（满意度、成本、根因分类），但由于 GT 的设计意图是"确认没有异常"，Agent 的"在均匀中寻找差异"策略虽然分析质量不低，却与评分标准错位。步骤 5 的根因 × 优先级熵值分析是最具深度的推理。

---

## 5. 独立质量评估：不对照 GT，Agent 结果是否是好的分析？

> 以下评估**完全不参考 GT**，仅从数据分析方法论角度判断 Agent 的分析质量。

### 5.1 flag-1：⭐⭐⭐⭐ (4/5)

**优点**: 发现 11.4% 的路由脱耦（有运维审计价值）；文本分析精确找到 printer（98.2%）和 australia（93.2%）；人员×类别×时长的三维交叉给出了可操作的资源调配建议；最终在步骤 5 通过三维交叉覆盖了地理维度。  
**不足**: 未锁定到 Printer546——在实际运维中定位具体故障源是最终目标。  
**判断**: 高质量运维分析报告。广度充分，某些维度超出预期，但缺了"最后一英里"。

### 5.2 flag-10：⭐⭐⭐⭐⭐ (5/5)

**优点**: 正确推导 TTR 并计算全面描述统计；**直接计算事件量与 TTR 相关系数 r=0.94**——教科书级别；发现优先级系统失效（Priority 2 最慢）是重要管理洞察；异常值分析精准定位人因瓶颈（Beth Anglin 42%）；月度 11 倍增长量化极具冲击力。  
**不足**: 未明确按类别拆分 TTR 趋势验证"是否均匀增长"（不过步骤 3 的类别间 1.27 天差异间接说明了均匀性）。  
**判断**: **5 个 case 中最优秀的分析**。每步有清晰目的，结论间形成逻辑链条，完全达到高级数据分析师水准。

### 5.3 flag-11：⭐⭐⭐⭐⭐ (5/5)

**优点**: 精准锁定 Hardware 是唯一 TTR 恶化来源（7.42 倍差异、821% 增长）；7 月 525.5% 突增被准确捕捉；文本分析命中 email（98.2%）根因；地理×组织交叉发现 Australia Service Desk 929h 瓶颈；极端异常值 100% 为 Hardware 提供决定性证据。  
**不足**: 未单独量化事件数量激增（只看了 TTR 没看 volume）。  
**判断**: 深度极佳，多维度交叉验证形成令人信服的证据链。

### 5.4 flag-12：⭐⭐ (2/5)

**优点**: 人员负载分析方法论正确；发现 Wilson"负载最重+效率最低"悖论有管理价值；关键词同质化分析揭示路由缺陷。  
**严重不足**: **完全误解了分析目标**——5 步分析无一步分析 category 分布，而 Hardware 占 81.2% 是数据中最显著特征。对 Level 1（最简单）数据集的这种遗漏不可接受。  
**判断**: 方法没问题，但方向偏了。说明 LLM 的 goal 理解能力对最终结果有决定性影响。

### 5.5 flag-100：⭐⭐⭐ (3/5)

**优点**: 正确识别均匀分布（不平衡比 1.00）；发现 GT 未涉及的维度（Inquiry/Help 时间波动 CV=0.44、根因与类别名脱耦、Software 成本偏高）；熵值和 HHI 方法论严谨。  
**不足**: 未分析文本字段和 resolution_method；Network 解决时间为负值（-251.49h）是明显数据质量问题但未标记为异常。  
**判断**: 维度丰富但缺乏与 goal 的对齐。在"找不平衡"目标下，"没有不平衡"才是正确结论，Agent 反而在均匀中强找差异。

---

## 6. Benchmark 评分是否合理？

### 6.1 逐 Case 评估

| Case | 独立质量 | Benchmark 得分 | 判断 |
|------|---------|---------------|------|
| flag-1 | ⭐⭐⭐⭐ | insight=0.80 | ✅ 合理 |
| flag-10 | ⭐⭐⭐⭐⭐ | insight=0.75 | ⚠️ 偏低（Agent 覆盖了 GT 全部 4 条发现+额外发现，但路径差异仍被扣分） |
| flag-11 | ⭐⭐⭐⭐⭐ | insight=0.85 | ✅ 合理 |
| flag-12 | ⭐⭐ | insight=0.50 | ✅ 合理（Agent 确实遗漏了最核心的 category 分布） |
| flag-100 | ⭐⭐⭐ | insight=0.33 | ⚠️ 偏低（Agent 做了有价值分析但方向与 GT 错位） |

### 6.2 评分机制的结构性问题

1. **单向匹配偏差**: 评分只看"GT 的每条 insight 是否被覆盖"（recall），不看"Agent 的额外发现是否有价值"（precision）。Agent 在 flag-1 中发现的 11.4% 路由失败、flag-10 的优先级悖论、flag-100 的根因分类脱耦，这些均不计分。

2. **表述风格干扰匹配**: Agent 的 insight 使用大量精确数字和复合从句（如 "98.2% of 'printer' keyword occurrences"），GT 是简洁单句（如 "hardware incidents is significantly higher than others"）。信息密度差异可能导致 LLM Judge 匹配不精确。

3. **对"确认无异常"的评估不友好**: flag-100 这类数据集的 GT 是逐一验证"没问题"。Agent 如果用不同验证路径（如分析满意度而非打印机频率），即使同样得出"没问题"也会因步骤不同而丢分。

### 6.3 与论文基准的对比判断

论文中 AgentPoirot+gpt-4o 的 100 case Insight Score 为 0.60±0.03。我们 5 case 的 0.6467 处于同一水平。考虑到我们跑的 5 个 case 包含 2 个 Level 4（Hard）、1 个 Level 3、1 个 Level 1、1 个 baseline，难度分布不偏向简单题，这个得分水平说明 Agent 的分析能力是有竞争力的。flag-12 的低分（0.50）主要是 goal 理解偏差导致，而非分析能力不足。

---

## 7. 适配层代码差异分析：adapter_insightbench vs 项目主体 data_analysis

### 7.1 调用关系概览

**项目主体 main.py 的调用链**:
```
main.py → analyze_region() → _generate_query_instructions() → _execute_queries() → DataInspectorMCPTool
                                                                                         ↓
                                                                                   CodeAgent.run()
```

**Benchmark 适配器 adapter_insightbench.py 的调用链**:
```
adapter_insightbench.py → run_agent_on_dataset() → analyze_data() → _generate_query_instructions() → _execute_queries()
                                                                                                           ↓
                              ↓ 后处理                                                              DataInspectorMCPTool
                         _extract_insight()                                                              ↓
                         _generate_summary()                                                       CodeAgent.run()
```

### 7.2 核心差异逐项对比

#### 差异 1: 入口函数不同

| 维度 | 项目主体（main.py） | Benchmark 适配器 |
|------|-------------------|-----------------|
| 入口函数 | `analyze_region()` | `run_agent_on_dataset()` → `analyze_data()` |
| 数据来源 | Excel 考核评估总表 + 补充材料目录 | CSV 单文件 + JSON 元数据中的 goal |
| 数据格式 | `{表名: DataFrame}` 字典（可能含多个 Sheet） | `{"data": DataFrame}` 固定单表 |

**影响**: 适配器将 InsightBench 的单 CSV 文件包装为 `{"data": df}` 字典传入 `analyze_data()`，数据结构被简化。项目主体支持多 Excel 文件多 Sheet 的复杂结构。

#### 差异 2: 分析目标/任务描述的来源不同（影响最大）

| 维度 | 项目主体 | Benchmark 适配器 |
|------|---------|-----------------|
| 任务描述 | `task_instruction=""` 或由 region_name 隐含，使用 Jinja2 模板中的地区分析逻辑 | `task_instruction=goal`（来自 JSON 元数据，如 "Find the discrepancy..."） |
| 模板分支 | 模板中 `{% if task_instruction %}` 走通用分析分支 vs `{% else %}` 走地区考核分析分支 | 始终走 `{% if task_instruction %}` 通用分支 |
| 查询指令要求 | 地区分支：侧重"整体概览、单项指标、横向对比、趋势排名、异常突出表现"，利用补充材料 | 通用分支：侧重"数据分布概览、异常值、文本字段分析、时间趋势、相关性分析、根因诊断" |

**影响**: 这是最关键的差异。InsightBench 走的是 `data_analysis_user.j2` 模板中 `task_instruction` 非空的通用分析分支，其 prompt 要求覆盖"文本字段内容分析"、"时间趋势"、"推导指标"等维度。而项目主体走的是地区考核专用分支，更侧重指标排名和横向对比。

#### 差异 3: Schema 描述参数不同

| 维度 | 项目主体 | Benchmark 适配器 |
|------|---------|-----------------|
| `max_sample_rows` | 默认 0（不展示示例行） | 3（展示 3 行示例数据） |
| `max_unique_values` | 默认 0（不展示唯一值） | 10（展示最多 10 个唯一值） |

**影响**: 适配器给 LLM 提供了更丰富的数据预览信息（示例行和唯一值），帮助 LLM 更好地理解数据内容从而生成更贴合数据的查询指令。项目主体不展示示例数据，LLM 只能看到列名和数据类型。

#### 差异 4: 后处理流程的额外步骤

| 维度 | 项目主体 | Benchmark 适配器 |
|------|---------|-----------------|
| 查询结果后处理 | 直接拼接为字符串，传给 DocWriter | 需要额外两步 LLM 调用 |
| 是否提取 insight | ❌ 不提取，原样传递 | ✅ `_extract_insight()` 对每条查询结果用 LLM 提取一段 insight |
| 是否生成 summary | ❌ 在 DocWriter 阶段生成 | ✅ `_generate_summary()` 用 LLM 汇总所有 insight |
| 是否分类问题类型 | ❌ | ✅ `_infer_question_type()` 根据关键词推断类型（descriptive/predictive/prescriptive/diagnostic） |

**影响**: 适配器额外增加了 `_extract_insight()` 和 `_generate_summary()` 两个 LLM 调用步骤。这是为了将 CodeAgent 的原始 print 输出转化为 InsightBench 期望的 insight 格式。项目主体不需要这个步骤，因为它的后续处理（DocWriter + Rewriter）会完成类似的信息提炼和润色工作。

#### 差异 5: 整体流程步骤数

| 项目主体（main.py） | Benchmark 适配器 |
|-------------------|-----------------|
| 1. 读取考核评估总表 | 1. 读取 CSV + goal |
| 2. 添加排名列 | — |
| 3. 查找并读取补充材料 Excel | — |
| 4. 数据分析（LLM 规划 + CodeAgent 执行） | 2. 数据分析（同左） |
| 5. DocWriter 撰写报告初稿 | 3. _extract_insight()（每条查询提取 insight） |
| 6. Rewriter 改写润色 | 4. _infer_question_type()（分类问题类型） |
| 7. 保存最终报告 | 5. _generate_summary()（汇总生成 summary） |

**总结**: 核心分析引擎（`analyze_data()` → `_generate_query_instructions()` → `_execute_queries()` → `DataInspectorMCPTool` → `CodeAgent`）是完全复用的。差异主要在：
1. **输入适配**（CSV 单文件 vs Excel 多文件 + 补充材料）
2. **Prompt 模板分支**（通用分析 vs 地区考核分析）
3. **Schema 详细度**（含示例和唯一值 vs 仅结构）
4. **输出后处理**（提取 insight + 生成 summary vs 传给 DocWriter/Rewriter）

#### 差异 6: LLM 配置

| 维度 | 项目主体 | Benchmark 适配器 |
|------|---------|-----------------|
| 数据分析 LLM | `_create_planning_llm()` → 默认模型 | `OpenAILikeLLM(config=LLMConfig())` → 同默认模型 |
| 报告撰写 LLM | `_create_writing_llm()` → 高级闭源模型 | 无独立写作 LLM，insight 提取和 summary 用同一个 LLM |
| 改写润色 LLM | `_create_rewriting_llm()` → 高级闭源模型 | 无此步骤 |
| 评估 LLM | 无 | `evaluator.py` 使用 Kimi-K2.5 |

**影响**: 项目主体在报告撰写和润色阶段使用了高级闭源模型（通过环境变量 `ADVANCED_MODEL_NAME` 配置），而 Benchmark 适配器在 insight 提取和 summary 生成时只使用默认模型。这意味着 Benchmark 适配器生成的文本质量可能不如项目主体的最终输出。

### 7.3 差异影响评估

| 差异项 | 对 Benchmark 结果的影响程度 | 说明 |
|-------|-------------------------|------|
| 数据格式（单表 vs 多表） | 🟡 中 | InsightBench 本身就是单 CSV，无影响；但 prompt 中的 Sheet 引用变为固定的 "data" |
| Prompt 模板分支 | 🔴 高 | 通用分支的分析维度要求（文本分析、推导指标等）直接影响 LLM 生成的查询质量 |
| Schema 详细度 | 🔴 高 | 展示示例行和唯一值让 LLM 更了解数据内容，是 Agent 能做出合理分析的关键 |
| 后处理步骤 | 🟡 中 | _extract_insight() 的质量影响最终 insight 的表述，但核心分析由 CodeAgent 完成 |
| LLM 配置 | 🟢 低 | 核心分析阶段用的是相同的默认模型 |

---

## 8. 评估机制说明

### 8.1 评估流程

我们的评估实现位于 `run_on_benchmark/evaluator.py`，核心逻辑如下：

1. **加载预测结果和 Ground Truth**: 从 `insightbench_predictions.json` 和 `flag-*.json` 分别读取
2. **逐条 Insight 评估**: 对 GT 的每条 insight，遍历所有预测 insight，找到 LLM Judge 给出的最高分作为该 GT insight 的匹配分数
3. **Summary 评估**: 对 GT summary 和预测 summary 做整体语义匹配
4. **LLM Judge 评分**: 使用 Kimi-K2.5 模型，prompt 要求输出 0.0~1.0 的评分

**LLM Judge Prompt**:
```
You are an expert evaluator. Compare the prediction against the reference and rate the semantic similarity/quality.

Reference (insight/summary): [GT 文本]
Prediction (insight/summary): [预测文本]

Rate from 0.0 to 1.0 where:
- 1.0 = prediction captures the same key insight/finding as the reference
- 0.5 = prediction is partially relevant but misses important aspects
- 0.0 = prediction is irrelevant or completely wrong

Return ONLY a single number between 0.0 and 1.0, nothing else.
```

### 8.2 评估结果汇总

| 数据集 | 难度 | Insight Score | Summary Score | GT Insight 数 | 预测 Insight 数 |
|--------|------|--------------|---------------|---------------|-----------------|
| flag-1 | Level 4 | 0.80 | 1.0 | 6 | 5 |
| flag-10 | Level 3 | 0.75 | 1.0 | 4 | 5 |
| flag-11 | Level 4 | 0.85 | 1.0 | 6 | 5 |
| flag-12 | Level 1 | 0.50 | 0.5 | 4 | 5 |
| flag-100 | — | 0.33 | 0.5 | 6 | 5 |
| **平均** | | **0.6467** | **0.80** | | |
| **Overall** | | **0.7233** | | | |

### 8.3 评估机制的局限性

1. **方向性偏差不被识别**: Agent 可能做了有价值的分析（如发现 silo 化分配模式），但如果 GT 没有对应的 insight，该发现不会被计分
2. **最佳匹配策略的宽松性**: 对每条 GT insight 取所有预测中的最高分，这意味着只要有一条预测"沾边"就能拿分，即使其他预测完全无关
3. **LLM Judge 的稳定性**: 使用 LLM 评分存在随机性，同样的输入可能得到不同的分数
4. **Summary 评估粒度粗**: 整体语义匹配可能忽略细节差异

---

## 9. 总结与改进建议

### 9.1 核心发现

1. **Agent 在有明确异常的数据集上表现优秀**（flag-1/10/11 的 insight score 0.75-0.85），在"无异常"或"目标理解有歧义"的数据集上表现较差（flag-100/12 的 0.33-0.50）。

2. **Agent 的分析广度和量化深度往往超过 GT**，能做出 GT 未涉及的有价值发现（路由失败、优先级悖论、人员瓶颈、根因分类脱耦），但这些额外价值在当前评分体系下不计分。

3. **与论文 SOTA 对比**，5 case 平均 Insight Score（0.6467）高于论文中 AgentPoirot+gpt-4o 的 100 case 平均（0.60），初步表明 Agent 能力不弱于当前最佳。但样本量过小，不具统计意义。

4. **最大风险是 goal 理解偏差**（flag-12 案例），当 goal 措辞有歧义时，Agent 可能选择完全不同的分析方向，导致核心发现缺失。

5. **适配层设计合理**: `adapter_insightbench.py` 对核心分析引擎的复用程度高，差异主要在输入适配和输出格式化上，没有对分析逻辑做实质性修改。

### 9.2 改进方向

1. **增加 goal 理解鲁棒性**: 在生成查询前先让 LLM 将 goal 分解为具体子问题，或要求第一步必须做全局概览（所有字段的基础分布统计）

2. **引导深度挖掘**: 当前 5 条查询并行独立生成，可改为"前一步的发现驱动后续问题"的迭代式分析

3. **改进 CodeAgent 的鲁棒性**: 增强对列名拼写错误（如 `assignement_group`）的模糊匹配能力，增强自动推导指标的能力

4. **丰富评估维度**: 除了与 GT 匹配外，增加独立评估 Agent 发现的新 insight 的质量（评估"发现了 GT 没有的有价值洞察"）

---

*文档生成日期: 2026-04-23*
