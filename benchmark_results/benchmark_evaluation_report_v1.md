# Benchmark 评测报告：InsightBench 推理链路分析与系统适配评估

> **文档版本**: v1.0  
> **评测日期**: 2026-04-22  
> **评测范围**: InsightBench 3 个 case（flag-1、flag-10、flag-100）  
> **评测总分**: overall = 0.6389（insight_score = 0.6111, summary_score = 0.6667）

---

## 目录

1. [Benchmark 数据集介绍](#1-benchmark-数据集介绍)
2. [InsightBench 三个 Case 的原始数据与 Ground Truth 全文档](#2-insightbench-三个-case-的原始数据与-ground-truth-全文档)
3. [推理链路对比：LLM 生成 vs Ground Truth](#3-推理链路对比llm-生成-vs-ground-truth)
4. [每步推理的因果分析与深层变量发掘评估](#4-每步推理的因果分析与深层变量发掘评估)
5. [适配层代码差异分析：adapter_insightbench vs 项目主体 data_analysis](#5-适配层代码差异分析adapter_insightbench-vs-项目主体-data_analysis)
6. [评估机制说明](#6-评估机制说明)
7. [总结与改进建议](#7-总结与改进建议)

---

## 1. Benchmark 数据集介绍

### 1.1 InsightBench

**论文**: [InsightBench: Evaluating Business Analytics Agents Through Multi-Step Insight Generation](https://arxiv.org/pdf/2407.06423)（arXiv 2024, ServiceNow Research）

**定位**: 评估端到端数据分析 Agent 的多步洞察生成能力。它不只看代码是否正确，而是评估 Agent 能否从原始数据出发，通过多步分析生成有商业价值的 insight。

**数据集构成**:
- 共 **100 个数据集**（flag-1 到 flag-100），每个数据集模拟一个 ServiceNow 工单管理场景
- 每个数据集包含：
  - **CSV 数据文件**: 通常 500 条工单记录，包含 `category`、`state`、`opened_at`、`closed_at`、`assigned_to`、`short_description`、`priority`、`location` 等字段
  - **元数据（metadata）**: 包含分析目标（goal）、角色（role）、数据集描述（dataset_description）
  - **Ground Truth insight 列表（insight_list）**: 人工标注的 insight 序列，每条包含问题（question）、洞察（insight）、数据类型（data_type）、可操作建议（actionable_insight）、分析代码（code）、图表定义（plot）
  - **摘要（summary）**: 人工撰写的全局分析摘要
  - **精简 insight 列表（insights）**: 纯文本的 insight 列表，用于评估匹配

**数据设计特点**:
- flag-1 到 flag-99 中人为植入了特定的数据异常（如 Hardware 类别过多、打印机故障集中等），使得 GT 的 insight 有明确的正确答案
- flag-100 通常作为"无异常"基线，数据均匀分布，用于测试 Agent 是否能正确识别"没有问题"
- 同一个分析目标（goal）可能对应不同的数据分布（如 flag-1 和 flag-100 的 goal 相同但数据不同）

**评估指标**:
- **Insight Score**: 对每条 GT insight，从所有预测 insight 中找到最佳匹配，由 LLM Judge 打分（0.0~1.0），取所有 GT insight 的平均分
- **Summary Score**: 对 GT summary 与预测 summary 做 LLM 语义匹配打分（0.0~1.0）
- **Overall Score**: Insight Score 和 Summary Score 的算术平均

原始 InsightBench 支持 ROUGE-1 和 G-EVAL/LLaMA-3-Eval 两种评估方式。我们的实现使用 **LLM-as-Judge** 方式（Kimi-K2.5 模型做评估者）。

### 1.2 DACO

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

**评估指标**:
- **BLEU Score**: 评估生成答案与 GT 答案的文本相似度
- **Entailment Score**: 评估生成答案是否蕴含 GT 的关键信息
- **ChatGPT Helpfulness Score**: 使用 GPT 评估分析结果对用户查询的帮助程度

**DACO 与 InsightBench 的核心差异**:

| 维度 | InsightBench | DACO |
|------|-------------|------|
| 任务形式 | 开放式多步探索 | 对话式逐步代码生成 |
| 输入 | CSV + 一句话目标 | 数据库 + 角色化查询（"As a ..., I want to ..."） |
| 输出 | Insight 列表 + Summary | Findings + Suggestions + Code Trajectory |
| GT 标注 | 人工设计的问题链 + 代码 + 图表 + insight | 人工标注的多轮对话（含代码和输出） |
| 评估重点 | 是否发现了关键 insight | 分析结果对用户是否有帮助 |
| 代码角色 | 分析手段（中间产物） | 核心评估对象（代码质量也被评分） |

---

## 2. InsightBench 三个 Case 的原始数据与 Ground Truth 全文档

### 2.1 Case 1: flag-1 — 硬件工单数据集

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

### 2.2 Case 2: flag-10 — 工单解决时间趋势分析

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

### 2.3 Case 3: flag-100 — 均衡分布基线数据集

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

## 3. 推理链路对比：LLM 生成 vs Ground Truth

### 3.1 Case 1: flag-1 对比

#### GT 推理链路（6 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | descriptive | 各类别的事件分布如何？ | Hardware=336 件占 67%，远高于其他类别 |
| 2 | diagnostic | 为什么 Hardware 最多？（词云分析） | short_description 中 "printer"、"malfunctioning" 频繁出现 |
| 3 | diagnostic | Printer 出现多少次？ | 出现 225 次 |
| 4 | descriptive | 硬件事件是否集中在某地点？ | Australia=241 件，远超其他地点 |
| 5 | descriptive | 事件分布随时间有无趋势？ | Hardware 无增长趋势但始终最高 |
| 6 | diagnostic | 哪个 Printer ID 最多？ | Printer546=158 次 |

#### LLM Agent 推理链路（5 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | descriptive | 每个类别中工单的数量分布和百分比 | Hardware=336 件占 67.2%，Inquiry/Help 最少仅 32 件占 6.4% |
| 2 | descriptive | 每个分配组处理不同优先级工单的数量和比例 | Hardware 组处理 341 件（是 Service Desk 25 件的 13.6 倍），承担 106 件 Critical 优先级工单 |
| 3 | descriptive | 分析 short_description 中各类别最常出现的关键词 | 发现 "printer546" 出现 155 次，"australia" 出现 109 次 |
| 4 | descriptive | 计算各类别的平均解决时间 | Inquiry/Help 平均 8.51 天最慢，Network 6.22 天最快 |
| 5 | predictive | 按月分析工单创建时间的趋势 | Hardware 每月占比 54.8%-82.9%，6月峰值达 82.9% |

#### 逐步对比分析

**步骤 1 — 类别分布（两方均有）**:
- GT 和 Agent 都以类别分布为起点，得到了一致的结论（Hardware 占比最高）
- Agent 额外计算了百分比和最少类别，信息更详细
- **一致性**: ✅ 高度一致

**步骤 2 — 深入原因分析（方向分歧）**:
- GT 选择对 `short_description` 做词云分析，找文本层面的根因
- Agent 选择分析 `assignment_group × priority` 的交叉分布，从组织管理角度切入
- **差异原因**: Agent 的 LLM 根据 schema 列名自动规划，优先选择了结构化字段（assignment_group, priority），而非需要文本挖掘的 short_description
- **评价**: GT 的方向更直接触及问题根因（为什么 Hardware 多→因为打印机坏了），Agent 的方向回答了另一个有价值的问题（工作量分配是否合理）

**步骤 3 — 关键词分析 vs 统计检验**:
- GT 量化"Printer"出现次数（225次）
- Agent 发现了 "printer546"（155次）和 "australia"（109次），但这是在第 3 步才做的文本分析
- **评价**: Agent 最终也做了文本分析，但晚了一步；有趣的是 Agent 直接找到了具体的打印机 ID（printer546），跳过了 GT 第 3 步直接到了第 6 步的发现

**步骤 4 — 地理维度 vs 解决时间**:
- GT 分析 Hardware 事件的地理分布（Australia 占大头）
- Agent 分析各类别的平均解决时间（Inquiry/Help 最慢）
- **差异**: 完全不同的分析维度。GT 继续沿着"Hardware 为什么多"深挖，Agent 转向了效率分析
- **评价**: 两者都有价值，但 GT 的路径更紧扣 goal（找分配不平衡）

**步骤 5 — 时间趋势（两方类似）**:
- GT 和 Agent 都做了时间趋势分析
- GT 发现"无增长趋势但始终最高"
- Agent 发现"Hardware 每月占比 54.8%-82.9%，6月峰值"
- **一致性**: ✅ 结论一致，Agent 提供了更多量化细节

**GT 独有但 Agent 缺失的分析**:
- **Printer 频率量化**（GT步骤3）: Agent 在步骤 3 中间接覆盖了
- **地理集中性**（GT步骤4）: Agent 完全没有分析 location 字段
- **具体 Printer ID**（GT步骤6）: Agent 在步骤 3 中发现了 printer546

**Agent 独有但 GT 缺失的分析**:
- **assignment_group × priority 交叉分析**: GT 未涉及组织管理维度
- **平均解决时间分析**: GT 在 flag-1 中未分析解决时间
- **月度百分比波动**: GT 只说"无增长趋势"，Agent 给出了具体数字

**评测得分**: insight_score = 1.0, summary_score = 1.0

---

### 3.2 Case 2: flag-10 对比

#### GT 推理链路（4 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | diagnostic | TTR 随时间的趋势？ | TTR 线性增长（从 opened_at/closed_at 计算） |
| 2 | diagnostic | 事件量与 TTR 的相关性？ | 正相关 |
| 3 | time_series | TTR 增长是否各类别均匀？ | 是，均匀增长 |
| 4 | descriptive | Agent 生产力是否均匀？ | 是，各 agent 处理量大致相同 |

#### LLM Agent 推理链路（5 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | descriptive | 计算各事件的 TTR 并统计分布 | 99.8% 事件超过 3 天解决，平均 24.8 天，最短 1.9 天 |
| 2 | descriptive | 不同优先级下的平均 TTR 和事件数量 | Critical=23.66天、High=25.12天、Moderate=23.70天，优先级系统无效 |
| 3 | predictive | 按月分析 TTR 趋势 | TTR 从 1月的 79.7 小时增到 12月的 857.7 小时，增长约 11 倍，斜率 72.12 小时/月 |
| 4 | descriptive | 不同分配组和处理人的平均 TTR | Hardware 组 654.4 小时，Network 组 574.31 小时，差异仅 13.9% |
| 5 | descriptive | 对 short_description 做文本分析 | VPN 和 Email 工单量大但 TTR 低于平均；Login(648h) 和 Application(640h) TTR 最长 |

#### 逐步对比分析

**步骤 1 — TTR 基础统计 vs TTR 趋势**:
- GT 直接计算 TTR 并看时间趋势
- Agent 先计算 TTR 的整体分布统计（平均、最小、最大、分区间分布）
- **评价**: Agent 的起步更扎实——先看整体分布再看趋势是更严谨的分析方法。但 GT 更直接地切入了核心问题

**步骤 2 — 事件量相关性 vs 优先级分析**:
- GT 分析事件量和 TTR 的相关性（发现正相关）
- Agent 分析优先级与 TTR 的关系（发现优先级系统无效）
- **评价**: 这是两个不同但都有价值的因素。GT 找到了一个重要的驱动因素（事件量），Agent 发现了一个管理问题（优先级形同虚设）

**步骤 3 — 类别均匀性 vs 月度趋势**:
- GT 验证 TTR 增长在所有类别中均匀（排除类别特异性）
- Agent 做了月度 TTR 趋势分析，发现 11 倍增长
- **评价**: Agent 的发现（11 倍增长、斜率 72.12 小时/月）在量化程度上更优。GT 的类别均匀性验证是重要的排除性分析

**步骤 4 — Agent 生产力 vs 分配组差异**:
- GT 验证各 agent 生产力均匀
- Agent 分析分配组和个人的 TTR 差异（发现差异仅 13.9%）
- **评价**: 两者分析方向非常接近，结论也一致——问题不在于个别团队或个人

**步骤 5 — Agent 独有的文本分析**:
- GT 没有做文本分析
- Agent 对 `short_description` 做了关键词频率和 TTR 关联分析
- **评价**: 这是有价值的额外分析，发现 Login 和 Application 类问题 TTR 最长

**关键发现对比**:
- GT 发现了**事件量与 TTR 的正相关**（Agent 未直接验证此点）
- GT 验证了**各类别均匀增长**（Agent 未按类别拆分 TTR 趋势）
- Agent 发现了**优先级系统无效**（GT 未关注优先级）
- Agent 量化了**月度增长斜率**（GT 只说"线性增长"）

**评测得分**: insight_score = 0.5, summary_score = 1.0

---

### 3.3 Case 3: flag-100 对比

#### GT 推理链路（6 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | descriptive | 各类别事件分布？ | 完全均匀，每类 100 件 |
| 2 | diagnostic | 描述中有特定问题吗？ | 无，词云无突出模式 |
| 3 | diagnostic | Printer 出现多少次？ | 0 次 |
| 4 | descriptive | 硬件事件集中在某地点吗？ | 不集中（22/21/20/19/18） |
| 5 | descriptive | 有时间趋势吗？ | 无 |
| 6 | 其他 | 最有效的解决方法？ | Restart service 最常见 |

#### LLM Agent 推理链路（5 步）

| 步骤 | 类型 | 问题 | 核心发现 |
|------|------|------|----------|
| 1 | descriptive | 各类别中事件数量和占比 | 完全均匀，每类 100 件（20%），不平衡比 1.00:1 |
| 2 | prescriptive | 每个分配组的平均满意度和事件数 | 各组 100 件，满意度 2.96~3.20，Software 组最高 Network 组最低 |
| 3 | predictive | 月度事件量趋势 | 均值 38.46，标准差 6.69，10月为异常高点（52件） |
| 4 | descriptive | 各类别的平均成本和受影响用户数 | Software 成本最高（$31,276），Inquiry/Help 受影响用户最少 |
| 5 | diagnostic | 各根因分类在不同优先级中的分布 | Configuration Error 与高优先级强相关（占 22.8%），Hardware Failure 仅 40.96% |

#### 逐步对比分析

**步骤 1 — 类别分布（两方一致）**:
- 都发现了完全均匀分布（每类 100 件）
- **一致性**: ✅ 完全一致

**步骤 2 — 词云分析 vs 满意度分析**:
- GT 做词云确认无特定问题
- Agent 分析 `user_satisfaction_score` 字段，发现组间满意度差异
- **差异**: Agent 利用了 GT 未使用的字段（user_satisfaction_score），这是 Agent 的独特发现

**步骤 3 — Printer 频率 vs 月度趋势**:
- GT 量化 Printer 出现次数（0 次），确认无打印机问题
- Agent 分析月度事件量，发现 10 月为异常高点（52 件）
- **差异**: GT 刻意验证"flag-1 的问题在 flag-100 中不存在"，Agent 做了独立的时间分析

**步骤 4 — 地理分布 vs 成本与影响**:
- GT 确认地理分布均匀
- Agent 分析了 `estimated_cost` 和 `users_affected` 字段
- **差异**: Agent 使用了更多字段，但没有验证地理维度

**步骤 5 — 时间趋势 vs 根因分类×优先级**:
- GT 确认无时间趋势
- Agent 分析 `rca_category` 与 `priority` 的交叉分布，发现 Configuration Error 与高优先级强关联
- **差异**: Agent 发现了一个 GT 完全没有涉及的深层模式

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

**评测得分**: insight_score = 0.3333, summary_score = 0.0

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
| 1. 类别分布统计 | 2/5（纯描述统计） | 2/5（仅用 category） | 5/5（数据准确） | 基础扎实但无因果推断 |
| 2. 分配组×优先级交叉 | 3/5（发现负载不均衡的组织原因） | 4/5（用了 assignment_group + priority） | 4/5（逻辑自洽） | 从组织管理角度做了诊断 |
| 3. 文本关键词分析 | 4/5（从描述中提取根因线索） | 3/5（short_description 文本挖掘） | 4/5（找到 printer546 和 australia） | 直接触及了问题根因 |
| 4. 解决时间分析 | 3/5（发现效率悖论：简单问题反而慢） | 3/5（推导了 opened_at - closed_at） | 4/5（结论有数据支撑） | 有价值的效率洞察 |
| 5. 月度趋势分析 | 2/5（描述性趋势） | 2/5（opened_at + category） | 5/5（数据准确，结论合理） | 有效的时间维度验证 |

**总体评价**: Agent 在 flag-1 上的推理质量较高，特别是步骤 2 和 3 展现了多变量交叉分析和文本挖掘能力。步骤 4 发现的"简单问题解决慢"悖论是 GT 没有提到的有价值洞察。主要不足是没有分析 location 字段。

### 4.3 flag-10 各步评估

| 步骤 | 因果深度 | 变量广度 | 合理性 | 综评 |
|------|---------|---------|--------|------|
| 1. TTR 分布统计 | 2/5（描述统计） | 3/5（推导了 TTR） | 5/5（数据准确） | 正确推导了 TTR，基础扎实 |
| 2. 优先级×TTR | 4/5（发现优先级系统失效） | 3/5（priority + TTR） | 5/5（结论有力） | **关键发现**：优先级形同虚设 |
| 3. 月度 TTR 趋势 | 3/5（量化了增长） | 2/5（仅时间维度） | 5/5（11倍增长有冲击力） | 量化程度优于 GT |
| 4. 分配组/个人 TTR | 3/5（排除了组/人差异） | 3/5（assignment_group + assigned_to） | 4/5（逻辑严密的排除法） | 与 GT 结论一致 |
| 5. 文本×TTR | 4/5（关联分析） | 4/5（short_description + TTR） | 4/5（发现高复杂问题类型） | GT 未做此分析，有附加价值 |

**总体评价**: Agent 在 flag-10 上表现出色，所有步骤都成功执行（与之前存在代码失败的版本不同）。步骤 2 发现的"优先级系统无效"是一个重要的管理洞察。步骤 5 的文本×TTR 关联分析超出了 GT 的覆盖范围。主要不足是没有分析事件量与 TTR 的相关性（GT 步骤 2 的核心发现）。

### 4.4 flag-100 各步评估

| 步骤 | 因果深度 | 变量广度 | 合理性 | 综评 |
|------|---------|---------|--------|------|
| 1. 类别分布 | 2/5（纯描述） | 2/5（仅 category） | 5/5（正确识别均匀） | 准确但基础 |
| 2. 满意度分析 | 3/5（发现绩效差异） | 3/5（user_satisfaction_score） | 4/5（逻辑合理） | GT 未利用此字段 |
| 3. 月度异常检测 | 3/5（统计方法检测异常） | 2/5（opened_at） | 4/5（10月异常有意义） | 标准差方法合理 |
| 4. 成本与影响 | 3/5（发现 Software 高成本） | 4/5（estimated_cost + users_affected） | 4/5（交叉分析有价值） | 多维度交叉分析 |
| 5. 根因×优先级 | 4/5（发现 ConfigError 与高优先级关联） | 4/5（rca_category + priority） | 4/5（比例数据有说服力） | **最佳步骤**：深层因果关联 |

**总体评价**: Agent 在 flag-100 上展示了更丰富的分析维度（满意度、成本、根因分类），但由于 GT 的设计意图是"确认没有异常"，Agent 的"在均匀中寻找差异"策略虽然分析质量不低，却与评分标准错位。步骤 5 的根因×优先级分析是最具深度的推理。

---

## 5. 适配层代码差异分析：adapter_insightbench vs 项目主体 data_analysis

### 5.1 调用关系概览

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

### 5.2 核心差异逐项对比

#### 差异 1: 入口函数不同

| 维度 | 项目主体（main.py） | Benchmark 适配器 |
|------|-------------------|-----------------|
| 入口函数 | `analyze_region()` | `run_agent_on_dataset()` → `analyze_data()` |
| 数据来源 | Excel 考核评估总表 + 补充材料目录 | CSV 单文件 + JSON 元数据中的 goal |
| 数据格式 | `{表名: DataFrame}` 字典（可能含多个 Sheet） | `{"data": DataFrame}` 固定单表 |

**影响**: 适配器将 InsightBench 的单 CSV 文件包装为 `{"data": df}` 字典传入 `analyze_data()`，数据结构被简化。项目主体支持多 Excel 文件多 Sheet 的复杂结构。

#### 差异 2: 分析目标/任务描述的来源不同

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

### 5.3 差异影响评估

| 差异项 | 对 Benchmark 结果的影响程度 | 说明 |
|-------|-------------------------|------|
| 数据格式（单表 vs 多表） | 🟡 中 | InsightBench 本身就是单 CSV，无影响；但 prompt 中的 Sheet 引用变为固定的 "data" |
| Prompt 模板分支 | 🔴 高 | 通用分支的分析维度要求（文本分析、推导指标等）直接影响 LLM 生成的查询质量 |
| Schema 详细度 | 🔴 高 | 展示示例行和唯一值让 LLM 更了解数据内容，是 Agent 能做出合理分析的关键 |
| 后处理步骤 | 🟡 中 | _extract_insight() 的质量影响最终 insight 的表述，但核心分析由 CodeAgent 完成 |
| LLM 配置 | 🟢 低 | 核心分析阶段用的是相同的默认模型 |

---

## 6. 评估机制说明

### 6.1 评估流程

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

### 6.2 评估结果汇总

| 数据集 | Insight Score | Summary Score | GT Insight 数 | 预测 Insight 数 |
|--------|--------------|---------------|---------------|-----------------|
| flag-1 | 1.0000 | 1.0000 | 6 | 5 |
| flag-10 | 0.5000 | 1.0000 | 4 | 5 |
| flag-100 | 0.3333 | 0.0000 | 6 | 5 |
| **平均** | **0.6111** | **0.6667** | — | — |
| **Overall** | **0.6389** | | | |

### 6.3 评估机制的局限性

1. **方向性偏差不被识别**: Agent 可能做了有价值的分析（如发现 silo 化分配模式），但如果 GT 没有对应的 insight，该发现不会被计分
2. **最佳匹配策略的宽松性**: 对每条 GT insight 取所有预测中的最高分，这意味着只要有一条预测"沾边"就能拿分，即使其他预测完全无关
3. **LLM Judge 的稳定性**: 使用 LLM 评分存在随机性，同样的输入可能得到不同的分数
4. **Summary 评估粒度粗**: 整体语义匹配可能忽略细节差异

---

## 7. 总结与改进建议

### 7.1 核心发现

1. **问题生成策略差异是根本原因**: GT 采用"数据探索深挖"路径（分布→文本→具体实体→维度验证），Agent 采用"管理咨询横向展开"路径（描述→诊断→预测→建议）。这导致即使分析质量不低，insight 维度也与 GT 不同。

2. **Agent 的优势**:
   - 多变量交叉分析能力强（assignment_group × priority、rca_category × priority）
   - 能发现 GT 未涉及的模式（优先级系统失效、silo 化分配、满意度差异）
   - 量化程度更高（提供具体百分比、倍数、斜率等）

3. **Agent 的不足**:
   - 文本字段分析不够主动（不会优先对 short_description 做关键词/词云分析）
   - 不会分析 GT 重点关注的 resolution_method、location 等字段
   - 推理链条"广而浅"，不如 GT 的"窄而深"

4. **适配层设计合理**: `adapter_insightbench.py` 对核心分析引擎的复用程度高，差异主要在输入适配和输出格式化上，没有对分析逻辑做实质性修改。

### 7.2 改进方向

1. **增强文本字段分析**: 在 prompt 模板中更强调对文本描述字段（如 `short_description`）的分析优先级，或在 Schema 描述中标注文本字段的分析价值

2. **引导深度挖掘**: 可以在 prompt 中要求"后续问题应基于前一步的发现进行深挖"，而非并行独立生成 5 个问题

3. **改进 CodeAgent 的鲁棒性**: 增强对列名拼写错误（如 `assignement_group`）的模糊匹配能力，增强自动推导指标的能力（如从 opened_at/closed_at 自动计算 TTR）

4. **丰富评估维度**: 除了与 GT 匹配外，增加独立评估 Agent 发现的新 insight 的质量（评估"发现了 GT 没有的有价值洞察"）

---

*文档生成日期: 2026-04-22*
