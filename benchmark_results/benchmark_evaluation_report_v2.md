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
4. [独立质量评估：不对照 GT，Agent 结果是否是好的分析？](#4-独立质量评估不对照-gtagent-结果是否是好的分析)
5. [Benchmark 评分是否合理？](#5-benchmark-评分是否合理)
6. [适配层代码差异分析：adapter_insightbench vs 项目主体 data_analysis](#6-适配层代码差异分析adapter_insightbench-vs-项目主体-data_analysis)
7. [评估机制说明](#7-评估机制说明)
8. [总结与改进建议](#8-总结与改进建议)

---

## 1. Benchmark 数据集介绍与论文基准分数

### 1.1 InsightBench

**论文**: [InsightBench: Evaluating Business Analytics Agents Through Multi-Step Insight Generation](https://arxiv.org/abs/2407.06423)（ICLR 2025, ServiceNow Research）

**定位**: 评估端到端数据分析 Agent 的多步洞察生成能力。不只看代码是否正确，而是评估 Agent 能否从原始数据出发，通过多步分析生成有商业价值的 insight。

**数据集构成**:
- 共 **100 个数据集**（flag-1 至 flag-100），模拟 ServiceNow 企业工作流场景
- 覆盖 5 大主题：Incident Management（20 个）、Asset Management（20 个）、User Management（15 个）、Finance Management（15 个）、Goal Management（10 个）等
- 每个数据集包含：
  - **CSV 数据文件**: 500 条记录，字段包括 `category`、`state`、`opened_at`、`closed_at`、`assigned_to`、`short_description`、`priority`、`location` 等
  - **元数据（metadata）**: 分析目标（goal）、角色（role）、数据集描述
  - **Ground Truth insight 列表（insight_list）**: 人工标注序列，每条含问题（question）、洞察（insight）、数据类型（data_type）、可操作建议（actionable_insight）、代码（code）、图表定义（plot）
  - **摘要（summary）**: 人工撰写的全局分析摘要
  - **精简 insight 列表（insights）**: 纯文本列表，用于评估匹配
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
- 每条数据包含 `table_id`（数据库标识）、`data_id`（唯一 ID）、`messages`（对话格式消息列表）
- 数据库以 `.pkl` 文件存储，包含 `title`、`database`（表名→表内容字典）、`name`

**DACO 与 InsightBench 的核心差异**:

| 维度 | InsightBench | DACO |
|------|-------------|------|
| 任务形式 | 开放式多步探索 | 对话式逐步代码生成 |
| 输入 | CSV + 一句话目标 | 数据库 + 角色化查询 |
| 输出 | Insight 列表 + Summary | Findings + Suggestions + Code Trajectory |
| 评估重点 | 是否发现了关键 insight | 分析结果对用户是否有帮助 |
| 代码角色 | 分析手段（中间产物） | 核心评估对象 |

两者互补：DACO 测"能不能写对代码"，InsightBench 测"能不能找到关键发现"。

---

## 2. 五个 Case 的 Ground Truth 原始数据

### 2.1 Case 1: flag-1 — 硬件工单分布不均（难度 Level 4）

**Goal**: "Find the discrepancy and imbalance in distribution of incidents assigned across categories"  
**角色**: L2 Support Agent | **主题**: Incidents Management  
**数据**: 500 条 ServiceNow 工单记录

**GT Insight 列表（6 条）**:

| # | 类型 | 问题 | 核心发现 | 关键数据 |
|---|------|------|----------|---------|
| 1 | descriptive | 各类别事件分布？ | Hardware 显著高于其他 | Hardware=336, Network=51, Software=41, Database=40, Inquiry/Help=32 |
| 2 | diagnostic | 为什么 Hardware 最多？ | 打印机故障在描述中大量出现 | 词云高频词: printer, malfunctioning, Australia |
| 3 | diagnostic | Printer 出现多少次？ | 大部分 Hardware 事件与打印机相关 | "Printer" 出现 225 次 |
| 4 | descriptive | 硬件事件集中在哪？ | 集中在 Australia | Australia=241, 其他各约 20-25 |
| 5 | descriptive | 有时间趋势吗？ | 无增长趋势但始终最高 | 按月折线图 |
| 6 | diagnostic | 哪台打印机最多？ | Printer546 | Printer546=158 次 |

**GT Summary**: Hardware 占 67%；根因是打印机故障；地理集中在 Australia。  
**GT 推理逻辑**: 分布 → 文本根因（词云 → printer）→ 量化根因（225 次）→ 地理维度（Australia 241）→ 时间维度（无增长）→ 锁定设备（Printer546）

---

### 2.2 Case 2: flag-10 — TTR 增长趋势分析（难度 Level 3）

**Goal**: "Identify trends and underlying factors or correlations contributing to the increase in TTR."  
**角色**: Incidents Manager | **主题**: Incident Management

**GT Insight 列表（4 条）**:

| # | 类型 | 问题 | 核心发现 | 关键数据 |
|---|------|------|----------|---------|
| 1 | diagnostic | TTR 随时间的趋势？ | 线性增长 | resolution_time = closed_at - opened_at |
| 2 | diagnostic | 事件量与 TTR 相关性？ | 正相关 | 双轴折线图 |
| 3 | time_series | TTR 增长各类别均匀吗？ | 是，均匀增长 | 多线图 |
| 4 | descriptive | Agent 生产力均匀吗？ | 是，各 agent 处理量相近 | 柱状图 |

**GT Summary**: TTR 线性增长；与事件量正相关；系统性问题非类别特有。  
**GT 推理逻辑**: 计算 TTR → 看趋势（线性增长）→ 找相关因素（事件量正相关）→ 排除类别差异（均匀）→ 排除 Agent 差异（均匀）

---

### 2.3 Case 3: flag-11 — 分类别解决时间趋势（难度 Level 4）

**Goal**: "Analyze the incident data to identify trends and underlying causes for the increasing resolution time in certain category."  
**角色**: L2 Engineering Manager | **主题**: Incident Management

**GT Insight 列表（6 条）**:

| # | 类型 | 问题 | 核心发现 | 关键数据 |
|---|------|------|----------|---------|
| 1 | descriptive | Hardware 的 TTR 趋势？ | 从 2023-07 开始线性增长 | 按类别的 TTR 折线图 |
| 2 | descriptive | 各类别事件量随时间波动？ | Hardware 在 2023-06~08 激增至平常 4-5 倍 | 月度类别计数 |
| 3 | descriptive | 哪些时间窗口 Hardware 激增？ | 2023-07（47 件）和 2023-08（43 件），平均仅 6 件/月 | 柱状图 |
| 4 | descriptive | 激增有地理模式吗？ | 集中在 Australia | location 分布 |
| 5 | descriptive | 异常期间 TTR 趋势？ | Hardware TTR 在异常期显著上升 | 过滤后折线图 |
| 6 | descriptive | 哪种硬件最有问题？ | Email 服务器宕机 | 词云: email, outage, system |

**GT Summary**: Hardware TTR 从 2023-07 开始线性增长，与同期事件量激增吻合（7-8 月工单量是平时 4-5 倍）；集中在 Australia；根因是 email 服务器宕机。  
**GT 推理逻辑**: 看 TTR 趋势（发现 Hardware 异常）→ 看事件量趋势（7-8 月激增）→ 量化激增窗口 → 地理维度（Australia）→ 异常期 TTR 验证 → 文本分析找根因（email server）

---

### 2.4 Case 4: flag-12 — 简单版硬件工单分布（难度 Level 1，最简单）

**Goal**: "Find the discrepancy and imbalance in incidents assigned"  
**角色**: L1 Agent | **主题**: Incidents Management

**GT Insight 列表（4 条）**:

| # | 类型 | 问题 | 核心发现 | 关键数据 |
|---|------|------|----------|---------|
| 1 | descriptive | 各类别事件分布？ | Hardware 显著高于其他 | Hardware=406, Software=33, Network=22, Inquiry/Help=20, Database=19 |
| 2 | descriptive | 为什么 Hardware 最多？ | 打印机问题 | "Printer" 在 Hardware 描述中出现 166 次 |
| 3 | descriptive | 硬件事件集中在某地点？ | **数据集中没有 location 字段** | GT 明确: "location is not specified" |
| 4 | descriptive | 有时间趋势吗？ | 无显著增长，稳定且高 | 折线图 |

**GT Summary**: Hardware 占 71%；打印机相关问题突出；地理信息缺失需进一步调查。  
**GT 推理逻辑**: 分布（Hardware 81.2%）→ 文本根因（printer 166 次）→ 地理（字段缺失）→ 时间趋势（稳定）

---

### 2.5 Case 5: flag-100 — 均衡分布基线（对照实验）

**Goal**: "Find the discrepancy and imbalance in distribution of incidents assigned across categories"（与 flag-1 相同）  
**角色**: L2 Support Agent | **主题**: Incidents Management

**GT Insight 列表（6 条）**:

| # | 类型 | 问题 | 核心发现 |
|---|------|------|----------|
| 1 | descriptive | 各类别事件分布？ | 完全均匀，每类恰好 100 件 |
| 2 | diagnostic | 描述中有特定问题吗？ | 无，词云无突出模式 |
| 3 | diagnostic | Printer 出现多少次？ | 0 次 |
| 4 | descriptive | 硬件事件集中在哪？ | 不集中（22/21/20/19/18） |
| 5 | descriptive | 有时间趋势吗？ | 无 |
| 6 | 其他 | 最有效的解决方法？ | Restart service 最常见 |

**GT Summary**: 分布均匀；Restart service 最常见；无异常。  
**设计意图**: 与 flag-1 使用相同问题模板，但数据无异常，测试 Agent 能否正确识别"一切正常"。

---

## 3. Agent 生成结果与 GT 逐步对比

### 3.1 flag-1（得分: insight=0.80, summary=1.0）

#### Agent 推理链路（5 步）

| # | 问题摘要 | 核心发现 |
|---|---------|---------|
| 1 | category 与 assignment_group 的分布对比 | Hardware 336-341 件占 67-68%；发现 32 件 Inquiry/Help 无分配组、25 件 Service Desk 无分类（11.4% 路由失败） |
| 2 | 各 category 月度分布与环比增长率 | Hardware 月均 25.8 件稳定；小类别波动剧烈（Database/Software 有 300-700% 月度尖峰） |
| 3 | short_description 按 category 的高频关键词 | Hardware 词频总量 908（是 Network 6.1 倍）；"printer" 占 98.2%（275/280），"australia" 占 93.2% |
| 4 | assigned_to × category 交叉（含解决时长） | 5 人均 60-70% 工作量在 Hardware；Inquiry/Help 平均 204.2h 最慢；Charlie W. 负载最重（115 件） |
| 5 | priority × location × category 交叉 | Hardware 在 Australia 集中 241 件（71.7%）；92.3% 高/Critical 优先级；三重集中 |

#### 与 GT 覆盖对照

| GT 发现 | Agent 覆盖？ | 说明 |
|---------|------------|------|
| Hardware=336 占 67% | ✅ | 步骤 1 |
| 打印机是根因（词云） | ✅ | 步骤 3（printer 占 98.2%） |
| Printer 出现 225 次 | ✅ 间接 | 步骤 3 报告 275/280 |
| Australia=241 | ✅ | 步骤 5（71.7% 在 Australia） |
| 无时间增长趋势 | ✅ | 步骤 2（月均稳定） |
| Printer546=158 | ❌ | 未锁定到具体设备 ID |

**Agent 独有发现**: 11.4% 路由失败（category-assignment 脱耦）、人员负载矩阵、priority 分布偏斜。

**对比小结**: Agent 覆盖了 GT 6 条 insight 中的 5 条，并提供了 GT 未涉及的组织管理维度分析。主要遗漏是未锁定到 Printer546 这个具体设备 ID。

---

### 3.2 flag-10（得分: insight=0.75, summary=1.0）

#### Agent 推理链路（5 步）

| # | 问题摘要 | 核心发现 |
|---|---------|---------|
| 1 | TTR 描述统计（均值/中位数/分位） | 中位 614h，均值 594h，84 件未解决（16.8%） |
| 2 | 月度 TTR 趋势 + 与事件量的相关性 | TTR 从 1 月 3.32 天增到 12 月 35.74 天（11 倍）；**与事件量相关系数 r=0.94** |
| 3 | short_description 关键词 × TTR 关联 | 各类别 TTR 差异仅 1.27 天（software 最长 25.60，hardware 最短 24.33） |
| 4 | priority/assignment_group/assigned_to × TTR | Priority 2 (High) 反而最慢（602.80h）；Hardware/Software 组比 Network 慢 80h |
| 5 | TTR 异常值（95th percentile）根因分析 | Network 组占极端异常 73.7%；Beth Anglin 独占 42.1% |

#### 与 GT 覆盖对照

| GT 发现 | Agent 覆盖？ | 说明 |
|---------|------------|------|
| TTR 线性增长 | ✅ | 步骤 2（11 倍增长） |
| 事件量与 TTR 正相关 | ✅ | 步骤 2（**r=0.94**） |
| 各类别均匀增长 | ✅ 间接 | 步骤 3（类别间差异仅 1.27 天） |
| Agent 生产力均匀 | ✅ 间接 | 步骤 4（个人差异 77h，有限） |

**Agent 独有发现**: Priority 2 反而最慢（优先级悖论）；Beth Anglin 独占 42% 极端异常值；16.8% 未解决积压。

**对比小结**: Agent 覆盖了 GT 全部 4 条核心发现，特别是**直接计算了事件量与 TTR 的相关系数 r=0.94**。比上一版（代码失败）大幅改善。

---

### 3.3 flag-11（得分: insight=0.85, summary=1.0）

#### Agent 推理链路（5 步）

| # | 问题摘要 | 核心发现 |
|---|---------|---------|
| 1 | 各 category 平均解决时间 | Hardware 平均 1237.45h（是 Inquiry/Help 166.83h 的 **7.42 倍**）；标准差 816h |
| 2 | 月度 TTR 趋势（按 category） | Hardware 从 149h(1月) 涨到 2723h(次年1月)，**821% 增长**；7 月突增 525.5% |
| 3 | short_description 文本分析 | 82.5% 工单超 72h；"email" 出现在 98.2% 超时工单中；最严重案例 2803h（117 天） |
| 4 | priority × group × location × assigned_to 交叉 | **Australia Service Desk 是瓶颈**（929.43h），其他地区仅 315-381h |
| 5 | 极端异常值（>95th percentile）分析 | **100% 极端案例均为 Hardware**（33 件），69.7% Critical |

#### 与 GT 覆盖对照

| GT 发现 | Agent 覆盖？ | 说明 |
|---------|------------|------|
| Hardware TTR 从 2023-07 线性增长 | ✅ | 步骤 2（7 月 525.5% 突增） |
| 2023-07/08 Hardware 事件量激增 4-5 倍 | ❌ | 未单独分析事件量激增窗口 |
| 集中在 Australia | ✅ | 步骤 4（Australia 888.37h vs 其他 315-381h） |
| 异常期 TTR 上升 | ✅ | 步骤 1+2 |
| 根因是 email 服务器宕机 | ✅ | 步骤 3（"email" 98.2%，"outage" 29.7%） |
| Hardware 事件量月度波动 | ❌ | 未按月统计各 category 事件数 |

**Agent 独有发现**: Australia Service Desk 929.43h 是最差组合；100% 极端异常均为 Hardware；人员粒度分析（Fred Luddy 在澳洲高优先级工单 1303.39h）。

**对比小结**: Agent 从多个独立维度（时间、地理、文本、异常值）交叉验证了同一结论。主要遗漏是未单独量化事件量激增窗口（GT 步骤 2-3 的核心）。

---

### 3.4 flag-12（得分: insight=0.50, summary=0.50）

#### Agent 推理链路（5 步）

| # | 问题摘要 | 核心发现 |
|---|---------|---------|
| 1 | 各 assigned_to 的事件总数和负载比 | Luke Wilson 最重（116 件, 23.2%），Beth Anglin 最轻（85 件）；比值 1.36 |
| 2 | 按周的各人员事件创建量波动 | 大量零工单周（Beth Anglin 15 周零件）与突发峰值（Charlie W. 单周 6 件）交替 |
| 3 | 各人员平均解决时长 | Fred Luddy 最快（165.03h），Luke Wilson 最慢（195.55h）——但 Wilson 负载也最重 |
| 4 | assigned_to × priority 交叉 | Wilson 承担最多高优先级工单（96 件）；Critical 分布不均 |
| 5 | short_description 关键词分析 | 各人员关键词分布高度同质化（printer 占 40-50%），专业化指数极低（0.153-0.237） |

#### 与 GT 覆盖对照

| GT 发现 | Agent 覆盖？ | 说明 |
|---------|------------|------|
| Hardware=406 占 81% | ❌ | **完全未分析 category 分布** |
| 打印机问题（166 次） | ✅ 间接 | 步骤 5 发现 printer 占 40-50%，但未按 category 关联 |
| 无 location 字段 | ❌ | 未提及 |
| 无时间增长趋势 | ❌ | 步骤 2 分析了周级人员波动，未分析 category 时间趋势 |

**低分根因**: Agent 将 goal "Find the discrepancy and imbalance in incidents assigned" 理解为"人员分配不均衡"，而 GT 理解为"类别分布不均衡"。5 步分析全部围绕 `assigned_to` 展开，完全没有分析 `category` 分布——而这正是 GT 的第一步和最核心发现（Hardware 占 81.2%）。

---

### 3.5 flag-100（得分: insight=0.3333, summary=0.50）

#### Agent 推理链路（5 步）

| # | 问题摘要 | 核心发现 |
|---|---------|---------|
| 1 | category 频数分布和不平衡度 | 完全均匀（每类 100 件），不平衡比 1.00 |
| 2 | 月度 × category 的时间趋势 | Inquiry/Help 波动最大（CV=0.44），10 月异常高点 52 件 |
| 3 | assignment_group × assigned_to 交叉 | Hardware 组内不均最大（CV=0.29），Luke Wilson 27 件 vs Whitherspoon 12 件 |
| 4 | 各 category 平均成本与受影响用户 | Software 成本最高（$31,276）；Network 解决时间异常为负值 |
| 5 | rca_category × priority 交叉（含熵值） | 无结构性偏差（HHI 0.169-0.175），根因与类别名不匹配（Hardware 仅 15% Hardware Failure） |

#### 与 GT 覆盖对照

| GT 发现 | Agent 覆盖？ | 说明 |
|---------|------------|------|
| 每类 100 件，完全均匀 | ✅ | 步骤 1 |
| 词云无突出模式 | ❌ | 未做文本分析 |
| Printer 出现 0 次 | ❌ | 未验证 |
| 地理分布均匀 | ❌ | 未分析 location |
| 无时间趋势 | ✅ 间接 | 步骤 2 |
| Restart service 最常见 | ❌ | 未分析 resolution_method |

**对比小结**: Agent 正确识别了均匀分布，但采用了"在均匀中挖掘细粒度差异"的策略（满意度、成本、根因分类），与 GT"逐一确认没有异常"的策略错位。

---

## 4. 独立质量评估：不对照 GT，Agent 结果是否是好的分析？

> 以下评估**完全不参考 GT**，仅从数据分析方法论角度判断 Agent 的分析质量。

### 4.1 flag-1：⭐⭐⭐⭐ (4/5)

**优点**: 发现 11.4% 的路由脱耦（有运维审计价值）；文本分析精确找到 printer（98.2%）和 australia（93.2%）；人员×类别×时长的三维交叉给出了可操作的资源调配建议。  
**不足**: 未锁定到 Printer546——在实际运维中定位具体故障源是最终目标。  
**判断**: 高质量运维分析报告。广度充分，某些维度超出预期，但缺了"最后一英里"。

### 4.2 flag-10：⭐⭐⭐⭐⭐ (5/5)

**优点**: 正确推导 TTR 并计算全面描述统计；**直接计算事件量与 TTR 相关系数 r=0.94**——教科书级别；发现优先级系统失效（Priority 2 最慢）；异常值分析精准定位人因瓶颈（Beth Anglin 42%）；月度 11 倍增长量化极具冲击力。  
**不足**: 未明确按类别拆分 TTR 趋势验证"是否均匀增长"。  
**判断**: **5 个 case 中最优秀的分析**。每步有清晰目的，结论间形成逻辑链条，完全达到高级数据分析师水准。

### 4.3 flag-11：⭐⭐⭐⭐⭐ (5/5)

**优点**: 精准锁定 Hardware 是唯一 TTR 恶化来源（7.42 倍差异、821% 增长）；7 月 525.5% 突增被准确捕捉；文本分析命中 email（98.2%）根因；地理×组织交叉发现 Australia Service Desk 929h 瓶颈；极端异常值 100% 为 Hardware 提供决定性证据。  
**不足**: 未单独量化事件数量激增（只看了 TTR 没看 volume）。  
**判断**: 深度极佳，多维度交叉验证形成令人信服的证据链。

### 4.4 flag-12：⭐⭐ (2/5)

**优点**: 人员负载分析方法论正确；发现 Wilson"负载最重+效率最低"悖论有管理价值；关键词同质化分析揭示路由缺陷。  
**严重不足**: **完全误解了分析目标**——5 步分析无一步分析 category 分布，而 Hardware 占 81.2% 是数据中最显著特征。对 Level 1（最简单）数据集的这种遗漏不可接受。  
**判断**: 方法没问题，但方向偏了。说明 LLM 的 goal 理解能力对最终结果有决定性影响。

### 4.5 flag-100：⭐⭐⭐ (3/5)

**优点**: 正确识别均匀分布（不平衡比 1.00）；发现 GT 未涉及的维度（Inquiry/Help 时间波动 CV=0.44、根因与类别名脱耦、Software 成本偏高）；熵值和 HHI 方法论严谨。  
**不足**: 未分析文本字段和 resolution_method；Network 解决时间为负值（-251.49h）是明显数据质量问题但未标记。  
**判断**: 维度丰富但缺乏与 goal 的对齐。在"找不平衡"目标下，"没有不平衡"才是正确结论，Agent 反而在均匀中强找差异。

---

## 5. Benchmark 评分是否合理？

### 5.1 逐 Case 评估

| Case | 独立质量 | Benchmark 得分 | 判断 |
|------|---------|---------------|------|
| flag-1 | ⭐⭐⭐⭐ | insight=0.80 | ✅ 合理 |
| flag-10 | ⭐⭐⭐⭐⭐ | insight=0.75 | ⚠️ 偏低（Agent 覆盖了 GT 全部 4 条发现+额外发现，但路径差异仍被扣分） |
| flag-11 | ⭐⭐⭐⭐⭐ | insight=0.85 | ✅ 合理 |
| flag-12 | ⭐⭐ | insight=0.50 | ✅ 合理（Agent 确实遗漏了最核心的 category 分布） |
| flag-100 | ⭐⭐⭐ | insight=0.33 | ⚠️ 偏低（Agent 做了有价值分析但方向与 GT 错位） |

### 5.2 评分机制的结构性问题

1. **单向匹配偏差**: 评分只看"GT 的每条 insight 是否被覆盖"（recall），不看"Agent 的额外发现是否有价值"（precision）。Agent 在 flag-1 中发现的 11.4% 路由失败、flag-10 的优先级悖论、flag-100 的根因分类脱耦，这些均不计分。

2. **表述风格干扰匹配**: Agent 的 insight 使用大量精确数字和复合从句（如 "98.2% of 'printer' keyword occurrences"），GT 是简洁单句（如 "hardware incidents is significantly higher than others"）。信息密度差异可能导致 LLM Judge 匹配不精确。

3. **对"确认无异常"的评估不友好**: flag-100 这类数据集的 GT 是逐一验证"没问题"。Agent 如果用不同验证路径（如分析满意度而非打印机频率），即使同样得出"没问题"也会因步骤不同而丢分。

### 5.3 与论文基准的对比判断

论文中 AgentPoirot+gpt-4o 的 100 case Insight Score 为 0.60±0.03。我们 5 case 的 0.6467 处于同一水平。考虑到我们跑的 5 个 case 包含 2 个 Level 4（Hard）、1 个 Level 3、1 个 Level 1、1 个 baseline，难度分布不偏向简单题，这个得分水平说明 Agent 的分析能力是有竞争力的。flag-12 的低分（0.50）主要是 goal 理解偏差导致，而非分析能力不足。

---

## 6. 适配层代码差异分析：adapter_insightbench vs 项目主体 data_analysis

### 6.1 调用关系概览

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

### 6.2 核心差异逐项对比

#### 差异 1: 入口函数与数据格式

| 维度 | 项目主体（main.py） | Benchmark 适配器 |
|------|-------------------|-----------------|
| 入口函数 | `analyze_region()` | `run_agent_on_dataset()` → `analyze_data()` |
| 数据来源 | Excel 考核评估总表 + 补充材料目录 | CSV 单文件 + JSON 元数据中的 goal |
| 数据格式 | `{表名: DataFrame}` 字典（可能含多个 Sheet） | `{"data": DataFrame}` 固定单表 |

#### 差异 2: Prompt 模板分支（影响最大）

| 维度 | 项目主体 | Benchmark 适配器 |
|------|---------|-----------------|
| 任务描述 | `task_instruction=""`，走地区考核分支 | `task_instruction=goal`，走通用分析分支 |
| 查询指令要求 | 侧重"整体概览、单项指标、横向对比、趋势排名" | 侧重"数据分布概览、异常值、文本字段分析、时间趋势、相关性分析、根因诊断" |

#### 差异 3: Schema 描述参数

| 维度 | 项目主体 | Benchmark 适配器 |
|------|---------|-----------------|
| `max_sample_rows` | 默认 0（不展示） | 3（展示 3 行示例） |
| `max_unique_values` | 默认 0（不展示） | 10（展示最多 10 个唯一值） |

#### 差异 4: 后处理流程

| 维度 | 项目主体 | Benchmark 适配器 |
|------|---------|-----------------|
| 查询结果后处理 | 直接拼接为字符串，传给 DocWriter | 需额外两步 LLM 调用 |
| 是否提取 insight | ❌ 原样传递 | ✅ `_extract_insight()` 对每条结果用 LLM 提取一段 insight |
| 是否生成 summary | ❌ DocWriter 阶段生成 | ✅ `_generate_summary()` 用 LLM 汇总 |
| 是否分类问题类型 | ❌ | ✅ `_infer_question_type()` |

#### 差异 5: LLM 配置

| 维度 | 项目主体 | Benchmark 适配器 |
|------|---------|-----------------|
| 数据分析 | 默认模型 | 同默认模型 |
| 报告撰写 | 高级闭源模型 | 无独立写作 LLM |
| 改写润色 | 高级闭源模型 | 无此步骤 |

### 6.3 差异影响评估

| 差异项 | 影响程度 | 说明 |
|-------|---------|------|
| 数据格式（单表 vs 多表） | 🟡 中 | InsightBench 本身就是单 CSV |
| Prompt 模板分支 | 🔴 高 | 通用分支的维度要求直接影响查询质量 |
| Schema 详细度 | 🔴 高 | 展示示例行和唯一值是 Agent 能做合理分析的关键 |
| 后处理步骤 | 🟡 中 | _extract_insight() 影响最终表述 |
| LLM 配置 | 🟢 低 | 核心分析阶段用相同模型 |

**总结**: 核心分析引擎（`analyze_data()` → `_generate_query_instructions()` → `_execute_queries()` → `DataInspectorMCPTool` → `CodeAgent`）完全复用。差异在：输入适配、Prompt 分支、Schema 详细度、输出后处理。

---

## 7. 评估机制说明

### 7.1 评估流程

评估实现位于 `run_on_benchmark/evaluator.py`：
1. 对 GT 每条 insight，遍历所有预测 insight，取 LLM Judge 最高分
2. 所有 GT insight 分数取平均 → Insight Score
3. GT summary vs 预测 summary 整体匹配 → Summary Score
4. LLM Judge 使用 Kimi-K2.5，prompt 要求输出 0.0~1.0

### 7.2 评估结果汇总

| 数据集 | 难度 | Insight Score | Summary Score | GT Insight 数 | 预测 Insight 数 |
|--------|------|--------------|---------------|---------------|-----------------|
| flag-1 | Level 4 | 0.80 | 1.0 | 6 | 5 |
| flag-10 | Level 3 | 0.75 | 1.0 | 4 | 5 |
| flag-11 | Level 4 | 0.85 | 1.0 | 6 | 5 |
| flag-12 | Level 1 | 0.50 | 0.5 | 4 | 5 |
| flag-100 | — | 0.33 | 0.5 | 6 | 5 |
| **平均** | | **0.6467** | **0.80** | | |
| **Overall** | | **0.7233** | | | |

### 7.3 评估机制的局限性

1. **单向匹配**: 只评估 recall（GT 是否被覆盖），不评估 precision（Agent 额外发现是否有价值）
2. **LLM Judge 不稳定**: 同样输入可能得不同分数
3. **表述风格影响匹配**: Agent 的长段落 vs GT 的单句，信息密度差异干扰语义匹配
4. **Summary 粒度粗**: 整体匹配可能忽略细节差异

---

## 8. 总结与改进建议

### 8.1 核心发现

1. **Agent 在有明确异常的数据集上表现优秀**（flag-1/10/11 的 insight score 0.75-0.85），在"无异常"或"目标理解有歧义"的数据集上表现较差（flag-100/12 的 0.33-0.50）。

2. **Agent 的分析广度和量化深度往往超过 GT**，能做出 GT 未涉及的有价值发现（路由失败、优先级悖论、人员瓶颈、根因分类脱耦），但这些额外价值在当前评分体系下不计分。

3. **与论文 SOTA 对比**，5 case 平均 Insight Score（0.6467）高于论文中 AgentPoirot+gpt-4o 的 100 case 平均（0.60），初步表明 Agent 能力不弱于当前最佳。但样本量过小，不具统计意义。

4. **最大风险是 goal 理解偏差**（flag-12 案例），当 goal 措辞有歧义时，Agent 可能选择完全不同的分析方向，导致核心发现缺失。

### 8.2 改进方向

1. **增加 goal 理解鲁棒性**: 在生成查询前先让 LLM 将 goal 分解为具体子问题，或要求第一步必须做全局概览（所有字段的基础分布统计）

2. **引导深度挖掘**: 当前 5 条查询并行独立生成，可改为"前一步的发现驱动后续问题"的迭代式分析

3. **丰富评估维度**: 增加对 Agent 独有发现的质量评估（precision 维度，而非仅 recall 维度）

4. **适配层设计合理**: `adapter_insightbench.py` 对核心分析引擎的复用程度高，差异主要在输入适配和输出格式化上，没有对分析逻辑做实质性修改

---

*文档生成日期: 2026-04-23*
