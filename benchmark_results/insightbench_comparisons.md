# InsightBench 推理链路对比分析

> 本文档对比 Agent 的推理链路与 Benchmark GT（Ground Truth）的推理链路。
> 评测仅跑了 3 个 case：flag-1、flag-10、flag-100。

---

## 调用架构说明

**Agent 的实际执行流程（adapter_insightbench.py）：**

1. 读取 CSV 数据 + goal（一句话分析目标）
2. 生成 schema 描述（列名、类型、样本行）
3. **`_generate_questions()`**：用裸 LLM 根据 schema + goal 凭空生成 5 个分析问题（按 descriptive/diagnostic/predictive/prescriptive 分类）
4. 对每个问题调用 **裸 `CodeAgent`**：让它写 Python 代码 → 执行 → 返回 print 输出
5. 用 LLM 从代码输出中提取 insight，最后汇总生成 summary

**关键点：Agent 没有使用更高层的数据分析链路（如 data_analysis.py / main.py），问题完全由 LLM 根据 schema 自动生成，CodeAgent 只是执行器。**

**GT 的链路：** Benchmark 作者人工查看数据后设计问题链 → 人工编写分析代码 → 记录 insight。

---

## Case 1: flag-1

**数据集：** 500 条 ServiceNow 工单记录
**Goal：** Find the discrepancy and imbalance in distribution of incidents assigned across categories

### GT 推理链路（6 步）

| 步骤 | 类型 | 问题 | 发现 |
|------|------|------|------|
| 1 | descriptive | 各类别的事件分布如何？ | Hardware=336 件远高于其他（Software=41, Network=51, Inquiry/Help=32, Database=40） |
| 2 | diagnostic | 为什么 Hardware 类别最多？ | 对 `short_description` 做词云分析，发现 "printer"、"malfunctioning" 等词突出 |
| 3 | diagnostic | Printer 在事件描述中出现了多少次？ | "Printer" 出现 225 次，大部分 Hardware 事件与打印机有关 |
| 4 | descriptive | 硬件事件是否集中在某个地点？ | Australia=241 件，远高于其他地点（USA/UK/India/Canada 各约 20-25 件） |
| 5 | descriptive | 事件分布随时间是否有趋势？ | Hardware 没有明显增长趋势，但始终高于其他类别 |
| 6 | diagnostic | 哪个具体的 Printer ID 引发最多问题？ | Printer546=158 次，是最大的单一故障源 |

**GT 的推理逻辑：** 看分布 → 文本分析找根因 → 量化根因 → 地理维度 → 时间维度 → 锁定具体设备

### Agent 推理链路（5 步）

| 步骤 | 类型 | 问题 | 发现 |
|------|------|------|------|
| 1 | descriptive | 各类别×优先级的事件数和平均解决时间分布如何？ | 报告数据缺少 `resolution_time_days` 和 `incident_id` 字段，无法直接计算（**代码执行部分失败**，未从 opened_at/closed_at 推导） |
| 2 | diagnostic | 为什么某些 assignment group 处理的事件比例更高？是由地点、高频 caller 还是季节驱动的？ | Hardware 组处理 341 件（68.2%），其中 Australia 占 90.1%；四个高频 caller 占大头；月度份额在 54.8%-82.9% 波动，属系统性问题 |
| 3 | diagnostic | assignment group 和 incident priority 之间是否有相关性？ | chi-square=157.45, p<0.05，存在统计显著关联；Service Desk 只处理 1% 高优先级工单（占比最低） |
| 4 | predictive | 按当前趋势，哪些类别和 assignment group 未来负载最高？ | 无法计算解决时间（数据问题），但 Hardware 组 10.5:1 的负载比意味着它会持续承受最高工作量 |
| 5 | prescriptive | 应该制定怎样的重新分配规则来平衡负载？ | 建议打破 silo 化分配（每个类别 100% 对应单一 group），将 Hardware 类工单部分分流给 Service Desk，跨组交叉培训 |

**Agent 的推理逻辑：** 看分布+解决时间 → 分析 assignment group 负载驱动因素 → 统计检验相关性 → 预测未来负载 → 给出重分配建议

### 对比小结

| 维度 | GT | Agent |
|------|-----|-------|
| 问题来源 | 人工设计，逐步深挖 | LLM 自动生成，按四类覆盖 |
| 核心发现 | 打印机故障是根因（Printer546） | assignment group silo 化是分配不均的组织原因 |
| 是否分析文本字段 | ✅ 词云 + 关键词频率 | ❌ 完全未碰 short_description |
| 是否使用统计检验 | ❌ | ✅ chi-square 检验 |
| 代码失败 | 无 | 步骤 1 和 4 部分失败（未能推导解决时间） |
| 得分 | — | insight=0.46, summary=0.6 |

---

## Case 2: flag-10

**数据集：** 500 条 ServiceNow 工单记录
**Goal：** Identify trends and underlying factors or correlations contributing to the increase in TTR.

### GT 推理链路（4 步）

| 步骤 | 类型 | 问题 | 发现 |
|------|------|------|------|
| 1 | diagnostic | TTR 随时间的趋势如何？ | TTR 随时间**线性增长**（从 opened_at 和 closed_at 计算 resolution_time） |
| 2 | diagnostic | 事件量和 TTR 是否有相关性？ | 正相关——事件量增加时 TTR 也增加 |
| 3 | time_series | TTR 增长是否在所有类别中均匀？ | 是，各类别 TTR 均匀增长，非某类别特有问题 |
| 4 | descriptive | Agent 生产力是否均匀？ | 各 agent 处理的事件数大致相同，生产力均匀 |

**GT 的推理逻辑：** 直接计算 TTR（closed_at - opened_at）→ 看趋势 → 找相关因素 → 排除 agent 生产力问题

### Agent 推理链路（5 步）

| 步骤 | 类型 | 问题 | 发现 |
|------|------|------|------|
| 1 | descriptive | 各 assignment group 和 category 的工单分布及平均 TTR 是多少？ | **失败**——报告缺少 "AssignmentGroup"、"Category"、"TTR" 列（实际列名是小写的 assignment_group/category，TTR 需要从 opened_at/closed_at 推导） |
| 2 | diagnostic | priority 和 TTR 是否有相关性？按 assignment group 或 agent 有差异吗？ | **失败**——报告 'ttr' 列不存在，'assigned_agent' 列不存在 |
| 3 | predictive | 基于历史月度趋势，预测下一季度的工单量和平均 TTR？ | **失败**——第一次执行代码报错，LLM 修复后成功：预测了月度工单趋势，但 TTR 相关分析仍不完整 |
| 4 | prescriptive | 哪些处理高优先级工单的 agent 或 group 应被优先培训？ | **失败**——找到 462 条高优先级工单，但 TTR 计算全部返回 NaN，无法给出有效建议 |
| 5 | diagnostic | 工单创建的时间（星期几/时段）是否影响 TTR？ | **失败**——脚本将 `closed_by`（关闭人）误认为关闭时间列，TTR 计算错误 |

**Agent 的推理逻辑：** 尝试统计各维度 TTR → 全部因为没有正确推导 TTR 字段而失败

### 对比小结

| 维度 | GT | Agent |
|------|-----|-------|
| TTR 计算 | ✅ 直接从 closed_at - opened_at 计算 | ❌ 期望数据中有现成的 "TTR" 列，不会自己推导 |
| 核心发现 | TTR 线性增长、与事件量正相关、各类别均匀 | 无有效发现，所有 insight 都是"分析失败" |
| 代码失败 | 无 | 5 步中 4 步失败或结果为 NaN |
| 得分 | — | insight=0.0, summary=0.0 |

---

## Case 3: flag-100

**数据集：** 500 条 ServiceNow 工单记录
**Goal：** Find the discrepancy and imbalance in distribution of incidents assigned across categories

### GT 推理链路（6 步）

| 步骤 | 类型 | 问题 | 发现 |
|------|------|------|------|
| 1 | descriptive | 各类别的事件分布如何？ | 完全均匀，每类恰好 100 件 |
| 2 | diagnostic | 事件描述中是否有特定问题？ | 词云显示无特定问题突出 |
| 3 | diagnostic | Printer 在描述中出现多少次？ | 0 次（与 flag-1 对比，说明此数据集无打印机问题） |
| 4 | descriptive | 硬件事件是否集中在某个地点？ | 不集中，各地点分布均匀（22/21/20/19/18） |
| 5 | descriptive | 事件分布随时间是否有趋势？ | 无显著趋势 |
| 6 | (其他) | 各类型事件最有效的解决方法是什么？ | **Restart service** 是最常见的解决方法 |

**GT 的推理逻辑：** 与 flag-1 相同的问题模板 → 逐一确认"没有异常" → 额外分析 resolution_method

### Agent 推理链路（5 步）

| 步骤 | 类型 | 问题 | 发现 |
|------|------|------|------|
| 1 | descriptive | 各 assignment group 的事件分布（数量和百分比）及与 category 的对比？ | 每组恰好 100 件（Gini=0.0），但组内有类别专长化（Luke Wilson 27% Hardware, Charlie Whitherspoon 26% Database） |
| 2 | diagnostic | 为什么某些 group 收到不成比例的事件？与 location/priority/closed_by 有关吗？ | **部分失败**——报告 `assignment_group` 列不存在（实际列名拼写为 `assignement_group`），但发现各 location/priority/closed_by 都完全均衡 |
| 3 | predictive | 基于趋势预测哪些 group 未来负载最不均衡？ | **完全失败**——代码执行出错，LLM 修复时三次超时，最终放弃 |
| 4 | prescriptive | 应该如何平衡工作量？是否应该重新分配类别或交叉培训？ | 发现各组 100% silo 化（Database 组只做 Database 工单），建议打破专属分配模式 |
| 5 | diagnostic | 用户满意度是否因 assignment group 不同而异？与工作量失衡是否相关？ | 满意度在 2.96-3.20 间，但因工作量完全均衡（每人 20 件），无法计算相关性 |

**Agent 的推理逻辑：** 看分布 → 分析分配驱动因素 → 预测（失败）→ 建议重分配 → 分析满意度

### 对比小结

| 维度 | GT | Agent |
|------|-----|-------|
| 核心发现 | 一切均匀，无异常，Restart service 是最常见解决方法 | 一切均匀，但发现了 100% silo 化分配模式（GT 未提及） |
| 是否分析文本字段 | ✅ 词云 + Printer 频率 | ❌ 未分析 short_description |
| 是否分析 resolution_method | ✅ 最常见方法 = Restart service | ❌ 未涉及 |
| 代码失败 | 无 | 步骤 2 部分失败（列名拼写），步骤 3 完全失败（超时） |
| 独特发现 | — | silo 化分配模式（GT 没有） |
| 得分 | — | insight=0.16, summary=0.0 |

---

## 总体问题汇总

### 1. 问题生成策略差异

- **GT：** 沿着"数据探索"路径深挖（分布 → 文本内容 → 具体实体 → 时间/地点维度）
- **Agent：** 沿着"管理咨询"路径横向展开（descriptive → diagnostic → predictive → prescriptive）

Agent 的问题完全由 `_generate_questions()` 中的 LLM 根据 schema + goal 生成，它只能看到列名，看不到数据内容，因此：
- 不知道 `short_description` 里藏着关键信息（printer）
- 不知道 `resolution_method` 这个字段有分析价值
- 倾向于对结构化字段（assignment_group, priority, location）做统计，而非对文本字段做内容挖掘

### 2. CodeAgent 代码生成能力不足

- **不会自主推导指标：** 数据中没有 "TTR" 列时，不会从 opened_at/closed_at 自己算
- **列名匹配不健壮：** 遇到 `assignement_group`（拼写错误）直接报错而非模糊匹配
- **不会做文本分析：** 从未尝试对 short_description 做词频/词云/关键词提取

### 3. 有效 insight 与 GT 重叠率低

即使代码成功执行、结果正确，由于问题方向不同，产出的 insight 与 GT 的 insight 不在同一维度上，导致 LLM judge 评分时匹配不上。
