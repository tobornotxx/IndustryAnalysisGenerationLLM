# DataSTORM → MyDataStorm → PI Agent 版本谱系审计

> 审计日期：2026-09-16
>
> 范围：论文、作者官方代码、`MyDataStorm` 全历史、`IndustryAnalysisGenerationLLM/pi-agent` 迁移历史、现有 benchmark 适配器与 skill 状态。
>
> 本轮仅做静态调研；没有调用模型 API，也没有生成新的实验结果。

## 1. 结论先行

1. `MyDataStorm` 的最早提交 `f8be022`（2026-05-07）就是基于 DataSTORM 论文 v1 的独立复刻。它不是作者代码的 fork：作者官方仓库直到 2026-08-19 才公开当前唯一提交。
2. 如果要选一个“最干净的论文复刻源码点”，应固定为 `f8be022`；如果要选一个仍保持论文结构、但模型配置更容易运行的历史点，可选 `5d0c343`（2026-05-17）。
3. `d60eb3a` 起出现面向成本与运行效果的行为修改；`429538d` 引入双模式问题树后，项目进入明显的自研分叉；`f7e275a` 删除论文明确包含的 warm-start，说明此后不宜继续称为 DataSTORM 论文复刻。
4. 现在的 `pi-agent` 不是 DataSTORM 的 TypeScript 移植版，而是“晚期自研 MyDataStorm”的 PI agent-loop 迁移和继续演化。它保留了问题树、目标充分性、insight bank、自一致性总结等后期设计。
5. 最近实验中标成 `datastorm-reproduction` 的系统不是早期论文复刻，也不是作者官方实现。它实际加载 `MyDataStorm@43292dd`，并通过适配器施加 3 层、总问题 6、最多 10 条 insight、executor 5 turns 等预算；该版本还会在 goal-sufficiency 中自动加载手工候选 skill。这个系统应改名为 `legacy-custom-python` 或 `pi-python-predecessor`。
6. 当前的 `skill_package_v9.json` 明确标注为 `manual-transcription-from-v8 (NOT the extraction agent)`、`candidates-pending-validation`、`frozen=false`。仓库已有 provenance mining 与治理门禁骨架，但“自动从轨迹抽取、泛化、跨任务验证、冻结 skill”的完整系统尚未实现。因此，现阶段不能把它写成论文的已完成核心方法。
7. 后续论文的核心比较应是 `PI-core` 对 `PI + auto-extracted skills`；作者官方 DataSTORM 和官方 AgentPoirot 是外部 baseline；晚期 Python 版只是内部前驱/工程消融，不应冒充外部 baseline。

## 2. 论文与官方实现

### 2.1 论文版本

论文：**DataSTORM: Deep Research on Large-Scale Databases using Exploratory Data Analysis and Data Storytelling**。

- arXiv v1：2026-04-07。
- v2：2026-08-17。
- v3：2026-08-26；COLM 2026。
- 论文将系统描述为面向结构化数据库与互联网来源的 thesis-driven 深度研究：warm-start、迭代式 planner/executor 数据探索、query consistency、自底向上统计、全局 insight bank、论点生成/精炼与分阶段报告生成。

来源：

- 论文主页：<https://arxiv.org/abs/2604.06474>
- 作者 publication 页面：<https://cs.stanford.edu/~yuchengj/publications/>

### 2.2 为什么本地复刻对应 v1

`f8be022` 提交时间为 2026-05-07，处于 arXiv v1（4 月 7 日）和 v2（8 月 17 日）之间，因此它使用的只能是 v1 时期论文。仓库中的 `datastorm.pdf` 在 `f8be022`、`5d0c343` 与当前 `origin/main` 中 git blob 均为：

```text
075977375942c64ba7b9c412ea61842f3088caea
```

本地文件 SHA-256：

```text
47265E27E845208C4E64E2602CD855AAE0CA47691BE4AE6E2B3F46E9D6AA82FB
```

该 PDF 为 38 页。时间边界比页数更关键：它不是 8 月份的 v2/v3。

### 2.3 作者官方仓库

作者官方仓库：<https://github.com/stanford-oval/DataSTORM>

审计时固定的唯一提交：

```text
29ba2031290e40a388ce548582d4e43bbac05d72
2026-08-19T04:17:28Z
DataSTORM: deep research on large-scale databases
```

官方代码结构与本地复刻不同：

- `knowledge_storm/datatalk_agent/`：自然语言到 SQL/SUQL 的数据代理。
- `knowledge_storm/datastorm/`：树搜索、分支评分与剪枝、全局 insight、报告生成。
- `eval/`：论文评估工具。
- `results/`：论文公开结果，包含 InsightBench 全 100 个任务。

官方 README 把执行过程定义成树搜索：每个节点提出子问题、生成并执行 SQL、判断有趣程度、继续扩展有希望的分支，再将发现汇总为全局 insights 和报告。这与本地最早复刻的“分层 planner + ReAct executor”在论文概念上对应，但代码血缘不同。

### 2.4 官方结果与我们现有评估不是同一个量

官方 `results/README.md` 报告：

- DataSTORM：GPT-5，100 个 InsightBench 任务。
- AgentPoirot baseline：GPT-5，100 个任务。
- GPT-4o judge insight recall：DataSTORM `0.6187`，baseline `0.4708`。
- Qwen3-30B judge insight recall：DataSTORM `0.6913`，baseline `0.4994`。

但是论文的核心自动指标是 best-match recall，基本不惩罚预测过多。官方案例 11–15 分别输出 83、74、92、89、88 条预测；我们最近的设置最多只保留 10 条，并主要看 precision/F1。这两个数不能直接横比。

因此，后续实验至少要把以下两种问题分开：

- **论文复现问题**：是否在作者协议下接近作者公开结果。
- **受控系统比较问题**：在相同模型、预算、输出上限和本地评分器下，哪个系统更有效、更省成本。

## 3. MyDataStorm 的真实演化

### 3.1 论文复刻阶段

#### `f8be022` — 最干净的 v1 论文复刻点

时间：2026-05-07，root commit。

README 直接声明“基于论文的复现实现”，并写明 prompt 1:1 对应论文。代码包含：

- internet warm-start；
- 多层 planner/executor；
- ReAct 风格 SQL executor；
- query consistency；
- bottom-up statistics；
- insight bank；
- thesis generation/refinement；
- outline → draft → citation validation → revision 的报告流水线。

关键默认参数也与论文描述一致：

```text
max_layers = 5
first_layer_max_questions = 2
subsequent_layer_max_questions = 5
executor_max_turns = 15
max_insights = 50
```

这应当作为仓库历史中的“论文复刻锚点”。

#### `5d0c343` — 更适合实际运行的早期快照

时间：2026-05-17。

主要变化是统一 LLM 配置、支持 `api_base`、修正 token 参数。它仍保留原始架构，适合作为“早期复刻的可运行快照”；但如果论文要求精确说明源代码版本，应优先写 `f8be022`。

### 3.2 工程优化阶段

以下修改仍可被描述为“在复刻框架上进行工程优化”，但已经开始影响算法行为：

- `d60eb3a`（2026-05-26）：减少 API 调用、增加 early stopping、修改报告与解析路径。
- `23dba51`（2026-05-27）：向 executor 注入 schema，减少无效工具轮次。
- `36e3968`（2026-05-27）：并行执行问题和报告章节。
- `acb8b4d`（2026-05-28）：重做 insight filter，并强制精确问题数。

这些改动不再是“只换模型/修 bug”，后续若比较它们，需要作为明确的实现差异或消融变量报告。

### 3.3 自研框架分叉阶段

#### `429538d` — 明确的架构分叉点

时间：2026-06-08。

新增 question-tree，区分 follow-up 与 exploratory 两类扩展，并加入 question logger。这个提交可作为“自研 MyDataStorm”阶段的起点。

随后又发生：

- `084a0c2`：第一层强制优先基础 EDA。
- `f7e275a`：删除 warm-start；这是与原论文流程最清晰的结构性背离。
- `88f0d1f`：所有问题持续锚定原始研究目标。
- `755ec82`：暴露完整探索树。
- `ce28e0f`：加入 goal-sufficiency 自评来驱动深度和早停。
- `b28d9a1`：强化维度分解与 anti-premature-stop。
- `b6212fe`：把部分硬编码分析引导包装成 skill package。
- `787b7f7`：insight bank 不再把原始统计块送给 LLM。
- `ab0c07d`：回滚测试集泄漏和无依据 skill。

从 `f8be022` 到 `ab0c07d`，共有 25 个文件变化，增加 1819 行、删除 288 行。它已经不是“原论文复刻加少量修补”。

### 3.4 PI Agent 迁移阶段

`IndustryAnalysisGenerationLLM/pi-agent` 的迁移序列：

- `03b5f8a`（2026-08-31）：真实 API 闭环与常驻 Python worker pool。
- `35500d8`（2026-09-01）：planner 与 agent 主体。
- `18f9171`（2026-09-01）：补齐其余模块。
- `7a00c37`：修复 insight bank 匹配并补负结果/解释引导。
- `a1e74be`（2026-09-03）：形成较干净的 PI v4 基线。

因此，“手写 ReAct → PI loop”只准确描述**晚期自研 Python 版本到 PI 版本**的局部迁移。它不能被扩展成“原始 DataSTORM 与 PI 只有 agent loop 不同”，因为在 PI 迁移之前，问题树、warm-start、goal sufficiency、skill guidance、insight 处理等已经发生多轮改变。

## 4. 当前实验命名为何有误

现有 `DataStormAdapter` 的注释仍写“将 MyDataStorm 包装为 InsightBench 兼容 Agent”，但实验标签使用了 `datastorm-reproduction`。实际行为是：

- 导入当前 MyDataStorm worktree，而不是 `f8be022` 或 `5d0c343`。
- 当前源提交为 `43292dd`。
- `max_layers=3`，而论文复刻默认为 5。
- 每层 2 个问题，并另设总问题上限；最近有效轮使用 `max_questions=6`。
- `max_insights=10`，早期复刻默认为 50。
- `executor_max_turns=5`，早期复刻/论文设置为 15。
- 禁用 web 与 citation check。
- 使用后期 question tree、goal sufficiency、budget 等模块。
- goal-sufficiency 模块会无条件调用 `get_skill_package()` 并注入 `coverage-multidim-selfcheck` 与 `anti-premature-stop`；适配器没有 no-skill 开关。

所以之前那一轮得分仍有工程价值，但其解释只能是：

> 在统一本地 harness 和受限预算下，PI 实现与其晚期 Python 前驱的比较。

不能解释为：

> PI/PyAgent 对 DataSTORM 原论文或作者官方实现的比较。

建议立刻将实验系统标识更名为 `legacy-custom-python`，保留 manifest 中的源提交和配置，不删除旧结果。

## 5. Skill 系统当前完成度

### 5.1 已有部分

- `skill_package_v9.json`：5 条候选分析引导。
- `mining.py`：provenance episode/mining 的部分数据结构与逻辑。
- `governance.py`：purity audit 与 freeze gate。
- PI 与 Python 两侧可以加载同一份 skill package。
- PI runner 提供 `useSkills` 开关，可用于 on/off 消融。

### 5.2 缺失部分

- 从历史轨迹/成败样本自动定位可复用经验的 extractor。
- 把 case-specific 经验泛化为数据无关 trigger/action/rationale 的 generalizer。
- 去重、冲突检测、skill 组合与版本化。
- 在训练/开发 case 上做自动 on/off validation。
- 在至少两个不相关 case 上验证迁移后才晋升/冻结的闭环。
- 对 held-out 测试集严格禁止反向写回 skill 的数据治理。
- 对 skill 被触发、采取何种动作、带来何种轨迹变化的可解释日志。

### 5.3 论文表述边界

现在可以写：

> 我们已经实现候选 skill 的跨运行时表示、加载与治理门禁原型。

现在不能写：

> 系统已经能够自动从轨迹中提炼并验证可迁移 skill。

后一句必须等完整闭环与 held-out 迁移实验完成后再成立。

## 6. 修正后的系统谱系与实验角色

### 外部 baseline

1. **DataSTORM-official@29ba203**：作者代码，外部主要 baseline。
2. **AgentPoirot-official**：InsightBench 作者 baseline，外部传统 agent baseline。

### 历史/内部参照

3. **Paper-guided reproduction@f8be022**：证明早期复刻来源与历史，不必自动等同作者实现。
4. **Legacy-custom-Python@固定提交**：PI 的直接 Python 前驱，用于迁移正确性和工程消融。

### 论文方法

5. **PI-core**：不注入 skill 的主体框架。
6. **PI + auto-extracted skills**：拟议的核心方法。

论文的主因果问题应是 5 对 6。1 和 2 用来建立外部竞争力；4 用来说明 PI 化和后期结构改造的贡献；3 主要承担 provenance，而不是必须投入大预算反复跑的主 baseline。

## 7. 获准实验前还应完成的非 API 工作

1. 修正系统命名和 manifest schema，禁止再把当前 MyDataStorm 标为论文复刻。
2. 将 `f8be022`、`5d0c343`、`29ba203` 写入不可变 source registry。
3. 为晚期 Python 适配器增加显式 `use_skills` 开关，否则不能做干净的 PI-core 对照。
4. 为官方 DataSTORM 写独立 adapter；不要复用现有 `DataStormAdapter` 名称造成混淆。
5. 同时保留两套协议：
   - paper-protocol：尽量复现官方 breadth/depth/output 与 recall；
   - controlled-protocol：统一模型、问题预算、输出上限、judge 与重复次数。
6. 补齐自动 skill 提取闭环，再冻结训练得到的 package，之后才允许接触 held-out 目标。
7. 先用静态 fixture 和 fake LLM 测试 adapter、预算、开关、日志和产物格式；这些不需要 API。

## 8. 下一轮实验的准入条件

在开始任何新 API 实验前，至少应满足：

- 每个系统都有唯一、准确、不误导的 ID。
- 每次运行记录 repo、commit、dirty state、模型、thinking 模式、预算、skill package hash、随机重复编号。
- `PI-core` 与 `PI + skills` 除 skill 开关外保持同配置。
- 官方 DataSTORM 的“公开 GPT-5 结果复评”和“V4.1 matched-model 重跑”分开报告。
- 不把作者 recall 与本地 capped-output F1 直接放在同一列得出优劣结论。
- 训练/开发 case 与 held-out 测试 case 的信息流隔离可被审计。

满足这些条件后，实验结论才适合写进毕业论文，而不是仅作为调试记录。
