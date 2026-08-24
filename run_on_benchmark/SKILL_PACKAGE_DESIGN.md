# Skill 提炼设计:Skill 包 + Skill 提取 Agent + 经验回放

> 状态:设计稿(待确认决策后实施)。与 `TRANSFER_DESIGN.md` 配套。
> 日期:2026-07-30
> 关系:**本文定义"被迁移的是什么"**(skill 如何提炼、固化、冻结);`TRANSFER_DESIGN.md` 定义"怎么迁、怎么评"。提炼是迁移的前置。
> 置信度标注:[核实] = 已直接抓 arXiv 页面确认;[待核实] = 未逐字验证。

---

## 0. 为什么需要这份文档

导师说的"在 InsightBench 上训 skill",这里的"训"**不是训权重**(我们没有 finetune 模型),而是:

> **把零散的 prompt/机制调优,固化成一个明确的、可冻结的、数据无关的"行为技能包"。**

这个包就是被迁移的对象。**没有这一步,"迁移测试"是空话**——你无法区分:
- (a) 是"在 InsightBench 上训出的 skill 泛化到了新 benchmark",还是
- (b) "我又在新 benchmark 上重新调了一遍 prompt"。

(b) 不是迁移,是二次过拟合。本文档的全部设计,都是为了让 (a) 可被证明、(b) 被物理阻止。

---

## 1. 三个组件与总体架构

```
[Skill 提取 Agent] --离线,在 InsightBench 上跑一次--> [Skill 包 v8(冻结)]
     (训练器)                                              │ load
                                                           ▼
                                          [分析 Pipeline(MyDataStorm)] --跑新 benchmark--> 结果
                                               (推理引擎)
```

| 组件 | 类比 | 角色 |
|---|---|---|
| **Skill 提取 Agent** | 训练器 | meta-agent,分析 pipeline 的历史表现,蒸馏出 skill |
| **Skill 包** | 训练产物(模型权重) | 声明式、数据无关、可冻结的技能包 |
| **分析 Pipeline** | 推理引擎 | load 包、跑数据、产 insight |

**关键纪律**:迁移测试时,**提取 agent 退场,包冻结,pipeline 只 load 不改**。这是"训在 InsightBench、测在新 benchmark"能物理成立的唯一保证。

---

## 2. 第一步:定义 —— skill 落到物理载体

skill 不是抽象概念,是具体的文件 / prompt / 参数。把 `TRANSFER_DESIGN.md` §1.3 的 5 个行为单元映射到物理载体:

| 行为单元 | 物理载体 | 来源迭代 |
|---|---|---|
| Planner 探索决策 | `planner.py` tree 生成 + focus_aspects 注入;`max_layers=3, questions_per_layer=2` | 架构 |
| Goal-sufficiency 自评 | `goal_sufficiency.py` evaluate() prompt(四维分解 + 反过早停) | v8 |
| Executor 计算 | `executor.py` + `bridge.execute_python_from_sql` | 架构 |
| Schema 画像 + 趋势分解 | `csv_db_bridge.py`: `_temporal_profile` + `_ANALYTICAL_GUIDANCE` | v8 |
| Insight 表达 | insight_bank / report summary prompt + `summary_samples=3` 自一致 | v6/v7 |

**skill = 这 5 个载体的并集。** 其余(`run_benchmark.py` 入口、`unified_scorer.py`、loader)是**脚手架,不算 skill**——迁移测试期间脚手架可以改(要接新数据、新指标),skill 不能改。

→ **产出 1:skill manifest**(文件路径 + 关键 prompt 片段 + 参数 + commit hash)—— skill 的身份证。

---

## 3. 第二步:纯度审计 —— 提炼的核心动作

**"提炼"的关键不是加东西,是剥离过拟合。**

**规则**:skill 文本不得硬编码任何具体数据集的列名 / 列值 / 业务假设,只能用"上方 schema 列出的分类列"这类**泛化指代**。

### 3.1 已确认的实锤泄漏

`csv_db_bridge.py` 的 `_ANALYTICAL_GUIDANCE` 举例写的是:

```
(e.g. category, assigned_to, priority, assignment_group)
```

**这四个是 ServiceNow 工单字段名。** 虽然标了 "e.g.",但 LLM 看到这些具体词会被诱导往工单场景靠。

> ⚠️ **不剥掉这个,迁移到 DataGovBench 的政府数据(氡浓度、人口加权)时,agent 会下意识去找 "assigned_to" 这类列,必塌。**

**整改方案**:把举例换成**运行时从当前 schema 动态抽 2 个分类列名**注入,彻底数据无关。

### 3.2 审计 checklist

| 载体 | 状态 | 说明 |
|---|---|---|
| `_ANALYTICAL_GUIDANCE` 工单字段举例 | ⚠️ **已确认,必改** | 见 3.1 |
| `goal_sufficiency.py` 四维分解的"智能体/优先级"措辞 | ⚠️ 嫌疑,需重读核实 | 可能 = assigned_to / priority 的中文化 |
| `_temporal_profile` 自适应粒度 | ✓ 通过 | 纯按列 dtype 推断,无数据特化 |
| executor 沙箱预注入包 | ✓ 通过 | 通用数据科学包,无特化 |
| planner tree 结构 / 超参 | ✓ 通过 | 结构性,非内容特化 |

→ **产出 2:纯度审计报告**(每载体:通过 / 嫌疑 / 整改)。
> **整改完才准进冻结。这一步直接决定迁移干不干净。**

---

## 4. 第三步:Skill 包(产物)

### 4.1 设计意图

把现在**硬编码在代码里**的引导(含 3.1 那个泄漏),重构成**声明式、可审计、可移植**的 skill 包。顺手修掉纯度问题。

### 4.2 包结构

```jsonc
SkillPackage {
  "meta": {
    "version": "v8",
    "source_benchmark": "InsightBench",
    "produced_by": "SkillExtractionAgent",
    "commit_hash": "...",
    "purity_audit": "passed",
    "frozen": true
  },
  "skills":   [ /* 语义记忆:抽象规则,见 4.3 */ ],
  "episodes": [ /* 情景记忆:具体经历,见 §7 */ ],
  "config": {                      // 冻结超参(非 skill,但随包冻结)
    "max_layers": 3,
    "questions_per_layer": 2,
    "summary_samples": 3
  }
}
```

### 4.3 单个 skill 的结构

```jsonc
{
  "id": "decompose-trend-by-category",
  "trigger": "question concerns TREND / GROWTH / CHANGE OVER TIME",
  "action": "do not test only on overall total; decompose the trend by the primary "
            "categorical columns shown in the schema; report per-group + overall; "
            "flag any trend that appears in ONE subgroup but is absent overall",
  "rationale": "a trend may exist in a subgroup while being absent in the aggregate",
  "provenance": "v8; fixed flag-4 GT4 (0.1→1.0), flag-6 GT2 (0.3→1.0)",
  "purity": "passed — references 'categorical columns in schema', no hardcoded field names",
  "validated": { "on_vs_off_delta_recall": +0.18, "cases": ["flag-4", "flag-6"] }
}
```

**四个硬约束**(由提取 agent 保证,见 §5.4):

1. **程序性**:action 必须是"通用分析动作",不得是具体答案 → 防 skill 退化成背 GT。
2. **数据无关**:trigger / action 不得出现具体列名 / 列值 / 业务词。
3. **带 provenance**:记录它从哪个 case、哪次迭代来。
4. **带 validated**:记录 on/off 分数差,证明它真有用而非 placebo。

### 4.4 现有硬编码 → skill 的映射(落地清单)

| 现有载体 | → skill id | 备注 |
|---|---|---|
| `_ANALYTICAL_GUIDANCE`(趋势部分) | `decompose-trend-by-category` | **剥离 `assigned_to` 等工单字段** |
| `_ANALYTICAL_GUIDANCE`(失衡部分) | `decompose-imbalance-by-group` | 同上 |
| `goal_sufficiency` 四维 + 反过早停 | `coverage-4dim-selfcheck` | "智能体/优先级"措辞泛化 |
| `_temporal_profile` 自适应粒度 | `pick-adequate-time-bucket` | 已数据无关,直接迁 |
| `summary_samples=3` 等超参 | → `config`(非 skill) | |

### 4.5 运行时消费方式

**起步用简单方案**:pipeline 启动读包,把所有 skill 的 `trigger + action` 拼进 `get_schema_context()` / goal-sufficiency prompt,**取代现在硬编码的 `_ANALYTICAL_GUIDANCE`**。

- 包很小(~4-6 个 skill),全注入够用。
- **不搞运行时 RAG 检索**——那会引入新的自由度(检索什么、检索多少),等于给过拟合开新口子。AgentAda 那种 74-skill 检索库是另一个量级的设计,先不做。

---

## 5. 第四步:Skill 提取 Agent(本体)

### 5.1 它是什么

一个 **meta-agent**——它**不分析数据,而是分析"分析 pipeline 的表现"**,把成功/失败经验蒸馏成声明式 skill 写进包。

**它就是"训 skill"的那个训练器,用 agent 循环代替梯度下降。**

### 5.2 输入

- InsightBench 运行结果(v5→v8 轨迹、per-case 分数、pred vs GT)
- **成功轨迹**(pipeline 命中 GT 的 case)+ **失败 case**(漏掉的,如 flag-5 `closed_by`)
- 当前 skill 包(迭代用)
- GT insights + goals

### 5.3 处理循环(6 步)

```
mine         扫结果,找分数涨/跌的 case、找失败模式
             (例:flag-4/6 涨 ↔ 趋势分解;flag-5 漏 ↔ closed_by 没当分析轴)
hypothesize  从失败/成功模式提候选 skill("当 X 触发,做 Y")
generalize   剥列名/列值/业务词,做成数据无关的 trigger+action   ← 纯度在这一步发生
validate     在受影响 case 上跑 skill on/off 迷你 ablation,只留真涨分的
audit        查包里有没有漏网的硬编码字段(自检 + §3.2 checklist)
emit         写/更新 skill_package.json + 训练日志,打版本
```

### 5.4 防过拟合:三条护栏(关键)

**提取 agent 自身会过拟合。** 它可能提出一个 "找 closed_by 字段的分布" 的 skill —— 那是**背 GT,不是技能**。三条护栏:

1. **程序性约束**:action 必须是通用分析动作(分解 / 自检 / 选粒度),不得命名具体发现。提"找某字段分布"会被拒。
2. **validate(迷你 ablation)**:候选 skill 必须在 **≥2 个不相关 case** 上 on/off 都涨分才入库,防单 case 过拟合。
3. **purity audit**:trigger / action 出现任何具体列名 / 值 → 拒,要求重写泛化版。

### 5.5 架构

- LLM agent(deepseek 即可,meta 推理不需要多模态)+ 几个 tool:
  - `read_run_logs(case_id)` / `read_gt(case_id)`
  - `run_mini_eval(skill, case_ids)` → 调主 pipeline 在指定 case 上跑,返回 on/off 分数差
  - `write_skill_package(...)`
- 驱动方式:一次性批处理(扫全部 case → 提一版包)或增量(每轮 v 迭代更新)。

**纯度守门用 pro 不用 flash**:纯度审计 / 程序性约束 / episode 抽象是全流程最不该省钱的地方——判错一次整个包就脏了,后续所有迁移结论都不可信。

→ **产出 5:skill 训练日志**(每个 skill:从哪个 case 来、on/off 差、是否因过拟合被拒)。

### 5.6 每 case 新增 LLM 调用 = 0(重要)

设计原则:**不为新增指标付出每 case 的调用成本。** 具体落实:

| 原计划的 judge | 处置 | 理由 |
|---|---|---|
| precision + F1 | ✅ 做,**零额外调用** | 与 recall 共享同一 pairwise 矩阵(`TRANSFER_DESIGN.md` §6.2) |
| novelty(3 票) | ⏸️ **暂不做** | 唯一真三倍成本项(约 +45 次/case);加分项非主指标 |
| 4 维 rubric | ⏸️ 后置 | 仅 DataGovBench 6 case,总量小,不急 |
| 轨迹反馈质量判定(档 2) | ✅ 做,但**用规则不用 LLM** | SQL 报错 / 结果集空 / 行数过少 —— 规则即可判,无需 LLM |
| episode 检索匹配(档 1) | ❌ **删除** | 已定全注入(§4.5),包里就几条 episode,无需检索 |
| 纯度审计 / 程序性约束 / episode 抽象 | ✅ 做,**离线一次** | 只在提炼阶段跑,不进每 case 成本;用 pro |

**结论:每 case 的 LLM 调用数不因本次升级而增加。** 提炼阶段的额外成本是一次性的、离线的。

---

## 6. 第五步:冻结 —— 过拟合防火墙

审计整改后锁死:

- **skill-v8** = manifest 所有载体 + skill 包在某个 commit 的快照。
- 迁移测试期间(跑 InsightEval / DataGovBench)**禁止改 skill 载体**(planner / executor / goal_sufficiency / bridge 的 prompt 和参数),**只能改脚手架**(loader / scorer / 入口)。
- 任何"想改 skill"的冲动 = 又在针对新 benchmark 调 = 过拟合。**记录下来,不许动**;若确实必要,回到提取 agent 走 validate + audit 流程,不许手改包。

→ **产出 3:冻结声明 + commit hash**。

### 6.1 ⚠️ 冻结前必须做完的改动(重要)

以下改动**触及 skill 载体**,所以**必须在冻结之前完成**,否则破坏冻结原则:

| 改动 | 来源 | 说明 |
|---|---|---|
| 剥离 `_ANALYTICAL_GUIDANCE` 工单字段 | §3.1 纯度审计 | 必做 |
| `goal_sufficiency` 措辞泛化 | §3.2 待核实 | 核实后决定 |
| **bridge 表序列化改造** | DataGovBench 消融[核实] | 见下 |
| **run 内轨迹反馈** | 经验回放档 2 | 见 §7.3 |

> **bridge 序列化的冲突**:DataGovBench 消融实测——**按列类型序列化 +50 分**(数值列给 min/max、分类列给 unique 值、时间列给最早/最晚、其他给随机 10 样本),而**只加 schema 没用**(0.310→0.308)、**多给行数反而掉分**(0.275,"LLM 在大池子里找不到关键信息")。这与我们 `get_schema_context()` 现在的路线部分冲突。若要吸收,**必须在提炼阶段改完再冻结**,不能迁移中途改。

---

## 7. 经验回放:导师提出的方向如何整合

### 7.1 导师的说法

> "股票交易那边有些 agent 论文采用相似做法:把过往交易过程中**决策和结果的整条链路**,提供给模型下一次做决策时参考,是会有收益的。"

这在文献里对应 **experience replay / episodic memory + reflection**。核实过的代表作[核实]:

| 论文 | arXiv | 机制 |
|---|---|---|
| **FinMem** | 2311.13743 (AAAI SS 2024) | 分层记忆(浅/中/深),按 recency+relevancy+importance 排序;**"extended reflection" 用股价走势、交易回报、行动理由重新评估过往决策**,结果存入深层 |
| **FinCon** | 2407.06567 (NeurIPS 2024) | 风险控制组件**周期性自我批判,更新系统性投资信念**;信念作为"verbal reinforcement"影响未来行为 |
| FinAgent | 2402.18485 (KDD 2024) | 双层反思模块 + 多样化记忆检索(摘要级证据) |
| Reflexion | 2303.11366 (NeurIPS 2023) | 口头反思存入 **episodic memory buffer**,喂给后续尝试 |
| ExpeL | 2308.10144 (AAAI-24) | 从训练任务自主收集经验、抽取知识,推理时**回忆洞察与过往经验** |
| Generative Agents | 2304.03442 (UIST 2023) | memory stream 记录完整经历,综合成高层反思,按 recency+relevance+importance 检索 |

> ⚠️ **一处更正**:我此前口头提过 TradingAgents(2412.20138)作为例子。核实后发现:**该论文正文并未描述决策→结果的记忆机制**(机制在其代码库里),其 related work 反而把这个能力归给 FinMem/FinAgent。**引用时应用 FinMem 和 FinCon,不要用 TradingAgents 支撑此说法。**

**领域内佐证**:上一轮调研挖到的 **DataSeer**(InsightEval 同组后续,ACL 2026 Findings)正是用**双层记忆系统(压缩/合并/检索)**——说明这个 idea **在我们这个领域已经是最新 SOTA 的组件**,不是跨域硬套。

### 7.2 和现有设计的关系:同一记忆层级的两层

| | 现有 skill 包 | 导师说的做法 |
|---|---|---|
| 记忆类型 | **语义记忆** —— 抽象规则("遇到趋势问题按分类列分解") | **情景记忆** —— 具体经历("这次问了 X,跑出 Y,结果好/坏,教训 Z") |
| 形态 | 泛化、去具体化、数据无关 | 保留具体、带上下文 |
| 时机 | 离线蒸馏一次,冻结 | 决策时检索、常在线累积 |

**两者互补,不冲突。** 更重要的是:**§5 的 skill 提取 agent 本身已经是这个 idea 的一种形式**——它就是扫过往运行的"轨迹 + 分数结果"再蒸馏。

**区别在于:提取 agent 把 episode 蒸干成规则后就把 episode 扔了;交易 agent 是把 episode 留着、决策时检索出来当范例。** 导师的说法等于说:**别扔,留着也有收益。** 这是现有稿的真实缺口。

### 7.3 ⚠️ 一个致命的不对称:结果可不可观测

照搬会翻车的地方:

> **交易 agent 的结果信号是免费的、即时的、近乎 ground truth 的** —— 下单后市场直接告诉你盈亏。这是它整个循环转起来的燃料。
>
> **我们的任务在测试时没有这个信号。** agent 产出 insight 后,没有任何东西告诉它这条好不好 —— GT 和 judge 分数**只在离线的训练 benchmark 里有**。

因此必须分层:

- **离线(InsightBench,训练侧):有真结果信号**(G-Eval 分 vs GT)→ 导师这套完全可用,**这正是 skill 提取 agent 该待的位置**。
- **在线(迁移测试时):没有真结果信号**,只能用**代理信号**:SQL 报没报错、结果集是否空/过稀疏、goal_sufficiency 是否判定已覆盖、insight_bank 判新颖还是冗余。弱一些,但真实可得。

### 7.4 整合方案:三档(按对"冻结"原则的威胁排序)

#### 档 1 ✅ 推荐:episode bank 进 skill 包,一起冻结

skill 包从 `skills[]` 扩成 `skills[] + episodes[]`:

```jsonc
"episodes": [{
  "situation": "goal 涉及某实体随时间的工作量趋势;表含高基数分类列 + 时间列",
  "decision":  "先整体按月聚合,再按该分类列分组分别做趋势检验",
  "observation": "整体平缓,但单个分组呈单调上升",
  "outcome":   { "judged_score": 1.0, "gt_hit": true },
  "lesson":    "整体平缓不代表无趋势;分组后才浮现"
}]
```

- 离线从 InsightBench 挖好,**随包冻结**。
- planner 决策时检索出来当 few-shot 范例 —— 这就是交易 agent 的检索式决策。
- 因为 bank 是**离线建好并冻结**的,**迁移测试的干净性不受影响**。

> ⚠️ **一个真实的张力,必须说在前面**:episode 的价值恰恰在于它**具体**,而具体 = 携带 ServiceNow 内容 = **纯度污染**。
> 缓解:把实体**抽象成角色** —— 写"一个高基数分类列"而不是 `assigned_to`,写"某实体"而不是 `Beth Anglin`。
> 但**抽象过头,episode 就退化成规则,和 `skills[]` 没区别了**。
> **这个度是本档的核心设计风险,不能假装没有。**

#### 档 2 ✅ 推荐:run 内轨迹反馈(补的是真缺口)

单个数据集探索过程中,把"上一层问了什么 → 跑出什么 → 有没有产出有效 insight(还是空结果/冗余)"喂给下一层 planner。

- **现状是缺的**:我们现在只把 `missing_aspects` **前馈**(还缺什么),**但不反馈"已问过的问题里哪些有产出、哪些是哑弹"**。planner 不知道自己刚才白问了。
- 全部使用**测试时可得的代理信号**,是在没有 GT 的前提下对导师那套**最忠实的移植**。
- 不碰冻结边界(全在单实例内、是机制而非习得内容),但**改 planner = 改 skill 载体,所以要在冻结前做完**。

#### 档 3 ⚠️ 危险,但可作独立实验臂

跨实例在线累积:跑 InsightEval 100 个实例时,第 50 个用上前 49 个的经验。

- 最贴近交易场景(交易本质就是时序在线)。
- 但它**打破冻结前提**:结果依赖实例顺序,且"这是 InsightBench 训的 skill 迁移过来的,还是刚从前 49 个实例现学的?"变得**无法回答**。

**诚实的做法 —— 不放进主臂,而作为第二条明确标注的实验臂**:

| 实验臂 | 配置 |
|---|---|
| **臂 A** | 冻结 skill 包(纯迁移测试) |
| **臂 B** | 冻结 skill 包 + 在线跨实例累积 |

> **A vs B 的差值,恰好就是在我们这个领域直接检验导师那个假设("喂过往决策链路有收益")的实验。**
> 这比偷偷混进主臂强得多,而且是个能写进论文的对照。

---

## 8. Ablation 锚定 —— 证明是 skill 在迁移

光冻住不够,要证明**提炼出的具体行为**真带了迁移收益。在新靶子上做 skill-on vs skill-off:

| ablation 切片 | 关掉什么 | 预期 |
|---|---|---|
| full skill | — | 基线 |
| − 四维分解 | goal_sufficiency 退回旧版 | recall 掉(覆盖变差) |
| − 趋势分解准则 | 去掉 `decompose-trend-by-category` | 趋势类 GT 掉 |
| − goal-sufficiency 自评 | 关 missing_aspects→focus 反馈 | 深层覆盖掉 |
| − 自一致 summary | `summary_samples=1` | summary 方差升 |
| **− episodes** | 去掉 episode bank(档 1) | 验证情景记忆是否真有收益 |
| **− 轨迹反馈** | 关 run 内产出质量回传(档 2) | 验证是否真有收益 |

**full 显著高于各切片 → 证明是这些提炼行为在迁移**,不是"有个 planner+executor"就够。这才是可发表的证据。

**成本控制**:ablation 只在 InsightEval 上跑,且可抽样 20–30 实例子集;DataGovBench 只有 6 case,ablation 仅看定性方向。

→ **产出 4:ablation 表**(每靶子 full vs 各切片的 recall / F1)。

---

## 9. 交付产出汇总

| # | 产出 | 内容 |
|---|---|---|
| 1 | **skill manifest** | 文件路径 + 关键 prompt 片段 + 参数 + commit hash |
| 2 | **纯度审计报告** | 每载体:通过 / 嫌疑 / 整改 |
| 3 | **冻结声明** | commit hash + 冻结范围 + 禁改清单 |
| 4 | **ablation 表** | full vs 各切片,证明 skill 有效 |
| 5 | **skill 训练日志** | v5→v8 每次迭代:加了什么、修了哪个 case、泛化保留 or 过拟合剥离 |

**训练日志示例**:
- v8 四维分解 → 修 flag-4 GT4 / flag-6 GT2 → **泛化**(数据无关)→ 保留进 skill
- flag-5 `closed_by` 盲区 → **过拟合**(字段特化)→ 剥离,不进 skill

---

## 10. 工程量与实施顺序

### 已完成(2026-07-30)

| # | 内容 | 状态 |
|---|---|---|
| 0a | `score_insight_matrix()` —— recall/precision/F1 共享矩阵,零额外调用 | ✅ 已实施 + mock 单测通过 |
| 0b | `gt_first` prompt 模板 —— 稳定前缀 32.9%→91.6%,含 `GEVAL_PROMPT_ORDER` A/B 开关 | ✅ 已实施,**稳定性待用户论证** |
| 0c | `_record_usage()` / `get_usage_stats()` —— token + 缓存命中核算 | ✅ 已实施 |
| 0d | `result.json` 落盘三指标 + `scorer_usage` | ✅ 已实施 |

### 待做

| 阶段 | 内容 | 工期 |
|---|---|---|
| 0e | 跑 1 个 case 验证:三指标数值合理 + 拿到真实缓存命中率与输出 token | 0.5 天 |
| 0f | `gt_first` A/B 稳定性验证(用户负责论证口径,我出对照数据) | — |
| 1 | 纯度审计(读全 5 个载体,出报告) | 0.5 天 |
| 2 | skill 包 schema + loader + pipeline 消费注入(取代硬编码) | 1 天 |
| 3 | 冻结前 skill 载体改动(纯度整改 + bridge 序列化 + 轨迹反馈) | 1 天 |
| 4 | 提取 agent(6 步循环 + mini-eval tool) | 1.5 天 |
| 5 | 跑一版 v8 包 + episode bank 抽取 | 0.5 天 |
| 6 | 冻结 + 打 commit | 0.5 天 |

**总计 ~5.5 天**,之后进入 `TRANSFER_DESIGN.md` 的迁移测试阶段。

> ⚠️ **阶段 0e/0f 必须先于阶段 3**:如果 `gt_first` 被证明改变打分口径,需要先决定"回退"还是"重跑 v8 基线",否则后续所有 ablation 的基线都不可信。

---

## 11. 风险与诚实声明

| 风险 | 缓解 |
|---|---|
| 提取 agent 自身过拟合 | §5.4 三护栏(程序性约束 / validate / purity audit) |
| episode 具体性 vs 纯度的张力 | 实体抽象成角色;但度难把握,是核心风险,需人工复核 |
| validate 要重跑 case = API 成本 | 只在 InsightBench(100 case)离线跑一次,可接受 |
| skill 同义重复 | audit 阶段去重 |
| bridge 序列化改造影响 InsightBench 兼容 | 冻结前做完 + InsightBench 回归测试 |
| 档 3 在线累积破坏可归因性 | 不进主臂,作独立实验臂 B |

### 诚实声明

**这套设计不消灭过拟合。** 它把过拟合的来源:

> 从 **"不可见的 prompt 手调"** → 变成 **"可审计、带 provenance、带 on/off 验证的声明式 skill"**。

**可审计性本身就是进步** —— 审稿人能看到每个 skill 从哪来、验证过没有、纯不纯。这是我们能给出的最诚实的定位,不宜over-claim 成"解决了过拟合"。

---

## 12. 已确认决策(2026-07-30 拍板)

| # | 决策 | 结论 |
|---|---|---|
| 5 | skill 包运行时消费 | **✅ 全注入**。包小(~4-6 条),`trigger+action` 直接拼进 prompt,取代硬编码 `_ANALYTICAL_GUIDANCE`。不做 RAG 检索——避免引入"检索什么/多少"的新自由度。 |
| 6 | 提取 agent 的 validate | **✅ 真跑 mini-eval**。候选 skill 必须在 **≥2 个不相关 case** 上 on/off 都涨分才入库。只在 InsightBench 离线跑一次,成本可控。 |
| 7 | episode bank(情景记忆) | **✅ 做**(档 1),每条**人工复核纯度**。已知风险:具体性 vs 纯度的张力(§7.4),抽象过头会退化成规则——靠人工把关。 |
| 8 | bridge 表序列化改造 | **✅ 吸收**。按列类型序列化(数值 min/max、分类 unique、时间最早/最晚、其他随机 10 样本)。**必须冻结前做完 + InsightBench 回归测试确认不掉分。** |
| 9 | 在线累积实验臂 B | **✅ 做**,作**独立实验臂**,不混进主臂。A vs B 差值直接检验导师的"经验回放有收益"假设。 |

**配套决策**(来自 `TRANSFER_DESIGN.md` §13):judge 全线统一 deepseek G-Eval;字典阅读直接用 deepseek-v4-flash 自身多模态能力,不引入任何额外 API key。

---

## 附录 A:与 TRANSFER_DESIGN.md 的分工

| | 本文档 | TRANSFER_DESIGN.md |
|---|---|---|
| 回答 | **被迁移的是什么** | **怎么迁、怎么评** |
| 内容 | skill 定义 / 纯度审计 / 包结构 / 提取 agent / 冻结 / 经验回放 | benchmark 脉络 / 靶子选择 / 评分 / loader / 预期结果 |
| 时序 | **前置**(先提炼冻结) | 后续(再迁移测试) |

## 附录 A2:MyDataStorm 的来历(影响 related work 与借鉴判断)

**`MyDataStorm` 是对 DataSTORM 论文(arXiv 2604.06474)的复刻**,加上为跑通所做的特定定制 + 我们自己的后续优化(v5→v8)。

**因此**:
- **related work 里 DataSTORM 是我们的基座,不是"可借鉴的他人工作"。** 它的 planner-executor 分解、thesis-driven、多阶段报告等设计我们已经有了。
- 上一轮调研列出的"可借鉴点"需按此重新分类:
  - **已有(来自基座)**:planner-executor 分解、thesis-driven 叙事、多阶段编辑管线
  - **基座有但我们定制时未必保留 / 需核实**:查询一致性检查(跨并行分支对齐口径)、归纳式统计嵌入(把分位数/频次自动嵌入答案)—— **这两条值得回头查我们的复刻里有没有**,若没有,补回来属于"回到基座能力",不算引入新东西
  - **真正的外部借鉴**:DataSage 的多角色提问 / 多路 CoT、AgentAda 的 skill 库、DataGovBench 的 DAG 与表序列化、DataSeer 的双层记忆
- **我们的贡献定位**:不是"提出 DataSTORM",而是"在 DataSTORM 基座上提炼出可迁移的 skill 包,并验证其跨 benchmark 泛化"。

## 附录 B:引用出处(已核实)

**经验回放 / 记忆机制**:
- FinMem — arXiv 2311.13743,AAAI Spring Symposium 2024 [核实,机制明确]
- FinCon — arXiv 2407.06567,NeurIPS 2024 [核实,机制明确]
- FinAgent — arXiv 2402.18485,KDD 2024 [核实,摘要级]
- Reflexion — arXiv 2303.11366,NeurIPS 2023 [核实]
- ExpeL — arXiv 2308.10144,AAAI-24 [核实]
- Generative Agents — arXiv 2304.03442,UIST 2023 [核实]
- ⚠️ TradingAgents(2412.20138)—— **正文无决策→结果记忆机制,勿用于支撑此说法**

**领域内佐证**:
- DataSeer — ACL 2026 Findings(`2026.findings-acl.787`),双层记忆系统,InsightEval 同组后续
- DataGovBench 消融 — arXiv 2607.06482 [核实]:序列化 +50 分、schema 无用、多行有害
