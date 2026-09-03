# 交接文档 —— Insight Agent / Skill 提炼项目

> **写作日期**：2026-09-03
> **写作动机**：当前开发机即将弃用，工作需在新机器上延续。本文档目标是让一个
> 完全没有上下文的人（或 agent）读完后能接手，不依赖任何本地未提交的状态。
> **文档定位**：事实与分析分离。凡是"实测"的都给出数据来源文件路径；凡是
> 推测的都标注为假设。**已被证伪的结论也保留**（见 §9），因为不知道哪些路
> 走不通的人会重复踩坑。

---

## 0. 三十秒速览

| 项 | 状态 |
|---|---|
| **总目标** | 把多智能体数据探索 pipeline 当作一个"skill"，在 InsightBench 上训练/精炼，再迁移到更新的 benchmark 上测试 |
| **交付物** | (a) 能正常工作的 insight agent；(b) 一个**真正的 skill**（可执行脚本 + 说明），别的 agent 系统（Claude Code / Codex）拿去也能用 |
| **当前进度** | agent 主体已从手写 ReAct 迁移到 pi harness 并功能对齐；skill 包仍是**未验证的候选**；迁移测试**尚未开始** |
| **当前分数** | InsightBench 5 case（`pi_v4` 干净基线）：recall **0.892 (raw)** / 0.837 (bank)，F1 0.721 (bank)。v8 基线 0.862 (raw) → **已追平**（+0.029，但小于判分噪声，不宜声称超越） |
| **最大的未完成项** | Skill Extraction Agent（整个 skill 这条主线还没真正开始） |
| **最需要注意的坑** | 判分噪声（同数据同尺子重跑极差最大 0.117）；与 v8 对比**必须用 raw 口径**（v8 无 bank）；recall 涨的同时 precision 在跌（§5.2d） |

---

## 1. 项目目标与它的来历

### 1.1 导师给的方向

原始要求是：**把现有的多智能体数据探索 pipeline 视为一个 "skill"，先在
InsightBench 上训练/精炼它，然后在一个更新的 benchmark 上做迁移测试。**

这句话里有三个独立的承诺，容易被混为一谈，接手时请分开看：

1. **agent 要能干活** —— 在 InsightBench 上分数不能太差。
2. **要产出一个 skill** —— 而且是现代 agent 系统意义上的 skill，不是一段
   prompt 文本。
3. **要证明可迁移** —— 在一个 InsightBench 之外的 benchmark 上验证。

目前状态：**(1) 基本达成，(2) 只有候选、未验证，(3) 未开始。**

### 1.2 "skill" 的定义经历过一次纠正 —— 这点很重要

早期我（AI）把 skill 实现成"注入 system prompt 的一段文本"。用户明确否决：

> 我们现在的 skill 就是单纯注入 agent system prompt 的文本？那我理解这个就
> 本质是在 prompt optimization 了，按这样做我觉得希望会很渺茫。
> 我觉得是不是把 skill 定位为**能力**，或者说定位为**脚本 + 说明 prompt**，
> 也就是说，这 skill 本身要真的是现在常见 agent 系统中的 skill 而不是单纯的
> prompt，这样才好吧？

**这是本项目的定性约束**：skill = 可执行代码 + 描述，对标 `SKILL.md` 标准
（agentskills.io，vendor-neutral，Claude Code / Codex / Cursor / Gemini CLI /
Copilot 都已采纳）。三级渐进披露：**脚本源码永不进入 context，只有 stdout
进入**。

还有一条同等重要的约束：

> 注意我们的核心是，running on a bench will accumulate information to form a
> skill，因此形成 skill 不是应该从我们 running on benchmark 的过程中由 agent
> 系统去形成吗

**即 skill 必须由 agent 从跑 benchmark 的过程中"挖"出来，不能手写。**
我曾违反过这条（见 §9.2），代价是把 5 条手写 skill 全部降级为候选。

---

## 2. 代码地图 —— 东西都在哪

工作区有两个 git repo，**都在 `pi-migration` 分支**，都已推到 GitHub。

### 2.1 `IndustryAnalysisGenerationLLM`（主 repo）

`git@github.com:tobornotxx/IndustryAnalysisGenerationLLM.git`

```
pi-agent/                       ← 【当前主力】pi harness 上的新实现，共 1983 行
  package.json                  pi 0.84.4 精确锁版本（见 §2.4）
  PI_API_NOTES.md               pi API 的 4 个踩坑记录，改代码前必读
  src/
    agent.mjs        (530)      主 agent：explore() 编排 + UsageTracker + 工具封装
    modules.mjs      (373)      非 agentic 模块：insight_bank / thesis / 自评 / summary
    planner.mjs      (307)      问题生成，prompt 逐字移植自 templates.py
    py-worker.mjs    (217)      Python worker 池（常驻进程 + 借还队列）
    skills.mjs        (75)      读 MyDataStorm 的 skill 包（不复制，见 §2.3）
  python/worker.py   (161)      常驻 Python worker，行分隔 JSON 协议
  run_insightbench.mjs (176)    单 flag 跑分
  run_batch.mjs      (144)      批量 + 断点续跑 + 均值±标准差
  probes/                       API 探针（当初验证 pi 用法用的，保留作参考）

run_on_benchmark/               ← 评分器 + 设计文档 + 旧 Python 实现
  unified_scorer.py  (464)      【重要】判分器，唯一的尺子
  datastorm_adapter/
    adapter.py       (677)      旧 Python pipeline 的 benchmark 适配层
    csv_db_bridge.py (658)      CSV→SQLite + schema 生成（pi 版直接复用它）
    run_benchmark.py (495)
  PI_MIGRATION_PLAN.md    (403) 迁移计划，§2 性能要求 §8 五项决策
  TRANSFER_DESIGN.md      (495) 迁移测试设计，§8.5 成本账 §13 四项决策
  SKILL_PACKAGE_DESIGN.md (490) skill 包结构 / 提取 agent / 冻结 / episode bank
  SKILL_EXTRACTION_PLAN.md(226) 提取流程 + 5 道审计闸
  insight-bench/                ← gitlink（submodule），上游 benchmark

handover/                       ← 【本次新增】交接文档 + 实验产物
  HANDOVER.md                   本文件
  experiment_logs/              见 §2.5

results/                        ← gitignored，263M，只有精简版进 handover/
```

### 2.2 `MyDataStorm`（旧 Python 实现 + skill 包）

`git@github.com:tobornotxx/MyDataStorm.git`

```
datastorm/
  pipeline.py                   旧主流程
  agents/{planner,executor}.py  旧 ReAct executor（已被 pi 取代）
  modules/
    exploration.py              分层探索框架
    insight_bank.py             候选发现去重择优
    thesis.py                   thesis 生成/精炼
    goal_sufficiency.py         覆盖度自评（早停判断）
    consistency.py              自一致性
    statistics.py / report.py / warmstart.py / question_logger.py
  prompts/templates.py          【重要】所有 prompt 的源头，pi 版逐字移植自此
  skills/
    skill_package_v9.json       ← skill 包（5 条候选，均未验证）
    __init__.py                 Python 侧 loader
```

**这个 repo 的定位**：它是 DataStorm 论文的复刻 + 用户为跑通做的定制 + 后续
优化。pi 版是它的**替代**而非扩展，但 **prompt 和 skill 包仍以它为单一来源**。

### 2.3 一个刻意的设计：skill 包不复制

`pi-agent/src/skills.mjs` **直接读 `MyDataStorm/datastorm/skills/skill_package_v9.json`**，
没有做 TypeScript 副本。理由有两条：

1. **防漂移** —— 两份拷贝一定会不一致，而 skill 内容是实验变量，不一致就等于
   实验记录作废。
2. **保留溯源** —— skill 的 provenance（哪条从哪个 case 挖出来的）要能追。

同理，`pi-agent/python/worker.py` **直接 import `CsvDatabaseBridge`**，没有重
新实现 CSV→SQLite 与 schema 生成。

**接手注意**：这意味着两个 repo 默认**并列 clone 在同一父目录下**，路径关系是
`<parent>/IndustryAnalysisGenerationLLM/` 与 `<parent>/MyDataStorm/`。
路径已改为从文件位置推导（不再硬编码），偏离此布局时用
`SKILL_PACKAGE_PATH` 覆盖。

### 2.4 依赖与环境

```
Node        v26.0.0            （package.json 要求 ≥22.19）
Python      3.11.15            （venv 在 <parent>/.venv/）
pi          0.84.4             精确锁版本，非 ^ 范围
  @earendil-works/pi-agent-core  0.84.4
  @earendil-works/pi-ai          0.84.4
  @earendil-works/pi-telemetry   0.84.4  （传递依赖）
typebox     0.34.52
```

**为什么精确锁**：pi 版本迭代快，而我们只用它 8504 行里的 `Agent`
（`agent.js` + `agent-loop.js` ≈ 971 行）。小版本升级动 agent loop 会直接改变
实验结果，而我们的判分噪声已经很大（§5.3），无法区分"框架变了"和"我们改的"。

**密钥**：`pi-agent/.env` 与 `llm_config.json` 含真实 API key，两者都
gitignored 且确认未被 track。新机器需自行创建。`.env` 格式：
```
DEEPSEEK_API_KEY=sk-...
```

### 2.5 实验产物归档（`handover/experiment_logs/`）

`results/` 有 263M，全量入库不合适。归档策略：

```
experiment_logs/
  pi/                           ← pi 版全量结果（4.5M，含完整 nodes + score_matrix）
    pi_v1/  flag-1 单 case（首次跑通）
    pi_v2/  5 case，**混合态**（flag-1/2/3 旧 prompt + id bug 未修）—— 仅作 §5.4 对照
    pi_v3/  flag-4/5，prompt 修复后（§5.4 实验的处理组）
    pi_v4/  5 case ← 【**当前基线，唯一可对外汇报的**】含 batch_console.log
  history/                      ← 旧 Python 版只留 summary（每个 case 的完整树太大）
    test_run_v8/         ← 【历史最佳基线】10 case
    test_run_v10_baseline/  ← 只有 2 case 有效，其余 402 余额不足
    test_run_v9_L5_flag4/
```

**`pi_*/flag-N_result.json` 是最有价值的文件**，它包含：
- `nodes` —— 完整探索树（每个问题、答案、turn 数、调了哪些工具）
- `score_matrix` —— GT × pred 的两两打分矩阵，**可离线重算 recall/precision/F1
  而不花任何 API 钱**，也可做 per-GT 的 MISS 归因（§5.5 就是这么做的）
- `insight_bank` / `thesis` / `pred_summary` / `agent_usage`

**历史版本 v2–v7 的完整树没有归档**（合计 200M+）。如果需要，它们已永久丢失，
但 §5.1 的分数趋势表保留了结论。

---

## 3. 架构：改造前 vs 改造后

### 3.1 旧架构（MyDataStorm，Python）

```
分层探索循环（exploration.py）
  ├─ Planner            → 生成本层问题（LLM，1 次）
  ├─ ExecutorAgent      → 【手写 ReAct 循环】
  │     "Thought: ... Action: run_sql ... " → 正则 parse → 执行 → 拼回 prompt
  │     错误处理：遇到一个 case 打一个补丁（sandbox __import__、matplotlib
  │     backend、scipy 缺失…… 见 git log 的 e26532d / 14aede4 / e7e1f92）
  ├─ InsightBank        → 候选发现去重择优（LLM，1 次）
  ├─ GoalSufficiency    → 覆盖度自评，够了就早停（LLM，1 次）
  └─ Thesis             → 生成/精炼（LLM，1 次）
最后 Summary（自一致性，3 份草稿 + 合并 = 4 次）
```

### 3.2 新架构（pi-agent，TypeScript + Python worker）

```
分层探索循环（agent.mjs::explore）
  ├─ Planner (planner.mjs)      → prompt 逐字移植，行为等价
  ├─ 每个问题 → pi 的 Agent      → 【harness 驱动的 loop】
  │     结构化 tool_call（TypeBox schema → JSON Schema → API tools=）
  │     错误恢复由 harness 通用覆盖，不用逐 case 打补丁
  │     并发执行（Promise.all）→ Python worker 池
  ├─ InsightBank / Sufficiency / Thesis  → modules.mjs，prompt 逐字移植
  └─ Summary（自一致性）
```

**关键变化只有一处**：手写 ReAct → pi 的 agent loop。其余全是逐字移植。
这是刻意的 —— 一次只动一个变量，否则分数变化无法归因。

### 3.3 "引入 pi 是不是作弊？" —— 用户问过，这里给结论

用户的原话：

> 我拿来 piagent，是不是等于作弊引入了强 agent 框架，还是单纯引入了一些 agent
> 脚手架，本质还是我自己的提问-探索框架在起主要作用。

**答案：是脚手架，不是颠覆。** 依据：

1. pi 全库 8504 行，我们只用 `Agent`（≈971 行）。用的是 loop 调度、工具分发、
   重试，**没有用它的任何 planning / memory / 多 agent 编排**。
2. **分层探索、问题树、thesis 演进、insight bank、覆盖度自评、早停 —— 全部是
   我们自己的框架**，pi 对这些一无所知。pi 只负责"回答单个问题"这一层。
3. 所有 prompt 都是我们的（逐字移植自 `templates.py`）。

**但要诚实说明它带来的实质增益**：结构化 tool call 替掉正则 parse，以及通用
错误恢复。旧版 executor 的错误处理是逐 case 补丁堆出来的；pi 版在一次 API 用法
错误（`execute` 首参是 id 不是 params，见 §9.3 第 13 条）下模型自己重试了 5 次
并绕过去了。这个能力是真的，且是我们没写的。

### 3.4 结构化 tool call 与正则 parse 的区别（用户要求科普过）

```
TypeBox schema  →  JSON Schema  →  API 请求的 tools= 字段  →  服务端约束采样
                                                          →  结构化 tool_calls
```

要点：
- TypeBox 在 TypeScript 里的角色**等价于 Python 的 pydantic** —— 定义 schema
  并生成 JSON Schema。理解成"校验参数合法"是对的，但它的作用不止校验：生成的
  JSON Schema 会送到 API 的 `tools=` 字段。
- **服务端约束采样**：模型在生成时就被限制只能产出符合 schema 的结构，而不是
  生成完自由文本再由我们正则去抠。这是它比正则 parse 稳的根本原因。
- 注意这**不是** `response_format` / `response_schema`（那是限制整个响应体的
  格式），是 tool calling 通路。

---

## 4. 判分器 —— 唯一的尺子，必须先搞懂

`run_on_benchmark/unified_scorer.py`（464 行）。**所有分数都出自这里，理解它
比理解 agent 更重要**，因为它决定了什么叫"变好"。

### 4.1 算法

```
S(pred_i, gt_j)  ←  LLM-as-judge（deepseek-v4-flash，logprobs 加权）

recall     = E_gt  [ max_pred S ]     每条 GT 找最匹配的 pred，取平均
precision  = E_pred[ max_gt  S ]      每条 pred 找最匹配的 GT，取平均
F1         = 2PR/(P+R)
```

**关键工程点：一个矩阵出三个指标。** `score_insight_matrix()` 算一次
GT × pred 的两两矩阵，recall / precision / F1 全部从这个矩阵派生，
**precision 和 F1 是零额外 API 调用**。矩阵存在 result.json 的 `score_matrix`
里，所以事后重算/归因完全免费。

### 4.2 已做的优化（都已验证）

| 优化 | 效果 | 备注 |
|---|---|---|
| `ThreadPoolExecutor` 并行 | **13.2×**（mock 600 次调用） | `SCORER_MAX_WORKERS=16`；加了 `_USAGE_LOCK` / `_DETECT_LOCK` |
| prompt 顺序 `pred_first` | 缓存命中 7.6% → 21.6% → ~100% | 环境变量 `GEVAL_PROMPT_ORDER`，三种顺序实测过 |
| precision/F1 复用矩阵 | 省一半调用 | 见上 |

并行这件事有个教训：早期串行判分（248 次调用一个个来）导致长时间挂机烧钱，
用户原话 *"傻逼别花老子钱了。算不出来就别算了。"* —— **判分务必确认并行开着**。

### 4.3 ⚠️ 口径陷阱：v8 没有 bank 模式

pi 版对 pred 有两种口径：

| 口径 | 内容 | n_pred 量级 |
|---|---|---|
| `raw` | 每个探索节点的「问题 + 答案」 | 34–40 |
| `bank` | 只用 insight_bank 去重择优后的条目 | 11–13 |

**precision 的分母是 pred 条数，所以两者 precision 差异巨大**（pi_v4 flag-1：
raw 0.441 vs bank 0.570；pi_v2 flag-1 更极端：0.515 vs 0.867）。

**而 v8 的历史结果只有 raw 口径**（`test_run_v8/summary.json` 连
`insights_precision` 字段都没有）。所以拿 v8 做对照时**必须用 raw**：

- ✅ 正确比较：pi_v4 raw **0.8917** vs v8 raw 0.8624 → **+0.029（追平）**
- ❌ 错误比较：pi_v4 bank 0.8367 vs v8 raw 0.8624 → −0.026（口径错配，无意义）

**我在对话中曾用错配口径报出"pi 比 v8 差 0.112"**（当时还叠加了混合态数据的
问题）。两个错误都已纠正，写进文档以免再犯。**任何与 v8 的对比一律用 raw。**

---

## 5. 实验结果与逐段分析

> 所有数字均可从 `handover/experiment_logs/` 离线复算，无需 API。

### 5.1 历史脉络（旧 Python 版，v2 → v8）

| 版本 | recall | n_pred | 关键改动 |
|---|---|---|---|
| v2 | 0.56 | 6.8 | 起点 |
| … | … | … | sandbox 修复、planner 可配、探索树打分 |
| v8 | **0.8624** | 14–32 | 历史最佳，10 case 全量 |

**分析 —— 这段最重要的教训是"pred 变多不是问题"**：pred 从 6.8 涨到 24，
recall 同步从 0.56 涨到 0.88。我曾把 pred 增长当成问题报告，用户直接反驳
*"是不是知识你的妄想"*，数据证明用户对：**在纯 recall 尺子下，更多 pred 是在
帮忙**。后来引入 precision 才让"pred 太多"第一次有了代价（v2 的 precision
只有 0.5676）。

**这条对接手人的意义**：如果你只看 recall，你会一路把 pred 做多；只有
recall/precision/F1 一起看才有意义。而 F1 从 v8 开始才有数据。

### 5.2 【当前基线】干净的 5 case（`experiment_logs/pi/pi_v4/`，layers=5）

**这是唯一可对外汇报的基线** —— 全部 5 个 case 用同一份代码（HEAD 含
insight_bank id 修复 + 负结果/解读引导）一次跑完，无混合状态。
2026-09-03 跑，$0.391，42.8 分钟。

| flag | raw R | raw P | raw F1 | bank R | bank P | bank F1 | v8 R (raw) | 成本 |
|---|---|---|---|---|---|---|---|---|
| flag-1 | 0.9667 | 0.4407 | 0.6054 | 0.9667 | 0.5703 | 0.7174 | 0.9667 | $0.082 |
| flag-2 | 0.9750 | 0.3584 | 0.5241 | 0.8500 | 0.6062 | 0.7077 | 1.0000 | $0.094 |
| flag-3 | 0.9667 | 0.7137 | 0.8211 | 0.9667 | 0.7201 | 0.8253 | 1.0000 | $0.055 |
| flag-4 | 0.7753 | 0.4976 | 0.6062 | 0.6748 | 0.7063 | 0.6902 | 0.7250 | $0.085 |
| flag-5 | 0.7750 | 0.3649 | 0.4962 | 0.7252 | 0.6088 | 0.6619 | 0.6202 | $0.075 |
| **mean** | **0.8917** | 0.4751 | 0.6106 | **0.8367** | 0.6423 | 0.7205 | **0.8624** | $0.391 |

⚠️ **flag-3 的 bank n=30 等于 raw** —— 第 4 层择优瞬时失败走了兜底，该行的
bank 数字实际是 raw 口径，不要当作 bank 有效样本。见 (c)。

**逐段分析：**

**(a) 同口径 recall 反超 v8：0.8917 vs 0.8624（+0.029），3/5 个 case 胜。**
这推翻了此前"pi 版比 v8 差"的结论 —— 那个结论建立在混合态数据上。
但 **+0.029 小于判分噪声均值极差 0.037（§5.3）**，所以诚实的说法是
**"pi 版已追平 v8，尚不能声称超越"**。方差还从 0.159 收窄到 0.095，
这个改善比均值本身更有意义（结果更稳）。

**(b) 干净基线相对混合态 pi_v2 普涨，且涨幅集中在此前最差的 case。**
bank recall：flag-5 +0.175、flag-1 +0.117、flag-2 +0.075、flag-4 +0.067、
flag-3 ±0。**flag-5 从 0.5500 涨到 0.7252，一举超过 v8 的 0.6202。**
两个来源：insight_bank id 匹配修复（§9.3a，此前静默丢弃整批择优结果，
flag-5 是受害最重的 case）+ 负结果/解读引导（§5.4）。

**(c) flag-3 暴露出兜底逻辑本身的缺陷（已修，见下）。** 日志：
```
· layer 3 done: insight bank 12/20
· insight filter yielded nothing; falling back to all 30 findings
· layer 4 done: insight bank 30/30
· goal-sufficiency: findings answer the goal — stopping early at layer 4
```
第 4 层择优瞬时失败 → 兜底**把第 3 层已选好的 12 条丢掉**，换成全部 30 个
原始节点 → 自评看到一大堆发现后当层就误判"够了"、提前停在第 4 层。
上一层的 bank 显然是比"全部节点"好得多的退路。**已修**：改为优先保留
上一层的 bank，只有首层就失败才退回全部节点（4 种情形已单测覆盖）。

**(d) ⚠️ recall 涨但 precision 跌 —— 这是本次最值得警惕的信号。**
bank precision 0.6615 → 0.6423，而且**分裂得很整齐**：
- flag-1 −0.296、flag-2 −0.225（**两个 recall 涨最多的**）
- flag-3 +0.202、flag-4 +0.088、flag-5 +0.135

同时 bank 条数普遍变多（12→13、13→16、11→16）。**合理解读**：修复后
insight_bank 真正开始工作，留下了更多条目，捞回了 GT（recall ↑）但也放进了
更多没对上 GT 的内容（precision ↓）。F1 净增 +0.029，所以整体是赚的；
但**这正是 §5.1 那个"pred 变多推高 recall"的老现象换了个位置重演**，
需要盯住。建议把 `maxInsights`（当前 12）做一次扫描。

**(e) 成本 $0.391，缓存命中 76–85%**，与 pi_v2 基本一致。成本不是瓶颈。

**(f) summary 分数依然不可用**：0.72 ± 0.271，flag-2 只有 0.20 而它的 insight
recall 是 0.85。**继续不建议用 summary 分数做任何决策。**

<details>
<summary>历史参考：混合态 pi_v2（flag-1/2/3 旧 prompt + bug 未修）</summary>

| flag | raw R | raw P | raw F1 | bank R | bank P | bank F1 |
|---|---|---|---|---|---|---|
| flag-1 | 0.9167 | 0.5147 | 0.6593 | 0.8500 | 0.8667 | 0.8583 |
| flag-2 | 0.7750 | 0.5616 | 0.6513 | 0.7750 | 0.8308 | 0.8019 |
| flag-3 | 1.0000 | 0.6800 | 0.8095 | 0.9667 | 0.5182 | 0.6747 |
| flag-4 | 0.7500 | 0.5100 | 0.6071 | 0.6082 | 0.6182 | 0.6131 |
| flag-5 | 0.5750 | 0.3458 | 0.4318 | 0.5500 | 0.4737 | 0.5090 |
| **mean** | 0.8033 | 0.5224 | 0.6318 | 0.7500 | 0.6615 | 0.6914 |

留档理由：§5.4 的 prompt 实验以它为对照；且它是"混合态数据会得出错误结论"
的实例 —— 基于它我曾报告"pi 比 v8 差 0.059"，干净基线证明是**追平**。
</details>

### 5.3 ⚠️ 判分噪声 —— 这是本项目最重要的测量学发现

**同一份数据、同一个判分器、重复跑**：

| flag | 三次 recall | 极差 |
|---|---|---|
| flag-1 | 0.967 / 0.850 / 0.950 | **0.117** |
| flag-4 | 0.608 / 0.562 | 0.046 |
| flag-5 | 0.550 / 0.575 | 0.025 |

**均值极差 0.037，最大 0.117 —— 与 pi-vs-v8 的差距同量级。**

**这意味着什么（请接手人务必内化）：**
- 任何 **< 0.05 的单 case 提升都不可信**，除非重复测量。
- 5 case 的**均值**加上**方向一致性**（5/5 同向）才勉强可读。
- 想要可靠结论，要么加 case 数（InsightBench 有 10 个 flag，目前只用了 5），
  要么重复测量取均值。**这是最该优先解决的方法论问题。**

我曾一度把判分不稳归因于"reasoning 随机性"，但实测同输入 3/3 次零方差 —— 不稳
是**输入特定的**（某些 GT-pred 对本身在判分边界上），不是系统性的。

### 5.4 prompt 修复实验（`pi_v3`，flag-4/5）

改动内容（`agent.mjs::ANSWER_SYSTEM_PROMPT` + `modules.mjs::filterInsights`）：
1. 显式要求陈述**负结果/空结果**（"A 与 B 无相关"、"分布均匀"、"指标稳定"
   本身就是发现）
2. 数据支持时给出**简短解读**（含义 + 指向的行动）
3. insight_bank 相应加 "Do NOT de-prioritize" 段 —— 否则第 1 条产出的负结果会
   在择优环节被当作 trivial 丢掉

| flag | 口径 | ΔR | ΔP | ΔF1 |
|---|---|---|---|---|
| flag-4 | raw | +0.038 | **+0.108** | +0.085 |
| flag-4 | bank | **+0.129** | +0.094 | +0.111 |
| flag-5 | raw | −0.025 | **+0.115** | +0.070 |
| flag-5 | bank | +0.050 | +0.112 | +0.084 |

**分析：这是目前唯一一个我认为可信的正向改动。** 理由：

1. flag-4 bank 的 +0.129 **远超该 case 的噪声（0.046）**。
2. **precision 在 4/4 个格子里全部上升（+0.094 至 +0.115）**，且幅度一致。
   这条最关键 —— 如果只是"多写了点东西碰运气"，precision 会掉。precision 涨说明
   新增的内容**本身是对的**。
3. flag-5 raw 的 recall 微跌 −0.025 在噪声内（该 case 噪声 0.025），不构成反证。

**⚠️ 曾被我当成"具体成功案例"的一条，后来自我推翻了**：flag-4 GT7（条件推断
类）在 v3 里 0.10 → 0.70，我据此说"这类 GT 加了解读引导就命中了"。但 `pi_v4`
用**同一份 prompt** 重跑，它掉回 **0.20**（见 §5.5）。**单条 GT 的 delta 是
不可引用的证据。** 上面那张 4×3 的表（尤其 precision 4/4 上升）才是这个改动
成立的依据。

**⚠️ 这个实验本身只在 flag-4/5 上做过**（v2 → v3 对照）。全 5 case 的干净
基线是 `pi_v4`，见 §5.2。

### 5.5 剩余 MISS 的逐条归因（零成本，用 `score_matrix` 离线做的）

干净基线 `pi_v4` 共 25 条 GT，得分分布：

```
0.8–1.0 : 19 条   ← 绝大多数已命中
0.6–0.8 :  1 条
0.4–0.6 :  2 条
0.2–0.4 :  1 条
0.0–0.2 :  2 条   ← 硬骨头
```

⚠️ **分布与混合态几乎一致，但成分变了**。跨版本逐 GT 对比（阈值 0.15）：

| GT | v2/v3 | v4 | Δ | |
|---|---|---|---|---|
| flag-5 GT4 | 0.30 | 0.80 | **+0.50** | 改善 |
| flag-2 GT2 | 0.10 | 0.40 | **+0.30** | 改善 |
| flag-4 GT7 | 0.70 | **0.20** | **−0.50** | **退步** |

**flag-4 GT7 的退步值得警惕。** 它正是 §5.4 我举为"prompt 修复成功案例"的
那一条（条件推断类，0.10→0.70）。**同一份 prompt、同一个 case，重跑一次就掉回
0.20。** 这不是代码变了 —— 是**运行间方差**（agent 每次探索路径不同）叠加判分
噪声。

**这条对接手人的意义**：§5.4 那个"+0.129"的结论仍然成立（precision 4/4 上升，
且 flag-4 整体 bank recall 在 v4 里是 0.6748 > v2 的 0.6082），**但不要引用
单条 GT 的改善作为证据** —— 单条 GT 是最不稳的粒度。

**5 条低分 GT 逐条分析：**

| flag | GT | 分 | 归因 |
|---|---|---|---|
| flag-5 GT1 | "Incident distribution across categories is more or less uniform" | 0.10 | **A: goal 不对齐** |
| flag-4 GT6 | "1. **Regular Updates and Maintenance**: Establish a routine…" | 0.10 | **B: 非实证型 GT** |
| flag-4 GT7 | "If the number of Hardware incidents over time is linearly increasing, it suggests…" | 0.20 | 条件推断型，**方差极大**（0.10↔0.70↔0.20） |
| flag-4 GT1 | "The increase in volume of incidents is seen **slightly**, need to investigate further" | 0.40 | **A: goal 不对齐**（程度词） |
| flag-2 GT2 | "There is a negative correlation between volume of incidents and TTR" | 0.40 | **A**，但 v4 已改善 +0.30 |

#### 成因 A：GT 与 goal 不对齐，而 planner 被明确要求不许偏离

以 flag-5 为例：
- **goal**："identify factors that influence the **time to resolution**"
- **GT1**："incident **数量**分布大致均匀" ← 问的是**计数**，不是解决时间

实测统计（从 `nodes` 里数的）：
- flag-5：36 个问题中**只有 1 个**涉及数量分布，且那一个问的是
  `sys_updated_by` 的**解决时间**分布
- flag-2 GT2：39 个问题中 **7 个**匹配相关性模式 —— 相关性**确实探索了**，
  但那个"负相关"的结论没有被作为发现陈述出来
- flag-4 GT1：40 个问题中 **4 个**涉及总体量级

**根因定位到 planner prompt 里我逐字移植的这段**：

```
Common failure modes to AVOID:
- Do NOT explore dimensions (e.g., priority, business hours, day-of-week) just
  because they are "interesting" or available in the data, if they are not asked
  for by the goal.
- Do NOT wander into unrelated territory: if the goal is about volume trends,
  stay on volume trends; if it is about resolution time, stay on resolution time.
```

**最后那句几乎逐字命中 flag-5 的情形** —— goal 讲 resolution time，GT1 问
incident count，prompt 明确说"别去"。

**这不是 bug，是取舍。** 这段是 v2→v8 调优的产物，当时在**纯 recall 尺子下
有效**（防发散、集中火力）。但它会漏掉"goal 没问、GT 却有"的条目。

**⚠️ 一个必须承认的未知**：**v8 用的是同一段 prompt**，所以 v8 在这些 GT 上
也未必命中。但 v8 那批实验**没有落盘 `score_matrix`**，所以**无法确认这是 pi
独有的损失还是两边共有**。这是个测量空洞，接手人如果要深究这条，需要重跑 v8
并保存矩阵。

#### 成因 B：GT 是报告体的行动建议

flag-4 GT6 原文带编号加粗：`1. **Regular Updates and Maintenance**: ...`
这是**报告结尾的建议清单格式**。我们的 agent 在回答单个探索问题时，结构上
产不出这种东西 —— 它需要一条 report / 叙事链路。v3 的 prompt 改动对它无效
（0.10 → 0.10）。

**但这条 GT 是否该追，值得商榷** —— 它不是数据发现，追它更像是在拟合
InsightBench 的 GT 格式。

---

## 6. 明显需要做的（近期，1–2 周）

按"性价比 × 阻塞程度"排序。

### 6.1 ✅ 已完成：干净的 5 case 基线（`pi_v4`）

2026-09-03 跑完，$0.391 / 42.8 分钟。结论见 §5.2 —— **recall 追平 v8**
（raw 0.8917 vs 0.8624），此前"pi 比 v8 差"的结论来自混合态数据，已推翻。
同时暴露并修掉了兜底逻辑的一个缺陷（§5.2c）。

复跑命令（改 `--out` 换目录）：
```bash
cd pi-agent
set -a && . ./.env && set +a
# caffeinate -i 保证锁屏不中断
nohup caffeinate -i node run_batch.mjs --flags 1,2,3,4,5 --layers 5 \
      --out ../results/pi_v5 --force > /tmp/pi_v5.log 2>&1 &
```

### 6.1b 【最高优先】验证兜底修复 + 查 precision 下滑

两件事一起跑一次就能看：

1. **兜底修复的效果** —— flag-3 上次走了兜底（bank n=30 等于 raw），修复后
   应当保留上一层的 12 条左右，且不会在第 4 层误判早停。
2. **precision 下滑（§5.2d）** —— bank precision 0.6615 → 0.6423，且
   flag-1/2（recall 涨最多的两个）跌了 0.22–0.30。建议同时扫
   `maxInsights`（当前 12，试 8 / 12 / 16）看 F1 的拐点在哪。

### 6.2 【高】把 case 数从 5 扩到 10，并做重复测量

**为什么**：§5.3 的噪声分析说明 5 case 单次跑不足以判断 <0.05 的改动。
InsightBench 有 10 个 flag，v8 跑的就是 10 个。

**建议**：10 case × 2 次重复 = 20 次跑，约 $1.5。这是**买测量精度**，
比任何单点优化都值。

### 6.3 【高】判分器加 novelty 指标

InsightEval 的 novelty 指标是 3-LLM 投票 ≥2 通过。目前我们只有
recall/precision/F1。**为什么需要**：F1 无法区分"复述 GT"和"发现了 GT 之外
的真东西"，而后者才是 insight agent 的价值。

### 6.4 【中】决定成因 A 要不要处理 —— 这是个**需要用户判断**的定性问题

可选做法：在 planner 的 "stay aligned" 之外加一条，每层保留 1 个问题做基础
EDA（数量分布、整体量级），即便 goal 没直接问。

- **可能修好 3 条 MISS**（flag-5 GT1、flag-2 GT2、flag-4 GT1）
- ⚠️ **风险**：这段约束是调优出来的，放宽可能让探索发散、precision 下降
- ⚠️ **更重要的风险**：这**有点像照着 GT 调 prompt**

**必须由用户拍板，不能由 AI 单方面改。** 依据是用户此前的定性裁决：

> 就算真有收益，也不过是用一堆数字去作弊罢了

（当时的语境是移除统计块，见 §9.4）以及

> 我们最后要的核心是一个正常工作的 agent

**判断标准应该是**："让 agent 在探索时做基础 EDA"是一个**通用的分析能力**，
还是**专为 InsightBench 的 GT 分布服务的过拟合**？我倾向前者（任何数据分析
师都会先看分布），但这需要用户确认。

---

## 7. 中期规划（1–2 月）—— skill 这条主线

**这是整个项目的核心承诺，但目前基本没开始。** 现有 5 条 skill 全部是
`status: candidate`、`validated.method: "none"`。

### 7.1 Skill Extraction Agent（`SKILL_EXTRACTION_PLAN.md` 步骤 1–6）

流程：**mine → generalize → validate → audit → freeze**

```
mine       从 benchmark 运行记录（experiment_logs 里的 nodes/探索树）中挖出
           "哪些分析动作反复带来高分发现"
generalize 去掉数据特化：列名、领域术语一律替换为运行时注入的占位符
validate   ⚠️ 硬性要求：在 ≥2 个不相关 case 上 on/off 都涨分
audit      5 道闸（见 7.2）
freeze     冻结版本号，之后的实验以此为固定变量
```

**关键设计**：脚本源码不进 context，只有 stdout 进 —— 这是 `SKILL.md` 三级
渐进披露的要求，也是"skill 是能力不是 prompt"的技术体现。

### 7.2 五道审计闸（`SKILL_EXTRACTION_PLAN.md`）

1. **无数据特化** —— 不含任何具体列名/域名词
2. **无 scorer 特化** —— 不是为了迎合判分方式（见 §8 的两类过拟合）
3. **有 provenance** —— 能追到是从哪个 case 的哪段挖出来的
4. **validate 过关** —— ≥2 个不相关 case 上 on/off 都涨分
5. **脚本能 smoke-test** —— 可执行且能跑通（这条是后加的）

### 7.3 已验证但未采纳的两个想法

| 想法 | 来源 | 为什么值得做 |
|---|---|---|
| **DAG 结构** | DataGovBench 的 Insight Agent | 探索树目前是分层的，DAG 能表达跨层依赖 |
| **report / 叙事链路** | §5.5 成因 B | 建议类 GT 需要它；也是"agent 可用"的自然要求 |

### 7.4 Episode bank / 经验回放（`SKILL_PACKAGE_DESIGN.md` §7）

设想的实验设计有三档，目前只有档 2 的雏形：
- **档 1**：episode bank —— 把跑过的 case 存下来供检索
- **档 2**：离线提炼 skill 包（当前所处位置，但 skill 未验证）
- **档 3**：在线累积（arm B）—— 跑 benchmark 的过程中实时形成 skill

**档 3 才是最贴合导师原话的形态**（"running on a bench will accumulate
information to form a skill"）。

---

## 8. 远期规划 —— 迁移测试

### 8.1 benchmark 选择（已定，见 `TRANSFER_DESIGN.md` §13）

| 角色 | benchmark | 说明 |
|---|---|---|
| **训练集** | InsightBench（**ICLR 2025**） | 当前所用 |
| **主测试集** | **InsightEval** | 同一 insight 质量脉络的后继者 |
| **次测试集** | **DataGovBench** | 多表 + 外部 PDF/字典接入 |

调研中确认的完整后继谱系：InsightEval、MedInsightBench、DataGovBench、
D3-Gym、AgentAda。

**⚠️ InsightBench 是 ICLR 2025**（我一度误判为"arXiv 2024、非顶会"，被用户
纠正：*"insight bench 是 iclr 25 发布了的。比他更早的不必考虑了"*）。这个事实
直接否决了把 DiscoveryBench 当主目标的方案。

> 注：`memory/` 里那条 "transfer benchmark direction" 记录的是更早的
> DiscoveryBench/DSBench 方案，**已被本节取代**。

### 8.2 🚨 测试集泄漏禁令 —— 最严重的一条纪律

**InsightEval / DataGovBench 是最终测试集，它们的任何结论都不得回流到
pipeline / skill 代码里。**

这条是**血的教训**。我曾把 DataGovBench 的消融结论写进
`csv_db_bridge.py` 的序列化逻辑，用户的反应：

> 那是我们最后的测试集，你踏马把做那里的经验弄到我们现有的 skills 里去了？

**而且它当场就打坏了 flag-1**：`Printer546` 从 schema 样本里消失，
recall 0.9667 → 0.50。已回滚（commit `53032da`）。

`skill_package_v9.json` 的 meta 里专门记了这件事：
```json
"test_set_leakage_removed": "已回滚 csv_db_bridge 的按列类型序列化改造 ——
  其依据来自 DataGovBench(2607.06482) 消融结论，而该 benchmark 是最终测试集，
  其经验不得回灌进 skill 本体。"
```

### 8.3 两类过拟合 —— 审 skill 时的判据

| 类型 | 定义 | 例子 |
|---|---|---|
| **数据特化** | 硬编码列名/领域术语 | `_ANALYTICAL_GUIDANCE` 里的 `category, assigned_to, priority, assignment_group`；`goal_sufficiency` 里的 ITSM 术语 `TTR` |
| **scorer 特化** | 输出形态迎合判分方式 | 往答案里堆统计数字，扩大字符串匹配面（见 §9.4） |

**数据特化已清理**：全部改为运行时从真实 schema 注入 `{categorical_columns}`。
**意外收获**：`closed_by`（v8 的盲点）现在会自动进入分组轴。

---

## 9. 走过的弯路 —— 请勿重复

**这一节是本文档最值得读的部分。** 20 条错误里挑出对接手人有实际价值的。

### 9.1 被证伪的假设（别再假设它们成立）

| 假设 | 证伪数据 |
|---|---|
| "bank 口径一律优于 raw" | recall 上 bank **总是 ≤ raw**（择优会丢东西）；precision 上 bank 总是更高。所以"更好"取决于看哪个指标，且 `pi_v4` 里 bank F1 (0.721) > raw F1 (0.611) —— **但与 v8 对比只能用 raw**（§4.3） |
| "问题前缀会稀释结论" | 只差 0.015，**flag-5 反向**，噪声内 |
| "pred 越多越糟" | v2→v8 pred 6.8→24，**recall 同步 0.56→0.88** |
| "判分器因 reasoning 随机而不可靠" | 同输入 **3/3 次零方差**；不稳是输入特定的 |
| "reasoning token 是成本问题" | 只占 5–10%，且在便宜的 output 侧 |
| "`_temporal_profile` 会误判文本列为时间列" | 实测 `short_description` 日期解析率 **0%**，没误判 |

### 9.2 手写 skill —— 违反了自己写的设计文档

我手写了 5 条 skill 塞进包里，违反 `SKILL_PACKAGE_DESIGN.md` 自己定的
mine→generalize→validate→audit 流程。处理：删掉毫无依据的
`prefer-productive-followups`，剩下 5 条全部降级为 `candidate`
（commit `ab0c07d`）。

**接手人注意**：`skill_package_v9.json` 的 `produced_by` 字段诚实地写着
`"manual-transcription-from-v8 (NOT the extraction agent)"`。**不要把这 5 条
当成已验证的成果去汇报。**

### 9.3 静默失败的 bug —— 两个都不报错，最难查

**(a) `insight_bank` id 严格匹配（已修，commit `7a00c37`）**

`filterInsights` 用 `known.has(id)` 精确匹配 node_id。模型有时会包一层
`{"insights": {...}}` 或改写 id 写法（`Q_1_00` vs `q_1_00`），于是**整批结果
被丢弃，且不报任何错**。实测 flag-5 每层返回 0，后续 planner / thesis / 自评
全部失去输入。

修法：两级容错（剥外层包裹 + `normalizeId` 规范化匹配）+ 兜底（择优返回空时
退回全部节点）。已用 6 个 case 单测覆盖：plain / wrapped / renamed id / 对象值 /
编造 id 丢弃 / 空输入。

**这个 bug 让之前所有 bank 分数的可信度打折** —— 见 §6.1。

**(b) `insight_bank` 算了但没用（早期）**

计算了 bank 但 pred 仍然用全部 34 个节点。这解释了 v2 的 precision 为什么掉到
0.5676。修好后：34→12 条，recall −0.017，precision 0.57→0.89。

**(c) `run_batch.mjs` 的 n 标签（已修，commit `aa1b53c`）**

`agg()` 把缺分的 case 静默剔出均值，但表头照旧打 `rows.length`。于是 flag-1
补分前那次汇总把 **4 个 case 的均值报成 "n = 5 cases, recall 0.7250"**
（正确的 5-case 值是 0.7500）。已改为每个指标各自打 `n`，不符时显式告警。

### 9.4 移除统计块 —— 一个"看起来涨分实际作弊"的例子

旧实现往 LLM 消费点喂原始统计块（`min: 26 max: 46 mean: 38.46`）。它带来
**+0.037 recall**。但用户判定这是作弊：

> 就算真有收益，也不过是用一堆数字去作弊罢了

**机制**：那串数字在做**字符串层面的匹配**，去撞 GT 里的 "volume increased
slightly"。这是典型的 **scorer 特化**。已从全部 4 个消费点移除。

**接手人应内化的判断标准**：涨分之前先问"**这个改动让 agent 变强了，还是让它
更会考试了？**"

### 9.5 pi API 的 4 个误解（`PI_API_NOTES.md` 有完整版）

| 我以为 | 实际 |
|---|---|
| `createModels({providers:[...]})` | 空构造 + `models.setProvider(p)` |
| `await models.get(...)` | `models.getModel(provider, id)`，**同步** |
| `Agent` 有默认 stream | **必须传 `streamFn`** |
| `for await (agent.prompt())` | 返回 `Promise<void>`，事件走 `subscribe(cb)` |

**外加两个**：
- **`execute(toolCallId, params, ...)` 首参是 ID 不是 params。** 我第一版把
  arg 1 当 params，Python 收到字面量 `undefined` → `NameError`。
  （副产品：模型自己重试了 5 次并绕过去，反而证明了通用错误恢复能力。）
- **`agent.initialState.messages` 跑完仍为空**，助手消息只能从 `message_end`
  事件收集。第一版因此产出空答案。

### 9.6 `cacheRead` 与 `input` 是互斥的，不是包含

我曾假设 `cacheRead ⊆ input`，算出 cache_hit_rate = 4.9（>1）。实测：

```
第1轮  input=1213  cacheRead=0
第2轮  input=61    cacheRead=1152    ← 命中部分从 input 移到 cacheRead
恒等式 input + cacheRead = totalTokens - output   两轮都成立
```

**命中率分母必须是 `input + cacheRead`。** 计价上命中只要未命中的 1/50
（`cacheRead 0.0028` vs `input 0.14`）。

### 9.7 内存优化的第一版是无效的（归因错了）

想让多个 worker 共享 SQLite 文件省内存，结果 2116→2073MB（没变）且加载时间
翻倍。**真实原因**：`pd.read_csv` 的 DataFrame 占 192MB，而 SQLite 文件的 41MB
**在磁盘上，不在 RSS 里**。

修法：reuse 模式的 worker 只读 5000 行样本。21 万行实测：
- `shared`：1634MB / 15.2s
- `isolated`：2143MB / 9.2s（大表省 24%）

**⚠️ 附带的坑**：reuse 模式必须**从 SQLite 算基数**，否则 5000 行样本会把
21 万基数的列误报为低基数列，进而被错误地选作分组轴。

---

## 10. 新机器上从零跑起来

```bash
# 1. 两个 repo 必须并列 clone（skills.mjs / worker.py 靠相对路径互相引用）
mkdir -p ~/llf-study && cd ~/llf-study
git clone git@github.com:tobornotxx/IndustryAnalysisGenerationLLM.git
git clone git@github.com:tobornotxx/MyDataStorm.git
cd IndustryAnalysisGenerationLLM && git checkout pi-migration
cd ../MyDataStorm && git checkout pi-migration

# 2. benchmark submodule（本次交接才补上 .gitmodules，见下方注记）
cd ../IndustryAnalysisGenerationLLM
git submodule update --init run_on_benchmark/insight-bench

# 3. Python 环境（判分器 + worker 用）
cd ~/llf-study && python3.11 -m venv .venv
./.venv/bin/pip install pandas numpy scipy scikit-learn statsmodels \
    ruptures pingouin openai

# 4. Node 依赖
cd IndustryAnalysisGenerationLLM/pi-agent && npm ci

# 5. 密钥（两个文件都 gitignored，须手工创建）
echo 'DEEPSEEK_API_KEY=sk-...' > .env
# 以及 <repo>/llm_config.json（格式见 datastorm/llm_config.example.json）

# 6. 冒烟测试：单 case
set -a && . ./.env && set +a
node run_insightbench.mjs --flag 1 --layers 3 --out /tmp/smoke

# 7. 全量基线（§6.1）
node run_batch.mjs --flags 1,2,3,4,5 --layers 5 --out ../results/pi_v4 --force
```

**关于 submodule**：仓库里有两个 gitlink（`run_on_benchmark/insight-bench`
和 `run_on_benchmark/daco`），但**一直缺 `.gitmodules`**，导致
`git submodule` 整个命令族报
`fatal: no submodule mapping found`，新 clone 只会得到两个空目录。
本次交接补上了 `.gitmodules`（`daco` 是搁置的探索线，只有调研文档提到、
无代码依赖，但不注册会连带让 insight-bench 也取不下来）。

**路径**：`run_insightbench.mjs` 与 `skills.mjs` 原本硬编码
`/Users/liulife/...` 绝对路径（共 3 处），本次已改为从文件位置推导，
并在启动时检查 Python 与 benchmark 数据是否存在、给出覆盖提示。
偏离默认布局时用环境变量覆盖，**不要改代码**。

### 环境变量一览

| 变量 | 默认 | 作用 |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | 必填 |
| `PRED_MODE` | `bank` | `raw` / `bank`，见 §4.3 |
| `USE_SKILLS` | 开 | `0` 关闭 skill 注入（做 on/off 消融用） |
| `SCORER_MAX_WORKERS` | 16 | 判分并行度，**别设 1** |
| `GEVAL_PROMPT_ORDER` | `pred_first` | 缓存友好顺序，别改 |
| `SKILL_PACKAGE_PATH` | 推导 | skill 包位置（MyDataStorm 不在同级时用） |
| `REPO_ROOT` | 推导 | 主 repo 根目录 |
| `BENCH_DIR` | 推导 | InsightBench 目录 |
| `PYTHON_BIN` | 推导 | 判分器/worker 用的 Python |

---

## 11. 未解决的问题清单

1. **v8 的 per-GT 数据缺失** —— 无法确认 §5.5 成因 A 是 pi 独有还是两版共有。
   要深究需重跑 v8 并保存 `score_matrix`。
2. **成因 A 要不要处理** —— 定性问题，需用户拍板（§6.4）。
3. **precision 随 recall 上升而下滑** —— §5.2(d)。bank precision
   0.6615→0.6423，flag-1/2 跌 0.22–0.30。是"择优留多了"还是"探索发散了"未定，
   建议扫 `maxInsights`。
4. **兜底修复未验证** —— §5.2(c) 的修复只过了单测，没跑真实 case（§6.1b）。
5. **单条 GT 的 delta 不可用作证据** —— flag-4 GT7 同 prompt 重跑
   0.70→0.20（§5.5）。这比 §5.3 的判分噪声更严重，因为它叠加了 agent 的
   **运行间探索路径方差**。想用 per-GT 做归因必须重复测量。
6. **flag-6..10 完全没在 pi 版上跑过** —— v8 跑了 10 个，pi 只跑了 5 个。
7. **summary 分数不可用** —— `pi_v4` 里 0.72 ± 0.271，样本量 1，建议先不看。
8. **skill 一条都没验证** —— 整个 skill 主线的起点还没迈出。
9. **迁移测试未开始** —— InsightEval / DataGovBench 都还没接。
10. ~~`run_insightbench.mjs` 硬编码绝对路径~~ —— 本次交接已改为路径推导 +
    环境变量覆盖 + 启动检查。
11. **`run_on_benchmark/daco` 是搁置的探索线** —— gitlink 在册（DACO，
    NeurIPS 2024 D&B），但磁盘上没 checkout、无代码依赖，只有
    `benchmark_survey.md` / `benchmark_run_guide.md` 提到它。要么接着做，
    要么明确删掉 gitlink。

---

## 12. 关键 commit 索引

| commit | 内容 |
|---|---|
| `aa1b53c` | run_batch 的 n 标签修复 |
| `7a00c37` | insight_bank id 匹配修复 + 负结果/解读引导（即 §5.4 的实验） |
| `18f9171` | pi 迁移步骤 5：补齐全部模块 |
| `35500d8` | pi 迁移步骤 4：planner + agent 主体 |
| `f5a2977` | worker 池内存优化（含错误归因的纠正） |
| `03b5f8a` | pi 迁移步骤 1–3：真实 API 闭环 + 常驻 worker 池 |
| `53032da` | **回滚测试集泄漏**（§8.2） |
| `e3667a4` | 判分并行化 + 剥离统计块（§9.4） |
| `ab0c07d` | （MyDataStorm）回滚测试集泄漏与无依据的 skill（§9.2） |
