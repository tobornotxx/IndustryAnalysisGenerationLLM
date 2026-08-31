# 迁移到 pi harness —— 实施方案

> 状态：方案稿（待确认后实施）
> 日期：2026-08-25
> 相关文档：`SKILL_EXTRACTION_PLAN.md`（skill 提取规划）、`TRANSFER_DESIGN.md`（benchmark 迁移）、`SKILL_PACKAGE_DESIGN.md`（skill 包定义）

---

## 0. 为什么要迁移

现状的问题不是某个 bug，而是**架构层面缺一个稳定的 agent harness**：

- `executor.py` 是手写 ReAct 循环，`max_turns=5` 硬编码，action 空间固定 5 个
- 没有通用的错误恢复机制 —— 每遇到一个新失败 case 就手工改代码打补丁
- skill 现在只是往 prompt 里塞的自然语言文本，**本质是 prompt optimization**。而我们在 InsightBench 上已经调了 v2→v8 八个版本，正是这条路到顶了才要换 benchmark。用同一手段做 skill，拿不到新东西。
- skill 不可移植：换个 agent 系统（Claude Code / Codex）完全用不了

**目标**：用成熟可定制的框架做基底，我们只做上层调优。产出两个东西 ——
一个正常工作的 insight agent，一个**真正的 skill**（能力，而非 prompt），哪怕拿给别的 agent 系统也能跑。

---

## 1. 为什么选 pi（不选 dsh）

调研以**实际编译并跑通原型**为依据（clone 两个仓库、编译 pi、跑通完整架构），不是读文档推测。

### dsh 的结构性阻碍（不是"不稳定"这么简单）

**dsh 没有进程内嵌入路径。** `docs/architecture.md:45` 明确拒绝绕过 `dsh` 的 Node 应用路径；`dsh-sdk-client` 是**把 dsh 当子进程拉起来**，走换行分隔 JSON-RPC。文档承认的两个限制：

- **No mid-turn cancel**（回合中途无法取消）
- **No per-prompt result** —— `run()` 返回"该区间内最后提交的 assistant 文本，而非因果对应该 prompt 的响应"

对于**控制流本身就是产品**的 agent，这把"harness 作为主体"整个反过来了：planner/thesis/report 只能住在 harness 外面，隔着一根有损的线驱动它。

其余风险：247 个 workspace 包 + Cordis DI（设计论文是 arXiv 预印本）；README 原话 **"THERE WILL BE COMPATIBILITY-BREAKING CHANGES"**；日均 100–219 commits；npm 全是 pre-release（`latest` 是 `0.1.1-rc.2`，仓库已到 `0.1.2-alpha.1`）；最近 400 个 commit 里 27 个是 refactor/rename/remove，含 `refactor(apiproxy)!:` 这种带 `!` 的破坏性变更，以及全库 `rename code-mode to ptc`（改的是**面向模型的词汇**，不只是内部实现）。

> 有个值得一提的发现：**dsh 自己依赖 pi 的 LLM 层** —— `@deepseek-ai/dsh-llm-pi-ai` 里 `"@earendil-works/pi-ai": "^0.84.2"`。选 pi 等于用 DeepSeek 官方自己在用的那层。

### pi 侧四个关键点均已实测通过

| # | 验证项 | 实测结果 |
|---|---|---|
| 1 | **DeepSeek 是一等内置 provider** | `packages/ai/src/providers/deepseek.ts` 仅 15 行；模型目录里正好是 `deepseek-v4-flash` / `deepseek-v4-pro`（1M context），且**易踩的 wire 细节已编码**：`supportsDeveloperRole:false`、`maxTokensField:"max_tokens"`、`thinkingFormat:"deepseek"`、`requiresReasoningContentOnAssistantMessages:true` |
| 2 | **内置 coding tools 可完全替换** | 构造 `new Agent({initialState:{systemPrompt, model, tools:[myTool]}})` 后，mock endpoint 收到 `REQ_TOOLS ["run_python"]` —— 只有我们的工具 |
| 3 | **长驻 Python 进程可行** | `python3 -u` + 行分隔 JSON，SQLite 常驻：4 次调用共 1344ms，单次冷启动 import pandas 1206ms → **1.2 秒导入成本只付一次**，且跨调用状态保住 |
| 4 | **Skills 按 agentskills.io 规范实现且脚本能执行** | 带 `scripts/profile.py` 的 skill：frontmatter 解析成功，**脚本通过 harness 自己的 bash tool 跑起来** → `SKILL_SCRIPT_RAN rows=2 cols=['g','v']` |

### 需要接受的风险

pi 是 **0.x 锁步版本**（`AGENTS.md:129`）：`patch` = 修复+新增，**`minor` = 破坏性变更，无 major**。0.74.0→0.84.3 已 42 个版本。

**缓解**：pin 精确版本 + 每次升级读 changelog 的 `### Breaking Changes` 段。关键区别是 pi 的破坏性变更**有文档记录**，dsh 那边只是"承诺会破坏"。

**建议先用稳定的 `Agent` 类，`AgentHarness` 暂不用**（后者是较新子系统，文档带 "Open questions" 附录）。

---

## 2. 性能要求（必须带过去，不能重蹈覆辙）

这些是之前实测踩出来的，迁移后**必须保持或改进**。

### 2.1 并行

| 层级 | 现状 | 迁移后 |
|---|---|---|
| **case 级** | `ProcessPoolExecutor(max_workers=N)`，`--workers 4` | 保持（外层调度不变） |
| **层内问题级** | `ThreadPoolExecutor`，同层多问题并发 | pi 的 tool 调用天然并发；需确认 Python worker 并发安全 |
| **判分级** | `SCORER_MAX_WORKERS=16` 线程池 | **不动**（判分留在 Python，是离线工具） |

> **判分并行的收益是实测的**：248 次调用从十几分钟降到 59 秒，mock 测 600 次调用 **13.2x 加速**。

⚠️ **迁移新增的并发问题**：长驻 Python worker 是**单进程**，行分隔 JSON 协议是**串行**的。若同层多个问题要并发执行 SQL/Python，需要 worker 池（N 个 worker）或在 worker 内部用线程。**这是迁移必须解决的设计点，不能默认单 worker 就够。**

### 2.2 Token 预算（已实施，逻辑要搬过去）

| 参数 | 值 | 作用 |
|---|---|---|
| `_SAMPLE_CHAR_BUDGET` | 500 | 每列样本字符预算，逐个累加直到首次超预算。短值列能列全，长文本列自动收敛 |
| `_CELL_MAX_CHARS` | 100 | 单值截断上限，超出标注并提示 agent 可自行查询 |
| `_ID_LIKE_SAMPLE_BUDGET` | 120 | `unique/rows > 0.9` 的列（ID/主键/时间戳）只给小预算 —— 样本高度同构，列满不增信息量 |
| `_SCHEMA_TOTAL_BUDGET` | 40000 | schema 文本总闸，防多表膨胀 |

**实测效果**：flag-1 schema 3777 字符，关键实体 `Printer546` 保留。

> **这套必须原样搬过去。** 它的教训是血的：我曾按 DataGovBench 论文把样本压缩掉，导致 flag-1 的 `Printer546` 从 schema 消失，agent 再也找不到它。

### 2.3 Prompt 缓存

判分 prompt 用 `pred_first` 顺序（`GEVAL_PROMPT_ORDER`）：把**长的 pred** 连同固定 instructions 前置作稳定前缀，短的 GT 放末尾。

实测前缀占比：`answer_first` 7.6% → `gt_first` 21.6% → **`pred_first` ~100%**。

**迁移后同样要注意 pi 侧的 prompt 前缀稳定性**：pi 的 context 是消息序列，稳定的 system prompt + schema 应在前，变化的探索历史在后。

### 2.4 实测成本基线（迁移后要对比）

来自 v9 flag-4（`max_layers=5`，31 条 pred，8 条 GT）：

| 阶段 | 调用数 | 输入 token |
|---|---|---|
| Pipeline（探索+报告+summary） | 229 | ~888K |
| Scoring（G-Eval，并行） | 249 | ~305K，cache 23% |

**迁移的一个预期收益**：Agent Skills 的三级渐进披露 —— metadata（~100 tok，常驻）→ SKILL.md 正文（激活时载入）→ **scripts 源码永不进 context，只有 stdout 进**。确定性的分析逻辑不再是每次都要重新生成的 token。

### 2.5 用量记账

`_record_usage()` 记录 `prompt_tokens` / `completion_tokens` / `cache_hit_tokens`（兼容 DeepSeek 和 OpenAI 两种字段），并发下加锁。

> 迁移后**必须保留等价能力**。之前完全没有这些字段，导致"缓存命中多少"根本无法回答，只能靠猜。

---

## 3. 目标架构

```
┌─────────────────────────────────────────────────────────────┐
│  pi harness (TypeScript)  —— agent loop / 重试 / skill 加载   │
│                                                              │
│  new Agent({ systemPrompt, model: deepseek-v4-flash,         │
│              tools: [runPython, runSql, ...] })              │
│                                                              │
│  钩子映射（我们手写的机制 → pi 官方钩子）：                     │
│    goal_sufficiency 早停  → shouldStopAfterTurn              │
│    insight_bank 去重      → transformContext                 │
│    insights / thesis      → CustomAgentMessages              │
│                              (留在 transcript 但不发给模型)    │
│    executor 重试          → pi 内置 retry                     │
└──────────────┬──────────────────────────┬───────────────────┘
               │ tool call                │ bash tool
               ▼                          ▼
┌──────────────────────────┐   ┌──────────────────────────────┐
│ Python worker (常驻)      │   │ skills/<name>/               │
│ 行分隔 JSON 协议:         │   │   SKILL.md   (frontmatter)   │
│   load_csv → SQLite      │   │   scripts/*.py  ← 可执行代码  │
│   sql / py               │   │                              │
│ 沙箱预装: pandas/numpy/   │   │ 三级渐进披露:                 │
│ scipy/sklearn/statsmodels│   │   metadata → 正文 → 脚本      │
│ /duckdb/ruptures/...     │   │ 脚本源码不进 context           │
└──────────────────────────┘   └──────────────────────────────┘

           ┌────────────────────────────────────┐
           │ 评测 (保持 Python，离线工具)          │
           │ unified_scorer.py                  │
           │  · recall/precision/F1 共享矩阵     │
           │  · SCORER_MAX_WORKERS=16 并行       │
           │  · pred_first 缓存优化              │
           │  · usage 记账                       │
           └────────────────────────────────────┘
```

**分工原则**：TypeScript 管编排与 agent 行为，Python 只做**数据分析执行**（那些库没有 JS 等价物）和**离线评测**。

---

## 4. 组件迁移对照

> **原则（决策 4）**：只有 `executor` 必须删、`planner` 值得 TS 原生重写。
> **其余 Python 模块全部包装成 pi 的 tool / skill 复用**，不重写 —— 现有 prompt 调优成果不丢。

| 现有组件 | 处置 | 说明 |
|---|---|---|
| `agents/executor.py` | **删除** | 由 pi 的 agent loop 取代（这是迁移的主要动因） |
| `agents/planner.py` | **重写为 TS** | 核心链路，值得原生化；prompt 模板 Jinja2 → TS 模板字符串 |
| `csv_db_bridge.py` | **改造成常驻 worker** | schema 生成 + 预算机制 + 时间画像原样保留；暴露为 `runSql` / `runPython` tool |
| `modules/goal_sufficiency.py` | **包装成 tool** | 挂 `shouldStopAfterTurn` |
| `modules/insight_bank.py` | **包装成 tool** | 或挂 `transformContext` |
| `modules/thesis.py` | **包装成 tool** | 内部仍是 Python 调 LLM |
| `modules/report.py` | **包装成 tool** | 同上 |
| `modules/consistency.py` | **包装成 tool** | 同上 |
| `modules/statistics.py` | **包装成 skill 脚本** | 统计计算留 Python |
| `unified_scorer.py` | **完全不动** | 离线评测工具 |
| `run_benchmark.py` | 改造 | 外层 case 级并行调度保留，内部改为调 TS agent |
| `skills/skill_package_v9.json` | **推翻重做** | 改为 `skills/<name>/SKILL.md` + `scripts/`（见 §5） |

⚠️ **包装成 tool 的模块内部自己调 LLM，不走 pi 的 model 层**，因此 pi 的 usage 统计
不含它们 —— §2.5 的 Python 侧记账必须保留，否则又会出现"token 花在哪不知道"的情况。

---

## 5. Skill 形态的根本改变

### 之前（prompt 文本，已被否决）

```jsonc
{ "id": "decompose-trend-by-category",
  "action": "Do not test it only on the overall total. Also decompose the trend by ..." }
```
→ 本质是 prompt optimization，不可移植，无法单独测试。

### 之后（Agent Skills 标准）

```
skills/trend-per-group/
  SKILL.md                    # frontmatter: name, description
  scripts/detect_trend.py     # 真实现：Mann-Kendall per group
```

**关键收益**：

| | prompt 文本 | SKILL.md + scripts |
|---|---|---|
| 形态 | 一段劝告 | **可执行能力** |
| 可验证 | 只能跑 A/B 猜 | **脚本可单元测试**，跑不通直接拒 |
| 可移植 | 无 | **Claude Code / Codex / Cursor / Gemini CLI 等已适配** |
| token | 每次全量进 prompt | **源码不进 context，只有 stdout 进** |

> agentskills.io 规范已是厂商中立标准（规范仓库 24.8k stars），已适配系统包括 Claude Code、OpenAI Codex/ChatGPT、Cursor、Gemini CLI、Copilot、OpenHands、Databricks、Snowflake 等。**"拿给别的 agent 也能跑"今天就成立。**

### 对 skill 提取 agent 的影响

`SKILL_EXTRACTION_PLAN.md` 的四道审核闸继续有效，且**新增一道最硬的闸**：

| 闸 | 判据 |
|---|---|
| 1 程序性 | action 是通用分析动作，不是具体发现 |
| 2 数据无关 | 不含具体列名/列值/领域术语（代码强制） |
| 3 来源合法 | provenance 只能指向 InsightBench 运行数据；出现测试集名称直接拒（代码强制） |
| 4 非平凡 | 不是"要仔细分析"这类空话 |
| **5 代码可运行（新）** | **脚本必须在样本数据上 smoke test 通过，跑不通直接拒** |

第 5 道闸比任何 LLM 审核都硬 —— 这是从"prompt 变能力"带来的直接好处。

---

## 6. 技术阻碍与对策

| 阻碍 | 严重度 | 对策 |
|---|---|---|
| **Python worker 并发** | 🔴 必须解决 | 单 worker + 串行协议无法支撑同层并发。需 worker 池（N 个进程）或 worker 内线程。**设计时定，不能事后补** |
| pi 0.x minor 即破坏 | 🟡 | pin 精确版本；升级前读 Breaking Changes；先用 `Agent` 不用 `AgentHarness` |
| 真实 API 未验证 | 🟡 | 调研用 mock 验证 wire format，**真实成本/延迟/限流未测**。第一步就要用真 key 跑通 |
| Node + Python 双栈 | 🟡 | 部署多一层依赖；worker 生命周期管理（崩溃重启、超时） |
| 现有 Python 逻辑重写风险 | 🟡 | 分步迁移，每步和现有结果对比；prompt 模板可直接搬（Jinja2 → TS 模板字符串） |
| 判分口径已变 | 🟡 | v8 的 0.8767 已不可比（剥统计块所致）。迁移后需重建基线 |

### 其他已知事项

- ~~`types.py` 遮蔽标准库~~ —— 已修
- pi 无正式 deprecation 窗口 / LTS 分支政策，只有锁步规则

---

## 7. 实施步骤

分支：`pi-migration`（两个 repo 均已建），`main` 保持可回退。

| # | 内容 | 验证方式 | 花钱 |
|---|---|---|---|
| 1 | pi 装起来，用**真实 DeepSeek key** 跑通最小 agent（一个 echo tool） | 看到正确响应 | 少量 |
| 2 | Python worker：`load_csv`/`sql`/`py`，搬 `csv_db_bridge` 的 schema 生成 + 预算机制 | schema 输出与现有逐字对比，`Printer546` 在 | 否 |
| 3 | `runSql`/`runPython` tool 包装 worker；**实现 worker 池 + 共享 DB 文件** | 并发调用不串数据；内存符合 §8 估算 | 少量 |
| 4 | planner 重写 TS；跑单 case 端到端出 insights | 与现有 pipeline 输出对比 | 中 |
| 5 | 把 goal_sufficiency / insight_bank / thesis / report / consistency **包装成 tool**；挂 `shouldStopAfterTurn` + `transformContext` | 早停与去重行为和现有一致 | 中 |
| 6 | 接 `unified_scorer.py` 评分 | 单 case 三指标与 v10 对比 | 中 |
| 7 | 写 1-2 个 SKILL.md + scripts，验证脚本被执行 | 日志见脚本 stdout | 少量 |
| 8 | **5 个 case** 回归，重建基线 | 均值 + 方差 | **主要成本** |

**步骤 2 不花钱**，步骤 1/3/7 花很少。**步骤 8 是主要开销。**

⚠️ **样本量提醒**：已知 flag-1 历史极差 **0.63**、flag-3 仅 **0.03** —— **case 间差异远大于版本间差异**。
5 个 case 只够看趋势和量级，**不足以判断 0.03 量级的差异**。任何"变好/变差"的结论都要带上这个限制。
（另：上次 10 case 跑了 39 分钟就把 API 余额烧完，5 个是成本上的现实选择。）

---

## 8. 已确认决策（2026-08-25）

| # | 决策 | 结论 |
|---|---|---|
| 1 | **Python worker 并发** | **worker 池（N 个独立进程）**。见下方内存实测。 |
| 2 | **迁移策略** | **git 分支 `pi-migration`**（两个 repo 均已建），`main` 保持可回退 |
| 3 | **重写顺序** | **先 planner**（核心链路），尽早暴露集成问题 |
| 4 | **agent 主体全部用 pi** | 现有 Python 模块**包装成 tool / skill 提供给 agent**，从而复用而非重写 |
| 5 | **回归 case 数** | **5** |

### 决策 1 的依据：worker 内存

本机 16 GB。

**初次估算**（仅 import 主要几个库）：单 worker 205 MB —— 库 202 MB + 500 行数据 3 MB。

**实测修正（2026-08-25，4-worker 池，已 warmup）**：

```
4 workers RSS: 365MB, 364MB, 365MB, 363MB  →  总计 1458 MB
```

**单 worker 实际 365 MB，比估算高 78%。** 差异来源：`execute_python_from_sql` 的沙箱会
预注入**全部**数据科学包（numpy/pandas/scipy/sklearn/statsmodels/sympy/networkx/
xgboost/lightgbm/polars/duckdb/lifelines/pingouin/ruptures/category_encoders/imblearn
+ matplotlib/seaborn），远多于估算脚本里 import 的那几个。

| worker 数 | InsightBench（实测/外推） |
|---|---|
| 2 | ~730 MB |
| **4** | **1458 MB（实测）** ✅ 16 GB 下无压力 |
| 8 | ~2.9 GB |

⚠️ **外推 DataGovBench（21 万行 × 18 列 × 5 表）须按 365 MB 修正**：

| worker 数 | 各自 in-memory 副本 | 共享 SQLite 文件 |
|---|---|---|
| 1 | ~6.7 GB | ~6.7 GB |
| 4 | **~26.8 GB** ❌ 超 16 GB | **~7.7 GB**（365×4 + 6.3 GB 共享）✅ |

**对策（迁移时必须实现）**：worker 之间**共享 SQLite 文件**，而非各自 in-memory 副本。
现有 `CsvDatabaseBridge` 本就用 `tempfile.mkstemp(suffix=".db")` 而非 `:memory:`，
但**当前 `loadAll()` 让每个 worker 各建一个临时 DB** —— 大数据集下必须改成传入同一
DB 路径共享。小数据集（InsightBench）无需改。

### 并发实测（决策 1 得到验证）

单 worker 的行分隔 JSON 协议是**串行**的，因此池是必需的：

| 负载 | 单 worker | 4-worker 池（冷） | 4-worker 池（已 warmup） |
|---|---|---|---|
| 3 × 1 秒任务并发 | 7.26 s | 5.52 s | **1.01 s** ✅ 真并行 |
| 4 × 空操作并发 | — | — | 0.001 s |
| 8 × SQL 并发 | — | — | 0.003 s |
| 6 × 1 秒任务（超池大小） | — | 4.92 s | 分两批 |

**关键：必须 warmup。** 沙箱首次调用要为预注入那一大堆包付约 1 秒/worker
（实测池整体 warmup 4.03 s）。不预热则「3 个 1 秒任务」要 5.5 s ——
每个 worker 各自摊 import 成本。`PyWorkerPool.warmup()` 把这笔一次性开销
挪到池初始化阶段。

### 决策 4 的落地方式（重要，简化了 §4 的工作量）

你的原话：*"agent 主体都用 pi，你这些东西本身包装一下提供给 agent 不就行了吗，一样可以复用。"*

这改变了迁移策略 —— **不是把 Python 模块重写成 TS，而是把它们包装成 pi 的 tool / skill**：

| 现有 Python 模块 | 之前的计划 | **改为** |
|---|---|---|
| `csv_db_bridge` | 改造成 worker | worker + `runSql`/`runPython` tool |
| `modules/statistics.py` | 移入 worker | 包装成 skill 脚本 |
| `modules/thesis.py` | 重写 TS | **包装成 tool**（内部仍是 Python 调 LLM） |
| `modules/report.py` | 重写 TS | **包装成 tool** |
| `modules/consistency.py` | 重写 TS | **包装成 tool** |
| `modules/insight_bank.py` | 重写 TS | **包装成 tool**，或挂 `transformContext` |
| `modules/goal_sufficiency.py` | 重写 TS | **包装成 tool**，挂 `shouldStopAfterTurn` |
| `agents/planner.py` | 重写 TS | **先 TS 重写**（决策 3：它是核心链路，值得原生化） |
| `agents/executor.py` | 删除 | **删除**（由 pi 的 loop 取代 —— 这是迁移主因） |

**只有 executor 必须删、planner 值得原生重写。其余全部包装复用。**

好处：
- 工作量大幅下降，且现有逻辑（含所有 prompt 调优成果）不丢
- 每个模块作为独立 tool，pi 的重试/错误恢复自动覆盖它们
- 符合 §5 的 skill 理念：**能力 = 可执行代码 + 描述**

代价：这些 tool 内部自己调 LLM（不走 pi 的 model 层），所以 pi 的 usage 统计
不包含它们 —— **§2.5 的 Python 侧记账必须保留**。

---

## 附录：两个必须守住的方法论纪律

这两条是之前实际犯过的错，迁移后依然适用。

1. **测试集经验绝不回灌**
   `InsightBench = 训练集`（在此提炼 skill）；`InsightEval / DataGovBench = 测试集`（只验证泛化）。
   曾把 DataGovBench 论文的消融结论写进 bridge —— 那不再是泛化，是照着答案改。已回滚（`53032da`）。
   → 提取 agent 的闸 3 必须是**代码级硬约束**。

2. **skill 必须从运行过程中形成**
   曾跳过提取 agent 手工抄 prompt 成 skill，导致出现 `prefer-productive-followups` 这种无任何依据的条目。已删除（`ab0c07d`）。
   → `running on a benchmark accumulates information to form a skill` 是设计的立身之本。
