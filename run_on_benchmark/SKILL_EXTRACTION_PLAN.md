# Skill 提取 Agent —— 实施规划

> 状态：规划稿（先对齐，再写代码）
> 日期：2026-08-25
> 上游文档：`SKILL_PACKAGE_DESIGN.md`（skill 包结构定义）、`TRANSFER_DESIGN.md`（迁移测试设计）

---

## 0. 我们到底在做什么（目标对齐）

**两个 agent，一次性验证。**

| # | 产出 | 是什么 |
|---|---|---|
| 1 | **Insight Agent with skill** | 现有 MyDataStorm pipeline + skill 包，读数据 → 产 NL insight |
| 2 | **Skill Extraction Agent** | 从 InsightBench 的**运行历史**中提炼 skill，靠约束+审核保证不过拟合 |

**验证方式：一次性实验。** 提取出 skill 包后，跑一次「带包 vs 不带包」对比，人看结果判断是否有效。

> ⚠️ **明确不做**：自动化的「提取 → 验证 → 更新 skill」闭环流水线。那是过度工程，成本高且偏离目标。
> 提取 agent 的重点是 **约束和审核**，不是自动化。

---

## 1. 核心原则（这次栽过的坑，写成硬约束）

### 1.1 skill 必须从「运行过程」中形成

这是设计的立身之本：**running on a benchmark will accumulate information to form a skill。**

- ✅ 合法输入：InsightBench 的运行日志、per-case 分数、pred vs GT 对比、成功/失败轨迹
- ❌ 非法输入：我（或任何人）读代码后手工抄录的 prompt
- ❌ 非法输入：从论文里读来的经验

> **本次的反面教材**：`skill_package_v9.json` 里 5 条 skill 全是我手工从 v8 代码抄的，
> `produced_by = "manual-transcription-from-v8 (NOT the extraction agent)"`。
> 它们只能算候选，必须由提取 agent 重新从运行数据中产出（或至少验证）。

### 1.2 测试集经验绝不回灌（test set leakage）

实验设计是：
```
InsightBench  = 训练集   → 在此提炼 skill
InsightEval   = 测试集   ┐
DataGovBench  = 测试集   ┘ → 只用于验证泛化，其经验不得反向进入 skill
```

> **本次的反面教材**：我把 DataGovBench 论文的消融结论（"按列类型序列化 +50 分、
> 多给样本行反而掉分"）写进了 `csv_db_bridge`，即 skill 载体。哪怕它在 DataGovBench
> 上真能提分，那个提分也不再是泛化——是照着答案改的。已于 `53032da` 回滚。
>
> 更糟的是它连训练集上都错：那结论来自 21 万行多表政府数据，InsightBench 是 500 行
> 单表，规模差 400 倍。flag-1 的关键实体 `Printer546` 原本就在 schema 样本里，被我
> 压缩掉后 agent 再也找不到它。

**因此**：提取 agent 必须把 provenance 校验做成**代码级硬约束**，不能靠人自觉。

### 1.3 不能是「为评分器定制」

过拟合有两类，我此前的审计只查了第一类：

| 类型 | 表现 | 例子 |
|---|---|---|
| **数据特化** | 硬编码列名/列值/领域术语 | `assigned_to`、`TTR`（已清除） |
| **评分器特化** | 为某种打分方式定制输出形态 | 罗列全部统计量去扩大与 GT 的字符匹配面（已清除） |

第二类更隐蔽：它能提分，所以看起来"有效"。判据是问一句——
**"这个行为在没有 judge 的真实分析场景里，还算好的分析习惯吗?"**

---

## 2. 输入：什么算「运行历史」

提取 agent 只读这些，不读代码：

| 来源 | 文件 | 提供什么 |
|---|---|---|
| per-case 结果 | `results/<run>/summary.json` | 各 flag 的 recall/precision/F1 |
| 详细结果 | `results/<run>/flag-N/result.json` | goal、pred_insights、gt_insights、score_matrix |
| 探索轨迹 | `results/<run>/flag-N/question_tree.json` | 每层问了什么、答了什么、哪些有产出 |
| 运行日志 | `results/<run>/flag-N/run.log` | executor 的实际动作、失败重试 |
| 跨版本对比 | 多个 `results/test_run_v*/` | 哪次迭代让哪个 case 涨/跌 |

**关键数据是 `score_matrix`**（已在 `e3667a4` 落盘）：`matrix[i][j] = S(pred_j, gt_i)`。
有了它就能定位「哪条 GT 没被覆盖、最接近的 pred 是什么」——这是发现盲区的直接依据。

---

## 3. 处理流程（5 步，一次性跑完）

```
mine        扫结果，找两类信号：
              · 命中：某条 GT 被高分覆盖 → 是什么行为促成的？
              · 漏掉：某条 GT 所有 pred 都低分 → 缺了什么分析动作？
              ↓
hypothesize 从模式提候选 skill，形式固定为「当 <trigger> 时，做 <action>」
              ↓
generalize  剥掉列名/列值/领域术语，改成 {categorical_columns} 之类占位符
              ↓
audit       四道闸逐条过（见 §4）。不过的直接拒，记录拒因
              ↓
emit        写 skill_package.json + extraction_log.md（含被拒项）
```

**不含 validate 步骤。** 验证是整包提完后的一次性实验（§5），不在提取循环内。

---

## 4. 四道审核闸（提取 agent 的核心）

每条候选 skill 必须全部通过，任一不过即拒绝并记录拒因。

| # | 闸 | 判据 | 实现 |
|---|---|---|---|
| 1 | **程序性** | action 是通用分析**动作**（分解/自检/选粒度），不是具体**发现** | LLM 判 + 关键词兜底 |
| 2 | **数据无关** | trigger/action 不含具体列名、列值、领域术语 | **代码强制**：正则扫黑名单 + 扫训练集实际列名/高频值 |
| 3 | **来源合法** | provenance 必须指向 InsightBench 运行数据的具体 case/run | **代码强制**：必须匹配 `flag-\d+` 或 `v\d+`；出现测试集名称直接拒 |
| 4 | **非平凡** | 不能是"要仔细分析""考虑多个角度"这类空话 | LLM 判 + 最小信息量检查 |

### 闸 2 的具体做法（不能只靠 LLM 自觉）

```python
# 黑名单：训练集的实际列名 + 高频值 + 领域术语
forbidden = set(df.columns) | top_frequent_values | {"TTR", "incident", "ITSM", ...}
# 占位符白名单
allowed_placeholders = {"{categorical_columns}", "{time_columns}", ...}
```
skill 文本里出现任何 forbidden 词 → 拒绝，要求重写为占位符形式。
**这是代码检查，不是 prompt 里的一句叮嘱。**

### 闸 3 的具体做法

```python
TEST_SET_MARKERS = ["InsightEval", "DataGovBench", "2511.22884", "2607.06482",
                    "MedInsightBench", "Data Analysis in the Wild"]
# provenance 中出现任一 → 直接拒绝，不给重写机会
```
这条**没有例外**。测试集的任何信息都不能作为 skill 的来源依据。

---

## 5. 验证：一次性实验，不是流水线

提取完成后跑一次对比：

| 臂 | 配置 |
|---|---|
| **A** | 不加载 skill 包（pipeline 裸跑） |
| **B** | 加载 skill 包 |

- 同一批 case（InsightBench），同 `max_layers`，同 judge
- 比 recall / precision / F1
- **人看结果判断**是否有效，不做自动迭代

**成本**：2 次跑，不是 per-skill ablation 的 4×N 次。

⚠️ **样本量提醒**：已知 flag-1 历史极差 0.63、flag-3 仅 0.03，**case 间差异远大于版本间差异**。
单 case 对比无意义，至少要 5-10 个 case 看均值。这也是当前最大的成本项。

---

## 6. 落地形式

**不引入开源 agent 框架**（LangChain / AutoGen 等）。理由：

- 提取 agent 需要的工具只有三个：读文件、跑 pipeline、算分。**三个我们都有了**
  （`run_benchmark.py` / `unified_scorer.py` / `result.json`）
- 流程是固定的，可以直接写成脚本；框架带来的通用 tool-calling 编排解决的不是我们的瓶颈
- 多一层抽象 = 多一批失败模式

**文件布局**：

| 文件 | 职责 |
|---|---|
| `MyDataStorm/datastorm/skills/extractor.py` | 提取 agent 主体（mine/hypothesize/generalize/audit/emit） |
| `MyDataStorm/datastorm/skills/audit.py` | 四道闸，可独立测试（闸 2/3 是纯代码，可写单元测试） |
| `MyDataStorm/datastorm/skills/__init__.py` | 已有：skill 包 loader + render_guidance |
| `run_on_benchmark/run_skill_extraction.py` | CLI 入口 |

**重要**：提取 agent 是**离线工具**，不属于被测 pipeline，不能进 skill 载体。

---

## 7. 与现有代码的关系

| 组件 | 现状 | 本规划要做的 |
|---|---|---|
| skill 包 JSON | 5 条候选（手工抄录，status=candidate） | 由提取 agent 重新产出/验证 |
| skill loader + `render_guidance` | ✅ 已可用 | 不动 |
| bridge / goal_sufficiency 的消费点 | ✅ 已接入 skill 包 | 不动 |
| `score_matrix` 落盘 | ✅ 已有 | mine 阶段的主要输入 |
| 判分并行 | ✅ 13x 加速 | 让验证实验成本可接受 |

---

## 8. 实施步骤

| # | 内容 | 产出 |
|---|---|---|
| 1 | `audit.py` 四道闸 + 单元测试（闸 2/3 纯代码，可零成本测） | 审核模块 |
| 2 | `extractor.py` 的 mine：从 score_matrix 定位命中/漏掉的 GT | 结构化信号 |
| 3 | `extractor.py` 的 hypothesize + generalize（LLM） | 候选 skill |
| 4 | 串起来 + CLI，在现有 v8/v10 结果上跑一次 | skill 包 + extraction_log |
| 5 | 人工复核提取结果（这一步不能省） | 确认的 skill 包 |
| 6 | 一次性 A/B 验证（5-10 case） | 有效性结论 |

**步骤 1-4 不花 API 钱**（除 3 的 LLM 调用，量很小）。**步骤 6 是主要成本。**

---

## 9. 待确认

1. **mine 的输入范围**：只用最新一次运行，还是跨 v2-v10 全部历史？
   （跨版本能看到"哪次改动让哪个 case 涨"，信号更强，但数据口径不一致——v10 剥了统计块）
2. **提取 agent 用哪个模型**：建议 pro（审核判错一次整个包就脏了，最不该省的地方）
3. **验证实验的 case 数**：5 还是 10？直接决定成本
4. **候选 skill 怎么处置**：现有 5 条是「让提取 agent 独立重新提取后对比」，还是「作为已有候选交给它审核」？
   前者更干净（真正从运行数据形成），后者更省。

## 附录：本次记录在案的两个方法论错误

1. **test set leakage** —— 把 DataGovBench 消融结论写进 bridge。已回滚（`53032da`）。
   教训：测试集的任何结论，只能在迁移测试跑完并记录结果**之后**讨论，且不得回灌。
2. **跳过提取 agent 直接手写 skill** —— 导致出现 `prefer-productive-followups`
   这种无任何依据的条目。已删除（`ab0c07d`）。
   教训：skill 必须从运行数据中形成，这是设计的立身之本，不是流程形式。
