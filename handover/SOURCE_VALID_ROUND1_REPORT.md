# Source-valid Round 1 实验报告

实验 ID：`v41_source_valid_round1_20260916`

日期：2026-09-16

实验分支：`codex/thesis-experiment-upgrades-industry`

评分协议：`local-deepseek-v41-nonthinking-v1`（已判定尺度失真；本报告语义分数等待思考模式复评替换）

> **勘误（2026-09-16）：** 非思考评分器会把明确同义的复杂 insight 系统性压低，且对原始长版与忠实压缩版给出异常大的分差。本报告中的语义 Recall、Precision、F1、Summary 及其系统排序暂不可引用；运行成功率、耗时、轨迹和确定性指标不受影响。

## 1. 结论先行

这一轮证明了新的实验框架可以完整运行，并能在相同预算下比较本地 PI、PI+skills、DataSTORM 复刻版和 AgentPoirot 官方实现。但是，它没有证明任何系统稳定领先。

四个系统的语义 F1 均值集中在 0.269–0.298。DataSTORM 和 AgentPoirot 相对 PI-core 的均值优势只有 0.017 左右；PI+skills 相对 PI-core 反而低 0.011。这里只有 2 个 case，每个 case 2 次成功生成，所以这些差异不能解释为真实性能提升。

相反，本轮最明确的发现是：生成随机性远大于评分器随机性。所有输出的平均 judge 标准差为 0.0128，而同一系统、同一 case 的平均 agent-run 标准差为 0.1058，约为前者的 8.3 倍。后续实验必须优先增加独立生成重复，而不是堆更多 judge 重复。

论文可以继续，但论点应从“改进后得分更高”改为：

> 在固定探索预算下，构建一个可复现、可迁移、可审计的数据分析智能体实验框架；比较层级式假设驱动探索、DataSTORM 式研究树和官方 AgentPoirot 在覆盖、稳定性、效率及失败模式上的差异。

这是比追求两个 case 上的小幅均值提升更可信、也更适合专业硕士论文的叙事。

## 2. 实验设计

- 数据：InsightBench 的 `flag-11`、`flag-12`，冻结为 `source-valid`；前 10 个 case 已被开发过程查看答案，因此只标记为 `dev-contaminated`，不进入本轮结论。
- 系统：`pi-core`、`pi-core-skills`、`datastorm-reproduction`、`agentpoirot-official`。
- 预算：每次最多 6 个主分析问题、最多 10 条输出 insight。
- 重复：每个系统、每个 case 取 2 次成功 agent run；每个输出做 2 次独立 judge run。
- 生成模型：DeepSeek V4.1 Flash，API 名 `deepseek-flash`。
- 评分模型：同一模型，但强制关闭思考模式并限制输出到 50 tokens；避免默认思考造成无意义的超长评分开销。
- 主指标：语义 insight precision、recall 和 F1。
- 辅助指标：token overlap、数字覆盖、重复率、输出长度、summary 分数、运行成功率、耗时和调用量。
- AgentPoirot 官方仓库未暴露可直接复用的 summary 生成入口，因此仅参与 insight 指标比较；它的 summary 分数不与其他系统比较。

## 3. 主要结果

### 3.1 系统级均值

| 系统 | 语义 F1 | Recall | Precision | Token overlap F1 | 数字覆盖率 | 平均每条 insight tokens | Summary |
|---|---:|---:|---:|---:|---:|---:|---:|
| PI-core | 0.2803 | 0.2750 | 0.2896 | 0.0621 | 0.8750 | 232.08 | 0.5500 |
| PI-core + skills | 0.2691 | 0.2792 | 0.2621 | 0.0902 | 0.7500 | 133.86 | 0.4500 |
| DataSTORM reproduction | 0.2971 | 0.2719 | 0.3458 | 0.0706 | 1.0000 | 187.29 | 0.3500 |
| AgentPoirot official | 0.2979 | 0.3385 | 0.2700 | 0.1269 | 0.8750 | 60.57 | 不适用 |

所有成功输出的重复率均为 0。数字覆盖率只在含可提取参考数字的 `flag-11` 上有值，不能当成两个 case 的总体成绩。确定性指标与语义指标给出的排序不同：AgentPoirot 的短 insight 更接近参考答案表面措辞，因此 token overlap 较高；PI 输出更长、分析更深入，却不一定更容易命中短参考 insight。这说明 token overlap 只能作为表达层辅助指标，不能替代语义质量。

### 3.2 case 级 F1 与生成波动

| 系统 | flag-11 | agent-run SD | flag-12 | agent-run SD |
|---|---:|---:|---:|---:|
| PI-core | 0.2037 | 0.0113 | 0.3569 | 0.3119 |
| PI-core + skills | 0.2958 | 0.1226 | 0.2425 | 0.1370 |
| DataSTORM reproduction | 0.2308 | 0.0743 | 0.3634 | 0.1291 |
| AgentPoirot official | 0.2874 | 0.0495 | 0.3085 | 0.0108 |

PI-core 在 `flag-12` 的两次运行均值分别为 0.5775 和 0.1364。仅这一处就足以说明：只跑一次会把随机命中误写成“显著改进”。

### 3.3 与 PI-core 的配对差异

| Challenger | flag-11 差异 | flag-12 差异 | 两 case 平均差异 | 配对置换 p |
|---|---:|---:|---:|---:|
| PI-core + skills | +0.0921 | -0.1145 | -0.0112 | 1.0 |
| DataSTORM reproduction | +0.0271 | +0.0065 | +0.0168 | 0.5 |
| AgentPoirot official | +0.0837 | -0.0484 | +0.0176 | 1.0 |

`n=2` 时置信区间和 p 值几乎没有推断价值。这里保留它们是为了验证统计管线，不用于声称显著性。

## 4. 轨迹和失败模式分析

### 4.1 PI-core

PI 使用两层问题树：第一层做类别、时间、人员或地点切分，第二层围绕已发现异常继续验证机制。`flag-11` 的轨迹能够从 Hardware 在 2023 年 7 月后的 TTR 异常继续追问 backlog、关闭批次、人员与地点效应。这种轨迹比平铺 10 个图更接近“研究过程”，也是论文中最有价值的可解释性材料。

问题在于第二层高度依赖第一层叙事。某次运行若第一层提出了错误机制，后续问题会持续验证该机制，形成路径依赖。`flag-12` 两次 F1 相差 0.44，就是这种生成随机性和路径依赖的直接表现。

### 4.2 PI-core + skills

skills 使问题更偏向统计检验、混杂控制和反事实验证，并把 insight 从平均 232 tokens 压缩到 134 tokens。但两 case 上一升一降，没有稳定净收益。当前证据只能说明 skill 改变了探索风格，不能说明它提高了质量。

下一轮应把 skill 拆成可消融的单元，而不是只比较总开关：异常确认、混杂控制、机制验证、结论压缩分别开关，才能判断究竟哪一部分有效。

### 4.3 DataSTORM reproduction

DataSTORM 在固定 6 问预算后可以稳定结束，但单次约需 7.7 分钟。`flag-11` 的一次轨迹中，首轮月度类别查询返回了 60 行，报告只抓住可见的前半段，随后错误地得出“没有单一类别持续恶化”，漏掉了参考答案最核心的 Hardware 异常。这不是模型知识不足，而是中间结果截断和证据选择问题。

因此，DataSTORM 的下一项必要修复不是增加更多 LLM 调用，而是让查询结果先经过确定性的压缩与异常摘要，再交给规划器。

### 4.4 AgentPoirot official

AgentPoirot 每次生成 10 条短 insight，覆盖广、token overlap 高；但轨迹中出现过类别选择不一致、基于小样本的夸大，以及问题标题与实际分析字段不一致。`flag-12` 还发生过一次上游绘图文件缺失导致的整次失败。适配层现已允许缺图时退回文本 insight，补跑成功；失败记录仍保留用于可靠性统计。

官方框架没有暴露与当前统一接口等价的 summary 输出，因此不能为了“凑齐指标”私自增加非官方 summary 实现。

## 5. 运行可靠性与成本

| 系统 | 成功正式运行 | 额外失败/中止 | 平均成功耗时 |
|---|---:|---:|---:|
| PI-core | 4 | 0 | 56.9 秒 |
| PI-core + skills | 4 | 0 | 58.9 秒 |
| DataSTORM reproduction | 4 | 1 次预算诊断中止 | 462.8 秒 |
| AgentPoirot official | 4 | 1 次缺图失败 | 476.9 秒 |

PI-core 四次生成共 149 calls、606,916 input tokens、75,883 output tokens，记录成本 0.069432 美元；PI+skills 共 151 calls、588,026 input tokens、72,271 output tokens，记录成本 0.065427 美元。

DataSTORM 和 AgentPoirot 的上游客户端目前没有统一 token/cost 记录，因此不能做公平成本比较。这是下一轮正式大样本实验前必须补齐的观测缺口。

32 次 judge run 共 1,588 calls、906,656 input tokens、11,116 output tokens。按当时 V4.1 Flash 高峰价估算约 1.23 元人民币。关闭默认思考模式后，judge 单次 completion 被稳定限制在很小范围；此前试验中默认思考曾产生 10 万级 completion tokens，因此冻结非思考评分协议是必要的。

## 6. 参考答案与评分有效性

`flag-12` 参考答案只有 4 条，覆盖的是：Hardware 数量异常、Printer 关键词、location 缺失和时间趋势。数据中 category 的 Hardware 数量是 406，而 assignment_group 的 Hardware 数量是 405；不同系统选择不同字段时，可能得到语义合理但与参考措辞不一致的结果。

这说明当前分数衡量的是“对一组稀疏参考 insight 的覆盖”，不是完整的数据分析质量。参考答案还包含解释性较弱或相互含混的叙述，因此不能把本地 judge 分数当作客观真值。

论文中应明确采用三角验证：

1. 冻结 LLM judge，用于大规模、低成本的相对比较；
2. 不含额外 LLM 调用的确定性指标，用于核对数字、重复、长度和表面覆盖；
3. 人工盲评，用于最终确认正确性、相关性、证据充分性和行动价值。

## 7. 论文框架建议

### 研究问题

- RQ1：固定问题预算下，层级式假设驱动探索能否比平铺式探索获得更好的 insight 覆盖与证据链？
- RQ2：skill 化的分析策略能否跨 case、跨 benchmark 稳定迁移，而不是只在已看过答案的开发集上提升？
- RQ3：不同框架的主要差异来自平均质量、生成稳定性、运行可靠性还是成本效率？
- RQ4：LLM judge、确定性指标与人工评价之间的一致性有多高？

### 可成立的贡献

- 将早期 DataSTORM 复刻工程重构为固定预算、可复现、可审计的统一实验协议；
- 引入官方 AgentPoirot 实现作为外部框架基线，而不是只和自己的复刻系统比较；
- 将轨迹、输出与评分随机性分离，报告 case-level 配对统计和失败率；
- 在源 benchmark 与新 benchmark 上检验 skill 的迁移，而不是只报告前 10 个开发 case；
- 分析自动评价的有效边界，并用盲评校准。

不建议把“超过某个基线若干分”设为论文唯一贡献，因为当前结果与评估条件都不足以支持这种叙事。

## 8. 下一轮最小必要升级

按优先级执行：

1. **补齐统一观测**：给 DataSTORM 和 AgentPoirot 的 API 客户端记录 calls、cached/uncached input、output、耗时和重试；结构化记录代码执行失败与 fallback。没有这一步就不能比较效率。
2. **修复中间证据截断**：对宽表和长查询结果做确定性的 group summary、top anomaly 和截断标记，尤其防止 DataSTORM 因只看到前若干行而漏掉异常。
3. **做小规模人工盲评校准**：随机打乱系统名，对本轮 16 个成功输出按正确性、相关性、证据充分性、非冗余和行动价值评分；同时标注参考答案漏项。此项放在下一阶段执行。
4. **迁移到真正未见的 target-test**：以 InsightEval 官方数据为主，先做 20 个 case × 3 个 agent seeds × 4 系统；若资源允许再扩到完整测试集。每个输出先保留 1 次 judge，只有结论敏感或边界样本再做第二次。
5. **只做必要消融**：PI-core、PI+完整 skills、去掉层级追问、去掉证据压缩四组即可。不要同时扫大量 prompt 超参。
6. **统计以 case 为单位**：报告 paired difference、case bootstrap、成功率和 agent-run 方差；把 seed 作为重复测量，不把 judge 重复伪装成独立样本。
7. **最终再做人工盲评**：根据 target-test 自动结果分层抽样，验证自动指标是否真的与人类判断一致。

进入下一轮的门槛：统一 usage 记录通过测试、所有系统能在固定预算下 dry-run、至少 3 个 smoke cases 无结构性失败、冻结模型/提示/评分器版本后再正式跑数。

## 9. 当前可写与不可写的结论

可以写：实验框架已具备复现和外部基线比较能力；层级轨迹提供了可审计的因果追问过程；随机性是当前主要不确定性来源；DataSTORM 和 AgentPoirot 暴露了不同的工程失败模式；V4.1 Flash 使统一模型条件下的大规模实验成本可控。

不能写：PI 或 skills 已显著优于基线；DataSTORM/AgentPoirot 已显著优于 PI；两 case 的均值能代表整套 benchmark；本地 LLM judge 分数等同于真实分析质量。
