# 毕业论文最小必要升级：实现状态与实验协议

更新日期：2026-09-15  
实现分支：`codex/thesis-experiment-upgrades`  
原则：先建立可追溯、可复现、可统计的证据链，再讨论分数是否提升。

## 1. 已完成的最小升级

### 模型与评分边界

- 新实验统一记录规范模型名 `deepseek-flash`，兼容旧别名但会明确告警。
- 业务生成模型与本地 DeepSeek 评分器分开配置，换业务模型不会暗中换 Judge。
- 价格估算区分北京时间高峰/非高峰，并使用 DeepSeek V4.1 Flash 当前价格。

### 不可覆盖的实验记录

- 生成、评分、汇总已经拆开；重复评分不会重新运行 Agent。
- 每次生成保存 `manifest.json`、`prediction.json`、`trajectory.jsonl`、`usage.json`。
- 路径固定为：

  ```text
  results/experiments/<experiment>/<system>/<case>/agent_run_<n>/
  ```

- 已存在的运行目录和评分文件默认拒绝覆盖；失败状态与错误也保留。
- manifest 记录仓库/基准提交、dirty 状态、模型、prompt/skill hash、预算和重复编号。

### 无额外 LLM 调用的指标与统计

- 本地指标：双向 token overlap、ROUGE-1、ROUGE-L、数字事实覆盖、输出数量/长度/重复率。
- 正式本地 Judge 协议固定为 `local-deepseek-v41-thinking-v1`；显式开启思考并给推理过程保留 4096 个输出 token。`local-deepseek-v41-nonthinking-v1` 只保留为一次无效的诊断协议，不得进入论文主结果。
- 官方 InsightEval 指标：直接加载作者仓库的 ROUGE-1 最佳匹配 recall、precision、Insight F1。
- 聚合顺序为 Judge 重复 → Agent 重复 → case，避免把一次随机波动当成独立样本。
- 汇总包含 case-level bootstrap 95% CI、配对置换检验、Cohen's dz、Agent 方差、Judge 方差和失败状态。

### 重复实验、消融与数据防污染

- 支持生成重复矩阵与评分重复矩阵，支持断点续跑，失败不会被静默删除。
- 支持固定 `max_questions` 和 `max_insights`。
- 可独立控制技能包、Insight Bank、goal sufficiency；系统名区分 `pi-core` 与 `pi-core-skills`。
- 数据用途只能从以下五类选择：

  - `dev-contaminated`：已反复查看的开发 case，不能用于泛化结论；
  - `source-train`：仅用于轨迹提炼；
  - `source-valid`：决定候选改动是否保留；
  - `source-test`：冻结后的一次性源域测试；
  - `target-test`：完全不用于调参的迁移测试。

- 默认把 InsightBench 运行标记为 `dev-contaminated`；InsightEval 强制为 `target-test`，传入其他用途会失败。

### 对比系统与开源实现

- DataSTORM：使用本仓库已有的论文引导复刻版，并修复了旧适配器参数与当前接口不一致的问题。
- PI：比较关闭/开启冻结技能包的 `pi-core` 与 `pi-core-skills`。
- AgentPoirot：固定 ServiceNow 官方 Apache-2.0 实现提交 `a731fb1397d0e5e8716681fb7ee45797c7ed110f`；只替换 GPT-only 传输层以连接同一 DeepSeek OpenAI-compatible API，不改官方编排和提示词。
- InsightBench：固定官方 `overhaul` 分支提交 `f29a9a6f2f71ac831d9815919147d62623ab7df4`。
- InsightEval：固定论文作者实现提交 `7bf3e160bb4eefb4cf6a0c878f0e5656bdcdf6be`，直接复用其本地指标。该仓库当前未声明代码许可证，因此这里只保留 Git submodule 引用；公开分发或复制代码前需再次确认许可。

### 技能提炼治理（MyDataStorm 分支）

- 只能从成功且标记为 `source-train` 的不可变运行中提炼 episode。
- 每条 episode 保留 run、case、commit、prompt hash、skill hash 和轨迹来源。
- 冻结前检查运行时字段污染，并要求至少两个不同 `source-valid` case 的配对消融都为正向。
- 现有手工候选技能不会被自动冒充为“已验证冻结技能”。

## 2. 建议的正式实验矩阵

数据清单已冻结在 [`run_on_benchmark/splits/thesis_split_v1.json`](../run_on_benchmark/splits/thesis_split_v1.json)：`flag-1` 至 `flag-10` 为 `dev-contaminated`，第一轮 `source-valid` 为 `flag-11`、`flag-12`。

先做不调用 API 的 dry-run，确认路径、模型、split 和预算。之后正式运行建议分三步。

### A. 小规模方差与成本试验

- 系统：DataSTORM、PI-core、PI-core-skills、AgentPoirot。
- 数据：2–4 个未参与具体调参的 `source-valid` case。
- 重复：每个系统/case 先做 2 次 Agent；每个固定预测做 2 次 Judge。
- 目的：估计 Agent/Judge 方差和实际成本，不对分数下论文结论。

### B. 源域主实验

- 冻结模型、prompt、技能包、问题预算和 scorer 配置。
- 在 `source-test` 上每组至少 3 次 Agent × 3 次 Judge。
- 先按 case 聚合，再做配对比较；同时报告完成率、成本和确定性指标。
- 主比较：DataSTORM vs PI-core vs PI-core-skills；AgentPoirot 作为外部开源基线。

### C. 目标域迁移

- 不看 InsightEval reference insight 做任何调参。
- 四个系统用同一生成模型、问题预算和重复数跑 `target-test`。
- 使用作者官方本地 Insight F1 作为可复现主指标；本地 DeepSeek Judge 可作为补充证据，但不得伪装成论文官方 G-Eval。
- 目标域跑完后不再改方法并重跑同一测试集。

## 3. 常用无 API 验证命令

以下命令只检查计划和本地代码，不调用模型服务。

```powershell
# PI 在 InsightBench 上的计划（默认明确标记为被污染开发集）
node pi-agent/generate_insightbench.mjs --flag 11 --dry-run

# PI 在 InsightEval 上的目标域计划
node pi-agent/generate_insightbench.mjs --benchmark-kind insighteval --instance 1 --dry-run

# AgentPoirot / DataSTORM 的统一生成计划
python -m run_on_benchmark.generate_baseline `
  --system agentpoirot-official `
  --benchmark-kind insighteval `
  --benchmark-dir run_on_benchmark/InsightEval-official `
  --case 1 --dry-run

# 对一份固定 InsightEval 预测执行作者官方本地指标并拒绝覆盖结果
python -m run_on_benchmark.insighteval_adapter `
  --instance 1 `
  --prediction <prediction.json> `
  --output <insighteval_score.json>

# 回归测试
python -m unittest discover -s tests -v
npm --prefix pi-agent test
```

## 4. 论文主线与结论边界

这套实现支持的稳健主线不是“每个 case 都显著涨分”，而是：

1. 在官方实现不完整或工程细节未公开时，构建可审计的开放式数据洞察系统与复现实验协议；
2. 分析问题树、目标充分性、Insight Bank 和可迁移技能对覆盖、冗余、可靠性、成本的影响；
3. 区分 Agent 随机性、Judge 随机性与真实方法差异；
4. 用语义评分、确定性指标、官方目标域指标和后续人工盲评形成多证据链；
5. 检验源域提炼的技能是否能迁移，而不把“必须显著提升”作为论文成立的前提。

如果 PI-core-skills 没有稳定胜过 PI-core，仍可如实得到有价值的结论：自动提炼技能的收益受任务、预算或轨迹质量约束；系统与评估框架本身仍是完整的工程和实证贡献。

## 5. 仍未执行且必须保留的边界

- 没有调用任何生成或评分 API，因此尚无 V4.1 Flash 正式结果。
- 没有制作最终人工盲评 `.docx`；应等正式候选结果冻结后再随机化导出。
- 没有接入 InsightBench 上游 G-Eval，也没有把本地 Judge 分数与其他论文的官方分数直接横比。
- 没有根据 InsightEval 答案改 prompt 或技能；后续也必须保持这一点。
- 正式运行前仍需由研究者确定未污染的 source-valid/source-test case 清单和可接受预算。

## 6. 下一步唯一必要的人工作业

1. 列出哪些 InsightBench case 曾被看过答案或用于调参；它们全部归入 `dev-contaminated`。
2. 从剩余 case 中一次性冻结 source-valid 与 source-test 清单。
3. 确认 2×2 小试预算后再启用 API。
4. 小试通过后冻结配置，执行 3×3 主实验与一次目标域迁移。
5. 最后生成盲评文档，由人工复核代表性样本。
