# PI Agent 自动 Skill 生命周期

这套实现把 skill 学习与最终推理解耦。所有修改先发生在 `source-train` 和
`source-valid`；只有通过验证并冻结的包才能以 `pi-auto-skills` 身份运行。

## 生命周期

```text
immutable source-train runs
  → mine
  → PI extraction meta-agent
  → generalization agent
  → candidate package
  → purity audit
  → source-valid on/off runs
  → validate
  → freeze
  → pi-auto-skills
```

### 1. 轨迹挖掘（无 API）

```powershell
npm run skills -- mine `
  --experiment-dir ../results/experiments/source_train `
  --output skills/work/corpus.json
```

只接收 manifest 中标为 `source-train`、且 frozen split registry 也确认属于
`source-train` 的成功运行。每条 episode 保留 run id、case、commit/prompt/skill
哈希、评分和完整问题轨迹。

### 2. 提取与泛化（需要 API，默认被锁住）

```powershell
npm run skills -- extract `
  --corpus skills/work/corpus.json `
  --output skills/work/candidates.json `
  --version auto-candidates-v1 `
  --model deepseek-flash `
  --reasoning medium `
  --allow-api
```

没有 `--allow-api` 时命令会直接拒绝执行。提取阶段使用真正的 PI meta-agent：
它只能通过 `inspect_episode` 读取训练 episode，并必须调用
`submit_skill_candidates` 提交候选。之后由独立 generalization agent 去除具体列、
实体、日期、数值和 benchmark 信息。

### 3. 纯度审计（无 API）

```powershell
npm run skills -- vocabulary `
  --benchmark-dir ../run_on_benchmark/insight-bench `
  --cases 1,2,3,4,5,6,7,8 `
  --output skills/work/forbidden_terms.json

npm run skills -- audit `
  --package skills/work/candidates.json `
  --forbidden-terms skills/work/forbidden_terms.json `
  --output skills/work/audit.json
```

禁词由 `source-train` 的 schema、角色和元数据自动生成；通用词（如 category、
time、count）不会因为恰好也是列名而被误杀。审计检查 runtime 文本中的
source-specific 词、字面标识符、非法 stage 和缺失 provenance。审计失败的包
不能冻结。

### 4. 跨 case 验证（结果聚合本身无 API）

验证记录中每个 skill 至少需要两个 `source-valid` case；正式设置每个 control 与
treated arm 至少三次独立生成。所有 case 的 treated-control delta 都必须为正，
并且成本比不能超过门槛。

```powershell
npm run skills -- prepare-validation `
  --package skills/work/candidates.json `
  --output-dir skills/work/validation_packages `
  --output skills/work/validation_plan.json `
  --cases 9,10,11,12 `
  --agent-runs 3

# 此命令会产生 API 调用；先用 --dry-run 审查完整任务矩阵
npm run skills:validate -- `
  --plan skills/work/validation_plan.json `
  --benchmark-dir ../run_on_benchmark/insight-bench `
  --dry-run

# 生成完成后每个输出只评分一次；评分器会调用 API，先 dry-run 审查缺失/已完成项
npm run skills:score -- `
  --plan skills/work/validation_plan.json `
  --benchmark-dir ../run_on_benchmark/insight-bench `
  --dry-run

# 生成和评分完成后，先收集每个 agent run 的配对结果
npm run skills -- collect-validation `
  --plan skills/work/validation_plan.json `
  --out-root ../results/experiments `
  --output skills/work/validation_records.json

npm run skills -- validate `
  --records skills/work/validation_records.json `
  --min-cases 2 `
  --min-runs 3 `
  --max-cost-ratio 1.5 `
  --output skills/work/validations.json
```

validation planner 会为每条候选 skill 创建只含该 skill 的不可变 package，并安排
`pi-core` 与 `pi-skill-candidate` 的配对运行。collector 先在每个 agent run 内聚合
judge，再按 case 形成 control/treated 数组，避免把 judge 重复误当独立样本。

### 5. 冻结（无 API）

```powershell
npm run skills -- freeze `
  --package skills/work/candidates.json `
  --audit skills/work/audit.json `
  --validations skills/work/validations.json `
  --version auto-v1 `
  --output skills/frozen/auto-v1.json
```

冻结包包含 structured provenance、逐 skill 验证结果、纯度审计和内容哈希。
`pi-auto-skills` 会在 dry-run 阶段拒绝任何未冻结或未验证的包。

## 三个实验身份

- `pi-core`：不加载 skill。
- `pi-manual-skills`：加载仓库内的手工候选包，只用于说明手工 guidance 的效果。
- `pi-auto-skills`：只接受 extraction agent 产出且通过验证的 frozen package。

运行时不会把所有 skill 无条件塞进 prompt。系统先按 stage 和 trigger terms 做
确定性选择，并把实际选择的 skill id、package hash 和版本写入 manifest。

## 思考模式

所有 PI agent、planner、summary 与 skill meta-agent 默认显式传入
`reasoning=medium`。这不是仅写在 manifest 中：PI SDK 会据此向 DeepSeek 发送
thinking enabled；不传该参数时 SDK 会明确关闭思考模式。
