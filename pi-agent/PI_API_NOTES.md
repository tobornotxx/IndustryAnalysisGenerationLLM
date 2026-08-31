# pi API 实测记录

> 实测环境：`@earendil-works/pi-agent-core@0.84.4` / `@earendil-works/pi-ai@0.84.4`，Node v26，真实 DeepSeek API
> 目的：记录**实际验证过**的 API 形状。文档与调研推测和实际有出入，以本文为准。

---

## 安装

```bash
npm install --save-exact \
  @earendil-works/pi-agent-core@0.84.4 \
  @earendil-works/pi-ai@0.84.4 \
  @sinclair/typebox            # tool 的 parameters 需要它
```

`engines: { node: ">=22.19.0" }`。按迁移方案要求 **pin 精确版本**（0.x 的 minor 即破坏性变更）。

---

## DeepSeek provider：内置，无需自建

`deepseekProvider()` 在 **subpath** 导出，不在 `pi-ai` 顶层：

```js
import { deepseekProvider } from "@earendil-works/pi-ai/providers/deepseek";
```

内置模型目录含 **`deepseek-v4-flash`、`deepseek-v4-pro`、`deepseek-v4-flash-vision-exp`**，均 1M context。

`deepseek-v4-flash` 完整定义（实测 dump）：

```jsonc
{
  "id": "deepseek-v4-flash",
  "api": "openai-completions",
  "baseUrl": "https://api.deepseek.com",
  "reasoning": true,
  "cost": { "input": 0.14, "output": 0.28, "cacheRead": 0.0028, "cacheWrite": 0 },
  "contextWindow": 1000000,
  "maxTokens": 384000,
  "compat": {
    "supportsStore": false,
    "supportsDeveloperRole": false,
    "maxTokensField": "max_tokens",
    "requiresReasoningContentOnAssistantMessages": true,
    "thinkingFormat": "deepseek"
  },
  "thinkingLevelMap": { "minimal": null, "low": "low", "medium": null, "high": "high", "max": "max" }
}
```

`compat` 里那几项正是手写 OpenAI-compat 客户端会静默出错的地方，**已经预置好，不用我们管**。

### ⚠️ 一个对成本有实际影响的发现

`cacheRead: 0.0028` vs `input: 0.14` —— **缓存命中的 token 只要 1/50 价格**。

这意味着 `TRANSFER_DESIGN.md` §6.3 做的 `pred_first` 前缀优化（缓存前缀占比 ~100%）
价值比原先估计的大得多：命中部分近乎免费。迁移后 pi 侧也应保持消息前缀稳定
（system prompt + schema 在前，变化的探索历史在后）。

认证走环境变量 **`DEEPSEEK_API_KEY`**（`envApiKeyAuth`）。

---

## 正确的最小骨架

```js
import { Agent } from "@earendil-works/pi-agent-core";
import { createModels } from "@earendil-works/pi-ai";
import { deepseekProvider } from "@earendil-works/pi-ai/providers/deepseek";

const models = createModels();              // 不接受 { providers: [...] }
models.setProvider(deepseekProvider());     // 用 setProvider 注册
const model = models.getModel("deepseek", "deepseek-v4-flash");   // 同步，非 async

const agent = new Agent({
  initialState: { systemPrompt, model, tools: [myTool] },
  streamFn: (m, ctx, opts) => models.streamSimple(m, ctx, opts),  // 必须显式提供
});

const unsub = agent.subscribe((ev) => { /* ev.type ... */ });
await agent.prompt("...");                  // 返回 Promise<void>，不是 async iterable
unsub();
```

### 踩过的四个坑

| 我以为 | 实际 |
|---|---|
| `createModels({ providers: [...] })` | 空构造 + `models.setProvider(p)` |
| `await models.get(...)` | `models.getModel(provider, id)`，**同步** |
| `Agent` 自带默认 stream | **必须传 `streamFn`**，否则抛 "No default stream function configured"。`models.streamSimple` 的签名与 `StreamFn` 完全一致，直接桥接 |
| `for await (const ev of agent.prompt(...))` | `prompt` 返回 `Promise<void>`；事件走 `agent.subscribe(cb)` |

---

## Tool 定义（`AgentTool`）

```js
import { Type } from "@sinclair/typebox";

const runPython = {
  name: "run_python",
  label: "Run Python",                        // 必填，UI 显示用
  description: "...",
  parameters: Type.Object({                   // TypeBox schema，不是裸 JSON Schema
    code: Type.String({ description: "..." }),
  }),
  async execute(_toolCallId, params, signal, onUpdate) {
    //          ^^^^^^^^^^^ 第一个参数是 toolCallId，参数在第二个！
    const out = execFileSync(PY, ["-c", params.code], { encoding: "utf8", signal });
    return {
      content: [{ type: "text", text: out.trim() || "(no output)" }],
      details: { code: params.code },         // 结构化信息，给日志/UI，不进模型
    };
  },
  // executionMode?: "sequential" | "parallel"   ← 逐 tool 覆盖并发模式
};
```

**最容易错的一点**：`execute(toolCallId, params, ...)`。我第一版把参数当第一个形参，
结果 `params.code` 是 `undefined`，Python 收到字面量 `undefined` 报 `NameError`。

**错误处理**：接口注释明确要求 **throw**，不要把错误编码进 `content`。
实测 throw 后模型能看到错误并自行重试（下面有观察）。

---

## 事件类型（实测全集）

```
agent_start → turn_start → message_start → message_update … → message_end
            → tool_execution_start → tool_execution_end
            → turn_end → agent_end
```

注意是 `tool_execution_start` / `tool_execution_end`，**不是** `tool_start` / `tool_end`。

---

## Agent 的钩子（实测存在，迁移方案要用的都在）

从 `Agent` 类公开成员实测：

```
prompt  continue  abort  subscribe  reset  waitForIdle
shouldStopAfterTurn      ← goal_sufficiency 早停挂这里
transformContext         ← insight_bank 去重挂这里
beforeToolCall  afterToolCall
convertToLlm             ← 自定义消息类型（insights/thesis 留在 transcript 但不发给模型）
toolExecution            ← "sequential" | "parallel"
followUp  steer  followUpMode  steeringMode
maxRetryDelayMs  thinkingBudgets  onResponse  onPayload
sessionId  initialState  streamFn
```

`PI_MIGRATION_PLAN.md` §3 计划映射的钩子**全部真实存在**。

---

## 内置 tools 可完全替换（已验证）

`pi-agent-core` 导出 `createBashTool` / `createReadTool` / `createWriteTool` / `createEditTool`
—— 都是**工厂函数，opt-in**，不传就没有。实测只给 `tools: [runPython]` 时，
模型可用工具就只有 `run_python`。

---

## 通用错误恢复（这是换 harness 的主要动因，已观察到）

第一次跑（tool 参数拿错，Python 每次都 `NameError`）时，**模型连续重试了 5 次**，
每次都拿到 traceback 后调整。这是手写 loop 没有的行为 —— 我们之前是遇到一个
失败 case 就手工打补丁。

---

## 端到端验证结果

`probe-minimal.mjs`，真实 API：

```
MODEL: deepseek-v4-flash | ctx: 1000000
→ TOOL_START: run_python
→ TOOL_END: {"content":[{"type":"text","text":"Mean: 3.875\nStandard Deviation: 2.748376143938713"}], ...}
FINAL: Mean 3.875 / Std ~2.7484（并主动说明用的是样本标准差 ddof=1）
```

结论：**真实 DeepSeek API + agent loop + 自定义 Python tool 闭环可用。**

---

## 尚未验证

| 项 | 状态 |
|---|---|
| `loadSkills` + SKILL.md + `scripts/` 执行 | 调研验证过，本地未测（步骤 7） |
| 长驻 Python worker（行分隔 JSON） | 调研验证过，本地未测（步骤 2-3） |
| worker 池 + 共享 SQLite 文件 | 未实现（步骤 3） |
| `shouldStopAfterTurn` / `transformContext` 实际行为 | 成员存在，未实测（步骤 5） |
| 并发 tool 调用（`executionMode: "parallel"`） | 未测 —— 关系到同层多问题并发 |
| pi 的 usage / token 统计接口 | 未查 —— §2.5 要求保留记账能力 |
