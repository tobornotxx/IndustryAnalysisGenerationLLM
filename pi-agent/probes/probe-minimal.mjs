/**
 * 最小 agent 验证：真实 DeepSeek API + 一个自定义 Python tool。
 *
 * 验证目标（PI_MIGRATION_PLAN.md 步骤 1）：
 *   1. DeepSeek 内置 provider 能用真实 key 跑通
 *   2. 内置 coding tools 可完全替换为我们自己的工具
 *   3. tool 调用 → Python 执行 → 结果回传给模型，闭环可用
 */
import { Agent } from "@earendil-works/pi-agent-core";
import { createModels } from "@earendil-works/pi-ai";
import { deepseekProvider } from "@earendil-works/pi-ai/providers/deepseek";
import { Type } from "@sinclair/typebox";
import { execFileSync } from "node:child_process";

const PY = "/Users/liulife/llf-study/.venv/bin/python";

// —— 自定义 tool ——
// AgentTool 接口要点（从 dist/types.d.ts 实测确认）：
//   · parameters 是 TypeBox schema（不是裸 JSON Schema）
//   · execute(toolCallId, params, signal?, onUpdate?) —— 第一个参数是 id，不是参数
//   · 返回 { content: [{type:"text",text}], details }
//   · 失败应 throw，而不是把错误编码进 content
const runPython = {
  name: "run_python",
  label: "Run Python",
  description:
    "Execute a Python snippet and return its stdout. " +
    "pandas/numpy/scipy/sklearn are available. Use print() to output results.",
  parameters: Type.Object({
    code: Type.String({ description: "Python source to execute" }),
  }),
  async execute(_toolCallId, params, signal) {
    const out = execFileSync(PY, ["-c", params.code], {
      encoding: "utf8",
      timeout: 30_000,
      signal,
    });
    const text = out.trim() || "(no output)";
    return { content: [{ type: "text", text }], details: { code: params.code } };
  },
};

const models = createModels();
models.setProvider(deepseekProvider());
const model = models.getModel("deepseek", "deepseek-v4-flash");
if (!model) throw new Error("model not found");
console.log("MODEL:", model.id, "| ctx:", model.contextWindow);
console.log("COMPAT:", JSON.stringify(model.compat));
console.log("");

const agent = new Agent({
  initialState: {
    systemPrompt:
      "You are a data analysis assistant. Use the run_python tool to compute " +
      "answers rather than doing arithmetic yourself.",
    model,
    tools: [runPython],
  },
  // StreamFn 签名与 models.streamSimple 一致，直接桥接
  streamFn: (m, ctx, opts) => models.streamSimple(m, ctx, opts),
});

// 事件通过 subscribe 获取；prompt 返回 Promise<void>
const seen = [];
const unsub = agent.subscribe((ev) => {
  seen.push(ev.type);
  if (ev.type === "tool_execution_start")
    console.log("→ TOOL_START:", ev.toolName ?? ev.name ?? JSON.stringify(ev).slice(0, 120));
  else if (ev.type === "tool_execution_end")
    console.log("→ TOOL_END:", JSON.stringify(ev.result ?? ev.output ?? ev).slice(0, 220));
});

await agent.prompt(
  "Using pandas, build a DataFrame with column v = [3, 1, 4, 1, 5, 9, 2, 6] " +
    "and tell me its mean and standard deviation.",
);
unsub?.();

console.log("");
console.log("EVENT TYPES:", [...new Set(seen)].join(", "));

const msgs = agent.initialState?.messages ?? agent.state?.messages ?? [];
const last = msgs[msgs.length - 1];
console.log("");
console.log(
  "FINAL:",
  typeof last?.content === "string"
    ? last.content
    : JSON.stringify(last?.content)?.slice(0, 600),
);
