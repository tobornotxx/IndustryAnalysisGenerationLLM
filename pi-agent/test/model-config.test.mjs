import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  canonicalizeDeepSeekModel,
  DEFAULT_REASONING_EFFORT,
  deepSeekV41FlashCost,
  isDeepSeekPeak,
  makeDeepSeekV41FlashModel,
} from "../src/model-config.mjs";
import {
  createDeepSeek,
  findDeepSeekApiKey,
  makeGenerateJson,
  makeGenerateText,
} from "../src/agent.mjs";

test("retired DeepSeek aliases resolve to the canonical V4.1 Flash ID", () => {
  assert.equal(canonicalizeDeepSeekModel("deepseek-v4-pro"), "deepseek-flash");
  assert.equal(canonicalizeDeepSeekModel("deepseek-v4-flash"), "deepseek-flash");
  assert.equal(canonicalizeDeepSeekModel("custom-model"), "custom-model");
});

test("official peak windows use peak pricing", () => {
  assert.equal(isDeepSeekPeak(new Date("2026-09-15T01:30:00Z")), true); // Tue 09:30 CST
  assert.equal(isDeepSeekPeak(new Date("2026-09-15T04:30:00Z")), false); // Tue 12:30 CST
  assert.deepEqual(deepSeekV41FlashCost(new Date("2026-09-15T01:30:00Z")), {
    input: 0.3,
    output: 1.2,
    cacheRead: 0.006,
    cacheWrite: 0,
  });
});

test("current model metadata uses the canonical ID without mutating catalog data", () => {
  const legacy = { id: "deepseek-v4-flash", name: "old", input: ["text"], cost: {} };
  const current = makeDeepSeekV41FlashModel(legacy, new Date("2026-09-15T04:30:00Z"));
  assert.equal(current.id, "deepseek-flash");
  assert.deepEqual(current.input, ["text", "image"]);
  assert.equal(current.cost.output, 0.6);
  assert.equal(legacy.id, "deepseek-v4-flash");
});

test("DeepSeek calls default to explicit thinking mode", () => {
  const handle = createDeepSeek();
  assert.equal(DEFAULT_REASONING_EFFORT, "medium");
  assert.equal(handle.reasoning, "medium");
  assert.equal(handle.model.reasoning, true);
});

test("DeepSeek key prefers env and falls back to the project legacy config", () => {
  const root = mkdtempSync(join(tmpdir(), "pi-deepseek-config-"));
  const nested = join(root, ".worktrees", "industry");
  const configDir = join(root, "MyDataStorm", "datastorm");
  mkdirSync(nested, { recursive: true });
  mkdirSync(configDir, { recursive: true });
  writeFileSync(join(configDir, "llm_config.json"), JSON.stringify({ api_key: "legacy-key" }));

  assert.equal(findDeepSeekApiKey({ env: {}, startDir: nested }), "legacy-key");
  assert.equal(
    findDeepSeekApiKey({ env: { DEEPSEEK_API_KEY: "env-key" }, startDir: nested }),
    "env-key",
  );
});

function fakeModels(events) {
  return {
    streamSimple() {
      return (async function* stream() { for (const event of events) yield event; })();
    },
  };
}

test("generation helpers fail closed on provider errors and empty responses", async () => {
  const failed = { stopReason: "error", errorMessage: "Connection error", content: [] };
  await assert.rejects(
    makeGenerateJson({ models: fakeModels([{ type: "error", message: failed }]), model: {} })("x"),
    /Connection error/,
  );
  const empty = { stopReason: "stop", content: [] };
  await assert.rejects(
    makeGenerateText({ models: fakeModels([{ type: "done", message: empty }]), model: {} })("x"),
    /no text/,
  );
});
