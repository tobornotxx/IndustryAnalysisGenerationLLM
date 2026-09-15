import test from "node:test";
import assert from "node:assert/strict";
import {
  canonicalizeDeepSeekModel,
  deepSeekV41FlashCost,
  isDeepSeekPeak,
  makeDeepSeekV41FlashModel,
} from "../src/model-config.mjs";

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
