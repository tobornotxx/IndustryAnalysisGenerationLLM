export const DEEPSEEK_CANONICAL_MODEL = "deepseek-flash";

const RETIRED_ALIASES = new Set([
  "deepseek-v4-flash",
  "deepseek-v4-flash-vision-exp",
  "deepseek-v4-pro",
]);

export function canonicalizeDeepSeekModel(model) {
  return RETIRED_ALIASES.has(model) ? DEEPSEEK_CANONICAL_MODEL : model;
}

export function isDeepSeekPeak(date = new Date()) {
  const china = new Date(date.getTime() + 8 * 60 * 60 * 1000);
  const day = china.getUTCDay();
  const hour = china.getUTCHours();
  return day >= 1 && day <= 5 && ((hour >= 9 && hour < 12) || (hour >= 14 && hour < 18));
}

export function deepSeekV41FlashCost(date = new Date()) {
  const multiplier = isDeepSeekPeak(date) ? 2 : 1;
  return {
    input: 0.15 * multiplier,
    output: 0.6 * multiplier,
    cacheRead: 0.003 * multiplier,
    cacheWrite: 0,
  };
}

/**
 * pi-ai 0.84.4 predates the canonical V4.1 model ID.  Reuse its tested
 * DeepSeek transport metadata but send the current API model name and prices.
 */
export function makeDeepSeekV41FlashModel(legacyModel, date = new Date()) {
  if (!legacyModel) throw new Error("pi-ai DeepSeek legacy model metadata is unavailable");
  return {
    ...legacyModel,
    id: DEEPSEEK_CANONICAL_MODEL,
    name: "DeepSeek V4.1 Flash",
    input: ["text", "image"],
    cost: deepSeekV41FlashCost(date),
  };
}
