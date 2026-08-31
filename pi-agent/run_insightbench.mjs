/**
 * 在 InsightBench 上跑 pi-agent 并用现有 Python 评分器打分。
 *
 * 评分刻意仍用 Python 的 unified_scorer（PI_MIGRATION_PLAN §4）：它是离线工具，
 * 不参与 agent 运行；沿用同一把尺子才能和 v8/v10 的历史结果对比。
 *
 * 用法：
 *   set -a && . ./.env && set +a
 *   node run_insightbench.mjs --flag 1 --layers 3 --out ../results/pi_v1
 */
import { writeFileSync, mkdirSync, readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { explore } from "./src/agent.mjs";

const REPO = "/Users/liulife/llf-study/IndustryAnalysisGenerationLLM";
const BENCH = `${REPO}/run_on_benchmark/insight-bench`;
const PY = "/Users/liulife/llf-study/.venv/bin/python";

function arg(name, dflt) {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : dflt;
}

const flagNum = Number(arg("flag", 1));
const layers = Number(arg("layers", 3));
const outDir = arg("out", `${REPO}/results/pi_v1`);
const poolSize = Number(arg("pool", 4));

// —— 读 InsightBench 的数据集定义 ——
const flagId = `flag-${flagNum}`;
const meta = JSON.parse(
  readFileSync(`${BENCH}/data/notebooks/${flagId}.json`, "utf8"),
);
const goal = meta.metadata?.goal ?? "Find interesting trends in this dataset";
const csvPath = `${BENCH}/${meta.dataset_csv_path}`;
const userCsv = meta.user_dataset_csv_path ? `${BENCH}/${meta.user_dataset_csv_path}` : null;
const gtInsights = meta.insights ?? [];

console.log(`=== ${flagId} | layers=${layers} | pool=${poolSize} ===`);
console.log(`goal: ${goal.slice(0, 120)}`);
console.log(`GT insights: ${gtInsights.length}`);

const t0 = Date.now();
const result = await explore({
  csvPath,
  userCsvPath: userCsv,
  tableName: "incidents",
  goal,
  maxLayers: layers,
  questionsPerLayer: 2,
  poolSize,
  pythonBin: PY,
  workerScript: new URL("./python/worker.py", import.meta.url).pathname,
  onLog: (m) => console.log("  ·", m),
});
const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

// pred insight = 每个探索节点的「问题 + 答案」，与 Python 侧 raw 模式一致，
// 保持可比（见 adapter.py 的 _extract_insights）
const predInsights = result.findings
  .filter((f) => f.answer && !f.answer.startsWith("Execution failed"))
  .map((f) => `${f.question} ${f.answer}`.trim());

console.log(`\nexplored in ${elapsed}s → ${predInsights.length} pred insights`);
console.log("usage:", JSON.stringify(result.usage));

// —— 交给 Python 评分器 ——
mkdirSync(outDir, { recursive: true });
const payload = `${outDir}/${flagId}_pred.json`;
writeFileSync(
  payload,
  JSON.stringify({ pred_insights: predInsights, gt_insights: gtInsights }, null, 2),
);

console.log(`\nscoring ${predInsights.length} × ${gtInsights.length} pairs …`);
const scoreScript = `
import json, sys
sys.path.insert(0, "${REPO}/run_on_benchmark")
from unified_scorer import score_insight_matrix, get_usage_stats, get_scorer_config
d = json.load(open("${payload}"))
r = score_insight_matrix(d["pred_insights"], d["gt_insights"])
print(json.dumps({
    "recall": r["recall"], "precision": r["precision"], "f1": r["f1"],
    "n_pairs": r["n_pairs"], "matrix": r["matrix"],
    "scorer": get_scorer_config(), "scorer_usage": get_usage_stats(),
}))
`;
const scoreOut = execFileSync(PY, ["-c", scoreScript], {
  encoding: "utf8",
  maxBuffer: 64 * 1024 * 1024,
});
const scores = JSON.parse(scoreOut.trim().split("\n").pop());

const out = {
  flag: flagId,
  goal,
  layers,
  elapsed_sec: Number(elapsed),
  insights_recall: scores.recall,
  insights_precision: scores.precision,
  insights_f1: scores.f1,
  n_pred_insights: predInsights.length,
  n_gt_insights: gtInsights.length,
  agent_usage: result.usage,
  scorer: scores.scorer,
  scorer_usage: scores.scorer_usage,
  pred_insights: predInsights,
  gt_insights: gtInsights,
  score_matrix: scores.matrix,
  nodes: result.nodes,
};
writeFileSync(`${outDir}/${flagId}_result.json`, JSON.stringify(out, null, 2));

console.log("\n" + "=".repeat(58));
console.log(`${flagId}  recall=${scores.recall.toFixed(4)}  precision=${scores.precision.toFixed(4)}  F1=${scores.f1.toFixed(4)}`);
console.log(`pred=${predInsights.length}  GT=${gtInsights.length}  elapsed=${elapsed}s  cost=$${result.usage.cost_usd}`);
console.log(`agent: ${result.usage.calls} calls, cache hit ${(result.usage.cache_hit_rate * 100).toFixed(0)}%`);
console.log(`saved → ${outDir}/${flagId}_result.json`);
console.log("=".repeat(58));
