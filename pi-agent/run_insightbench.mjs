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
const gtSummary = meta.summary ?? "";

console.log(`=== ${flagId} | layers=${layers} | pool=${poolSize} ===`);
console.log(`goal: ${goal.slice(0, 120)}`);
console.log(`GT insights: ${gtInsights.length} | GT summary: ${gtSummary.length} chars`);

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
  useSkills: process.env.USE_SKILLS !== "0",
  onLog: (m) => console.log("  ·", m),
});
const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

// pred insight 的两种口径：
//   raw  —— 每个探索节点的「问题 + 答案」，与 Python adapter.py 的 raw 模式一致
//   bank —— 只用 insight_bank 去重择优后的条目
// precision 的分母是 pred 条数，所以两者会给出很不同的 precision。
// 两个都算，用数据决定该采哪个口径。
const rawPred = result.nodes
  .filter((n) => n.answer && !n.answer.startsWith("Execution failed"))
  .map((n) => `${n.question} ${n.answer}`.trim());

const byId = new Map(result.nodes.map((n) => [n.id, n]));
const bankPred = Object.entries(result.insightBank).map(([id, text]) => {
  const n = byId.get(id);
  // 保留问题上下文，与 raw 口径同形态，只是条数更少
  return n ? `${n.question} ${text}`.trim() : text;
});

const predMode = process.env.PRED_MODE ?? "bank";
const predInsights = predMode === "bank" && bankPred.length ? bankPred : rawPred;

console.log(`\nexplored in ${elapsed}s`);
console.log(`layers run: ${result.layersRun}${result.stoppedEarly ? " (stopped early)" : ""}`);
console.log(`pred: raw=${rawPred.length} bank=${bankPred.length} → scoring "${predMode}" (${predInsights.length})`);
console.log(`thesis: ${result.thesis?.title ?? "(none)"}`);
console.log("usage:", JSON.stringify(result.usage));

// —— 交给 Python 评分器 ——
mkdirSync(outDir, { recursive: true });
const payload = `${outDir}/${flagId}_pred.json`;
writeFileSync(
  payload,
  JSON.stringify({
    pred_insights: predInsights,
    pred_insights_raw: rawPred,
    pred_insights_bank: bankPred,
    gt_insights: gtInsights,
    pred_summary: result.summary,
    gt_summary: gtSummary,
  }, null, 2),
);

console.log(`\nscoring both pred modes × ${gtInsights.length} GT + summary …`);
const scoreScript = `
import json, sys
sys.path.insert(0, "${REPO}/run_on_benchmark")
from unified_scorer import score_insight_matrix, score_summary, get_usage_stats, get_scorer_config
d = json.load(open("${payload}"))
out = {}
for mode in ("raw", "bank"):
    preds = d["pred_insights_" + mode]
    if not preds:
        continue
    r = score_insight_matrix(preds, d["gt_insights"])
    out[mode] = {"recall": r["recall"], "precision": r["precision"],
                 "f1": r["f1"], "n_pred": len(preds), "matrix": r["matrix"]}
out["summary"] = score_summary(d["pred_summary"], d["gt_summary"]) if d["pred_summary"] and d["gt_summary"] else 0.0
out["scorer"] = get_scorer_config()
out["scorer_usage"] = get_usage_stats()
print(json.dumps(out))
`;
const scoreOut = execFileSync(PY, ["-c", scoreScript], {
  encoding: "utf8",
  maxBuffer: 64 * 1024 * 1024,
});
const scores = JSON.parse(scoreOut.trim().split("\n").pop());
const primary = scores[predMode] ?? scores.raw;

const out = {
  flag: flagId,
  goal,
  layers,
  layers_run: result.layersRun,
  stopped_early: result.stoppedEarly,
  pred_mode: predMode,
  elapsed_sec: Number(elapsed),
  insights_recall: primary.recall,
  insights_precision: primary.precision,
  insights_f1: primary.f1,
  score_summary: scores.summary,
  scores_by_mode: {
    raw: scores.raw && { recall: scores.raw.recall, precision: scores.raw.precision, f1: scores.raw.f1, n_pred: scores.raw.n_pred },
    bank: scores.bank && { recall: scores.bank.recall, precision: scores.bank.precision, f1: scores.bank.f1, n_pred: scores.bank.n_pred },
  },
  n_pred_insights: predInsights.length,
  n_gt_insights: gtInsights.length,
  n_insight_bank: Object.keys(result.insightBank).length,
  thesis: result.thesis,
  skill_package: result.skillPackage,
  agent_usage: result.usage,
  scorer: scores.scorer,
  scorer_usage: scores.scorer_usage,
  pred_summary: result.summary,
  gt_summary: gtSummary,
  pred_insights: predInsights,
  pred_insights_raw: rawPred,
  pred_insights_bank: bankPred,
  gt_insights: gtInsights,
  insight_bank: result.insightBank,
  score_matrix: primary.matrix,
  nodes: result.nodes,
};
writeFileSync(`${outDir}/${flagId}_result.json`, JSON.stringify(out, null, 2));

console.log("\n" + "=".repeat(70));
for (const mode of ["raw", "bank"]) {
  const s = scores[mode];
  if (!s) continue;
  const mark = mode === predMode ? " ←" : "";
  console.log(
    `${flagId} [${mode.padEnd(4)}] n=${String(s.n_pred).padStart(2)}  ` +
      `recall=${s.recall.toFixed(4)}  precision=${s.precision.toFixed(4)}  F1=${s.f1.toFixed(4)}${mark}`,
  );
}
console.log(`${flagId} summary=${scores.summary.toFixed(4)}`);
console.log(`elapsed=${elapsed}s  cost=$${result.usage.cost_usd}  ` +
  `agent ${result.usage.calls} calls, cache ${(result.usage.cache_hit_rate * 100).toFixed(0)}%`);
console.log(`saved → ${outDir}/${flagId}_result.json`);
console.log("=".repeat(70));
