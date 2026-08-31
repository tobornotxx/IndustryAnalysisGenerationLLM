/** 步骤 4：端到端跑通一层探索（小规模，先验证可用性）。 */
import { explore } from "../src/agent.mjs";
const R = "/Users/liulife/llf-study/IndustryAnalysisGenerationLLM/run_on_benchmark";
const t0 = Date.now();
const r = await explore({
  csvPath: `${R}/insight-bench/data/notebooks/csvs/flag-1.csv`,
  tableName: "incidents",
  goal: "Find the discrepancy and imbalance in distribution of incidents assigned across categories",
  maxLayers: Number(process.env.LAYERS ?? 1),
  questionsPerLayer: 2,
  poolSize: 4,
  pythonBin: "/Users/liulife/llf-study/.venv/bin/python",
  workerScript: new URL("../python/worker.py", import.meta.url).pathname,
  onLog: (m) => console.log("  ·", m),
});
console.log(`\n=== ${r.findings.length} findings in ${((Date.now()-t0)/1000).toFixed(1)}s ===`);
r.findings.forEach((f, i) => {
  console.log(`\n[${i + 1}] Q: ${f.question.slice(0, 110)}`);
  console.log(`    tools: ${f.toolCalls.join(",") || "(none)"} | turns: ${f.turns}`);
  console.log(`    A: ${f.answer.slice(0, 260).replace(/\n/g, " ")}`);
});
console.log("\nusage:", JSON.stringify(r.usage));
