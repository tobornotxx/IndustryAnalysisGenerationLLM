/**
 * 批量跑多个 flag 并汇总。
 *
 * 断点续跑：已存在 <flag>_result.json 的跳过（跑一次几分钟且花钱，
 * 中途失败不该全部重来）。
 *
 * 用法：
 *   set -a && . ./.env && set +a
 *   node run_batch.mjs --flags 1,2,3,4,5 --layers 5 --out ../results/pi_v2
 */
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { spawnSync } from "node:child_process";

function arg(name, dflt) {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : dflt;
}

const flags = arg("flags", "1,2,3,4,5").split(",").map(Number);
const layers = arg("layers", "5");
const outDir = arg("out", "../results/pi_v2");
const force = process.argv.includes("--force");

mkdirSync(outDir, { recursive: true });
console.log(`=== batch: flags=[${flags}] layers=${layers} → ${outDir} ===\n`);

const t0 = Date.now();
for (const f of flags) {
  const resultPath = `${outDir}/flag-${f}_result.json`;
  if (!force && existsSync(resultPath)) {
    console.log(`--- flag-${f}: 已有结果，跳过 ---\n`);
    continue;
  }
  console.log(`--- flag-${f} ---`);
  const r = spawnSync(
    process.execPath,
    ["run_insightbench.mjs", "--flag", String(f), "--layers", layers, "--out", outDir],
    { stdio: "inherit", env: process.env },
  );
  if (r.status !== 0) console.log(`!!! flag-${f} 失败 (exit ${r.status})，继续下一个\n`);
  else console.log("");
}

// ── 汇总 ──
const rows = [];
for (const f of flags) {
  const p = `${outDir}/flag-${f}_result.json`;
  if (!existsSync(p)) continue;
  try {
    rows.push(JSON.parse(readFileSync(p, "utf8")));
  } catch {
    /* 跳过损坏的 */
  }
}

if (!rows.length) {
  console.log("没有可汇总的结果");
  process.exit(0);
}

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const std = (xs) => {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
};

console.log("=".repeat(84));
console.log("汇总".padStart(42));
console.log("=".repeat(84));
console.log(
  `${"flag".padEnd(8)}${"n_raw".padStart(6)}${"n_bank".padStart(7)}` +
    `${"R(bank)".padStart(9)}${"P(bank)".padStart(9)}${"F1(bank)".padStart(10)}` +
    `${"F1(raw)".padStart(9)}${"summary".padStart(9)}${"$".padStart(8)}${"层".padStart(4)}`,
);
console.log("-".repeat(84));
for (const r of rows) {
  const b = r.scores_by_mode?.bank ?? {};
  const w = r.scores_by_mode?.raw ?? {};
  console.log(
    `${r.flag.padEnd(8)}${String(w.n_pred ?? "-").padStart(6)}${String(b.n_pred ?? "-").padStart(7)}` +
      `${(b.recall ?? 0).toFixed(4).padStart(9)}${(b.precision ?? 0).toFixed(4).padStart(9)}` +
      `${(b.f1 ?? 0).toFixed(4).padStart(10)}${(w.f1 ?? 0).toFixed(4).padStart(9)}` +
      `${(r.score_summary ?? 0).toFixed(4).padStart(9)}` +
      `${(r.agent_usage?.cost_usd ?? 0).toFixed(3).padStart(8)}` +
      `${String(r.layers_run ?? "-").padStart(4)}${r.stopped_early ? "*" : ""}`,
  );
}
console.log("-".repeat(84));

const agg = (path) => rows.map((r) => path(r)).filter((x) => typeof x === "number");
const bankF1 = agg((r) => r.scores_by_mode?.bank?.f1);
const rawF1 = agg((r) => r.scores_by_mode?.raw?.f1);
const bankR = agg((r) => r.scores_by_mode?.bank?.recall);
const bankP = agg((r) => r.scores_by_mode?.bank?.precision);
const sums = agg((r) => r.score_summary);
const costs = agg((r) => r.agent_usage?.cost_usd);

const fmt = (xs) => `${mean(xs).toFixed(4)} ± ${std(xs).toFixed(4)}`;
console.log(`n = ${rows.length} cases`);
console.log(`  recall   (bank)  ${fmt(bankR)}`);
console.log(`  precision(bank)  ${fmt(bankP)}`);
console.log(`  F1       (bank)  ${fmt(bankF1)}`);
console.log(`  F1       (raw )  ${fmt(rawF1)}`);
console.log(`  summary          ${fmt(sums)}`);
console.log(`  总成本            $${costs.reduce((a, b) => a + b, 0).toFixed(3)}`);
console.log(`  总耗时            ${((Date.now() - t0) / 60000).toFixed(1)} 分钟`);
console.log("=".repeat(84));
console.log("\n⚠️ 方差是这次的关键产出：case 间差异（如 flag-1 历史极差 0.63 vs");
console.log("   flag-3 仅 0.033）远大于版本间差异，没有方差就无法判断提升是否真实。");

writeFileSync(
  `${outDir}/summary.json`,
  JSON.stringify(
    {
      n_cases: rows.length,
      layers: Number(layers),
      recall_bank: { mean: mean(bankR), std: std(bankR) },
      precision_bank: { mean: mean(bankP), std: std(bankP) },
      f1_bank: { mean: mean(bankF1), std: std(bankF1) },
      f1_raw: { mean: mean(rawF1), std: std(rawF1) },
      summary: { mean: mean(sums), std: std(sums) },
      total_cost_usd: costs.reduce((a, b) => a + b, 0),
      per_flag: rows.map((r) => ({
        flag: r.flag,
        layers_run: r.layers_run,
        stopped_early: r.stopped_early,
        bank: r.scores_by_mode?.bank,
        raw: r.scores_by_mode?.raw,
        summary: r.score_summary,
        cost_usd: r.agent_usage?.cost_usd,
      })),
    },
    null,
    2,
  ),
);
console.log(`\nsaved → ${outDir}/summary.json`);
