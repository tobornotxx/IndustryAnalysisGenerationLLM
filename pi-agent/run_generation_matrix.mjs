/** Schedule repeated pi-core generations with resume-by-skip semantics. */
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { buildRunDirectory, validateDataSplit } from "./src/experiment.mjs";
import { NativeSkillRuntime } from "./src/native-skills.mjs";
import { assertCaseSplit } from "./src/split-registry.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = process.env.REPO_ROOT ?? resolve(HERE, "..");
function arg(name, dflt) {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : dflt;
}
const dryRun = process.argv.includes("--dry-run");
const benchmarkKind = arg("benchmark-kind", "insightbench");
if (!["insightbench", "insighteval"].includes(benchmarkKind)) {
  throw new Error(`unsupported benchmark-kind: ${benchmarkKind}`);
}
const defaultBenchmarkDir = benchmarkKind === "insighteval"
  ? `${REPO}/run_on_benchmark/InsightEval-official`
  : `${REPO}/run_on_benchmark/insight-bench`;
const benchmarkDir = arg("benchmark-dir", process.env.BENCH_DIR ?? defaultBenchmarkDir);
const split = validateDataSplit(arg(
  "split", benchmarkKind === "insighteval" ? "target-test" : "source-train",
));
const defaultCases = benchmarkKind === "insighteval" ? "1,2"
  : split === "source-train" ? "1,2"
    : split === "source-valid" ? "9,10" : "13,14";
const caseNumbers = arg("cases", arg("flags", defaultCases)).split(",").map(Number);
const systems = arg("systems", "pi-core,pi-manual-skills").split(",").filter(Boolean);
const repeats = Number(arg("agent-runs", 3));
const experimentId = arg("experiment", "v41_baseline");
const skillDirectory = arg("skill-dir", process.env.SKILL_DIR ?? "");
const outRoot = arg("out-root", `${REPO}/results/experiments`);
const layers = arg("layers", "3");
const questions = arg("questions", "2");
const maxQuestions = arg("max-questions", "");
const poolSize = arg("pool", "4");
const maxInsights = arg("max-insights", "10");
const summarySamples = arg("summary-samples", "3");
const reasoning = arg("reasoning", "medium");
const useInsightBank = arg("use-insight-bank", "1");
const goalSufficiency = arg("goal-sufficiency", "1");

if (systems.includes("pi-auto-skills") && !skillDirectory) {
  throw new Error("pi-auto-skills matrix requires --skill-dir");
}
if (systems.includes("pi-auto-skills")) await NativeSkillRuntime.load([skillDirectory], { cwd: HERE });

if (!Number.isInteger(repeats) || repeats < 1 || caseNumbers.some((value) => !Number.isInteger(value) || value < 1)) {
  throw new Error("cases and agent-runs must be positive integers");
}
if (benchmarkKind === "insighteval" && split !== "target-test") {
  throw new Error("InsightEval is reserved for target-test");
}
const benchmarkId = benchmarkKind === "insighteval" ? "insighteval-official" : "insightbench-overhaul";
for (const caseNumber of caseNumbers) {
  const caseId = benchmarkKind === "insighteval" ? `insighteval-${caseNumber}` : `flag-${caseNumber}`;
  assertCaseSplit(benchmarkId, caseId, split);
}

const tasks = [];
for (const systemId of systems) {
  if (!["pi-core", "pi-manual-skills", "pi-auto-skills"].includes(systemId)) {
    throw new Error(`unsupported generation system: ${systemId}`);
  }
  for (const caseNumber of caseNumbers) {
    for (let agentRun = 1; agentRun <= repeats; agentRun += 1) {
      const caseId = benchmarkKind === "insighteval" ? `insighteval-${caseNumber}` : `flag-${caseNumber}`;
      const runDir = buildRunDirectory({ outRoot, experimentId, systemId, caseId, agentRun });
      let existingStatus = null;
      if (existsSync(`${runDir}/manifest.json`)) {
        try { existingStatus = JSON.parse(readFileSync(`${runDir}/manifest.json`, "utf8")).status; }
        catch { existingStatus = "corrupt"; }
      } else if (existsSync(runDir)) {
        existingStatus = "incomplete";
      }
      tasks.push({ systemId, caseNumber, caseId, agentRun, runDir, existingStatus });
    }
  }
}

if (dryRun) {
  console.log(JSON.stringify({
    experiment_id: experimentId, benchmark_kind: benchmarkKind, benchmark_dir: benchmarkDir, split,
    config: { layers, questions, max_questions: maxQuestions || null, pool: poolSize, max_insights: maxInsights, summary_samples: summarySamples, reasoning, thinking_mode: true },
    tasks,
  }, null, 2));
  process.exit(0);
}

const counts = { success: 0, failed: 0, skipped: 0 };
for (const task of tasks) {
  if (task.existingStatus) {
    counts.skipped += 1;
    console.log(`skip ${task.systemId}/${task.caseId}/run-${task.agentRun}: ${task.existingStatus}`);
    continue;
  }
  const args = [
    "generate_insightbench.mjs", "--benchmark-kind", benchmarkKind, "--benchmark-dir", benchmarkDir,
    benchmarkKind === "insighteval" ? "--instance" : "--flag", String(task.caseNumber),
    "--experiment", experimentId,
    "--system", task.systemId, "--agent-run", String(task.agentRun), "--layers", layers,
    "--split", split,
    "--questions", questions, "--pool", poolSize,
    "--reasoning", reasoning,
    "--max-insights", maxInsights, "--summary-samples", summarySamples,
    "--out-root", outRoot, "--use-skills", task.systemId === "pi-core" ? "0" : "1",
    "--use-insight-bank", useInsightBank, "--goal-sufficiency", goalSufficiency,
  ];
  if (task.systemId === "pi-auto-skills") {
    args.push("--skill-dir", skillDirectory);
  }
  if (maxQuestions) args.push("--max-questions", maxQuestions);
  const result = spawnSync(process.execPath, args, { cwd: HERE, stdio: "inherit", env: process.env });
  if (result.status === 0) counts.success += 1;
  else counts.failed += 1;
}
console.log(JSON.stringify({ experiment_id: experimentId, counts }, null, 2));
process.exitCode = counts.failed ? 1 : 0;
