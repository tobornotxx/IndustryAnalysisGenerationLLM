/** Schedule repeated pi-core generations with resume-by-skip semantics. */
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { buildRunDirectory, validateDataSplit } from "./src/experiment.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = process.env.REPO_ROOT ?? resolve(HERE, "..");
function arg(name, dflt) {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : dflt;
}
const dryRun = process.argv.includes("--dry-run");
const flags = arg("flags", "11,12").split(",").map(Number);
const systems = arg("systems", "pi-core,pi-core-skills").split(",").filter(Boolean);
const repeats = Number(arg("agent-runs", 3));
const experimentId = arg("experiment", "v41_baseline");
const split = validateDataSplit(arg("split", "dev-contaminated"));
const outRoot = arg("out-root", `${REPO}/results/experiments`);
const layers = arg("layers", "3");
const questions = arg("questions", "2");
const maxQuestions = arg("max-questions", "");
const poolSize = arg("pool", "4");
const maxInsights = arg("max-insights", "10");
const summarySamples = arg("summary-samples", "3");
const useInsightBank = arg("use-insight-bank", "1");
const goalSufficiency = arg("goal-sufficiency", "1");

if (!Number.isInteger(repeats) || repeats < 1 || flags.some((flag) => !Number.isInteger(flag))) {
  throw new Error("flags and agent-runs must be positive integers");
}

const tasks = [];
for (const systemId of systems) {
  if (!["pi-core", "pi-core-skills"].includes(systemId)) {
    throw new Error(`unsupported generation system: ${systemId}`);
  }
  for (const flag of flags) {
    for (let agentRun = 1; agentRun <= repeats; agentRun += 1) {
      const caseId = `flag-${flag}`;
      const runDir = buildRunDirectory({ outRoot, experimentId, systemId, caseId, agentRun });
      let existingStatus = null;
      if (existsSync(`${runDir}/manifest.json`)) {
        try { existingStatus = JSON.parse(readFileSync(`${runDir}/manifest.json`, "utf8")).status; }
        catch { existingStatus = "corrupt"; }
      } else if (existsSync(runDir)) {
        existingStatus = "incomplete";
      }
      tasks.push({ systemId, flag, agentRun, runDir, existingStatus });
    }
  }
}

if (dryRun) {
  console.log(JSON.stringify({
    experiment_id: experimentId, split,
    config: { layers, questions, max_questions: maxQuestions || null, pool: poolSize, max_insights: maxInsights, summary_samples: summarySamples },
    tasks,
  }, null, 2));
  process.exit(0);
}

const counts = { success: 0, failed: 0, skipped: 0 };
for (const task of tasks) {
  if (task.existingStatus) {
    counts.skipped += 1;
    console.log(`skip ${task.systemId}/${task.flag}/run-${task.agentRun}: ${task.existingStatus}`);
    continue;
  }
  const args = [
    "generate_insightbench.mjs", "--flag", String(task.flag), "--experiment", experimentId,
    "--system", task.systemId, "--agent-run", String(task.agentRun), "--layers", layers,
    "--split", split,
    "--questions", questions, "--pool", poolSize,
    "--max-insights", maxInsights, "--summary-samples", summarySamples,
    "--out-root", outRoot, "--use-skills", task.systemId === "pi-core-skills" ? "1" : "0",
    "--use-insight-bank", useInsightBank, "--goal-sufficiency", goalSufficiency,
  ];
  if (maxQuestions) args.push("--max-questions", maxQuestions);
  const result = spawnSync(process.execPath, args, { cwd: HERE, stdio: "inherit", env: process.env });
  if (result.status === 0) counts.success += 1;
  else counts.failed += 1;
}
console.log(JSON.stringify({ experiment_id: experimentId, counts }, null, 2));
process.exitCode = counts.failed ? 1 : 0;
