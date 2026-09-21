/** Score completed skill-validation predictions without regenerating them. */
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { buildRunDirectory } from "./src/experiment.mjs";
import { deduplicateRunTasks } from "./src/skill-validation.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = process.env.REPO_ROOT ?? resolve(HERE, "..");
const PYTHON = process.env.PYTHON_BIN ?? "python";

function arg(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

const planPath = arg("plan");
if (!planPath) throw new Error("--plan is required");
const plan = JSON.parse(readFileSync(planPath, "utf8"));
const outRoot = resolve(arg("out-root", resolve(REPO, "results/experiments")));
const benchmarkDir = resolve(arg("benchmark-dir", resolve(REPO, "run_on_benchmark/insight-bench")));
const scorerId = arg("scorer-id", "local-deepseek-v41-thinking-v2");
const judgeRun = Number(arg("judge-run", "1"));
const modes = arg("modes", "primary");
const dryRun = process.argv.includes("--dry-run");
const skillFilter = new Set((arg("skills", "") ?? "").split(",").filter(Boolean));
const treatedOnly = process.argv.includes("--treated-only");
if (!Number.isInteger(judgeRun) || judgeRun < 1) throw new Error("judge-run must be a positive integer");

const selectedPlans = (plan.plans ?? []).filter(
  (skillPlan) => !skillFilter.size || skillFilter.has(skillPlan.skill_id),
);
if (skillFilter.size && selectedPlans.length !== skillFilter.size) {
  throw new Error("--skills contains an id not present in the validation plan");
}
const plannedTasks = selectedPlans.flatMap((skillPlan) => skillPlan.tasks)
  .filter((task) => !treatedOnly || task.arm === "treated");
const tasks = deduplicateRunTasks(plannedTasks.map((task) => {
  const runDir = buildRunDirectory({
    outRoot, experimentId: task.experiment_id, systemId: task.system_id,
    caseId: task.case_id, agentRun: task.agent_run,
  });
  const predictionPath = join(runDir, "prediction.json");
  const scorePath = join(runDir, "scores", scorerId, `judge_run_${judgeRun}.json`);
  return {
    ...task, predictionPath, scorePath,
    predictionExists: existsSync(predictionPath), scoreExists: existsSync(scorePath),
  };
}));

if (dryRun) {
  console.log(JSON.stringify({
    schema_version: 1, scorer_id: scorerId, judge_run: judgeRun,
    counts: {
      total: tasks.length,
      ready: tasks.filter((task) => task.predictionExists && !task.scoreExists).length,
      missing_prediction: tasks.filter((task) => !task.predictionExists).length,
      already_scored: tasks.filter((task) => task.scoreExists).length,
    },
    tasks,
  }, null, 2));
  process.exit(0);
}

const counts = { success: 0, failed: 0, skipped: 0, missing: 0 };
for (const task of tasks) {
  if (task.scoreExists) { counts.skipped += 1; continue; }
  if (!task.predictionExists) { counts.missing += 1; continue; }
  const result = spawnSync(PYTHON, [
    "-m", "run_on_benchmark.score_prediction",
    "--prediction", task.predictionPath,
    "--benchmark-dir", benchmarkDir,
    "--judge-run", String(judgeRun),
    "--scorer-id", scorerId,
    "--modes", modes,
  ], { cwd: REPO, stdio: "inherit", env: process.env });
  counts[result.status === 0 ? "success" : "failed"] += 1;
}
console.log(JSON.stringify({ scorer_id: scorerId, judge_run: judgeRun, counts }, null, 2));
process.exitCode = counts.failed ? 1 : 0;
