/** Execute or dry-run a validation plan created by skill_pipeline.mjs. */
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { buildRunDirectory } from "./src/experiment.mjs";
import { deduplicateRunTasks } from "./src/skill-validation.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
function arg(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}
const has = (name) => process.argv.includes(`--${name}`);
const planPath = arg("plan");
if (!planPath) throw new Error("--plan is required");
const plan = JSON.parse(readFileSync(planPath, "utf8"));
const benchmarkDir = arg("benchmark-dir", resolve(HERE, "../run_on_benchmark/insight-bench"));
const outRoot = arg("out-root", resolve(HERE, "../results/experiments"));
const dryRun = has("dry-run");
const skillFilter = new Set(arg("skills", "").split(",").filter(Boolean));
const treatedOnly = has("treated-only");
const configArgs = [
  "--layers", arg("layers", "3"), "--questions", arg("questions", "2"),
  "--max-questions", arg("max-questions", "6"), "--max-insights", arg("max-insights", "10"),
  "--summary-samples", arg("summary-samples", "3"), "--reasoning", arg("reasoning", "medium"),
];

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
  let existingStatus = null;
  if (existsSync(`${runDir}/manifest.json`)) {
    existingStatus = JSON.parse(readFileSync(`${runDir}/manifest.json`, "utf8")).status ?? "unknown";
  } else if (existsSync(runDir)) existingStatus = "incomplete";
  return { ...task, runDir, existingStatus };
}));

if (dryRun) {
  console.log(JSON.stringify({ schema_version: 1, split: plan.split, tasks }, null, 2));
  process.exit(0);
}

const counts = { success: 0, failed: 0, skipped: 0 };
for (const task of tasks) {
  if (task.existingStatus) { counts.skipped += 1; continue; }
  const args = [
    "generate_insightbench.mjs", "--flag", task.case_id.replace("flag-", ""),
    "--experiment", task.experiment_id, "--system", task.system_id,
    "--agent-run", String(task.agent_run), "--split", "source-valid",
    "--benchmark-dir", benchmarkDir, "--out-root", outRoot,
    "--use-skills", task.arm === "treated" ? "1" : "0", ...configArgs,
  ];
  if (task.skill_package) args.push("--skill-package", task.skill_package);
  const result = spawnSync(process.execPath, args, { cwd: HERE, stdio: "inherit", env: process.env });
  counts[result.status === 0 ? "success" : "failed"] += 1;
}
console.log(JSON.stringify({ counts }, null, 2));
process.exitCode = counts.failed ? 1 : 0;
