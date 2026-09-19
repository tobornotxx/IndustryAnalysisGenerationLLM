import { existsSync, mkdirSync, readFileSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { writeJsonExclusive } from "./skill-lifecycle.mjs";
import { assertCaseSplit } from "./split-registry.mjs";

function nested(value, dotted) {
  return dotted.split(".").reduce((current, key) => current?.[key], value);
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function csvHeader(line) {
  const fields = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      if (quoted && line[index + 1] === '"') { value += '"'; index += 1; }
      else quoted = !quoted;
    } else if (char === "," && !quoted) {
      fields.push(value.trim()); value = "";
    } else value += char;
  }
  fields.push(value.trim());
  return fields.map((field) => field.replace(/^\uFEFF/, "")).filter(Boolean);
}

const GENERIC_COLUMNS = new Set([
  "category", "group", "date", "time", "value", "count", "number", "name", "type", "status", "state",
]);

export function buildForbiddenVocabulary({ benchmarkDir, caseIds }) {
  const terms = new Set(["insightbench", "servicenow", ...caseIds]);
  for (const caseId of caseIds) {
    assertCaseSplit("insightbench-overhaul", caseId, "source-train");
    const meta = readJson(join(benchmarkDir, "data", "notebooks", `${caseId}.json`));
    const csvPath = resolve(benchmarkDir, meta.dataset_csv_path);
    const firstLine = readFileSync(csvPath, "utf8").split(/\r?\n/, 1)[0] ?? "";
    for (const field of csvHeader(firstLine)) {
      if (!GENERIC_COLUMNS.has(field.toLowerCase())) terms.add(field);
    }
    const metadata = meta.metadata ?? {};
    for (const literal of [metadata.category, metadata.role, metadata.header]) {
      if (literal && String(literal).length >= 4) terms.add(String(literal));
    }
  }
  return [...terms].map(String).sort((left, right) => left.localeCompare(right));
}

export function prepareValidationPlan(candidatePackage, {
  outputDir, caseIds = ["flag-9", "flag-10", "flag-11", "flag-12"], agentRuns = 3,
} = {}) {
  if (!candidatePackage.skills?.length) throw new Error("candidate package has no skills");
  if (candidatePackage.meta?.frozen) throw new Error("validation planning expects an unfrozen candidate package");
  if (!Number.isInteger(agentRuns) || agentRuns < 2) throw new Error("agentRuns must be at least 2");
  caseIds.forEach((caseId) => assertCaseSplit("insightbench-overhaul", caseId, "source-valid"));
  mkdirSync(outputDir, { recursive: true });
  const plans = [];
  const controlExperimentId = `skillval-${candidatePackage.meta?.version ?? "auto"}-shared-control`;
  for (const skill of candidatePackage.skills) {
    const packagePath = resolve(outputDir, `candidate-${skill.id}.json`);
    const packageValue = {
      meta: {
        version: `${candidatePackage.meta?.version ?? "auto"}-${skill.id}-validation`,
        produced_by: candidatePackage.meta?.produced_by ?? "pi-skill-extraction-agent",
        status: "validation-candidate",
        frozen: false,
        parent_version: candidatePackage.meta?.version,
      },
      skills: [skill],
      episodes: candidatePackage.episodes ?? [],
      config: candidatePackage.config ?? {},
    };
    writeJsonExclusive(packagePath, packageValue);
    const experimentId = `skillval-${candidatePackage.meta?.version ?? "auto"}-${skill.id}`;
    const tasks = [];
    for (const caseId of caseIds) {
      for (let agentRun = 1; agentRun <= agentRuns; agentRun += 1) {
        tasks.push({ experiment_id: controlExperimentId, skill_id: skill.id, arm: "control", system_id: "pi-core", case_id: caseId, agent_run: agentRun });
        tasks.push({ experiment_id: experimentId, skill_id: skill.id, arm: "treated", system_id: "pi-skill-candidate", case_id: caseId, agent_run: agentRun, skill_package: packagePath });
      }
    }
    plans.push({ skill_id: skill.id, experiment_id: experimentId, skill_package: packagePath, tasks });
  }
  return {
    schema_version: 1,
    split: "source-valid",
    candidate_version: candidatePackage.meta?.version,
    agent_runs: agentRuns,
    case_ids: caseIds,
    shared_control_experiment_id: controlExperimentId,
    plans,
  };
}

function scoreValues(runDir, scorerId, metric) {
  const scoreDir = join(runDir, "scores", scorerId);
  if (!existsSync(scoreDir)) return [];
  return readdirSync(scoreDir)
    .filter((name) => /^judge_run_\d+\.json$/.test(name)).sort()
    .map((name) => Number(nested(readJson(join(scoreDir, name)), metric)))
    .filter(Number.isFinite);
}

export function collectValidationRecords(plan, {
  outRoot, scorerId = "local-deepseek-v41-thinking-v1", metric = "semantic.primary.f1",
} = {}) {
  const records = [];
  for (const skillPlan of plan.plans ?? []) {
    for (const caseId of plan.case_ids ?? []) {
      const arms = { control: [], treated: [] };
      const costs = { control: [], treated: [] };
      for (const task of skillPlan.tasks.filter((item) => item.case_id === caseId)) {
        const runDir = join(outRoot, task.experiment_id, task.system_id, caseId, `agent_run_${task.agent_run}`);
        const manifestPath = join(runDir, "manifest.json");
        if (!existsSync(manifestPath) || readJson(manifestPath).status !== "success") continue;
        const values = scoreValues(runDir, scorerId, metric);
        if (!values.length) continue;
        arms[task.arm].push(mean(values));
        const usagePath = join(runDir, "usage.json");
        costs[task.arm].push(existsSync(usagePath) ? Number(readJson(usagePath).cost_usd ?? 0) : 0);
      }
      records.push({
        skill_id: skillPlan.skill_id,
        benchmark_id: "insightbench-overhaul",
        case_id: caseId,
        split: "source-valid",
        control: arms.control,
        treated: arms.treated,
        control_cost: costs.control,
        treated_cost: costs.treated,
        scorer_id: scorerId,
        metric,
      });
    }
  }
  return records;
}
