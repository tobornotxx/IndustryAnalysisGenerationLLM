import {
  cpSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync,
} from "node:fs";
import { randomUUID } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import { writeJsonExclusive } from "./skill-lifecycle.mjs";
import { assertCaseSplit } from "./split-registry.mjs";
import { NativeSkillRuntime } from "./native-skills.mjs";

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
  "location", "priority",
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
  experimentTag = "",
} = {}) {
  if (!candidatePackage.skills?.length) throw new Error("candidate package has no skills");
  if (candidatePackage.meta?.frozen) throw new Error("validation planning expects an unfrozen candidate package");
  if (!Number.isInteger(agentRuns) || agentRuns < 2) throw new Error("agentRuns must be at least 2");
  caseIds.forEach((caseId) => assertCaseSplit("insightbench-overhaul", caseId, "source-valid"));
  mkdirSync(outputDir, { recursive: true });
  const plans = [];
  const validationVersion = [candidatePackage.meta?.version ?? "auto", experimentTag].filter(Boolean).join("-");
  const controlExperimentId = `skillval-${validationVersion}-shared-control`;
  for (const skill of candidatePackage.skills) {
    const packagePath = resolve(outputDir, `candidate-${skill.id}.json`);
    const packageValue = {
      meta: {
        version: `${validationVersion}-${skill.id}-validation`,
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
    const experimentId = `skillval-${validationVersion}-${skill.id}`;
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
    experiment_tag: experimentTag || null,
    agent_runs: agentRuns,
    case_ids: caseIds,
    shared_control_experiment_id: controlExperimentId,
    plans,
  };
}

export async function prepareDirectoryValidationPlan(skillRoot, {
  outputDir, caseIds = ["flag-9", "flag-10", "flag-11", "flag-12"], agentRuns = 3,
  experimentTag = "native-v1", skillNames = [],
} = {}) {
  if (!Number.isInteger(agentRuns) || agentRuns < 2) throw new Error("agentRuns must be at least 2");
  caseIds.forEach((caseId) => assertCaseSplit("insightbench-overhaul", caseId, "source-valid"));
  const root = resolve(skillRoot);
  const runtime = await NativeSkillRuntime.load([root]);
  if (!runtime.skills.length) throw new Error("candidate skill directory contains no valid SKILL.md files");
  const selectedNames = new Set((skillNames ?? []).map(String).map((name) => name.trim()).filter(Boolean));
  const selectedSkills = selectedNames.size
    ? runtime.skills.filter((skill) => selectedNames.has(skill.name))
    : runtime.skills;
  const missingNames = [...selectedNames].filter((name) => !runtime.skills.some((skill) => skill.name === name));
  if (missingNames.length) throw new Error(`unknown candidate skill(s): ${missingNames.join(", ")}`);
  mkdirSync(outputDir, { recursive: true });
  const validationVersion = experimentTag || "native-v1";
  const controlExperimentId = `skillval-${validationVersion}-shared-control`;
  const plans = [];
  for (const skill of selectedSkills) {
    const candidateRoot = resolve(outputDir, `candidate-${skill.name}`);
    if (existsSync(candidateRoot)) throw new Error(`refusing to overwrite validation skill directory: ${candidateRoot}`);
    mkdirSync(candidateRoot, { recursive: true });
    cpSync(dirname(skill.filePath), join(candidateRoot, skill.name), {
      recursive: true, errorOnExist: true, force: false,
    });
    const experimentId = `skillval-${validationVersion}-${skill.name}`;
    const tasks = [];
    for (const caseId of caseIds) {
      for (let agentRun = 1; agentRun <= agentRuns; agentRun += 1) {
        tasks.push({
          experiment_id: controlExperimentId,
          skill_id: skill.name,
          arm: "control",
          system_id: "pi-core",
          case_id: caseId,
          agent_run: agentRun,
        });
        tasks.push({
          experiment_id: experimentId,
          skill_id: skill.name,
          arm: "treated",
          system_id: "pi-skill-candidate",
          case_id: caseId,
          agent_run: agentRun,
          skill_dir: candidateRoot,
        });
      }
    }
    plans.push({
      skill_id: skill.name,
      experiment_id: experimentId,
      skill_dir: candidateRoot,
      skill_hash: (await NativeSkillRuntime.load([candidateRoot])).hash,
      tasks,
    });
  }
  return {
    schema_version: 2,
    skill_format: "pi-skill-directories-v1",
    split: "source-valid",
    source_skill_root: root,
    source_skill_hash: runtime.hash,
    experiment_tag: validationVersion,
    agent_runs: agentRuns,
    case_ids: caseIds,
    shared_control_experiment_id: controlExperimentId,
    plans,
  };
}

export async function freezeDirectorySkillSet(skillRoot, validations, {
  outputDir, version,
} = {}) {
  if (!outputDir || !version) throw new Error("freeze requires outputDir and version");
  const sourceRoot = resolve(skillRoot);
  const targetRoot = resolve(outputDir);
  if (existsSync(targetRoot)) throw new Error(`refusing to overwrite frozen skill directory: ${targetRoot}`);
  const source = await NativeSkillRuntime.load([sourceRoot]);
  const accepted = source.skills.filter((skill) => validations?.[skill.name]?.passed === true);
  if (!accepted.length) throw new Error("no physical Skill passed validation");
  const staging = `${targetRoot}.staging-${randomUUID()}`;
  mkdirSync(staging, { recursive: true });
  try {
    for (const skill of accepted) {
      cpSync(dirname(skill.filePath), join(staging, skill.name), {
        recursive: true, errorOnExist: true, force: false,
      });
    }
    const staged = await NativeSkillRuntime.load([staging]);
    const selectedValidations = Object.fromEntries(
      accepted.map((skill) => [skill.name, validations[skill.name]]),
    );
    const manifest = {
      schema_version: 1,
      version,
      status: "frozen-validated",
      frozen: true,
      produced_by: "pi-skill-creator-agent",
      source_skill_root: sourceRoot,
      source_skill_hash: source.hash,
      skills: accepted.map((skill) => skill.name).sort(),
      validations: selectedValidations,
      content_hash: staged.hash,
    };
    writeFileSync(join(staging, "skill-set-manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
    await NativeSkillRuntime.load([staging], { requireFrozen: true });
    renameSync(staging, targetRoot);
    return manifest;
  } catch (error) {
    rmSync(staging, { recursive: true, force: true });
    throw error;
  }
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
  outRoot, scorerId = "local-deepseek-v41-thinking-v2", metric = "semantic.primary.f1",
} = {}) {
  const records = [];
  for (const skillPlan of plan.plans ?? []) {
    for (const caseId of plan.case_ids ?? []) {
      const arms = { control: [], treated: [] };
      const costs = { control: [], treated: [] };
      const activation = { read: [], executed: [] };
      for (const task of skillPlan.tasks.filter((item) => item.case_id === caseId)) {
        const runDir = join(outRoot, task.experiment_id, task.system_id, caseId, `agent_run_${task.agent_run}`);
        const manifestPath = join(runDir, "manifest.json");
        if (!existsSync(manifestPath)) continue;
        const manifest = readJson(manifestPath);
        if (manifest.status !== "success") continue;
        const values = scoreValues(runDir, scorerId, metric);
        if (!values.length) continue;
        arms[task.arm].push(mean(values));
        const usagePath = join(runDir, "usage.json");
        costs[task.arm].push(existsSync(usagePath) ? Number(readJson(usagePath).cost_usd ?? 0) : 0);
        if (task.arm === "treated") {
          const runtime = manifest.skill_package ?? {};
          activation.read.push((runtime.read_names ?? []).includes(skillPlan.skill_id) ? 1 : 0);
          activation.executed.push((runtime.executions ?? []).some(
            (execution) => execution.skill_name === skillPlan.skill_id,
          ) ? 1 : 0);
        }
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
        treated_skill_read: activation.read,
        treated_skill_executed: activation.executed,
        scorer_id: scorerId,
        metric,
      });
    }
  }
  return records;
}

export function deduplicateRunTasks(tasks) {
  const unique = new Map();
  for (const task of tasks) {
    const key = task.runDir ?? [task.experiment_id, task.system_id, task.case_id, task.agent_run].join("/");
    if (!unique.has(key)) unique.set(key, task);
  }
  return [...unique.values()];
}
