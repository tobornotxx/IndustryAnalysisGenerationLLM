import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  buildForbiddenVocabulary, collectValidationRecords, deduplicateRunTasks, freezeDirectorySkillSet,
  prepareDirectoryValidationPlan, prepareValidationPlan,
} from "../src/skill-validation.mjs";
import { NativeSkillRuntime } from "../src/native-skills.mjs";

const candidatePackage = {
  meta: { version: "auto-candidates-v1", produced_by: "pi-skill-extraction-agent", frozen: false },
  skills: [{
    id: "compare-groups", trigger: "compare groups", trigger_terms: ["compare"],
    stages: ["executor"], action: "Compare groups.", rationale: "Aggregates hide variation.",
    provenance: { source_runs: ["run-1"], source_cases: ["flag-1"] }, status: "candidate",
  }],
  episodes: [], config: {},
};

test("forbidden vocabulary is derived from training schema without generic column words", () => {
  const root = mkdtempSync(join(tmpdir(), "skill-vocabulary-"));
  mkdirSync(join(root, "data", "notebooks"), { recursive: true });
  writeFileSync(join(root, "table.csv"), "number,category,location,priority,assigned_to,opened_at\n1,Hardware,HQ,1,Alice,2024-01-01\n");
  writeFileSync(join(root, "data", "notebooks", "flag-1.json"), JSON.stringify({
    dataset_csv_path: "table.csv", metadata: { category: "Incident Management", role: "L2 Manager" },
  }));
  const terms = buildForbiddenVocabulary({ benchmarkDir: root, caseIds: ["flag-1"] });
  assert.ok(terms.includes("assigned_to"));
  assert.ok(terms.includes("opened_at"));
  assert.ok(!terms.includes("category"));
  assert.ok(!terms.includes("number"));
  assert.ok(!terms.includes("location"));
  assert.ok(!terms.includes("priority"));
});

test("shared control tasks execute and score only once", () => {
  const tasks = [
    { experiment_id: "control", system_id: "pi-core", case_id: "flag-9", agent_run: 1, skill_id: "a" },
    { experiment_id: "control", system_id: "pi-core", case_id: "flag-9", agent_run: 1, skill_id: "b" },
    { experiment_id: "treated-a", system_id: "pi-skill-candidate", case_id: "flag-9", agent_run: 1 },
  ];
  assert.equal(deduplicateRunTasks(tasks).length, 2);
});

test("validation planner creates immutable single-skill packages and paired tasks", () => {
  const root = mkdtempSync(join(tmpdir(), "skill-validation-plan-"));
  const plan = prepareValidationPlan(candidatePackage, {
    outputDir: root, caseIds: ["flag-9", "flag-10"], agentRuns: 3,
  });
  assert.equal(plan.plans[0].tasks.length, 12);
  assert.equal(plan.plans[0].tasks.filter((task) => task.arm === "control").length, 6);
  assert.equal(
    new Set(plan.plans.flatMap((item) => item.tasks.filter((task) => task.arm === "control").map((task) => task.experiment_id))).size,
    1,
  );
  assert.ok(existsSync(plan.plans[0].skill_package));
  assert.throws(() => prepareValidationPlan(candidatePackage, {
    outputDir: root, caseIds: ["flag-13"], agentRuns: 3,
  }), /frozen as source-test/);
});

test("directory validation plan isolates each physical PI Skill", async () => {
  const root = mkdtempSync(join(tmpdir(), "native-skill-validation-"));
  const skillRoot = join(root, "skills");
  for (const name of ["first-method", "second-method"]) {
    const dir = join(skillRoot, name);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "SKILL.md"), [
      "---", `name: ${name}`, `description: Apply ${name} when its analytical method is relevant.`, "---",
      `Use this skill when ${name} is relevant. Do not use it otherwise.`,
    ].join("\n"));
  }
  const outputDir = join(root, "validation");
  const plan = await prepareDirectoryValidationPlan(skillRoot, {
    outputDir, caseIds: ["flag-9"], agentRuns: 2, experimentTag: "test-native",
  });
  assert.equal(plan.schema_version, 2);
  assert.equal(plan.plans.length, 2);
  assert.equal(plan.plans[0].tasks.length, 4);
  assert.equal(plan.plans[0].tasks.filter((task) => task.arm === "treated").length, 2);
  assert.ok(existsSync(join(plan.plans[0].skill_dir, plan.plans[0].skill_id, "SKILL.md")));
  await assert.rejects(prepareDirectoryValidationPlan(skillRoot, {
    outputDir, caseIds: ["flag-9"], agentRuns: 2, experimentTag: "test-native",
  }), /overwrite validation skill directory/);

  const selectedOutputDir = join(root, "selected-validation");
  const selected = await prepareDirectoryValidationPlan(skillRoot, {
    outputDir: selectedOutputDir, caseIds: ["flag-9"], agentRuns: 3,
    experimentTag: "selected-native", skillNames: ["first-method"],
    controlExperimentId: "skillval-prior-shared-control",
  });
  assert.deepEqual(selected.plans.map((item) => item.skill_id), ["first-method"]);
  assert.equal(selected.plans[0].tasks.length, 6);
  assert.equal(selected.shared_control_experiment_id, "skillval-prior-shared-control");
  assert.ok(selected.plans[0].tasks
    .filter((task) => task.arm === "control")
    .every((task) => task.experiment_id === "skillval-prior-shared-control"));
  await assert.rejects(prepareDirectoryValidationPlan(skillRoot, {
    outputDir: join(root, "unknown-validation"), caseIds: ["flag-9"], agentRuns: 3,
    skillNames: ["missing-method"],
  }), /unknown candidate skill/);
});

test("directory freeze includes only validated Skills and detects later tampering", async () => {
  const root = mkdtempSync(join(tmpdir(), "native-skill-freeze-"));
  const skillRoot = join(root, "skills");
  for (const name of ["accepted-method", "rejected-method"]) {
    const dir = join(skillRoot, name);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "SKILL.md"), [
      "---", `name: ${name}`, `description: Apply ${name} when relevant.`, "---",
      `Use this skill when ${name} applies. Do not use it otherwise.`,
    ].join("\n"));
  }
  const frozenRoot = join(root, "frozen");
  const manifest = await freezeDirectorySkillSet(skillRoot, {
    "accepted-method": { passed: true, mean_delta: 0.1 },
    "rejected-method": { passed: false, mean_delta: -0.1 },
  }, { outputDir: frozenRoot, version: "test-v1" });
  assert.deepEqual(manifest.skills, ["accepted-method"]);
  const frozen = await NativeSkillRuntime.load([frozenRoot], { requireFrozen: true });
  assert.deepEqual(frozen.skills.map((skill) => skill.name), ["accepted-method"]);
  const frozenSkillPath = join(frozenRoot, "accepted-method", "SKILL.md");
  writeFileSync(frozenSkillPath, `${readFileSync(frozenSkillPath, "utf8")}\nTampered.\n`, "utf8");
  await assert.rejects(NativeSkillRuntime.load([frozenRoot], { requireFrozen: true }), /content hash/);
});

test("validation collector pairs agent-run means and costs by case", () => {
  const root = mkdtempSync(join(tmpdir(), "skill-validation-results-"));
  const planDir = join(root, "plan");
  const plan = prepareValidationPlan(candidatePackage, {
    outputDir: planDir, caseIds: ["flag-9", "flag-10"], agentRuns: 2,
  });
  for (const task of plan.plans[0].tasks) {
    const run = join(root, "runs", task.experiment_id, task.system_id, task.case_id, `agent_run_${task.agent_run}`);
    mkdirSync(join(run, "scores", "judge"), { recursive: true });
    writeFileSync(join(run, "manifest.json"), JSON.stringify({
      status: "success",
      skill_package: task.arm === "treated" ? {
        read_names: task.agent_run === 1 ? [task.skill_id] : [],
        executions: task.agent_run === 1 ? [{ skill_name: task.skill_id }] : [],
      } : null,
    }));
    writeFileSync(join(run, "usage.json"), JSON.stringify({ cost_usd: task.arm === "treated" ? 1.1 : 1 }));
    writeFileSync(join(run, "scores", "judge", "judge_run_1.json"), JSON.stringify({
      semantic: { primary: { f1: task.arm === "treated" ? 0.6 : 0.5 } },
    }));
  }
  const records = collectValidationRecords(plan, { outRoot: join(root, "runs"), scorerId: "judge" });
  assert.equal(records.length, 2);
  assert.deepEqual(records[0].control, [0.5, 0.5]);
  assert.deepEqual(records[0].treated, [0.6, 0.6]);
  assert.deepEqual(records[0].treated_cost, [1.1, 1.1]);
  assert.deepEqual(records[0].treated_skill_read, [1, 0]);
  assert.deepEqual(records[0].treated_skill_executed, [1, 0]);
});
