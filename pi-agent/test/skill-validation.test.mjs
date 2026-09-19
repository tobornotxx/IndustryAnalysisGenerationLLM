import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  buildForbiddenVocabulary, collectValidationRecords, prepareValidationPlan,
} from "../src/skill-validation.mjs";

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
  writeFileSync(join(root, "table.csv"), "number,category,assigned_to,opened_at\n1,Hardware,Alice,2024-01-01\n");
  writeFileSync(join(root, "data", "notebooks", "flag-1.json"), JSON.stringify({
    dataset_csv_path: "table.csv", metadata: { category: "Incident Management", role: "L2 Manager" },
  }));
  const terms = buildForbiddenVocabulary({ benchmarkDir: root, caseIds: ["flag-1"] });
  assert.ok(terms.includes("assigned_to"));
  assert.ok(terms.includes("opened_at"));
  assert.ok(!terms.includes("category"));
  assert.ok(!terms.includes("number"));
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

test("validation collector pairs agent-run means and costs by case", () => {
  const root = mkdtempSync(join(tmpdir(), "skill-validation-results-"));
  const planDir = join(root, "plan");
  const plan = prepareValidationPlan(candidatePackage, {
    outputDir: planDir, caseIds: ["flag-9", "flag-10"], agentRuns: 2,
  });
  for (const task of plan.plans[0].tasks) {
    const run = join(root, "runs", task.experiment_id, task.system_id, task.case_id, `agent_run_${task.agent_run}`);
    mkdirSync(join(run, "scores", "judge"), { recursive: true });
    writeFileSync(join(run, "manifest.json"), JSON.stringify({ status: "success" }));
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
});
