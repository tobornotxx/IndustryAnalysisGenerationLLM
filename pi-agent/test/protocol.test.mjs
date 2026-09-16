import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { packageHash } from "../src/skill-lifecycle.mjs";
import { validateFormalProtocol } from "../src/protocol.mjs";

const protocol = {
  schema_version: 1,
  status: "frozen-before-formal-api-runs",
  generation: { model: "deepseek-flash", reasoning: "medium", thinking_mode: true, agent_runs_per_case: 3 },
  scoring: { judge_runs_per_prediction: 1 },
  required_order: ["source-train", "source-valid", "source-test", "target-test"],
  stages: {
    "source-train": { benchmark_id: "insightbench-overhaul", case_ids: ["flag-1"], systems: ["pi-core"] },
    "source-valid": { benchmark_id: "insightbench-overhaul", case_ids: ["flag-9"], systems: ["pi-skill-candidate"] },
    "source-test": { benchmark_id: "insightbench-overhaul", case_ids: ["flag-13"], systems: ["pi-core", "pi-auto-skills"] },
    "target-test": { benchmark_id: "insighteval-official", case_ids: ["insighteval:1-2"], systems: ["pi-core", "pi-auto-skills"] },
  },
};
const systems = { systems: { "pi-core": {}, "pi-skill-candidate": {}, "pi-auto-skills": {} } };

test("formal protocol checks model, replication, systems, and frozen splits", () => {
  assert.equal(validateFormalProtocol(protocol, systems, { stage: "source-train" }).passed, true);
  const invalid = structuredClone(protocol);
  invalid.generation.reasoning = "off";
  invalid.stages["source-test"].case_ids = ["flag-9"];
  const result = validateFormalProtocol(invalid, systems, { stage: "source-train" });
  assert.equal(result.passed, false);
  assert.match(result.errors.join("\n"), /thinking mode|split mismatch/);
});

test("held-out stages require an untampered frozen skill package", () => {
  const root = mkdtempSync(join(tmpdir(), "protocol-package-"));
  const path = join(root, "auto.json");
  const pkg = {
    meta: { version: "auto-v1", status: "frozen-validated", frozen: true },
    skills: [{ id: "generic", action: "Check another dimension", stages: ["planner"] }],
  };
  pkg.meta.package_hash = packageHash(pkg);
  writeFileSync(path, JSON.stringify(pkg));
  assert.equal(validateFormalProtocol(protocol, systems, { stage: "source-test", skillPackagePath: path }).passed, true);
  const tampered = JSON.parse(JSON.stringify(pkg));
  tampered.skills[0].action = "Changed after freeze";
  writeFileSync(path, JSON.stringify(tampered));
  assert.match(
    validateFormalProtocol(protocol, systems, { stage: "source-test", skillPackagePath: path }).errors.join("\n"),
    /content hash/,
  );
});
