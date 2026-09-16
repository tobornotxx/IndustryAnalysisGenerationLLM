import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  buildRunDirectory, createRunDirectory, loadBenchmarkCase, makeManifest, sha256Files, validateDataSplit,
  writeJsonAtomic,
} from "../src/experiment.mjs";

test("run directory encodes experiment, system, case and repeat", () => {
  const path = buildRunDirectory({
    outRoot: "results", experimentId: "exp", systemId: "pi-core",
    caseId: "flag-11", agentRun: 3,
  });
  assert.match(path.replaceAll("\\", "/"), /results\/exp\/pi-core\/flag-11\/agent_run_3$/);
});

test("data split names prevent accidental test claims", () => {
  assert.equal(validateDataSplit("source-train"), "source-train");
  assert.equal(validateDataSplit("target-test"), "target-test");
  assert.throws(() => validateDataSplit("test"), /unsupported data split/);
});

test("official InsightEval cases normalize to the common target schema", () => {
  const root = mkdtempSync(join(tmpdir(), "insighteval-case-"));
  mkdirSync(join(root, "data", "jsons"), { recursive: true });
  mkdirSync(join(root, "data", "csvs"), { recursive: true });
  writeFileSync(join(root, "data", "csvs", "data_1.csv"), "value\n1\n");
  writeFileSync(join(root, "data", "jsons", "data_1.json"), JSON.stringify({
    goal: "Find a fact", metadata: { table_path: "./csvs/data-1.csv" },
  }));
  const item = loadBenchmarkCase({ benchmarkKind: "insighteval", benchmarkDir: root, caseNumber: 1 });
  assert.equal(item.caseId, "insighteval-1");
  assert.equal(item.benchmarkId, "insighteval-official");
  assert.match(item.csvPath.replaceAll("\\", "/"), /data\/csvs\/data_1.csv$/);
});

test("existing run directories cannot be overwritten", () => {
  const root = mkdtempSync(join(tmpdir(), "immutable-run-"));
  const path = join(root, "run");
  createRunDirectory(path);
  assert.throws(() => createRunDirectory(path), /refusing to overwrite/);
});

test("atomic JSON and hashes are deterministic", () => {
  const root = mkdtempSync(join(tmpdir(), "manifest-"));
  const path = join(root, "manifest.json");
  writeJsonAtomic(path, makeManifest({ runId: "fixed", createdAt: "fixed", status: "success" }));
  assert.equal(JSON.parse(readFileSync(path, "utf8")).run_id, "fixed");
  assert.equal(sha256Files([path]), sha256Files([path]));
});

test("ablation switches are represented by distinct run paths at the system level", () => {
  const withoutSkills = buildRunDirectory({
    outRoot: "results", experimentId: "ablation", systemId: "pi-core",
    caseId: "flag-20", agentRun: 1,
  });
  const withSkills = buildRunDirectory({
    outRoot: "results", experimentId: "ablation", systemId: "pi-core-skills",
    caseId: "flag-20", agentRun: 1,
  });
  assert.notEqual(withoutSkills, withSkills);
});
