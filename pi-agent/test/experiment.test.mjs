import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  buildRunDirectory, createRunDirectory, makeManifest, sha256Files, validateDataSplit,
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
  assert.equal(validateDataSplit("dev-contaminated"), "dev-contaminated");
  assert.equal(validateDataSplit("target-test"), "target-test");
  assert.throws(() => validateDataSplit("test"), /unsupported data split/);
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
