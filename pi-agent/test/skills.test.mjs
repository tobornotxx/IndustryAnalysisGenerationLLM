import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SkillPackage } from "../src/skills.mjs";

const manual = {
  meta: { status: "candidate", frozen: false },
  skills: [
    { id: "trend", stages: ["executor"], trigger_terms: ["trend", "time"], action: "group {categorical_columns}" },
    { id: "always-check", stages: ["sufficiency"], always: true, action: "check coverage" },
  ],
};

test("runtime selects stage- and context-relevant skills", () => {
  const pkg = new SkillPackage(manual);
  assert.deepEqual(pkg.select("executor", "find the trend over time").map((skill) => skill.id), ["trend"]);
  assert.deepEqual(pkg.select("sufficiency", "anything").map((skill) => skill.id), ["always-check"]);
  assert.match(pkg.renderSkills(pkg.select("executor", "trend"), { categoricalColumns: "region" }), /group region/);
});

test("auto-skill runs reject candidate or manual packages", () => {
  assert.throws(() => new SkillPackage(manual, { requireFrozen: true }), /frozen and validated/);
});

test("frozen packages can be loaded in strict mode", () => {
  const root = mkdtempSync(join(tmpdir(), "frozen-skills-"));
  const path = join(root, "skills.json");
  writeFileSync(path, JSON.stringify({ ...manual, meta: { status: "frozen-validated", frozen: true } }));
  const pkg = SkillPackage.load(path, { requireFrozen: true });
  assert.equal(pkg.skills.length, 2);
  assert.match(pkg.hash, /^[a-f0-9]{64}$/);
});
