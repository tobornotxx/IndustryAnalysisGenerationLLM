import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { NativeSkillRuntime, createNativeSkillTools } from "../src/native-skills.mjs";
import { ResearchState } from "../src/research-loop.mjs";

function makeSkillRoot() {
  const root = mkdtempSync(join(tmpdir(), "pi-native-skill-"));
  const skill = join(root, "explain-shift");
  mkdirSync(join(skill, "scripts"), { recursive: true });
  writeFileSync(join(skill, "SKILL.md"), [
    "---",
    "name: explain-shift",
    "description: Decompose an aggregate change into within-group and composition effects.",
    "---",
    "SECRET FULL METHOD: inspect both components before interpreting the shift.",
  ].join("\n"));
  writeFileSync(join(skill, "scripts", "decompose.py"), "print({'rows': len(sql_results), 'args': skill_args})\n");
  return root;
}

function findTool(tools, name) {
  return tools.find((tool) => tool.name === name);
}

test("native skill catalog reveals metadata but full content only on read", async (t) => {
  const root = makeSkillRoot();
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const runtime = await NativeSkillRuntime.load([root]);
  assert.match(runtime.catalogPrompt, /explain-shift/);
  assert.match(runtime.catalogPrompt, /Decompose an aggregate change/);
  assert.doesNotMatch(runtime.catalogPrompt, /SECRET FULL METHOD/);
  const full = runtime.read("explain-shift");
  assert.match(full, /SECRET FULL METHOD/);
  assert.deepEqual(runtime.reads.map((item) => item.name), ["explain-shift"]);
});

test("skill scripts require an explicit read and remain inside the skill directory", async (t) => {
  const root = makeSkillRoot();
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const runtime = await NativeSkillRuntime.load([root]);
  assert.throws(() => runtime.resolveScript("explain-shift", "scripts/decompose.py"), /read skill/);
  runtime.read("explain-shift");
  assert.throws(() => runtime.resolveScript("explain-shift", "../outside.py"), /escapes/);
  assert.match(runtime.resolveScript("explain-shift", "scripts/decompose.py"), /decompose\.py$/);
});

test("executed skill scripts are attached to question evidence", async (t) => {
  const root = makeSkillRoot();
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const runtime = await NativeSkillRuntime.load([root]);
  const state = new ResearchState({ goal: "explain change", maxQuestions: 1 });
  state.openQuestion({ question: "What drives the change?", category: "exploratory" });
  const pool = {
    async call(kind, { code }) {
      assert.equal(kind, "python");
      assert.match(code, /skill_args/);
      return { output: "rows=10" };
    },
  };
  const tools = createNativeSkillTools(runtime, { state, pool });
  await findTool(tools, "read_skill").execute("1", { name: "explain-shift" });
  await findTool(tools, "run_skill_python").execute("2", {
    skill_name: "explain-shift",
    script: "scripts/decompose.py",
    question_id: "q_001",
    sql: "select * from main_table",
    arguments: { group: "region" },
  });
  assert.equal(runtime.executions.length, 1);
  assert.match(state.nodes[0].evidence[0].tool, /run_skill_python:explain-shift/);
});
