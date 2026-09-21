import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { NativeSkillRuntime } from "../src/native-skills.mjs";
import {
  SkillCreatorWorkspace,
  createSkillCreatorTools,
  validateCreatorEpisodes,
} from "../src/skill-creator-agent.mjs";

const episodes = [
  {
    episode_id: "run-low", case_id: "flag-1", split: "source-train", score: 0.2,
    score_provenance: { status: "valid", scorer_id: "judge-v2", judge_calls: 1 },
    steps: [{ question: "What changed?", answer: "A changed.", tool_names: ["run_sql"] }],
  },
  {
    episode_id: "run-high", case_id: "flag-2", split: "source-train", score: 0.8,
    score_provenance: { status: "valid", scorer_id: "judge-v2", judge_calls: 1 },
    steps: [{ question: "What drives the change?", answer: "Composition explains it.", tool_names: ["run_python"] }],
  },
];

function resultJson(result) {
  return JSON.parse(result.content[0].text);
}

test("creator rejects missing or degenerate score evidence", () => {
  assert.throws(
    () => validateCreatorEpisodes(episodes.map((episode) => ({ ...episode, score: 0 }))),
    /all episode scores are equal/,
  );
  assert.throws(
    () => validateCreatorEpisodes(episodes.map((episode) => ({ ...episode, score_provenance: undefined }))),
    /validated per-episode scores/,
  );
});

test("creator writes a physical PI Skill with exact episode provenance", async (t) => {
  validateCreatorEpisodes(episodes);
  const parent = mkdtempSync(join(tmpdir(), "pi-skill-creator-"));
  const output = join(parent, "created-skills");
  t.after(() => rmSync(parent, { recursive: true, force: true }));
  const workspace = new SkillCreatorWorkspace({ outputDir: output, episodes });
  const tools = createSkillCreatorTools(workspace);
  const find = (name) => tools.find((tool) => tool.name === name);

  await find("inspect_episode").execute("1", { episode_id: "run-low" });
  await find("inspect_episode").execute("2", { episode_id: "run-high" });
  await find("create_skill").execute("3", {
    name: "decompose-observed-shifts",
    description: "Separate composition changes from within-group changes before explaining an aggregate shift.",
    capability_gap: "Weak runs described the aggregate while stronger runs tested competing decomposition explanations.",
    evidence_episode_ids: ["run-low", "run-high"],
    instructions: [
      "# Decompose observed shifts",
      "Use this skill when an aggregate metric changes and group composition may also have changed.",
      "1. State the aggregate pattern and competing within-group versus composition explanations.",
      "2. Compute both components and compare their contributions.",
      "3. Use the result to prioritize follow-up questions and retain alternative explanations.",
      "Do not use this skill when the data has no meaningful grouping or comparison period.",
    ].join("\n\n"),
  });
  await find("write_skill_asset").execute("4", {
    skill_name: "decompose-observed-shifts",
    path: "scripts/decompose.py",
    content: "print({'groups': int(sql_results.shape[0])})\n",
  });
  await find("write_skill_asset").execute("5", {
    skill_name: "decompose-observed-shifts",
    path: "tests/test_decompose.py",
    content: "# Offline fixture test for the decomposition script.\n",
  });
  const validation = resultJson(await find("validate_skill_set").execute("6", {}));
  assert.equal(validation.passed, true);
  const submitted = resultJson(await find("submit_skill_set").execute("7", {}));
  assert.equal(submitted.passed, true);
  assert.equal(existsSync(join(output, "decompose-observed-shifts", "SKILL.md")), true);

  const runtime = await NativeSkillRuntime.load([output]);
  assert.deepEqual(runtime.skills.map((skill) => skill.name), ["decompose-observed-shifts"]);
  const provenance = JSON.parse(readFileSync(
    join(output, "decompose-observed-shifts", "provenance.json"), "utf8",
  ));
  assert.deepEqual(provenance.evidence_episode_ids, ["run-low", "run-high"]);
  assert.match(readFileSync(join(output, "creator_trajectory.jsonl"), "utf8"), /inspect_episode/);
});

test("creator cannot cite episodes it did not inspect", (t) => {
  const parent = mkdtempSync(join(tmpdir(), "pi-skill-creator-"));
  t.after(() => rmSync(parent, { recursive: true, force: true }));
  const workspace = new SkillCreatorWorkspace({ outputDir: join(parent, "skills"), episodes });
  assert.throws(() => workspace.createSkill({
    name: "uninspected-evidence",
    description: "A valid description.",
    capabilityGap: "A gap.",
    instructions: "Use this skill when appropriate. Do not use it otherwise.",
    evidenceEpisodeIds: ["run-low"],
  }), /inspect evidence episode/);
  workspace.cleanup();
});

test("creator enforces the experiment's candidate Skill budget", (t) => {
  const parent = mkdtempSync(join(tmpdir(), "pi-skill-creator-"));
  t.after(() => rmSync(parent, { recursive: true, force: true }));
  const workspace = new SkillCreatorWorkspace({
    outputDir: join(parent, "skills"), episodes, maxSkills: 1,
  });
  workspace.inspectEpisode("run-low");
  const input = {
    description: "Use this skill when an observed aggregate change has rival explanations.",
    capabilityGap: "Weak runs did not distinguish rival explanations.",
    instructions: "Use this skill when explanations compete. Do not use it without computed evidence.",
    evidenceEpisodeIds: ["run-low"],
  };
  workspace.createSkill({ name: "first-method", ...input });
  assert.throws(
    () => workspace.createSkill({ name: "second-method", ...input }),
    /skill creation limit reached: 1/,
  );
  workspace.cleanup();
});

test("creator validation returns training-literal findings to the agent", async (t) => {
  const parent = mkdtempSync(join(tmpdir(), "pi-skill-creator-"));
  t.after(() => rmSync(parent, { recursive: true, force: true }));
  const workspace = new SkillCreatorWorkspace({
    outputDir: join(parent, "skills"), episodes, forbiddenTerms: ["private_training_column"],
  });
  workspace.inspectEpisode("run-low");
  workspace.createSkill({
    name: "leaky-method",
    description: "Use this skill when an observed aggregate change has rival explanations.",
    capabilityGap: "Weak runs did not distinguish rival explanations.",
    instructions: [
      "Use this skill when explanations compete.",
      "Compute private_training_column before deciding.",
      "Do not use it without computed evidence.",
    ].join("\n"),
    evidenceEpisodeIds: ["run-low"],
  });
  const validation = await workspace.validate();
  assert.equal(validation.passed, false);
  assert.match(validation.errors.join("\n"), /forbidden-training-literal.*private_training_column/);
  workspace.cleanup();
});

test("creator validation executes generated Python tests", async (t) => {
  const parent = mkdtempSync(join(tmpdir(), "pi-skill-creator-"));
  t.after(() => rmSync(parent, { recursive: true, force: true }));
  const workspace = new SkillCreatorWorkspace({ outputDir: join(parent, "skills"), episodes });
  workspace.inspectEpisode("run-low");
  workspace.createSkill({
    name: "tested-method",
    description: "Use this skill when an observed aggregate change has rival explanations.",
    capabilityGap: "Weak runs did not distinguish rival explanations.",
    instructions: "Use this skill when explanations compete. Do not use it without computed evidence.",
    evidenceEpisodeIds: ["run-low"],
  });
  workspace.writeAsset({
    skillName: "tested-method", path: "scripts/method.py", content: "def answer(): return 1\n",
  });
  workspace.writeAsset({
    skillName: "tested-method", path: "tests/test_method.py", content: "raise AssertionError('fixture failure')\n",
  });
  const validation = await workspace.validate();
  assert.equal(validation.passed, false);
  assert.match(validation.errors.join("\n"), /Python test failed.*fixture failure/s);
  workspace.cleanup();
});
