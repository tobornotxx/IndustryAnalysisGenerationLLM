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
    discovery_attribution: {
      available: true, match_threshold: 0.5, match_matrix: [[0.2]],
      reference_best_match: [0.2], prediction_best_match: [0.2], prediction_insights: ["aggregate"],
    },
    steps: [{ question: "What changed?", answer: "A changed.", tool_names: ["run_sql"] }],
  },
  {
    episode_id: "run-high", case_id: "flag-1", split: "source-train", score: 0.8,
    score_provenance: { status: "valid", scorer_id: "judge-v2", judge_calls: 1 },
    discovery_attribution: {
      available: true, match_threshold: 0.5, match_matrix: [[0.9]],
      reference_best_match: [0.9], prediction_best_match: [0.9], prediction_insights: ["composition"],
    },
    steps: [{ question: "What drives the change?", answer: "Composition explains it.", tool_names: ["run_python"] }],
  },
  {
    episode_id: "run-low-2", case_id: "flag-2", split: "source-train", score: 0.3,
    score_provenance: { status: "valid", scorer_id: "judge-v2", judge_calls: 1 },
    discovery_attribution: {
      available: true, match_threshold: 0.5, match_matrix: [[0.1]],
      reference_best_match: [0.1], prediction_best_match: [0.1], prediction_insights: ["surface trend"],
    },
    steps: [{ question: "Is there a trend?", answer: "Maybe.", tool_names: ["run_sql"] }],
  },
  {
    episode_id: "run-high-2", case_id: "flag-2", split: "source-train", score: 0.7,
    score_provenance: { status: "valid", scorer_id: "judge-v2", judge_calls: 1 },
    discovery_attribution: {
      available: true, match_threshold: 0.5, match_matrix: [[0.8]],
      reference_best_match: [0.8], prediction_best_match: [0.8], prediction_insights: ["regime change"],
    },
    steps: [{ question: "Which subgroup changed?", answer: "One regime changed.", tool_names: ["run_python"] }],
  },
];

function primeDiscovery(workspace) {
  for (const episode of episodes) workspace.inspectEpisode(episode.episode_id);
  workspace.compareEpisodes("run-high", "run-low");
  workspace.compareEpisodes("run-high-2", "run-low-2");
}

function discoverySkillInput() {
  return {
    description: "Use this skill when an aggregate pattern may hide a more decision-relevant subgroup or regime.",
    capabilityGap: "Weak runs stopped at aggregates while stronger runs found a new explanatory branch.",
    discoveryGain: "Across two cases, stronger runs expanded the search and found a newly covered subgroup or regime.",
    instructions: "Use this skill when an aggregate signal appears. Search and rank candidate branches. Do not use it without a measurable signal.",
    positiveEvidenceEpisodeIds: ["run-high", "run-high-2"],
    negativeEvidenceEpisodeIds: ["run-low", "run-low-2"],
  };
}

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

  for (const episode of episodes) {
    await find("inspect_episode").execute("inspect", { episode_id: episode.episode_id });
  }
  const comparison = resultJson(await find("compare_episodes").execute("compare-1", {
    positive_episode_id: "run-high", negative_episode_id: "run-low",
  }));
  assert.deepEqual(comparison.gained_reference_slots, [0]);
  assert.equal(comparison.gained_prediction_insights[0].insight, "composition");
  await find("compare_episodes").execute("compare-2", {
    positive_episode_id: "run-high-2", negative_episode_id: "run-low-2",
  });
  await find("create_skill").execute("3", {
    name: "decompose-observed-shifts",
    description: "Separate composition changes from within-group changes before explaining an aggregate shift.",
    capability_gap: "Weak runs described the aggregate while stronger runs tested competing decomposition explanations.",
    discovery_gain: "In two cases the stronger path opened a new subgroup or regime branch that covered a missed reference slot.",
    positive_evidence_episode_ids: ["run-high", "run-high-2"],
    negative_evidence_episode_ids: ["run-low", "run-low-2"],
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
    content: "def run(sql_results, skill_args):\n    return {'groups': int(sql_results.shape[0])}\n",
  });
  await find("write_skill_asset").execute("5", {
    skill_name: "decompose-observed-shifts",
    path: "tests/test_decompose.py",
    content: [
      "from scripts.decompose import run",
      "class Fixture:",
      "    shape = (2, 1)",
      "assert run(Fixture(), {})['groups'] == 2",
      "",
    ].join("\n"),
  });
  const validation = resultJson(await find("validate_skill_set").execute("6", {}));
  assert.equal(validation.passed, true, validation.errors.join("\n"));
  const submitted = resultJson(await find("submit_skill_set").execute("7", {}));
  assert.equal(submitted.passed, true);
  assert.equal(existsSync(join(output, "decompose-observed-shifts", "SKILL.md")), true);

  const runtime = await NativeSkillRuntime.load([output]);
  assert.deepEqual(runtime.skills.map((skill) => skill.name), ["decompose-observed-shifts"]);
  const provenance = JSON.parse(readFileSync(
    join(output, "decompose-observed-shifts", "provenance.json"), "utf8",
  ));
  assert.deepEqual(provenance.positive_evidence_episode_ids, ["run-high", "run-high-2"]);
  assert.deepEqual(provenance.negative_evidence_episode_ids, ["run-low", "run-low-2"]);
  assert.deepEqual(provenance.compared_case_ids.sort(), ["flag-1", "flag-2"]);
  assert.match(readFileSync(join(output, "creator_trajectory.jsonl"), "utf8"), /compare_episodes/);
});

test("creator cannot cite episodes it did not inspect", (t) => {
  const parent = mkdtempSync(join(tmpdir(), "pi-skill-creator-"));
  t.after(() => rmSync(parent, { recursive: true, force: true }));
  const workspace = new SkillCreatorWorkspace({ outputDir: join(parent, "skills"), episodes });
  assert.throws(() => workspace.createSkill({
    name: "uninspected-evidence",
    description: "A valid description.",
    capabilityGap: "A gap.",
    discoveryGain: "A stronger run found an extra pattern.",
    instructions: "Use this skill when appropriate. Do not use it otherwise.",
    positiveEvidenceEpisodeIds: ["run-high", "run-high-2"],
    negativeEvidenceEpisodeIds: ["run-low", "run-low-2"],
  }), /inspect evidence episode/);
  workspace.cleanup();
});

test("creator enforces the experiment's candidate Skill budget", (t) => {
  const parent = mkdtempSync(join(tmpdir(), "pi-skill-creator-"));
  t.after(() => rmSync(parent, { recursive: true, force: true }));
  const workspace = new SkillCreatorWorkspace({
    outputDir: join(parent, "skills"), episodes, maxSkills: 1,
  });
  primeDiscovery(workspace);
  const input = discoverySkillInput();
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
  primeDiscovery(workspace);
  workspace.createSkill({
    name: "leaky-method",
    ...discoverySkillInput(),
    instructions: [
      "Use this skill when explanations compete.",
      "Compute private_training_column before deciding.",
      "Do not use it without computed evidence.",
    ].join("\n"),
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
  primeDiscovery(workspace);
  workspace.createSkill({
    name: "tested-method",
    ...discoverySkillInput(),
    instructions: "Use this skill when explanations compete. Do not use it without computed evidence.",
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
