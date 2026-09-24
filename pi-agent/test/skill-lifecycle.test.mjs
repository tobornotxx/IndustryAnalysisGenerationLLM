import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  auditSkillPackage, extractAndGeneralizeSkills, freezeSkillPackage,
  mergeCandidates, mineEpisodes, normalizeCandidate, packageHash, validateAblations,
} from "../src/skill-lifecycle.mjs";

function candidate(overrides = {}) {
  return normalizeCandidate({
    id: "compare-subgroups",
    trigger: "comparison across groups",
    trigger_terms: ["compare", "group"],
    stages: ["executor"],
    action: "Compare the relevant metric across {categorical_columns}.",
    rationale: "Aggregate values can hide subgroup differences.",
    ...overrides,
  }, { source_runs: ["run-1"], source_cases: ["flag-1"] });
}

test("episode miner reads source-train immutable runs and ignores validation runs", () => {
  const root = mkdtempSync(join(tmpdir(), "skill-corpus-"));
  const writeRun = (caseId, split) => {
    const run = join(root, caseId, "agent_run_1");
    mkdirSync(join(run, "scores", "judge"), { recursive: true });
    writeFileSync(join(run, "manifest.json"), JSON.stringify({
      status: "success", split, benchmark_id: "insightbench-overhaul",
      case_id: caseId, run_id: `run-${caseId}`, system_id: "pi-core", agent_run: 1,
    }));
    writeFileSync(join(run, "trajectory.jsonl"), `${JSON.stringify({
      id: "q1", layer: 1, question: "q", answer: "a", toolCalls: [],
    })}\n`);
    const predictionPath = join(run, "prediction.json");
    writeFileSync(predictionPath, JSON.stringify({ case_id: caseId, pred_insights: ["a"] }));
    const predictionSha256 = createHash("sha256").update(
      readFileSync(predictionPath),
    ).digest("hex");
    writeFileSync(join(run, "scores", "judge", "judge_run_1.json"), JSON.stringify({
      prediction: predictionPath,
      prediction_sha256: predictionSha256,
      case_id: caseId,
      judge_run: 1,
      scorer_id: "judge",
      semantic: { primary: { f1: 0.5, matrix: [[0.75]] } },
      usage: { calls: 1 },
    }));
  };
  writeRun("flag-1", "source-train");
  writeRun("flag-9", "source-valid");
  const episodes = mineEpisodes(root, { scorerId: "judge" });
  assert.equal(episodes.length, 1);
  assert.equal(episodes[0].case_id, "flag-1");
  assert.equal(episodes[0].steps[0].question, "q");
  assert.equal(episodes[0].score_provenance.status, "valid");
  assert.equal(episodes[0].discovery_attribution.available, true);
  assert.deepEqual(episodes[0].discovery_attribution.reference_best_match, [0.75]);
  assert.deepEqual(episodes[0].discovery_attribution.prediction_insights, ["a"]);
});

test("answer-informed mining attaches only source-train benchmark reference insights", () => {
  const root = mkdtempSync(join(tmpdir(), "skill-corpus-"));
  const benchmark = mkdtempSync(join(tmpdir(), "skill-benchmark-"));
  const run = join(root, "flag-1", "agent_run_1");
  mkdirSync(join(run, "scores", "judge"), { recursive: true });
  mkdirSync(join(benchmark, "data", "notebooks"), { recursive: true });
  writeFileSync(join(run, "manifest.json"), JSON.stringify({
    status: "success", split: "source-train", benchmark_id: "insightbench-overhaul",
    case_id: "flag-1", run_id: "run-informed", system_id: "pi-core", agent_run: 1,
  }));
  writeFileSync(join(run, "trajectory.jsonl"), `${JSON.stringify({ id: "q1", layer: 1, question: "q", answer: "a" })}\n`);
  const predictionPath = join(run, "prediction.json");
  writeFileSync(predictionPath, JSON.stringify({ pred_insights: ["a"] }));
  const predictionSha256 = createHash("sha256").update(readFileSync(predictionPath)).digest("hex");
  writeFileSync(join(run, "scores", "judge", "judge_run_1.json"), JSON.stringify({
    prediction: predictionPath, prediction_sha256: predictionSha256, case_id: "flag-1", judge_run: 1,
    scorer_id: "judge", semantic: { primary: { f1: 0.5, matrix: [[0.75]] } }, usage: { calls: 1 },
  }));
  writeFileSync(join(benchmark, "data", "notebooks", "flag-1.json"), JSON.stringify({
    insights: ["The reference mechanism is present."],
  }));
  const episodes = mineEpisodes(root, {
    scorerId: "judge", benchmarkDir: benchmark, includeReferenceInsights: true,
  });
  assert.deepEqual(episodes[0].reference_insights, ["The reference mechanism is present."]);
  assert.equal(episodes[0].reference_access, "answer-informed-source-train");
});

test("extraction and generalization are separate injected agent calls", async () => {
  const calls = [];
  const generateJson = async (prompt) => {
    calls.push(prompt);
    if (calls.length === 1) return { candidates: [{ id: "raw", action: "inspect flag-1.category" }] };
    return {
      id: "compare-subgroups", trigger: "comparison across groups",
      trigger_terms: ["compare", "group"], stages: ["executor"],
      action: "Compare the metric across {categorical_columns}.",
      rationale: "Aggregates hide subgroup differences.", implementation: "prompt-guidance",
    };
  };
  const skills = await extractAndGeneralizeSkills([
    { episode_id: "run-1", case_id: "flag-1", split: "source-train", steps: [] },
  ], { generateJson });
  assert.equal(calls.length, 2);
  assert.equal(skills[0].id, "compare-subgroups");
  assert.deepEqual(skills[0].provenance.source_runs, ["run-1"]);
  assert.doesNotMatch(skills[0].action, /flag-1|category/);
});

test("candidate merge refuses semantic conflicts hidden behind one id", () => {
  assert.throws(
    () => mergeCandidates([candidate(), candidate({ action: "Do something incompatible." })]),
    /conflicting candidates/,
  );
});

test("common extraction stage aliases normalize to runtime stages", () => {
  const candidate = normalizeCandidate({
    id: "finish", trigger: "when done", trigger_terms: ["done"],
    stages: ["finalization", "tool-use"], action: "Finish cleanly", rationale: "Avoid drift",
  }, { source_runs: ["r1"], source_cases: ["flag-1"] });
  assert.deepEqual(candidate.stages, ["summary", "executor"]);
});

test("purity audit catches dataset literals and missing provenance", () => {
  const dirty = candidate({ action: "Inspect assigned_to and compare groups." });
  dirty.provenance.source_runs = [];
  const audit = auditSkillPackage({ skills: [dirty] }, { forbiddenTerms: ["assigned_to"] });
  assert.equal(audit.passed, false);
  assert.deepEqual(
    new Set(audit.findings.map((finding) => finding.kind)),
    new Set(["forbidden-term", "literal-identifier", "missing-source-runs"]),
  );
});

test("validation requires positive gains on multiple frozen validation cases", () => {
  const records = ["flag-9", "flag-10"].map((caseId) => ({
    skill_id: "compare-subgroups", benchmark_id: "insightbench-overhaul",
    case_id: caseId, split: "source-valid",
    control: [0.4, 0.42, 0.41], treated: [0.5, 0.51, 0.52],
    control_cost: [1, 1, 1], treated_cost: [1.1, 1.1, 1.1],
  }));
  const result = validateAblations(records, { minRunsPerArm: 3 });
  assert.equal(result["compare-subgroups"].passed, true);
});

test("validation reports autonomous Skill activation separately from score effects", () => {
  const result = validateAblations([{
    skill_id: "method", benchmark_id: "insightbench-overhaul", case_id: "flag-9", split: "source-valid",
    treated: [0.7, 0.4], control: [0.5, 0.5], treated_cost: [1, 1], control_cost: [1, 1],
    treated_skill_read: [1, 0], treated_skill_executed: [0, 0],
  }, {
    skill_id: "method", benchmark_id: "insightbench-overhaul", case_id: "flag-10", split: "source-valid",
    treated: [0.8, 0.6], control: [0.6, 0.6], treated_cost: [1, 1], control_cost: [1, 1],
    treated_skill_read: [1, 1], treated_skill_executed: [1, 0],
  }], { minCases: 2, minRunsPerArm: 2 });
  assert.equal(result.method.activation.read_rate, 0.75);
  assert.equal(result.method.activation.execution_rate, 0.25);
  assert.ok(Math.abs(result.method.activation.paired_delta_when_read - (0.2 + 0.2 + 0) / 3) < 1e-12);
});

test("freeze gate requires audit and validation then content-addresses package", () => {
  const skill = candidate();
  const pkg = { meta: { frozen: false }, skills: [skill], episodes: [], config: {} };
  const validation = { passed: true, split: "source-valid", n_cases: 2 };
  const frozen = freezeSkillPackage(pkg, {
    audit: { passed: true, findings: [] },
    validations: { [skill.id]: validation }, version: "auto-v1",
  });
  assert.equal(frozen.meta.frozen, true);
  assert.equal(frozen.meta.produced_by, "pi-skill-extraction-agent");
  assert.equal(frozen.meta.package_hash, packageHash(frozen));
  assert.throws(() => freezeSkillPackage(pkg, {
    audit: { passed: false }, validations: {}, version: "bad",
  }), /purity audit failed/);
});
