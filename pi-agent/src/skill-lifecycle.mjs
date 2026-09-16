import { createHash } from "node:crypto";
import {
  existsSync, mkdirSync, readFileSync, readdirSync, renameSync, writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";
import { randomUUID } from "node:crypto";
import { assertCaseSplit } from "./split-registry.mjs";

const ALLOWED_STAGES = new Set(["planner", "executor", "sufficiency", "insight_bank", "summary"]);
const RUNTIME_FIELDS = ["trigger", "action", "rationale", "implementation"];

function nested(value, dotted) {
  return dotted.split(".").reduce((current, key) => current?.[key], value);
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function numericValues(value) {
  const values = Array.isArray(value) ? value : [value];
  return values.map(Number).filter(Number.isFinite);
}

function walk(root, filename, output = []) {
  if (!existsSync(root)) return output;
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const path = join(root, entry.name);
    if (entry.isDirectory()) walk(path, filename, output);
    else if (entry.name === filename) output.push(path);
  }
  return output;
}

function slug(value) {
  return String(value ?? "")
    .toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 80);
}

function sortDeep(value) {
  if (Array.isArray(value)) return value.map(sortDeep);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.keys(value).sort().map((key) => [key, sortDeep(value[key])]));
}

export function canonicalJson(value) {
  return JSON.stringify(sortDeep(value));
}

export function packageHash(pkg) {
  const copy = structuredClone(pkg);
  if (copy.meta) delete copy.meta.package_hash;
  return createHash("sha256").update(canonicalJson(copy)).digest("hex");
}

export function mineEpisodes(experimentDir, {
  split = "source-train",
  scorerId = "local-deepseek-v41-thinking-v1",
  metric = "semantic.primary.f1",
} = {}) {
  if (split !== "source-train") throw new Error("skill mining is restricted to source-train");
  const episodes = [];
  for (const manifestPath of walk(experimentDir, "manifest.json").sort()) {
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    if (manifest.status !== "success" || manifest.split !== split) continue;
    assertCaseSplit(manifest.benchmark_id, manifest.case_id, split);
    const runDir = dirname(manifestPath);
    const scoreDir = join(runDir, "scores", scorerId);
    const scores = existsSync(scoreDir)
      ? readdirSync(scoreDir).filter((name) => /^judge_run_\d+\.json$/.test(name)).sort()
        .map((name) => nested(JSON.parse(readFileSync(join(scoreDir, name), "utf8")), metric))
        .map(Number).filter(Number.isFinite)
      : [];
    const trajectoryPath = join(runDir, "trajectory.jsonl");
    if (!scores.length || !existsSync(trajectoryPath)) continue;
    const steps = readFileSync(trajectoryPath, "utf8").split(/\r?\n/).filter(Boolean).map(JSON.parse);
    episodes.push({
      episode_id: manifest.run_id,
      benchmark_id: manifest.benchmark_id,
      case_id: manifest.case_id,
      split,
      system_id: manifest.system_id,
      agent_run: manifest.agent_run,
      score: mean(scores),
      n_judge_runs: scores.length,
      prompt_hash: manifest.prompt_hash,
      skill_hash: manifest.skill_hash,
      source_manifest: manifestPath,
      steps: steps.map((step) => ({
        node_id: step.id,
        layer: step.layer,
        category: step.category,
        question: step.question,
        answer: step.answer,
        turns: step.turns,
        tool_names: (step.toolCalls ?? []).map((call) => call?.name).filter(Boolean),
      })),
    });
  }
  return episodes;
}

export function normalizeCandidate(raw, provenance = {}) {
  const candidate = {
    id: slug(raw.id || raw.name || raw.trigger),
    trigger: String(raw.trigger ?? "").trim(),
    trigger_terms: [...new Set((raw.trigger_terms ?? []).map((term) => String(term).trim().toLowerCase()).filter(Boolean))],
    stages: [...new Set((raw.stages ?? []).map(String))],
    action: String(raw.action ?? "").trim(),
    rationale: String(raw.rationale ?? "").trim(),
    implementation: String(raw.implementation ?? "prompt-guidance").trim(),
    provenance: {
      source_runs: [...new Set([...(raw.provenance?.source_runs ?? []), ...(provenance.source_runs ?? [])])],
      source_cases: [...new Set([...(raw.provenance?.source_cases ?? []), ...(provenance.source_cases ?? [])])],
      extraction_prompt_hash: provenance.extraction_prompt_hash ?? raw.provenance?.extraction_prompt_hash,
    },
    status: "candidate",
  };
  if (!candidate.id || !candidate.trigger || !candidate.action || !candidate.rationale) {
    throw new Error("candidate requires id, trigger, action, and rationale");
  }
  if (!candidate.stages.length || candidate.stages.some((stage) => !ALLOWED_STAGES.has(stage))) {
    throw new Error(`candidate ${candidate.id} has invalid stages`);
  }
  if (!candidate.trigger_terms.length && !raw.always) {
    throw new Error(`candidate ${candidate.id} requires trigger_terms or always=true`);
  }
  if (raw.always) candidate.always = true;
  return candidate;
}

function extractionPrompt(episodes) {
  return [
    "You are the extraction agent for a data-analysis agent framework.",
    "Infer reusable procedural skills from the source-train trajectories below.",
    "A skill must describe when to take a general analytical action. It must not contain a dataset answer, literal column name, entity, date, threshold, or benchmark label.",
    "Return JSON: {candidates:[{id,trigger,trigger_terms,stages,action,rationale,implementation}]}",
    `EPISODES:\n${JSON.stringify(episodes)}`,
  ].join("\n\n");
}

function generalizationPrompt(candidate) {
  return [
    "You are the generalization agent. Rewrite this candidate as a dataset-independent procedural skill.",
    "Remove literal columns, entities, dates, values, answers, and benchmark names. Preserve the useful action.",
    "Use placeholders such as {categorical_columns} only when runtime schema substitution is necessary.",
    "Return one JSON object with id, trigger, trigger_terms, stages, action, rationale, implementation, and optional always.",
    JSON.stringify(candidate),
  ].join("\n\n");
}

export async function generalizeCandidates(rawCandidates, episodes, { generateJson, extractionPromptHash }) {
  const sourceRuns = episodes.map((episode) => episode.episode_id).filter(Boolean);
  const sourceCases = episodes.map((episode) => episode.case_id).filter(Boolean);
  const candidates = [];
  for (const raw of rawCandidates) {
    const generalized = await generateJson(generalizationPrompt(raw), { temperature: 0.1 });
    candidates.push(normalizeCandidate(generalized, {
      source_runs: sourceRuns,
      source_cases: sourceCases,
      extraction_prompt_hash: extractionPromptHash,
    }));
  }
  return mergeCandidates(candidates);
}

export async function extractAndGeneralizeSkills(episodes, { generateJson }) {
  if (!episodes.length) throw new Error("cannot extract skills from an empty corpus");
  if (episodes.some((episode) => episode.split !== "source-train")) {
    throw new Error("extraction corpus contains a non-training episode");
  }
  const prompt = extractionPrompt(episodes);
  const extracted = await generateJson(prompt, { temperature: 0.2 });
  if (!Array.isArray(extracted?.candidates)) throw new Error("extraction agent returned no candidates array");
  return generalizeCandidates(extracted.candidates, episodes, {
    generateJson,
    extractionPromptHash: createHash("sha256").update(prompt).digest("hex"),
  });
}

export function mergeCandidates(candidates) {
  const merged = new Map();
  for (const candidate of candidates) {
    const previous = merged.get(candidate.id);
    if (!previous) {
      merged.set(candidate.id, structuredClone(candidate));
      continue;
    }
    if (previous.action !== candidate.action || previous.trigger !== candidate.trigger) {
      throw new Error(`conflicting candidates share id ${candidate.id}`);
    }
    previous.trigger_terms = [...new Set([...previous.trigger_terms, ...candidate.trigger_terms])];
    previous.stages = [...new Set([...previous.stages, ...candidate.stages])];
    previous.provenance.source_runs = [...new Set([
      ...previous.provenance.source_runs, ...candidate.provenance.source_runs,
    ])];
    previous.provenance.source_cases = [...new Set([
      ...previous.provenance.source_cases, ...candidate.provenance.source_cases,
    ])];
  }
  return [...merged.values()];
}

export function auditSkillPackage(pkg, { forbiddenTerms = [] } = {}) {
  const findings = [];
  const terms = forbiddenTerms.map((term) => String(term).trim().toLowerCase()).filter(Boolean);
  for (const skill of pkg.skills ?? []) {
    for (const field of RUNTIME_FIELDS) {
      const text = String(skill[field] ?? "").toLowerCase();
      const hits = [...new Set(terms.filter((term) => text.includes(term)))];
      if (hits.length) findings.push({ skill_id: skill.id, field, kind: "forbidden-term", terms: hits });
      if (/`[^`]+`/.test(text)) findings.push({ skill_id: skill.id, field, kind: "literal-identifier" });
    }
    if (!skill.provenance?.source_runs?.length) {
      findings.push({ skill_id: skill.id, field: "provenance", kind: "missing-source-runs" });
    }
    if (!skill.stages?.length || skill.stages.some((stage) => !ALLOWED_STAGES.has(stage))) {
      findings.push({ skill_id: skill.id, field: "stages", kind: "invalid-stage" });
    }
  }
  return { passed: findings.length === 0, findings, n_terms: terms.length };
}

export function validateAblations(records, {
  split = "source-valid", minCases = 2, minRunsPerArm = 2,
  minMeanDelta = 0, maxCostRatio = 1.5,
} = {}) {
  const grouped = new Map();
  for (const record of records) {
    if (record.split !== split) continue;
    const rows = grouped.get(record.skill_id) ?? [];
    rows.push(record);
    grouped.set(record.skill_id, rows);
  }
  const results = {};
  for (const [skillId, rows] of grouped) {
    const cases = {};
    for (const row of rows) {
      assertCaseSplit(row.benchmark_id ?? "insightbench-overhaul", row.case_id, split);
      const treated = numericValues(row.treated);
      const control = numericValues(row.control);
      const treatedCost = numericValues(row.treated_cost ?? 0);
      const controlCost = numericValues(row.control_cost ?? 0);
      const enoughRuns = treated.length >= minRunsPerArm && control.length >= minRunsPerArm;
      const delta = treated.length && control.length ? mean(treated) - mean(control) : null;
      const costRatio = mean(controlCost) > 0 ? mean(treatedCost) / mean(controlCost) : 1;
      cases[row.case_id] = { delta, cost_ratio: costRatio, enough_runs: enoughRuns };
    }
    const values = Object.values(cases);
    const deltas = values.map((value) => value.delta).filter(Number.isFinite);
    const passed = values.length >= minCases
      && values.every((value) => value.enough_runs && value.delta > 0 && value.cost_ratio <= maxCostRatio)
      && mean(deltas) >= minMeanDelta;
    results[skillId] = {
      passed,
      split,
      protocol: { min_cases: minCases, min_runs_per_arm: minRunsPerArm, min_mean_delta: minMeanDelta, max_cost_ratio: maxCostRatio },
      n_cases: values.length,
      per_case: cases,
      mean_delta: deltas.length ? mean(deltas) : null,
    };
  }
  return results;
}

export function freezeSkillPackage(candidatePackage, { audit, validations, version }) {
  if (!audit?.passed) throw new Error("purity audit failed");
  if (!candidatePackage.skills?.length) throw new Error("cannot freeze an empty package");
  const frozen = structuredClone(candidatePackage);
  for (const skill of frozen.skills) {
    if (!skill.provenance?.source_runs?.length) throw new Error(`skill ${skill.id} lacks provenance`);
    const validation = validations?.[skill.id];
    if (!validation?.passed) throw new Error(`skill ${skill.id} lacks passed cross-case validation`);
    skill.validated = validation;
    skill.status = "validated";
  }
  frozen.meta = {
    ...(frozen.meta ?? {}),
    version,
    produced_by: "pi-skill-extraction-agent",
    status: "frozen-validated",
    frozen: true,
    frozen_at: new Date().toISOString(),
    purity_audit: audit,
  };
  frozen.meta.package_hash = packageHash(frozen);
  return frozen;
}

export function writeJsonExclusive(path, value) {
  if (existsSync(path)) throw new Error(`refusing to overwrite: ${path}`);
  mkdirSync(dirname(path), { recursive: true });
  const temporary = join(dirname(path), `.${randomUUID()}.tmp`);
  writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  renameSync(temporary, path);
}
