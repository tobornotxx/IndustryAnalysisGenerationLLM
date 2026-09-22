/** Offline-first CLI for mining, extracting, validating, auditing, and freezing skills. */
import { readFileSync } from "node:fs";
import { basename, join } from "node:path";
import {
  auditSkillPackage, freezeSkillPackage, generalizeCandidates, mineEpisodes,
  validateAblations, writeJsonExclusive,
} from "./src/skill-lifecycle.mjs";
import { createDeepSeek, makeGenerateJson, UsageTracker } from "./src/agent.mjs";
import { runSkillExtractionAgent } from "./src/skill-meta-agent.mjs";
import { runSkillCreatorAgent } from "./src/skill-creator-agent.mjs";
import { auditNativeSkillDirectory } from "./src/native-skills.mjs";
import {
  buildForbiddenVocabulary, collectValidationRecords, freezeDirectorySkillSet,
  prepareDirectoryValidationPlan, prepareValidationPlan,
} from "./src/skill-validation.mjs";

const [command] = process.argv.slice(2);
function arg(name, fallback = undefined) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}
const has = (name) => process.argv.includes(`--${name}`);
const required = (name) => {
  const value = arg(name);
  if (!value) throw new Error(`--${name} is required`);
  return value;
};
const readJson = (path) => JSON.parse(readFileSync(path, "utf8"));

if (command === "mine") {
  const episodes = mineEpisodes(required("experiment-dir"), {
    scorerId: arg("scorer-id", "local-deepseek-v41-thinking-v2"),
    metric: arg("metric", "semantic.primary.f1"),
  });
  writeJsonExclusive(required("output"), {
    schema_version: 1, split: "source-train", episodes,
  });
  console.log(`mined ${episodes.length} source-train episodes`);
} else if (command === "extract") {
  if (!has("allow-api")) {
    throw new Error("extract invokes the configured model; pass --allow-api only after experiment approval");
  }
  const corpus = readJson(required("corpus"));
  if (corpus.split !== "source-train") throw new Error("corpus must be source-train");
  const usage = new UsageTracker();
  const deepseek = createDeepSeek({
    model: arg("model", "deepseek-flash"), reasoning: arg("reasoning", "medium"),
  });
  const generateJson = makeGenerateJson({ ...deepseek, usage });
  const extracted = await runSkillExtractionAgent({ episodes: corpus.episodes ?? [], deepseek, usage });
  const skills = await generalizeCandidates(extracted.candidates, corpus.episodes ?? [], {
    generateJson, extractionPromptHash: extracted.prompt_hash,
  });
  const pkg = {
    meta: {
      version: arg("version", "auto-candidates"),
      produced_by: "pi-skill-extraction-agent",
      status: "candidates-pending-validation",
      frozen: false,
      extraction_usage: usage.toJSON(),
      extraction_turns: extracted.turns,
    },
    skills,
    episodes: (corpus.episodes ?? []).map((episode) => ({
      episode_id: episode.episode_id, case_id: episode.case_id, score: episode.score,
    })),
    config: {},
  };
  writeJsonExclusive(required("output"), pkg);
  console.log(`extracted ${skills.length} generalized candidate skills`);
} else if (command === "create") {
  if (!has("allow-api")) {
    throw new Error("create invokes the configured model; pass --allow-api only after experiment approval");
  }
  const corpus = readJson(required("corpus"));
  if (corpus.split !== "source-train") throw new Error("corpus must be source-train");
  const usage = new UsageTracker();
  const deepseek = createDeepSeek({
    model: arg("model", "deepseek-flash"), reasoning: arg("reasoning", "medium"),
  });
  const seedSkillDir = arg("seed-skill-dir");
  const seedSkill = seedSkillDir ? {
    name: basename(seedSkillDir),
    instructions: readFileSync(join(seedSkillDir, "SKILL.md"), "utf8"),
    provenance: readJson(join(seedSkillDir, "provenance.json")),
  } : null;
  const result = await runSkillCreatorAgent({
    episodes: corpus.episodes ?? [],
    deepseek,
    usage,
    outputDir: required("output-dir"),
    maxTurns: Number(arg("max-turns", 20)),
    maxSkills: Number(arg("max-skills", 3)),
    forbiddenTerms: arg("forbidden-terms")
      ? (readJson(arg("forbidden-terms")).terms ?? readJson(arg("forbidden-terms")))
      : [],
    seedSkill,
    revisionBrief: arg("revision-brief") ? readFileSync(arg("revision-brief"), "utf8") : "",
  });
  console.log(JSON.stringify({ ...result, usage: usage.toJSON() }, null, 2));
} else if (command === "audit") {
  const pkg = readJson(required("package"));
  const forbiddenPath = required("forbidden-terms");
  const forbiddenText = readFileSync(forbiddenPath, "utf8");
  let forbiddenTerms;
  try {
    const parsed = JSON.parse(forbiddenText);
    forbiddenTerms = Array.isArray(parsed) ? parsed : parsed.terms;
  } catch {
    forbiddenTerms = forbiddenText.split(/\r?\n/).filter(Boolean);
  }
  if (!Array.isArray(forbiddenTerms)) throw new Error("forbidden terms must be a JSON array/object or line list");
  writeJsonExclusive(required("output"), auditSkillPackage(pkg, { forbiddenTerms }));
} else if (command === "audit-directory") {
  const forbiddenText = readFileSync(required("forbidden-terms"), "utf8");
  let forbiddenTerms;
  try {
    const parsed = JSON.parse(forbiddenText);
    forbiddenTerms = Array.isArray(parsed) ? parsed : parsed.terms;
  } catch {
    forbiddenTerms = forbiddenText.split(/\r?\n/).filter(Boolean);
  }
  if (!Array.isArray(forbiddenTerms)) throw new Error("forbidden terms must be a JSON array/object or line list");
  writeJsonExclusive(required("output"), await auditNativeSkillDirectory(
    required("skill-dir"), { forbiddenTerms },
  ));
} else if (command === "vocabulary") {
  const caseIds = required("cases").split(",").map((value) => value.startsWith("flag-") ? value : `flag-${value}`);
  writeJsonExclusive(required("output"), {
    schema_version: 1,
    source: "source-train schema and metadata",
    terms: buildForbiddenVocabulary({ benchmarkDir: required("benchmark-dir"), caseIds }),
  });
} else if (command === "prepare-validation") {
  const packagePath = required("package");
  const outputDir = required("output-dir");
  const caseIds = arg("cases", "9,10,11,12").split(",").map((value) => value.startsWith("flag-") ? value : `flag-${value}`);
  const plan = prepareValidationPlan(readJson(packagePath), {
    outputDir, caseIds, agentRuns: Number(arg("agent-runs", 3)), experimentTag: arg("tag", ""),
  });
  writeJsonExclusive(required("output"), plan);
} else if (command === "prepare-directory-validation") {
  const outputDir = required("output-dir");
  const caseIds = arg("cases", "9,10,11,12").split(",").map((value) => value.startsWith("flag-") ? value : `flag-${value}`);
  const plan = await prepareDirectoryValidationPlan(required("skill-dir"), {
    outputDir,
    caseIds,
    agentRuns: Number(arg("agent-runs", 3)),
    experimentTag: arg("tag", "native-v1"),
    skillNames: arg("skills", "").split(",").map((value) => value.trim()).filter(Boolean),
    controlExperimentId: arg("control-experiment-id", null),
  });
  writeJsonExclusive(required("output"), plan);
} else if (command === "collect-validation") {
  const records = collectValidationRecords(readJson(required("plan")), {
    outRoot: required("out-root"),
    scorerId: arg("scorer-id", "local-deepseek-v41-thinking-v2"),
    metric: arg("metric", "semantic.primary.f1"),
  });
  writeJsonExclusive(required("output"), records);
} else if (command === "validate") {
  const records = readJson(required("records"));
  writeJsonExclusive(required("output"), validateAblations(records, {
    minCases: Number(arg("min-cases", 2)),
    minRunsPerArm: Number(arg("min-runs", 3)),
    minMeanDelta: Number(arg("min-mean-delta", 0)),
    maxCostRatio: Number(arg("max-cost-ratio", 1.5)),
  }));
} else if (command === "freeze") {
  const frozen = freezeSkillPackage(readJson(required("package")), {
    audit: readJson(required("audit")),
    validations: readJson(required("validations")),
    version: required("version"),
  });
  writeJsonExclusive(required("output"), frozen);
  console.log(`frozen ${frozen.skills.length} validated skills as ${frozen.meta.version}`);
} else if (command === "freeze-directory") {
  const manifest = await freezeDirectorySkillSet(
    required("skill-dir"),
    readJson(required("validations")),
    { outputDir: required("output-dir"), version: required("version") },
  );
  console.log(JSON.stringify(manifest, null, 2));
} else {
  throw new Error("command must be one of: mine, extract, create, vocabulary, audit, audit-directory, prepare-validation, prepare-directory-validation, collect-validation, validate, freeze, freeze-directory");
}
