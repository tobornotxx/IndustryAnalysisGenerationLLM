/** Offline-first CLI for mining, extracting, validating, auditing, and freezing skills. */
import { readFileSync } from "node:fs";
import {
  auditSkillPackage, freezeSkillPackage, generalizeCandidates, mineEpisodes,
  validateAblations, writeJsonExclusive,
} from "./src/skill-lifecycle.mjs";
import { createDeepSeek, makeGenerateJson, UsageTracker } from "./src/agent.mjs";
import { runSkillExtractionAgent } from "./src/skill-meta-agent.mjs";

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
    scorerId: arg("scorer-id", "local-deepseek-v41-thinking-v1"),
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
} else if (command === "audit") {
  const pkg = readJson(required("package"));
  const forbiddenPath = required("forbidden-terms");
  const forbiddenTerms = readFileSync(forbiddenPath, "utf8").split(/\r?\n/).filter(Boolean);
  writeJsonExclusive(required("output"), auditSkillPackage(pkg, { forbiddenTerms }));
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
} else {
  throw new Error("command must be one of: mine, extract, audit, validate, freeze");
}
