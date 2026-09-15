/** Generate one immutable pi-agent prediction. This command never scores it. */
import { readFileSync, existsSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { explore } from "./src/agent.mjs";
import {
  buildRunDirectory, createRunDirectory, gitState, makeManifest, sha256Files, writeJsonAtomic,
} from "./src/experiment.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = process.env.REPO_ROOT ?? resolve(HERE, "..");
const BENCH = process.env.BENCH_DIR ?? `${REPO}/run_on_benchmark/insight-bench`;
const PY = process.env.PYTHON_BIN ?? "python";

function arg(name, dflt) {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : dflt;
}
const has = (name) => process.argv.includes(`--${name}`);

const flagNum = Number(arg("flag", 1));
const layers = Number(arg("layers", 3));
const questions = Number(arg("questions", 2));
const maxQuestionsArg = arg("max-questions", "");
const maxQuestions = maxQuestionsArg === "" ? null : Number(maxQuestionsArg);
const poolSize = Number(arg("pool", 4));
const maxInsights = Number(arg("max-insights", 12));
const summarySamples = Number(arg("summary-samples", 3));
const agentRun = Number(arg("agent-run", 1));
const experimentId = arg("experiment", "v41_baseline");
const useSkills = arg("use-skills", process.env.USE_SKILLS ?? "1") !== "0";
const useInsightBank = arg("use-insight-bank", "1") !== "0";
const goalSufficiencyCheck = arg("goal-sufficiency", "1") !== "0";
const systemId = arg("system", useSkills ? "pi-core-skills" : "pi-core");
const model = arg("model", "deepseek-flash");
const outRoot = arg("out-root", `${REPO}/results/experiments`);
const dryRun = has("dry-run");

if (![flagNum, layers, questions, poolSize, maxInsights, summarySamples, agentRun]
  .concat(maxQuestions === null ? [] : [maxQuestions]).every(Number.isFinite)) {
  throw new Error("numeric arguments must be valid numbers");
}

const caseId = `flag-${flagNum}`;
const metaPath = `${BENCH}/data/notebooks/${caseId}.json`;
if (!existsSync(metaPath)) throw new Error(`benchmark case not found: ${metaPath}`);
const meta = JSON.parse(readFileSync(metaPath, "utf8"));
const goal = meta.metadata?.goal ?? "Find interesting trends in this dataset";
const csvPath = `${BENCH}/${meta.dataset_csv_path}`;
const userCsvPath = meta.user_dataset_csv_path ? `${BENCH}/${meta.user_dataset_csv_path}` : null;
if (!existsSync(csvPath)) throw new Error(`dataset not found: ${csvPath}`);

const runDir = buildRunDirectory({ outRoot, experimentId, systemId, caseId, agentRun });
const repoGit = gitState(REPO);
const benchmarkGit = gitState(BENCH);
const promptFiles = [
  `${HERE}/src/planner.mjs`, `${HERE}/src/modules.mjs`, `${HERE}/src/agent.mjs`,
];
const skillPath = process.env.SKILL_PACKAGE_PATH ?? "";
const config = {
  model, layers, questions_per_layer: questions, pool_size: poolSize,
  max_questions: maxQuestions, max_insights: maxInsights, summary_samples: summarySamples,
  use_skills: useSkills, use_insight_bank: useInsightBank,
  goal_sufficiency_check: goalSufficiencyCheck,
};
const baseManifest = makeManifest({
  experiment_id: experimentId, system_id: systemId, case_id: caseId,
  agent_run: agentRun, status: "planned", generation_model: model,
  scorer_model: null, repository: repoGit, benchmark: benchmarkGit,
  prompt_hash: sha256Files(promptFiles), skill_hash: skillPath ? sha256Files([skillPath]) : null,
  config,
});

if (dryRun) {
  console.log(JSON.stringify({ run_directory: runDir, manifest: baseManifest }, null, 2));
  process.exit(0);
}

createRunDirectory(runDir);
writeJsonAtomic(`${runDir}/manifest.json`, { ...baseManifest, status: "running" });
const t0 = Date.now();
try {
  const result = await explore({
    csvPath, userCsvPath, tableName: "incidents", goal, maxLayers: layers,
    questionsPerLayer: questions, maxQuestions, poolSize, model, pythonBin: PY,
    workerScript: new URL("./python/worker.py", import.meta.url).pathname,
    useSkills, useInsightBank, goalSufficiencyCheck, maxInsights, summarySamples,
    onLog: (message) => console.log("  ·", message),
  });
  const raw = result.nodes
    .filter((node) => node.answer && !node.answer.startsWith("Execution failed"))
    .map((node) => `${node.question} ${node.answer}`.trim());
  const byId = new Map(result.nodes.map((node) => [node.id, node]));
  const bank = Object.entries(result.insightBank).map(([id, text]) => {
    const node = byId.get(id);
    return node ? `${node.question} ${text}`.trim() : text;
  });
  const prediction = {
    schema_version: 1, system_id: systemId, case_id: caseId, goal,
    generation_model: model, pred_insights: bank.length ? bank : raw,
    pred_insights_raw: raw, pred_insights_bank: bank, pred_summary: result.summary,
  };
  writeJsonAtomic(`${runDir}/prediction.json`, prediction);
  writeJsonAtomic(`${runDir}/usage.json`, result.usage);
  writeFileSync(
    `${runDir}/trajectory.jsonl`,
    result.nodes.map((node) => JSON.stringify(node)).join("\n") + "\n", "utf8",
  );
  writeJsonAtomic(`${runDir}/manifest.json`, {
    ...baseManifest, status: "success", completed_at: new Date().toISOString(),
    elapsed_sec: Number(((Date.now() - t0) / 1000).toFixed(3)),
    layers_run: result.layersRun, stopped_early: result.stoppedEarly,
    skill_package: result.skillPackage,
  });
  console.log(`saved immutable prediction → ${runDir}`);
} catch (error) {
  writeJsonAtomic(`${runDir}/manifest.json`, {
    ...baseManifest, status: "failed", completed_at: new Date().toISOString(),
    elapsed_sec: Number(((Date.now() - t0) / 1000).toFixed(3)),
    error: { name: error?.name ?? "Error", message: error?.message ?? String(error) },
  });
  throw error;
}
