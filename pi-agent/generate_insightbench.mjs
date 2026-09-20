/** Generate one immutable pi-agent prediction. This command never scores it. */
import { existsSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { explore } from "./src/agent.mjs";
import {
  buildRunDirectory, createRunDirectory, gitState, loadBenchmarkCase, makeManifest,
  sha256Files, validateDataSplit, writeJsonAtomic,
} from "./src/experiment.mjs";
import { assertCaseSplit, splitRegistrySha256 } from "./src/split-registry.mjs";
import { NativeSkillRuntime, hashSkillDirectories } from "./src/native-skills.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = process.env.REPO_ROOT ?? resolve(HERE, "..");
const PY = process.env.PYTHON_BIN ?? "python";

function arg(name, dflt) {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : dflt;
}
const has = (name) => process.argv.includes(`--${name}`);

const benchmarkKind = arg("benchmark-kind", "insightbench");
const caseNumber = Number(arg(benchmarkKind === "insighteval" ? "instance" : "flag", 1));
const defaultBenchmark = benchmarkKind === "insighteval"
  ? `${REPO}/run_on_benchmark/InsightEval-official`
  : `${REPO}/run_on_benchmark/insight-bench`;
const BENCH = arg("benchmark-dir", process.env.BENCH_DIR ?? defaultBenchmark);
const layers = Number(arg("layers", 3));
const questions = Number(arg("questions", 2));
const maxQuestionsArg = arg("max-questions", "");
const maxQuestions = maxQuestionsArg === "" ? null : Number(maxQuestionsArg);
const poolSize = Number(arg("pool", 4));
const maxInsights = Number(arg("max-insights", 12));
const summarySamples = Number(arg("summary-samples", 3));
const agentRun = Number(arg("agent-run", 1));
const experimentId = arg("experiment", "v41_baseline");
const split = validateDataSplit(arg(
  "split", benchmarkKind === "insighteval" ? "target-test" : "source-train",
));
const useSkills = arg("use-skills", process.env.USE_SKILLS ?? "0") !== "0";
const useInsightBank = arg("use-insight-bank", "1") !== "0";
const goalSufficiencyCheck = arg("goal-sufficiency", "1") !== "0";
const systemId = arg("system", useSkills ? "pi-auto-skills" : "pi-core");
const skillDirectories = arg("skill-dir", process.env.SKILL_DIR ?? "")
  .split(",").map((value) => value.trim()).filter(Boolean);
const model = arg("model", "deepseek-flash");
const reasoning = arg("reasoning", "medium");
const outRoot = arg("out-root", `${REPO}/results/experiments`);
const dryRun = has("dry-run");

if (!["pi-core", "pi-manual-skills", "pi-auto-skills", "pi-skill-candidate"].includes(systemId)) {
  throw new Error(`unsupported PI system id: ${systemId}`);
}
if ((systemId === "pi-core") === useSkills) {
  throw new Error(`system ${systemId} is inconsistent with use-skills=${useSkills ? 1 : 0}`);
}
if (useSkills && !skillDirectories.length) throw new Error(`${systemId} requires --skill-dir`);
if (useSkills) await NativeSkillRuntime.load(skillDirectories, { cwd: HERE });

if (![caseNumber, layers, questions, poolSize, maxInsights, summarySamples, agentRun]
  .concat(maxQuestions === null ? [] : [maxQuestions]).every(Number.isFinite)) {
  throw new Error("numeric arguments must be valid numbers");
}

if (benchmarkKind === "insighteval" && split !== "target-test") {
  throw new Error("InsightEval is reserved for target-test and must not be used for tuning");
}
const benchmarkCase = loadBenchmarkCase({ benchmarkKind, benchmarkDir: BENCH, caseNumber });
const { benchmarkId, caseId, goal, csvPath, userCsvPath } = benchmarkCase;
assertCaseSplit(benchmarkId, caseId, split);
if (!existsSync(csvPath)) throw new Error(`dataset not found: ${csvPath}`);

const runDir = buildRunDirectory({ outRoot, experimentId, systemId, caseId, agentRun });
const repoGit = gitState(REPO);
const benchmarkGit = gitState(BENCH);
const promptFiles = [
  `${HERE}/src/planner.mjs`, `${HERE}/src/modules.mjs`, `${HERE}/src/agent.mjs`,
  `${HERE}/src/research-loop.mjs`, `${HERE}/src/native-skills.mjs`,
];
const config = {
  model, reasoning, thinking_mode: true, layers, questions_per_layer: questions, pool_size: poolSize,
  max_questions: maxQuestions, max_insights: maxInsights, summary_samples: summarySamples,
  use_skills: useSkills, use_insight_bank: useInsightBank,
  goal_sufficiency_check: goalSufficiencyCheck,
};
const baseManifest = makeManifest({
  experiment_id: experimentId, system_id: systemId, case_id: caseId, benchmark_id: benchmarkId,
  split,
  agent_run: agentRun, status: "planned", generation_model: model,
  scorer_model: null, repository: repoGit, benchmark: benchmarkGit,
  prompt_hash: sha256Files(promptFiles), skill_hash: useSkills ? hashSkillDirectories(skillDirectories) : null,
  split_registry_hash: splitRegistrySha256(),
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
    questionsPerLayer: questions, maxQuestions, poolSize, model, reasoning, pythonBin: PY,
    workerScript: fileURLToPath(new URL("./python/worker.py", import.meta.url)),
    useSkills, useInsightBank, goalSufficiencyCheck, maxInsights, summarySamples,
    skillDirectories: useSkills ? skillDirectories : [],
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
    schema_version: 1, system_id: systemId, case_id: caseId, benchmark_id: benchmarkId, split, goal,
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
