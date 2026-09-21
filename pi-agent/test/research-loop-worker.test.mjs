import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { PyWorkerPool } from "../src/py-worker.mjs";
import { ResearchState, createResearchTools } from "../src/research-loop.mjs";
import { NativeSkillRuntime, createNativeSkillTools } from "../src/native-skills.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const WORKER = join(HERE, "..", "python", "worker.py");
const VENV_PYTHON = process.platform === "win32"
  ? join(HERE, "..", "..", ".venv", "Scripts", "python.exe")
  : join(HERE, "..", "..", ".venv", "bin", "python");

function findTool(tools, name) {
  return tools.find((tool) => tool.name === name);
}

function resultText(result) {
  return result.content[0].text;
}

test("research loop tools preserve real SQL and Python evidence through submission", async (t) => {
  const root = mkdtempSync(join(tmpdir(), "pi-research-worker-"));
  const csv = join(root, "fixture.csv");
  writeFileSync(csv, "segment,value\nA,10\nA,20\nB,5\n", "utf8");
  t.after(() => rmSync(root, { recursive: true, force: true }));

  const python = process.env.PYTHON_BIN ?? (existsSync(VENV_PYTHON) ? VENV_PYTHON : "python");
  const pool = new PyWorkerPool({ python, workerScript: WORKER, size: 1 });
  t.after(async () => pool.close());
  await pool.ready();
  const loaded = await pool.loadAll({ csv_path: csv, table_name: "main_table" });
  assert.match(loaded.schema_context, /segment/);

  const state = new ResearchState({ goal: "compare segment totals", maxQuestions: 1 });
  const planner = {
    dbDescription: loaded.schema_context,
    async generateInitialQuestions() { return []; },
    async generateTreeBasedQuestions() { return { followUps: [], exploratory: [] }; },
  };
  const tools = createResearchTools({
    state,
    planner,
    pool,
    evaluateSufficiency: async ({ insights }) => ({
      sufficient: insights.some((value) => value.includes("30")),
      missingAspects: [],
      reasoning: "segment totals were computed",
    }),
  });

  await findTool(tools, "start_question").execute("1", {
    question: "What is the total value by segment?",
    category: "exploratory",
    parent_ids: [],
  });
  const sql = await findTool(tools, "run_sql").execute("2", {
    question_id: "q_001",
    sql: "SELECT segment, SUM(value) AS total FROM main_table GROUP BY segment ORDER BY segment",
  });
  assert.match(resultText(sql), /30/);
  assert.match(resultText(sql), /5/);

  const pythonResult = await findTool(tools, "run_python").execute("3", {
    question_id: "q_001",
    sql: "SELECT segment, SUM(value) AS total FROM main_table GROUP BY segment ORDER BY segment",
    code: "print(float(sql_results['total'].max() / sql_results['total'].min()))",
  });
  assert.match(resultText(pythonResult), /6\.0/);

  await findTool(tools, "record_finding").execute("4", {
    question_id: "q_001",
    finding: "Segment A totals 30 versus 5 for B, a sixfold difference.",
    limitations: ["The fixture contains only three rows."],
  });
  await findTool(tools, "review_progress").execute("5", {});
  const submission = JSON.parse(resultText(await findTool(tools, "submit_analysis").execute("6", {
    summary: "Segment A contributes 30 and segment B contributes 5.",
  })));
  assert.equal(submission.accepted, true);
  assert.equal(state.nodes[0].evidence.length, 2);
  assert.deepEqual(state.nodes[0].toolCalls, ["run_sql", "run_python"]);
});

test("native Skill runtime executes one adaptive run entrypoint against real SQL data", async (t) => {
  const root = mkdtempSync(join(tmpdir(), "pi-skill-worker-"));
  const csv = join(root, "fixture.csv");
  const skillDir = join(root, "skills", "distribution-audit");
  mkdirSync(join(skillDir, "scripts"), { recursive: true });
  writeFileSync(csv, "segment,value\nA,10\nA,20\nB,5\n", "utf8");
  writeFileSync(join(skillDir, "SKILL.md"), [
    "---", "name: distribution-audit",
    "description: Audit a numeric distribution when a group comparison depends on it.", "---",
    "Use this local diagnostic when group comparisons depend on a numeric value.",
  ].join("\n"), "utf8");
  writeFileSync(join(skillDir, "scripts", "audit.py"), [
    "from __future__ import annotations",
    "def run(sql_results, skill_args):",
    "    totals = sql_results.groupby(skill_args['group_col'])[skill_args['value_col']].sum()",
    "    return {'applicable': True, 'checks': ['group totals'], 'max_total': int(totals.max())}",
    "",
  ].join("\n"), "utf8");
  t.after(() => rmSync(root, { recursive: true, force: true }));

  const python = process.env.PYTHON_BIN ?? (existsSync(VENV_PYTHON) ? VENV_PYTHON : "python");
  const pool = new PyWorkerPool({ python, workerScript: WORKER, size: 1 });
  t.after(async () => pool.close());
  await pool.ready();
  await pool.loadAll({ csv_path: csv, table_name: "main_table" });

  const runtime = await NativeSkillRuntime.load([join(root, "skills")]);
  const state = new ResearchState({ goal: "audit totals", maxQuestions: 1 });
  state.openQuestion({ question: "Are group totals valid?", category: "exploratory" });
  const tools = createNativeSkillTools(runtime, { state, pool });
  await findTool(tools, "read_skill").execute("1", { name: "distribution-audit" });
  const result = await findTool(tools, "run_skill_python").execute("2", {
    skill_name: "distribution-audit", script: "scripts/audit.py", question_id: "q_001",
    sql: "SELECT segment, value FROM main_table",
    arguments: { group_col: "segment", value_col: "value" },
  });
  const payload = JSON.parse(resultText(result));
  assert.equal(payload.applicable, true);
  assert.equal(payload.max_total, 30);
  assert.equal(runtime.executionAttempts[0].status, "success");
  assert.equal(runtime.executions.length, 1);
  assert.equal(state.nodes[0].evidence.length, 1);
});
