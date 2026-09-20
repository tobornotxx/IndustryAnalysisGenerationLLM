import assert from "node:assert/strict";
import test from "node:test";
import { ResearchState, createResearchTools } from "../src/research-loop.mjs";

function findTool(tools, name) {
  return tools.find((tool) => tool.name === name);
}

function resultJson(result) {
  return JSON.parse(result.content[0].text);
}

function fixtures({ sufficient = true } = {}) {
  const state = new ResearchState({ goal: "explain the pattern", maxQuestions: 2 });
  const planner = {
    dbDescription: "table main_table(a, b)",
    async generateInitialQuestions() {
      return [{ question: "What is the distribution?", destination: "database" }];
    },
    async generateTreeBasedQuestions() {
      return {
        followUps: [{ question: "Which subgroup drives it?", parentIds: ["q_001"], destination: "database" }],
        exploratory: [],
      };
    },
  };
  const pool = {
    async call(kind) {
      return kind === "sql" ? { summary: "A=7, B=3" } : { output: "effect=0.4" };
    },
  };
  const evaluateSufficiency = async () => ({
    sufficient,
    missingAspects: sufficient ? [] : ["subgroup explanation"],
    reasoning: sufficient ? "covered" : "needs follow-up",
  });
  return {
    state,
    tools: createResearchTools({ state, planner, pool, evaluateSufficiency }),
  };
}

test("research tools preserve a parent-linked evidence trail", async () => {
  const { state, tools } = fixtures();
  const initial = resultJson(await findTool(tools, "propose_questions").execute("1", {}));
  assert.equal(initial.exploratory_questions[0].question, "What is the distribution?");

  const opened = resultJson(await findTool(tools, "start_question").execute("2", {
    question: initial.exploratory_questions[0].question,
    category: "exploratory",
    parent_ids: [],
  }));
  await findTool(tools, "run_sql").execute("3", {
    question_id: opened.question_id,
    sql: "select a, count(*) from main_table group by a",
  });
  await findTool(tools, "record_finding").execute("4", {
    question_id: opened.question_id,
    finding: "A accounts for 70% and B for 30%.",
    interpretation: "The pattern is concentrated in A.",
  });

  const followUps = resultJson(await findTool(tools, "propose_questions").execute("5", {}));
  assert.deepEqual(followUps.follow_up_questions[0].parent_ids, ["q_001"]);
  assert.equal(state.nodes[0].evidence[0].tool, "run_sql");
  assert.equal(state.nodes[0].status, "completed");
});

test("submission requires a current sufficient review while budget remains", async () => {
  const { tools } = fixtures({ sufficient: false });
  await findTool(tools, "start_question").execute("1", {
    question: "What is the distribution?", category: "exploratory", parent_ids: [],
  });
  await findTool(tools, "run_sql").execute("2", { question_id: "q_001", sql: "select 1" });
  await findTool(tools, "record_finding").execute("3", {
    question_id: "q_001", finding: "A exceeds B.",
  });
  const beforeReview = resultJson(await findTool(tools, "submit_analysis").execute("4", { summary: "A exceeds B." }));
  assert.equal(beforeReview.accepted, false);
  await findTool(tools, "review_progress").execute("5", {});
  const afterReview = resultJson(await findTool(tools, "submit_analysis").execute("6", { summary: "A exceeds B." }));
  assert.equal(afterReview.accepted, false);
  assert.deepEqual(afterReview.missing_aspects, ["subgroup explanation"]);
});

test("submission succeeds after a sufficient review", async () => {
  const { state, tools } = fixtures({ sufficient: true });
  await findTool(tools, "start_question").execute("1", {
    question: "What is the distribution?", category: "exploratory", parent_ids: [],
  });
  await findTool(tools, "run_sql").execute("2", { question_id: "q_001", sql: "select 1" });
  await findTool(tools, "record_finding").execute("3", {
    question_id: "q_001", finding: "A exceeds B.",
  });
  await findTool(tools, "review_progress").execute("4", {});
  const submitted = resultJson(await findTool(tools, "submit_analysis").execute("5", { summary: "A exceeds B." }));
  assert.equal(submitted.accepted, true);
  assert.equal(state.submission, "A exceeds B.");
});
