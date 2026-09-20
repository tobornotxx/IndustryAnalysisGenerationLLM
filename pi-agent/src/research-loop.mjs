import { Agent } from "@earendil-works/pi-agent-core";
import { Type } from "@sinclair/typebox";

function toolResult(value, details = undefined) {
  return {
    content: [{ type: "text", text: typeof value === "string" ? value : JSON.stringify(value) }],
    details,
  };
}

function cleanStrings(values) {
  return [...new Set((values ?? []).map(String).map((value) => value.trim()).filter(Boolean))];
}

export class ResearchState {
  constructor({ goal, maxQuestions }) {
    this.goal = goal;
    this.maxQuestions = maxQuestions;
    this.nodes = [];
    this.lastReview = null;
    this.submission = null;
    this.events = [];
  }

  get remainingQuestions() {
    return Math.max(0, this.maxQuestions - this.nodes.length);
  }

  get findings() {
    return this.nodes.filter((node) => node.status === "completed" && node.answer);
  }

  get activeQuestions() {
    return this.nodes.filter((node) => node.status === "open");
  }

  openQuestion({ question, category, parentIds = [], hypothesis = "" }) {
    const text = String(question ?? "").trim();
    if (!text) throw new Error("question must not be empty");
    if (!this.remainingQuestions) throw new Error("question budget exhausted");
    const parents = cleanStrings(parentIds);
    const known = new Set(this.nodes.map((node) => node.id));
    const unknown = parents.filter((id) => !known.has(id));
    if (unknown.length) throw new Error(`unknown parent question(s): ${unknown.join(", ")}`);
    const parentLayers = parents.map((id) => this.nodes.find((node) => node.id === id)?.layer ?? 0);
    const node = {
      id: `q_${String(this.nodes.length + 1).padStart(3, "0")}`,
      layer: parents.length ? Math.max(...parentLayers) + 1 : 1,
      category: category === "follow_up" ? "follow_up" : "exploratory",
      parentIds: parents,
      question: text,
      hypothesis: String(hypothesis ?? "").trim(),
      answer: "",
      interpretation: "",
      status: "open",
      toolCalls: [],
      evidence: [],
    };
    this.nodes.push(node);
    this.events.push({ type: "question_opened", question_id: node.id });
    return node;
  }

  requireOpen(questionId) {
    const node = this.nodes.find((candidate) => candidate.id === questionId);
    if (!node) throw new Error(`unknown question: ${questionId}`);
    if (node.status !== "open") throw new Error(`question is not open: ${questionId}`);
    return node;
  }

  recordToolEvidence(questionId, tool, input, output) {
    const node = this.requireOpen(questionId);
    node.toolCalls.push(tool);
    node.evidence.push({ tool, input, output });
    this.events.push({ type: "evidence_recorded", question_id: questionId, tool });
  }

  completeQuestion({ questionId, finding, interpretation = "", limitations = [] }) {
    const node = this.requireOpen(questionId);
    if (!node.evidence.length) throw new Error(`question ${questionId} has no computed evidence`);
    const answer = String(finding ?? "").trim();
    if (!answer) throw new Error("finding must not be empty");
    node.answer = answer;
    node.interpretation = String(interpretation ?? "").trim();
    node.limitations = cleanStrings(limitations);
    node.status = "completed";
    this.lastReview = null;
    this.events.push({ type: "question_completed", question_id: questionId });
    return node;
  }

  snapshot() {
    return {
      goal: this.goal,
      budget: {
        maximum_questions: this.maxQuestions,
        used_questions: this.nodes.length,
        remaining_questions: this.remainingQuestions,
      },
      questions: this.nodes.map((node) => ({
        id: node.id,
        layer: node.layer,
        category: node.category,
        parent_ids: node.parentIds,
        question: node.question,
        hypothesis: node.hypothesis,
        status: node.status,
        finding: node.answer,
        interpretation: node.interpretation,
        limitations: node.limitations ?? [],
      })),
      last_review: this.lastReview,
    };
  }
}

export function createResearchTools({ state, planner, pool, evaluateSufficiency }) {
  return [
    {
      name: "inspect_research_state",
      label: "Inspect research state",
      description: "Read the current question tree, findings, evidence gaps, and remaining question budget.",
      parameters: Type.Object({}),
      async execute() {
        return toolResult(state.snapshot());
      },
    },
    {
      name: "propose_questions",
      label: "Propose research questions",
      description:
        "Use the framework's question generator to propose follow-up and exploratory questions from the current tree. " +
        "The proposals are suggestions; choose only questions that materially advance the research goal.",
      parameters: Type.Object({
        focus_aspects: Type.Optional(Type.Array(Type.String())),
      }),
      async execute(_id, { focus_aspects: focusAspects = [] }) {
        if (!state.nodes.length) {
          const questions = await planner.generateInitialQuestions({
            topic: state.goal,
            dbDescription: planner.dbDescription,
          });
          return toolResult({
            exploratory_questions: questions.map((item) => ({
              question: item.question,
              parent_ids: [],
              destination: item.destination,
            })),
            follow_up_questions: [],
          });
        }
        const proposed = await planner.generateTreeBasedQuestions({
          topic: state.goal,
          dbDescription: planner.dbDescription,
          questionNodes: state.nodes,
          insights: state.findings.map((node) => node.answer),
          thesis: null,
          focusAspects: cleanStrings(focusAspects),
        });
        return toolResult({
          follow_up_questions: proposed.followUps.map((item) => ({
            question: item.question,
            parent_ids: item.parentIds,
            destination: item.destination,
          })),
          exploratory_questions: proposed.exploratory.map((item) => ({
            question: item.question,
            parent_ids: item.parentIds,
            destination: item.destination,
          })),
        });
      },
    },
    {
      name: "start_question",
      label: "Start question",
      description:
        "Add a selected analytical question to the research tree before querying data. " +
        "Use parent_ids for a follow-up; use an empty list for a new exploratory direction.",
      parameters: Type.Object({
        question: Type.String(),
        category: Type.Union([Type.Literal("exploratory"), Type.Literal("follow_up")]),
        parent_ids: Type.Array(Type.String()),
        hypothesis: Type.Optional(Type.String()),
      }),
      async execute(_id, input) {
        const node = state.openQuestion({
          question: input.question,
          category: input.category,
          parentIds: input.parent_ids,
          hypothesis: input.hypothesis,
        });
        return toolResult({ question_id: node.id, remaining_questions: state.remainingQuestions });
      },
    },
    {
      name: "run_sql",
      label: "Run SQL for a question",
      description:
        "Execute SQL against the dataset and attach the result to an open question. " +
        "Use for counts, filters, joins, aggregations, and grouped comparisons.",
      parameters: Type.Object({
        question_id: Type.String(),
        sql: Type.String(),
      }),
      async execute(_id, { question_id: questionId, sql }) {
        state.requireOpen(questionId);
        const result = await pool.call("sql", { sql });
        state.recordToolEvidence(questionId, "run_sql", { sql }, result.summary);
        return toolResult(result.summary, { question_id: questionId, sql });
      },
    },
    {
      name: "run_python",
      label: "Run Python for a question",
      description:
        "Run SQL and analyse its result in Python, attaching the output to an open question. " +
        "The query result is available as `sql_results`; scientific Python packages are preloaded.",
      parameters: Type.Object({
        question_id: Type.String(),
        sql: Type.String(),
        code: Type.String(),
      }),
      async execute(_id, { question_id: questionId, sql, code }) {
        state.requireOpen(questionId);
        const result = await pool.call("python", { sql, code });
        state.recordToolEvidence(questionId, "run_python", { sql, code }, result.output);
        return toolResult(result.output, { question_id: questionId, sql });
      },
    },
    {
      name: "record_finding",
      label: "Record finding",
      description:
        "Close an open question by recording a finding grounded in its computed evidence. " +
        "Keep causal mechanisms as interpretations unless the data design identifies a causal effect.",
      parameters: Type.Object({
        question_id: Type.String(),
        finding: Type.String(),
        interpretation: Type.Optional(Type.String()),
        limitations: Type.Optional(Type.Array(Type.String())),
      }),
      async execute(_id, input) {
        const node = state.completeQuestion({
          questionId: input.question_id,
          finding: input.finding,
          interpretation: input.interpretation,
          limitations: input.limitations,
        });
        return toolResult({ question_id: node.id, status: node.status });
      },
    },
    {
      name: "review_progress",
      label: "Review evidence coverage",
      description:
        "Audit whether the recorded findings answer the research goal and return specific gaps for further questions.",
      parameters: Type.Object({}),
      async execute() {
        const verdict = await evaluateSufficiency({
          goal: state.goal,
          insights: state.findings.map((node) => node.answer),
        });
        state.lastReview = {
          ...verdict,
          finding_count: state.findings.length,
          reviewed_after_question_count: state.nodes.length,
        };
        state.events.push({ type: "progress_reviewed", sufficient: verdict.sufficient });
        return toolResult(state.lastReview);
      },
    },
    {
      name: "submit_analysis",
      label: "Submit analysis",
      description:
        "Submit the final evidence-grounded answer. Submission is accepted after a sufficient review or when the question budget is exhausted.",
      parameters: Type.Object({
        summary: Type.String(),
      }),
      async execute(_id, { summary }) {
        const text = String(summary ?? "").trim();
        if (!text) return toolResult({ accepted: false, reason: "summary is empty" });
        if (state.activeQuestions.length) {
          return toolResult({ accepted: false, reason: "complete all open questions before submission" });
        }
        if (!state.findings.length) {
          return toolResult({ accepted: false, reason: "no evidence-backed findings have been recorded" });
        }
        const reviewIsCurrent = state.lastReview
          && state.lastReview.reviewed_after_question_count === state.nodes.length;
        if (state.remainingQuestions > 0 && (!reviewIsCurrent || !state.lastReview.sufficient)) {
          return toolResult({
            accepted: false,
            reason: "review_progress has not confirmed sufficient evidence",
            missing_aspects: state.lastReview?.missingAspects ?? [],
          });
        }
        state.submission = text;
        state.events.push({ type: "analysis_submitted" });
        return toolResult({ accepted: true });
      },
    },
  ];
}

const RESEARCH_SYSTEM_PROMPT = `You are an autonomous data-insight research agent.

Investigate the user's research goal by maintaining an explicit question tree and grounding every finding in SQL or Python output. You control the order of work. The host enforces only evidence provenance and a finite question budget.

Operating principles:
- Begin by inspecting the research state and use propose_questions when useful.
- Select questions that can distinguish competing explanations, not merely produce descriptive tables.
- Register each chosen question with start_question before querying data.
- Use follow_up questions to investigate anomalies, mechanisms, confounders, and rival explanations.
- Use exploratory questions for materially different angles that still serve the original goal.
- Record each completed investigation with record_finding before moving on.
- Treat causal explanations from observational data as hypotheses unless the design supports identification.
- Call review_progress when the evidence may be sufficient. If it reports gaps, continue investigating them.
- Finish only through submit_analysis. The final answer must connect concrete evidence to the research goal and state important uncertainty.
`;

export async function runResearchAgentLoop({
  goal,
  schemaContext,
  maxQuestions,
  maxTurns,
  deepseek,
  usage,
  tools,
  state,
  systemPromptExtension = "",
  onLog = () => {},
}) {
  const agent = new Agent({
    initialState: {
      systemPrompt: `${RESEARCH_SYSTEM_PROMPT}\n## Dataset\n${schemaContext}${systemPromptExtension}`,
      model: deepseek.model,
      tools,
    },
    streamFn: deepseek.streamFn,
  });
  let turns = 0;
  let lastAssistantText = "";
  const unsubscribe = agent.subscribe((event) => {
    if (event.type === "turn_end") turns += 1;
    else if (event.type === "tool_execution_start") onLog(`research tool: ${event.toolName}`);
    else if (event.type === "message_end") {
      usage?.record(event.message?.usage);
      if (event.message?.role === "assistant") {
        const text = (event.message.content ?? [])
          .filter((part) => part.type === "text")
          .map((part) => part.text)
          .join("")
          .trim();
        if (text) lastAssistantText = text;
      }
    }
  });
  agent.shouldStopAfterTurn = () => turns >= maxTurns || state.submission !== null;
  await agent.prompt(
    `Research this goal: ${goal}\nThe total question budget is ${maxQuestions}. ` +
    "Use the research tools to gather, review, and submit the analysis.",
  );
  unsubscribe?.();
  return {
    turns,
    summary: state.submission ?? lastAssistantText,
    submitted: state.submission !== null,
  };
}
