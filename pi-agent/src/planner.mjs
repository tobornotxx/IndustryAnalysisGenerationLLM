/**
 * Planner —— 探索问题生成。
 *
 * 从 MyDataStorm/datastorm/agents/planner.py + prompts/templates.py 迁移。
 *
 * **prompt 文本逐字搬运**：这些措辞是 v2→v8 八个版本在 InsightBench 上调出来的
 * （"stay aligned with the original goal"、"避免漂移到 priority/business-hours"、
 * Layer 1 必须做基础 EDA 等），是实打实的调优成果，不重写、不"改进"。
 * 唯一改动是 Jinja2 语法 → TS 模板字符串。
 *
 * planner 之所以在 pi 迁移里做 TS 原生重写（而非像 thesis/report 那样包装成 tool）：
 * 它是核心链路，值得原生化以便直接使用 pi 的 model 层与 usage 统计。
 */

/** 问题的路由目标。对应 Python 侧 QuestionDestination。 */
export const Destination = {
  DATABASE: "database",
  INTERNET: "internet",
};

// ── 第一层：初始问题（对应 INITIAL_QUESTIONS_GENERATION / Prompt 11）──
function initialQuestionsPrompt({ topic, dbDescription, numQuestions, article }) {
  return `You are conducting research on a goal/topic: "${topic}". The goal here is to extract previously unknown
insights by exploring and observing the information in the database with the following description: ${dbDescription}.

Generate EXACTLY ${numQuestions} questions that an investigator will be interested in.
- You MUST generate exactly ${numQuestions} questions — no more, no less.
- The questions will be used to generate search queries in the database to help answer them.
- The questions should be self-contained (include any specific years, months, locations, etc.)
  and related to the goal/topic: "${topic}".
- Each question should investigate one specific aspect. Do not include too many subquestions inside a single question.
- The questions should be independent of each other.

CRITICAL: Layer 1 questions MUST prioritize fundamental EDA (Exploratory Data Analysis). Include these types:
1. Distribution/breakdown by key categorical columns (e.g., "What is the distribution of X across categories?")
2. Time trends (e.g., "How does the volume/metric change over time? Is there an increasing or decreasing trend?")
3. Correlations or lack thereof between key numeric variables (e.g., "Is there a correlation between volume and resolution time?")
4. Per-group comparisons (e.g., "Is performance/metric uniform across all agents/groups, or does one stand out?")

Do NOT jump to advanced analysis (statistical tests, Gini coefficients, outlier detection) in Layer 1.
Start with basic counts, averages, and trend lines. Advanced analysis belongs in later layers.

Output a JSON object with EXACTLY this structure:
{
  "questions": [
    {"question": "...", "destination": "database"},
    ... (EXACTLY ${numQuestions} items)
  ]
}
${article ? `\nHere is more background information on the goal/topic based on the internet: "${article}".\n` : ""}`;
}

// ── 后续层：基于问题树的双模式生成（对应 TREE_BASED_QUESTION_GENERATION / Prompt 5b）──
function treeBasedPrompt({
  topic,
  dbDescription,
  m,
  n,
  questionTree,
  globalInsights,
  thesis,
  researchStrategy,
  focusAspects = [],
}) {
  const thesisBlock = thesis
    ? `
## Current Thesis
"${thesis}"
Research strategy: ${researchStrategy}
Prioritize questions that help build, test, or refine this argument.
Also ask questions that challenge or qualify the thesis - strong analysis addresses counter-arguments.
`
    : "";

  const focusBlock = focusAspects.length
    ? `
## Priority Gaps To Close This Round
A self-assessment found the findings so far do NOT yet fully answer the goal.
The following angles are still unexplored or under-investigated:
${focusAspects.map((a) => `- ${a}`).join("\n")}
Prioritize questions that directly close these gaps. They are the most important
questions to generate this round (while still serving the original goal).
`
    : "";

  return `# instruction
You are an analytical reasoning engine exploring a relational database.
Your task is to make progress on a specific research goal by generating targeted questions.

You have access to:
1. A QUESTION TREE showing all previously asked questions, organized by layer,
   with their answers and results.
2. A list of GLOBAL INSIGHTS that have been extracted from those answers.
3. A research THESIS (if one has been formed).

## CRITICAL: Stay Aligned With the Original Research Goal
The original research goal is: "${topic}"

EVERY question you generate — whether follow-up or exploratory — MUST directly serve this goal.
Before generating each question, ask yourself: "Does answering this question help achieve the original goal?"
- YES → include it.
- NO  → discard it and think of a better question that does.

Common failure modes to AVOID:
- Do NOT explore dimensions (e.g., priority, business hours, day-of-week) just because
  they are "interesting" or available in the data, if they are not asked for by the goal.
- Do NOT wander into unrelated territory: if the goal is about volume trends, stay on
  volume trends; if it is about resolution time, stay on resolution time.
- Exploratory questions should open NEW ANGLES on the SAME goal, not drift away from it.

Your task is to generate TWO groups of questions:

## Group A: FOLLOW-UP questions (AT LEAST ${m}, AT MOST 5 questions)
- These should deepen EXISTING lines of investigation found in the question tree
  that are relevant to the goal.
- Look at questions that revealed interesting partial results or anomalies,
  and drill deeper into those findings — as long as the drill-down still serves the goal.
- Each follow-up question MUST reference which existing question(s) it extends,
  using the "parent_ids" field with question node IDs from the tree.
- Follow-up questions should be more specific and targeted than their parents.
- Generate more than ${m} if the data suggests many goal-relevant lines of investigation
  are worth pursuing deeper.

## Group B: EXPLORATORY questions (AT LEAST ${n}, AT MOST 5 questions)
- These should open ENTIRELY NEW investigation directions NOT covered by existing questions,
  but still in service of the original goal.
- Valid new angles aligned with most goals: time trends, category breakdowns, correlations,
  geographic or group-level breakdowns, root-cause identification via text fields.
- Ensure these questions are self-contained and independent.
- Exploratory questions should have an empty "parent_ids" list.
- Generate more than ${n} if there are unexplored goal-relevant dimensions in the data.

## Routing
For each question, specify a "destination":
- "database": answerable by querying the database
- "internet": requires external context (use sparingly)

## Rules
- You MUST generate AT LEAST ${m} follow-up questions AND AT LEAST ${n} exploratory questions.
- Each type MUST NOT exceed 5 questions. If there aren't enough good questions, fewer is fine — but never exceed 5.
- Do NOT ask questions already covered by existing questions or global insights.
- Make each question clear, specific, and scoped to one aspect.
- Be decisive: generate the number of questions that is appropriate for the current state
  of exploration. Do NOT pad with weak questions just to hit a number.
${thesisBlock}${focusBlock}
Output a JSON object with this structure:
{
  "chain_of_thought": "first restate the original goal, then explain how each question group serves it",
  "follow_up_questions": [
    {
      "question": "...",
      "destination": "database",
      "parent_ids": ["<node_id_from_tree>", ...]
    }
  ],
  "exploratory_questions": [
    {
      "question": "...",
      "destination": "database",
      "parent_ids": []
    }
  ]
}

# input
Description of database content: ${dbDescription}

Research Goal (stay aligned with this at all times): ${topic}

QUESTION TREE (all previously asked questions with their answers):
${questionTree}

GLOBAL INSIGHTS extracted so far:
${globalInsights}
`;
}

/** 每类问题的硬上限（对应 Python 侧 MAX_PER_TYPE）。 */
const MAX_PER_TYPE = 5;

export class Planner {
  /**
   * @param {object} opts
   * @param {(prompt: string, opts?: object) => Promise<any>} opts.generateJson
   *   调 LLM 并返回解析后的 JSON。由调用方注入，便于复用 pi 的 model 层。
   * @param {number} opts.followUpPerLayer  m：跟进问题下限
   * @param {number} opts.exploratoryPerLayer  n：探索问题下限
   * @param {number} opts.firstLayerQuestions  第一层问题数
   */
  constructor({
    generateJson,
    followUpPerLayer = 2,
    exploratoryPerLayer = 2,
    firstLayerQuestions = 2,
  }) {
    this.generateJson = generateJson;
    this.m = followUpPerLayer;
    this.n = exploratoryPerLayer;
    this.firstLayerQuestions = firstLayerQuestions;
  }

  /** 第一层：从 warmstart 报告（可选）生成初始问题。 */
  async generateInitialQuestions({ topic, dbDescription, warmstartReport = "" }) {
    const maxQ = this.firstLayerQuestions;
    const prompt = initialQuestionsPrompt({
      topic,
      dbDescription,
      numQuestions: maxQ,
      article: warmstartReport || null,
    });
    const resp = await this.generateJson(prompt, { temperature: 0.7 });
    let questions = parseQuestions(resp?.questions ?? [], []);
    // 安全截断：保证不超过 maxQ（即使 LLM 返回更多）
    if (questions.length > maxQ) questions = questions.slice(0, maxQ);
    return questions;
  }

  /**
   * 后续层：基于完整问题树，同时产出跟进问题（m）与探索性问题（n）。
   * m/n 是下限，每类上限 5 个。
   */
  async generateTreeBasedQuestions({
    topic,
    dbDescription,
    questionNodes = [],
    insights = [],
    thesis = null,
    focusAspects = [],
  }) {
    const prompt = treeBasedPrompt({
      topic,
      dbDescription,
      m: this.m,
      n: this.n,
      questionTree: formatQuestionTree(questionNodes),
      globalInsights: formatInsights(insights),
      thesis: thesis?.title ?? null,
      researchStrategy: thesis?.researchStrategy ?? null,
      focusAspects,
    });

    const resp = await this.generateJson(prompt, { temperature: 0.7 });

    // 跟进问题：parent_ids 缺失时保持 undefined 语义（由 LLM 提供）
    let followUps = parseQuestions(resp?.follow_up_questions ?? [], null);
    // 探索问题：parent_ids 默认空数组（新方向，无父节点）
    let exploratory = parseQuestions(resp?.exploratory_questions ?? [], []);

    if (followUps.length > MAX_PER_TYPE) followUps = followUps.slice(0, MAX_PER_TYPE);
    if (exploratory.length > MAX_PER_TYPE) exploratory = exploratory.slice(0, MAX_PER_TYPE);

    return { followUps, exploratory };
  }
}

// ──────────────────────────────────────────────────────────────
// 解析与格式化（对应 planner.py 的 _parse_tree_questions / _format_*）
// ──────────────────────────────────────────────────────────────

function parseQuestions(raw, defaultParentIds) {
  if (!Array.isArray(raw)) return [];
  const out = [];
  for (const q of raw) {
    if (!q || typeof q !== "object") continue;
    const destStr = q.destination ?? Destination.DATABASE;
    const destination =
      destStr === Destination.INTERNET ? Destination.INTERNET : Destination.DATABASE;
    const parentIds = q.parent_ids ?? defaultParentIds ?? [];
    out.push({
      question: q.question ?? "",
      destination,
      parentIds: Array.isArray(parentIds) ? parentIds : [],
    });
  }
  return out;
}

/** 把问题树格式化为文本，展示问题的链和继承关系。 */
export function formatQuestionTree(nodes) {
  if (!nodes?.length) return "No previous questions yet.";
  const byLayer = new Map();
  for (const node of nodes) {
    if (!byLayer.has(node.layer)) byLayer.set(node.layer, []);
    byLayer.get(node.layer).push(node);
  }
  const lines = [];
  for (const layer of [...byLayer.keys()].sort((a, b) => a - b)) {
    lines.push(`=== Layer ${layer} ===`);
    for (const node of byLayer.get(layer)) {
      const parentInfo = node.parentIds?.length
        ? ` [derived from: ${node.parentIds.join(", ")}]`
        : " [exploratory/root]";
      lines.push(`  Node [${node.id}] (${node.category})${parentInfo}`);
      lines.push(`    Q: ${node.question}`);
      if (node.answer) lines.push(`    A: ${node.answer}`);
    }
  }
  return lines.join("\n");
}

/** 把洞察库格式化为文本。 */
export function formatInsights(insights) {
  if (!insights?.length) return "No insights yet.";
  return insights
    .map((ins, i) => `${i + 1}. ${typeof ins === "string" ? ins : ins.content}`)
    .join("\n");
}
