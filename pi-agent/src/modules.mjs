/**
 * 非 agentic 的分析模块 —— 从 MyDataStorm 迁移。
 *
 * 这些步骤都是「填 prompt → 调一次 LLM → 解析结果」，没有工具调用，
 * 所以不走 pi 的 agent loop，直接用 generateJson / generateText。
 *
 * **prompt 逐字搬运**（同 planner）：措辞是 v2→v8 在 InsightBench 上调出来的。
 * 唯一的例外见 insight-bank 的 note —— 那里原文含 ServiceNow 维度名，按纯度
 * 要求泛化了。
 */

// ══════════════════════════════════════════════════════════════
// Insight Bank —— 候选发现的去重与择优（对应 INSIGHT_BANK_FILTER / Prompt 8）
// ══════════════════════════════════════════════════════════════

/**
 * @param nodes 形如 [{id, question, answer}]
 * @returns 被选中的 node id → insight 文本
 *
 * 注意 selection criteria 第 4 条：原 prompt 写的是
 * "different dimensions (time, category, agent, priority)" —— 后三个是
 * ServiceNow 工单维度名，属于对 benchmark 的数据特化（与 skill 包清理的同一类
 * 问题）。这里改为从运行时 schema 注入的 {categoricalColumns}，语义不变。
 */
export async function filterInsights({
  generateJson,
  nodes,
  topic,
  dbDescription,
  thesis = null,
  maxInsights = 12,
  categoricalColumns = "the dataset's categorical columns",
  skillGuidance = "",
}) {
  if (!nodes.length) return {};

  const input = nodes
    .map((n) => `node_id: ${n.id}\nQuestion: ${n.question}\nAnswer: ${n.answer}`)
    .join("\n\n---\n\n");

  const thesisBlock = thesis
    ? `
## Current working thesis (for context, NOT as a filter):
"${thesis}"
Note: Do NOT filter out findings just because they don't support the thesis.
Surprising findings that contradict or are orthogonal to the thesis are often the most valuable.
`
    : "";

  const prompt = `# instruction
You are given a list of candidate insights derived from database exploration on a topic.
Your task is to select the most valuable insights, capped at ${maxInsights}.

## Selection criteria (in priority order):
1. **Anomalies and surprises**: findings that reveal unexpected patterns, outliers, or counter-intuitive results (e.g., one category growing while others shrink, a metric behaving opposite to expectation)
2. **Significant trends**: statistically significant changes over time, strong correlations, or clear distributional skews
3. **Actionable findings**: observations that point to root causes or have clear operational implications
4. **Breadth of coverage**: prefer a diverse set of insights covering different dimensions (over time, and across ${categoricalColumns}) over many insights about the same dimension

## De-prioritize:
- Redundant findings that repeat what another selected insight already says
- Trivial or expected observations (e.g., "data has 500 rows", "5 categories exist")
- Findings where the analysis failed or produced no result

## Do NOT de-prioritize:
- Negative or null results ("no correlation between A and B", "the distribution is
  uniform", "the metric is stable over time"). A well-established absence of an
  effect is a genuine finding, not a trivial observation — it often matters as much
  as a positive one.
- Findings that state what the data implies or what action it points to, when that
  interpretation is grounded in a reported result.

The topic is: ${topic}

The database context: ${dbDescription}
${thesisBlock}
${skillGuidance}
Output a JSON dict, where each key is a node_id and the value is the insight for that node_id.

# input
${input}
`;

  const resp = await generateJson(prompt, { temperature: 0.3 });

  // 模型有时会包一层（{"insights": {...}}）或改写 node_id 的写法，
  // 严格按 id 精确匹配会把整批结果丢掉（实测 flag-5 每层都返回 0）。
  // 这里做两级容错：先剥外层包裹，再对 id 做规范化匹配。
  const known = new Map(nodes.map((n) => [normalizeId(n.id), n.id]));
  let body = resp ?? {};
  // 若顶层只有一个键且其值是对象，视为包裹层剥掉
  const topKeys = Object.keys(body);
  if (topKeys.length === 1 && body[topKeys[0]] && typeof body[topKeys[0]] === "object"
      && !Array.isArray(body[topKeys[0]]) && !known.has(normalizeId(topKeys[0]))) {
    body = body[topKeys[0]];
  }

  const out = {};
  for (const [rawId, value] of Object.entries(body)) {
    const id = known.get(normalizeId(rawId));
    if (!id) continue;
    // 值可能是字符串，也可能是 {insight: "..."} 之类的小对象
    const text =
      typeof value === "string"
        ? value
        : typeof value?.insight === "string"
          ? value.insight
          : typeof value?.text === "string"
            ? value.text
            : null;
    if (text && text.trim()) out[id] = text.trim();
  }
  return out;
}

/** node_id 规范化：去掉非字母数字，小写。用于容忍模型改写 id 的写法。 */
function normalizeId(s) {
  return String(s).toLowerCase().replace(/[^a-z0-9]/g, "");
}

// ══════════════════════════════════════════════════════════════
// Thesis —— 生成与精炼（对应 THESIS_GENERATION / THESIS_REFINEMENT）
// ══════════════════════════════════════════════════════════════

const THESIS_PREAMBLE = `You are a senior analyst at a world-class publication (think The Economist, Foreign Affairs, or
FiveThirtyEight).`;

const THESIS_RULES = `Rules:
- Each thesis is a CONCISE TITLE - maximum 10 words. Think magazine cover line or op-ed headline, NOT a full
  sentence or a data summary.
- A good thesis takes a POSITION. It argues something. It should be possible to disagree with it. Avoid bland
  descriptive titles like "Trends in
  X" or "Overview of Y."
- Do NOT embed statistics, numbers, or data citations in the thesis title.
- The thesis should capture a non-obvious, thought-provoking argument that would make an informed reader want
  to read the full article.
- For each thesis, provide a research_strategy: a concrete plan for how a writer should develop this argument
  into a full analytical article.`;

export async function generateThesis({ generateJson, topic, dbDescription, insights }) {
  const prompt = `# instruction

${THESIS_PREAMBLE} You have been given a general
topic and a batch of findings produced by a preliminary data exploration agent.

Your job is NOT to describe what the data shows. Your job is to REASON about what the findings mean - to
identify non-obvious patterns, causal
claims, counter-narratives, strategic implications, or surprising tensions - and to distill them into
compelling, defensible thesis statements.

Each thesis should be the kind of bold, original argument that could anchor a top-tier analytical article
written for a general audience.

Generate at most 3 thesis candidates.

${THESIS_RULES}

Output a JSON object:
{"theses": [{"title": "...", "research_strategy": "..."}]}

# input
Description of database content: ${dbDescription}

Topic: ${topic}

Findings:
${formatFindings(insights)}
`;
  const resp = await generateJson(prompt, { temperature: 0.7 });
  const first = (resp?.theses ?? [])[0];
  if (!first?.title) return null;
  return { title: first.title, researchStrategy: first.research_strategy ?? "" };
}

export async function refineThesis({
  generateJson,
  topic,
  dbDescription,
  currentThesis,
  insights,
}) {
  const prompt = `# instruction

${THESIS_PREAMBLE}

You previously proposed a working thesis to guide research on a topic. Since then, a research agent has
gathered
additional findings from the database. Your task is to re-examine that thesis in light of the new evidence and
decide whether to:

1. Sharpen - narrow or deepen the original argument using new supporting evidence
2. Pivot - shift to a better-supported or more compelling argument uncovered by the new findings
3. Confirm - keep the thesis essentially unchanged if the evidence continues to support it strongly

Output exactly one refined thesis and the updated research strategy.

${THESIS_RULES}

Output a JSON object:
{"title": "...", "research_strategy": "..."}

# input
Description of database content: ${dbDescription}

Topic: ${topic}

Current thesis: "${currentThesis.title}"
Current research strategy: ${currentThesis.researchStrategy}

New findings:
${formatFindings(insights)}
`;
  const resp = await generateJson(prompt, { temperature: 0.7 });
  if (!resp?.title) return currentThesis;
  return {
    title: resp.title,
    researchStrategy: resp.research_strategy ?? currentThesis.researchStrategy,
  };
}

// ══════════════════════════════════════════════════════════════
// Goal Sufficiency —— 覆盖度自评（对应 goal_sufficiency.py）
// ══════════════════════════════════════════════════════════════

/**
 * 判断当前发现是否已足以回答 goal；不够则给出缺失角度。
 *
 * 覆盖审计规则来自 skill 包（coverage-multidim-selfcheck / anti-premature-stop），
 * 由调用方通过 skillGuidance 注入 —— 维度名在运行时从真实 schema 填充，
 * 不硬编码任何列名。
 *
 * 出错时保守返回 sufficient=false（倾向继续探索，不误停）。
 */
export async function evaluateSufficiency({
  generateJson,
  goal,
  insights,
  skillGuidance = "",
}) {
  const findings = formatFindings(insights);
  if (!findings.trim() || findings.startsWith("No findings")) {
    return { sufficient: false, missingAspects: [], reasoning: "no findings yet" };
  }

  const prompt = `You are auditing whether a data analysis has gathered enough evidence to fully answer its research goal.

RESEARCH GOAL:
${goal}

FINDINGS SO FAR:
${findings}

Judge ONLY against the goal itself — do not assume any external 'correct' answer. Ask yourself:
- Do these findings, taken together, actually answer the goal?
- Has every distinct angle the goal asks about been investigated (e.g. if the goal mentions a trend, was change-over-time examined; if it mentions a specific subgroup or time window, was that subgroup/window actually located and analyzed rather than assumed)?
- Are there obvious follow-up questions a careful analyst would still ask before concluding?
${skillGuidance}

Respond in JSON:
{
  "sufficient": true | false,
  "reasoning": "<one or two sentences>",
  "missing_aspects": ["<specific angle still unexplored>", ...]
}
When sufficient=true, missing_aspects must be empty.
`;

  try {
    const r = await generateJson(prompt, { temperature: 0.3 });
    const missing = Array.isArray(r?.missing_aspects)
      ? r.missing_aspects.map(String).map((s) => s.trim()).filter(Boolean)
      : [];
    return {
      sufficient: Boolean(r?.sufficient),
      missingAspects: missing,
      reasoning: String(r?.reasoning ?? "").trim(),
    };
  } catch (e) {
    return { sufficient: false, missingAspects: [], reasoning: `eval error: ${e.message}` };
  }
}

// ══════════════════════════════════════════════════════════════
// Summary —— 自一致性合并（对应 adapter._extract_summary + _summarize_self_consistent）
// ══════════════════════════════════════════════════════════════

function summaryPrompt(goal, sourceText, skillGuidance = "") {
  const goalLine = goal ? `RESEARCH GOAL: ${goal}\n\n` : "";
  return (
    "You are writing the summary of a data analysis. Below are the " +
    "findings produced while investigating the research goal.\n\n" +
    goalLine +
    "Write a coherent summary that ANSWERS the research goal, built " +
    "entirely from these findings.\n\n" +
    "Principles:\n" +
    "- Every claim must be supported by a specific finding below " +
    "(reference the concrete numbers/patterns the finding reports).\n" +
    "- Select and connect the findings that actually bear on the goal; " +
    "ignore findings that turned out irrelevant.\n" +
    "- Build an argument: state what the data shows, and why it answers " +
    "the goal. Let the findings dictate the shape of the explanation — " +
    "do not force any predetermined structure.\n" +
    "- Be faithful: if the findings are inconclusive or contradict an " +
    "expected pattern, say so rather than inventing a clean story.\n\n" +
    "FORMAT:\n" +
    "- A numbered list of 3-5 key points: 1. **Title**: explanation.\n" +
    "- Each point 1-3 sentences, grounded in the findings.\n\n" +
    skillGuidance +
    "Findings:\n" +
    sourceText.slice(0, 6000)
  );
}

/**
 * 生成 summary，可选自一致性合并。
 *
 * samples>1 时并行生成多份草稿再合并出稳定核心 —— 降低 summary 的运行间方差
 * （原实现的 summary_samples，默认 3）。
 */
export async function generateSummary({
  generateText,
  goal,
  nodes,
  samples = 3,
  skillGuidance = "",
}) {
  const sourceText = nodes
    .filter((n) => n.answer && !n.answer.startsWith("Execution failed"))
    .map((n) => `- Q: ${n.question}\n  Finding: ${n.answer}`)
    .join("\n");

  if (sourceText.length < 100) return "";

  const prompt = summaryPrompt(goal, sourceText, skillGuidance);

  if (samples <= 1) {
    return (await generateText(prompt, { temperature: 0.3 })).trim();
  }

  const drafts = (
    await Promise.all(
      Array.from({ length: samples }, (_, i) =>
        generateText(prompt, { temperature: i === 0 ? 0.3 : 0.5 }).catch(() => ""),
      ),
    )
  ).filter((d) => d && d.length > 50);

  if (!drafts.length) return "";
  if (drafts.length === 1) return drafts[0].trim();

  const mergePrompt =
    "Below are several independently written summaries of the same data analysis.\n\n" +
    "Produce ONE consolidated summary that keeps only the points that appear " +
    "consistently across drafts (those are the reliable core), and drops points " +
    "that appear in only one draft (those are likely artefacts of a single " +
    "generation).\n\n" +
    "Keep the same FORMAT: a numbered list of 3-5 key points, " +
    "1. **Title**: explanation, each 1-3 sentences.\n\n" +
    drafts.map((d, i) => `--- Draft ${i + 1} ---\n${d}`).join("\n\n");

  const merged = await generateText(mergePrompt, { temperature: 0.2 }).catch(() => "");
  return (merged && merged.length > 50 ? merged : drafts[0]).trim();
}

// ══════════════════════════════════════════════════════════════
// 工具函数
// ══════════════════════════════════════════════════════════════

export function formatFindings(insights) {
  if (!insights?.length) return "No findings available yet.";
  return insights
    .map((ins, i) => {
      if (typeof ins === "string") return `Finding ${i + 1}: ${ins}`;
      return `Finding ${i + 1}:\n  Question: ${ins.question ?? ""}\n  Insight: ${ins.answer ?? ins.content ?? ""}`;
    })
    .join("\n\n");
}
