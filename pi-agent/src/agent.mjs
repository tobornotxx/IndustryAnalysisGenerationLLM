/**
 * Insight Agent —— pi harness 之上的数据分析 agent。
 *
 * 架构（PI_MIGRATION_PLAN §3）：
 *   pi 的 Agent 拥有 loop / 工具调度 / 重试 / 错误恢复；
 *   Planner（TS 原生）负责生成探索问题；
 *   Python worker 池负责真正的数据分析执行。
 *
 * 与旧 executor 的本质区别：不再手写 ReAct 循环。每一层的问题交给 pi 的
 * agent loop 去跑，模型自己决定调几次工具、失败了怎么改 —— 那些"遇到一个
 * case 就打一个补丁"的错误处理由 harness 通用地覆盖。
 */
import { Agent } from "@earendil-works/pi-agent-core";
import { createModels } from "@earendil-works/pi-ai";
import { deepseekProvider } from "@earendil-works/pi-ai/providers/deepseek";
import { Type } from "@sinclair/typebox";
import { PyWorkerPool } from "./py-worker.mjs";
import { Planner } from "./planner.mjs";
import {
  DEEPSEEK_CANONICAL_MODEL,
  canonicalizeDeepSeekModel,
  makeDeepSeekV41FlashModel,
} from "./model-config.mjs";
import {
  filterInsights,
  generateThesis,
  refineThesis,
  evaluateSufficiency,
  generateSummary,
} from "./modules.mjs";
import {
  SkillPackage,
  describeCategoricalColumns,
  EXECUTOR_SKILL_IDS,
  SUFFICIENCY_SKILL_IDS,
} from "./skills.mjs";

/**
 * 累计 token 用量与成本。
 *
 * pi 的 usage 结构（实测）：
 *   { input, output, cacheRead, cacheWrite, reasoning, totalTokens,
 *     cost: { input, output, cacheRead, cacheWrite, total } }
 *
 * 关键：`input` 与 `cacheRead` 是**互斥的两部分**，不是包含关系 ——
 * 实测同前缀两次请求：
 *   第1轮 input=1213 cacheRead=0
 *   第2轮 input=61   cacheRead=1152    ← 命中的部分从 input 移到了 cacheRead
 * 恒等式 input + cacheRead = totalTokens - output 两轮都成立。
 * 因此命中率的分母必须是 (input + cacheRead)，用 input 会算出 >1 的值。
 *
 * cost 字段已按运行时模型价目表计算，直接累加即可。
 */
export class UsageTracker {
  constructor() {
    this.calls = 0;
    this.inputTokens = 0;
    this.completionTokens = 0;
    this.cacheReadTokens = 0;
    this.reasoningTokens = 0;
    this.costUsd = 0;
  }
  record(usage) {
    if (!usage) return;
    this.calls += 1;
    this.inputTokens += usage.input ?? 0;
    this.completionTokens += usage.output ?? 0;
    this.cacheReadTokens += usage.cacheRead ?? 0;
    this.reasoningTokens += usage.reasoning ?? 0;
    this.costUsd += usage.cost?.total ?? 0;
  }
  /** 总输入 = 未命中(input) + 命中(cacheRead)。 */
  get totalPromptTokens() {
    return this.inputTokens + this.cacheReadTokens;
  }
  get cacheHitRate() {
    const t = this.totalPromptTokens;
    return t ? this.cacheReadTokens / t : 0;
  }
  toJSON() {
    return {
      calls: this.calls,
      prompt_tokens: this.totalPromptTokens,
      prompt_uncached: this.inputTokens,
      prompt_cached: this.cacheReadTokens,
      completion_tokens: this.completionTokens,
      reasoning_tokens: this.reasoningTokens,
      cache_hit_rate: Number(this.cacheHitRate.toFixed(4)),
      cost_usd: Number(this.costUsd.toFixed(6)),
    };
  }
}

/** 建 DeepSeek 模型句柄 + 与 pi 的 streamFn 桥接。 */
export function createDeepSeek({ model = DEEPSEEK_CANONICAL_MODEL, now = new Date() } = {}) {
  const models = createModels();
  models.setProvider(deepseekProvider());
  const canonical = canonicalizeDeepSeekModel(model);
  let handle = models.getModel("deepseek", canonical);
  if (!handle && canonical === DEEPSEEK_CANONICAL_MODEL) {
    const legacy = models.getModel("deepseek", "deepseek-v4-flash");
    handle = makeDeepSeekV41FlashModel(legacy, now);
  }
  if (!handle) throw new Error(`model not found: deepseek/${model}`);
  return {
    models,
    model: handle,
    streamFn: (m, ctx, opts) => models.streamSimple(m, ctx, opts),
  };
}

/**
 * 直接向模型要一段 JSON（非 agentic 步骤：planner / thesis / 自评等）。
 * 不走 agent loop，因为这些步骤没有工具调用。
 */
export function makeGenerateJson({ models, model, usage }) {
  return async function generateJson(prompt, { temperature = 0.7 } = {}) {
    const stream = models.streamSimple(model, {
      messages: [{ role: "user", content: [{ type: "text", text: prompt }] }],
    });
    let final = null;
    for await (const ev of stream) {
      if (ev.type === "done" || ev.type === "error") final = ev.message;
    }
    usage?.record(final?.usage);
    const text = (final?.content ?? [])
      .filter((c) => c.type === "text")
      .map((c) => c.text)
      .join("");
    return parseJsonLoose(text);
  };
}

/** 直接向模型要一段纯文本（summary 等），不走 agent loop。 */
export function makeGenerateText({ models, model, usage }) {
  return async function generateText(prompt, { temperature = 0.3 } = {}) {
    const stream = models.streamSimple(
      model,
      { messages: [{ role: "user", content: [{ type: "text", text: prompt }] }] },
      { temperature },
    );
    let final = null;
    for await (const ev of stream) {
      if (ev.type === "done" || ev.type === "error") final = ev.message;
    }
    usage?.record(final?.usage);
    return (final?.content ?? [])
      .filter((c) => c.type === "text")
      .map((c) => c.text)
      .join("");
  };
}

/** 从可能带 ```json 围栏或前后缀文本的响应里抠出 JSON。 */
export function parseJsonLoose(text) {
  if (!text) return {};
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const body = fenced ? fenced[1] : text;
  try {
    return JSON.parse(body);
  } catch {
    // 退而求其次：取第一个 { 到最后一个 }
    const s = body.indexOf("{");
    const e = body.lastIndexOf("}");
    if (s >= 0 && e > s) {
      try {
        return JSON.parse(body.slice(s, e + 1));
      } catch {
        /* fallthrough */
      }
    }
    return {};
  }
}

/** 把 worker 池包装成 pi 的工具。 */
export function createDataTools(pool) {
  const runSql = {
    name: "run_sql",
    label: "Run SQL",
    description:
      "Execute a SQL query against the dataset (SQLite) and return the result rows " +
      "as text. Use this for aggregations, group-bys, filters, and counts.",
    parameters: Type.Object({
      sql: Type.String({ description: "SQL query to execute" }),
    }),
    async execute(_id, { sql }) {
      const r = await pool.call("sql", { sql });
      return { content: [{ type: "text", text: r.summary }], details: { sql } };
    },
  };

  const runPython = {
    name: "run_python",
    label: "Run Python",
    description:
      "Run a SQL query, then analyse its result with Python. The query result is " +
      "available as a pandas DataFrame named `sql_results`. " +
      "numpy/pandas/scipy/sklearn/statsmodels/ruptures/pingouin are preloaded. " +
      "Use print() to output what you want to observe. " +
      "Use this for statistics, trend tests, correlations, and any computation SQL can't do.",
    parameters: Type.Object({
      sql: Type.String({ description: "SQL selecting the rows to analyse" }),
      code: Type.String({ description: "Python code; `sql_results` is the DataFrame" }),
    }),
    async execute(_id, { sql, code }) {
      const r = await pool.call("python", { sql, code });
      return { content: [{ type: "text", text: r.output }], details: { sql } };
    },
  };

  return [runSql, runPython];
}

const ANSWER_SYSTEM_PROMPT = `You are a data analyst answering ONE specific question about a dataset.

Use the provided tools to compute the answer from the data — never guess or fabricate numbers.

When you have the answer, state the FINDING: what the data actually revealed.
- State conclusions, not procedures. Do not describe what you are about to do.
- Name the specific entities involved (which column, which category, which direction of change).
- Include the concrete numbers that support the finding.
- If the result is inconclusive or the data cannot answer the question, say so plainly.

State NEGATIVE and NULL results as explicitly as positive ones. "There is no
correlation between A and B", "the distribution is uniform across categories",
"the metric is stable over time" are findings in their own right — do not bury
them or present them only as the absence of something.

After the finding, when — and only when — the data genuinely supports it, add a
short INTERPRETATION: what the finding implies, and what action it points to.
- Derive it from the finding you just reported; do not introduce claims the data
  does not support.
- One or two sentences. Skip this entirely if the finding is inconclusive or
  carries no clear implication.`;

/**
 * 用 pi 的 agent loop 回答单个探索问题。
 *
 * 这里是旧 ExecutorAgent 的替代：不再手写 ReAct，由 harness 驱动
 * 「思考 → 调工具 → 看结果 → 再决定」，包括失败重试。
 */
export async function answerQuestion({
  question,
  schemaContext,
  deepseek,
  tools,
  usage,
  maxTurns = 8,
}) {
  const agent = new Agent({
    initialState: {
      systemPrompt: `${ANSWER_SYSTEM_PROMPT}\n\n## Dataset\n${schemaContext}`,
      model: deepseek.model,
      tools,
    },
    streamFn: deepseek.streamFn,
  });

  let turns = 0;
  const toolCalls = [];
  // 实测：agent.initialState.messages 在运行后仍为空，助手消息只能从
  // message_end 事件收集。取最后一条 assistant 文本作为回答。
  let lastAssistantText = "";
  const unsub = agent.subscribe((ev) => {
    if (ev.type === "turn_end") turns += 1;
    else if (ev.type === "tool_execution_start") toolCalls.push(ev.toolName);
    else if (ev.type === "message_end") {
      const msg = ev.message;
      usage?.record(msg?.usage);
      if (msg?.role === "assistant") {
        const text = (msg.content ?? [])
          .filter((c) => c.type === "text")
          .map((c) => c.text)
          .join("")
          .trim();
        if (text) lastAssistantText = text;
      }
    }
  });

  // 超过层数上限就停，避免个别问题拖垮整层
  agent.shouldStopAfterTurn = () => turns >= maxTurns;

  await agent.prompt(question);
  unsub?.();

  return { question, answer: lastAssistantText, turns, toolCalls };
}

/**
 * 跑完一个数据集的分层探索。
 *
 * 与原 pipeline（MyDataStorm ExplorationFramework）的对应：
 *   Layer 1          → planner.generateInitialQuestions
 *   Layer 2..m       → planner.generateTreeBasedQuestions（问题树 + thesis + 缺口引导）
 *   每层内并发执行   → worker 池 + pi agent 并发
 *   每层后           → insight_bank 去重择优
 *   每 p 层          → thesis 生成/精炼
 *   min_layers 之后  → goal_sufficiency 自评，够了就早停
 *   结束             → summary（自一致性合并）
 */
export async function explore({
  csvPath,
  userCsvPath = null,
  tableName = "main_table",
  goal,
  maxLayers = 3,
  questionsPerLayer = 2,
  poolSize = 4,
  model = DEEPSEEK_CANONICAL_MODEL,
  pythonBin,
  workerScript,
  loadMode = "isolated",
  // ── 与原 pipeline 对齐的开关 ──
  useSkills = true,
  goalSufficiencyCheck = true,
  goalSufficiencyMinLayers = 2,
  thesisInterval = 1,
  maxInsights = 12,
  summarySamples = 3,
  onLog = () => {},
}) {
  const usage = new UsageTracker();
  const deepseek = createDeepSeek({ model });
  const generateJson = makeGenerateJson({ ...deepseek, usage });
  const generateText = makeGenerateText({ ...deepseek, usage });
  const skillPkg = useSkills ? SkillPackage.load() : new SkillPackage(null);

  const pool = new PyWorkerPool({ python: pythonBin, workerScript, size: poolSize });
  await pool.ready();

  try {
    const loaded = await pool.loadAll(
      { csv_path: csvPath, table_name: tableName, user_csv_path: userCsvPath },
      { mode: loadMode },
    );
    await pool.warmup();

    // skill 引导：列名在运行时从真实 schema 填充，不硬编码任何字段名
    const catCols = describeCategoricalColumns(loaded.grouping_columns);
    const executorGuidance = skillPkg.renderGuidance(EXECUTOR_SKILL_IDS, {
      categoricalColumns: catCols,
    });
    const sufficiencyGuidance = skillPkg.renderGuidance(SUFFICIENCY_SKILL_IDS, {
      header: "COVERAGE AUDIT RULES",
      categoricalColumns: catCols,
    });
    const schemaContext = loaded.schema_context + executorGuidance;

    onLog(
      `dataset loaded: schema ${loaded.schema_context.length} chars` +
        `${executorGuidance ? ` + ${executorGuidance.length} chars skill guidance` : ""}, ` +
        `grouping=[${loaded.grouping_columns.slice(0, 4).join(", ")}]`,
    );
    if (useSkills) {
      onLog(
        `skills: ${skillPkg.meta.version ?? "?"} ` +
          `(${skillPkg.skills.length} available, status=${skillPkg.meta.status ?? "?"})`,
      );
    }

    const planner = new Planner({
      generateJson,
      followUpPerLayer: questionsPerLayer,
      exploratoryPerLayer: questionsPerLayer,
      firstLayerQuestions: questionsPerLayer,
    });
    const tools = createDataTools(pool);

    const nodes = [];
    let thesis = null;
    let insightBank = {}; // node_id → 精炼后的 insight 文本
    let missingAspects = [];
    let stoppedEarly = false;
    let layersRun = 0;

    for (let layer = 1; layer <= maxLayers; layer++) {
      layersRun = layer;

      // ── Step 1: 生成问题 ──
      let questions;
      if (layer === 1) {
        questions = (
          await planner.generateInitialQuestions({
            topic: goal,
            dbDescription: schemaContext,
          })
        ).map((q) => ({ ...q, category: "exploratory" }));
      } else {
        const { followUps, exploratory } = await planner.generateTreeBasedQuestions({
          topic: goal,
          dbDescription: schemaContext,
          questionNodes: nodes,
          insights: Object.values(insightBank),
          thesis,
          focusAspects: missingAspects,
        });
        questions = [
          ...followUps.map((q) => ({ ...q, category: "follow_up" })),
          ...exploratory.map((q) => ({ ...q, category: "exploratory" })),
        ];
      }
      onLog(`layer ${layer}/${maxLayers}: ${questions.length} questions`);

      // ── Step 2: 并发执行（pi 的 agent loop 取代手写 ReAct）──
      const answers = await Promise.all(
        questions.map((q) =>
          answerQuestion({
            question: q.question,
            schemaContext,
            deepseek,
            tools,
            usage,
          }).catch((e) => ({
            question: q.question,
            answer: `Execution failed: ${e.message}`,
            turns: 0,
            toolCalls: [],
          })),
        ),
      );

      answers.forEach((a, i) => {
        const q = questions[i];
        nodes.push({
          id: `q_${layer}_${String(i).padStart(2, "0")}`,
          layer,
          category: q.category,
          parentIds: q.parentIds ?? [],
          question: a.question,
          answer: a.answer,
          turns: a.turns,
          toolCalls: a.toolCalls,
        });
      });

      const avgTurns = (
        answers.reduce((s, a) => s + a.turns, 0) / Math.max(1, answers.length)
      ).toFixed(1);

      // ── Step 3: insight bank 去重择优 ──
      const selected = await filterInsights({
        generateJson,
        nodes,
        topic: goal,
        dbDescription: schemaContext,
        thesis: thesis?.title ?? null,
        maxInsights,
        categoricalColumns: catCols,
      }).catch((e) => {
        onLog(`insight filter failed (${e.message}); keeping all findings`);
        return null;
      });
      if (selected && Object.keys(selected).length) {
        insightBank = selected;
      } else if (Object.keys(insightBank).length) {
        // 择优瞬时失败时，上一层已选好的 bank 是比「全部节点」好得多的退路：
        // 它已经去过重、择过优。实测 flag-3 第 4 层择优失败，若退回全部节点，
        // bank 会从 12 条炸到 30 条（等于 raw 口径，precision 失去意义），
        // 且 goal-sufficiency 看到一大堆发现后当层就误判「够了」提前停。
        onLog(
          `insight filter yielded nothing; keeping previous layer's ` +
            `${Object.keys(insightBank).length} selected findings`,
        );
      } else {
        // 首层就失败：没有上一层可退，只能用全部节点，否则后续 planner /
        // thesis / 自评全部失去输入。
        insightBank = Object.fromEntries(
          nodes
            .filter((n) => n.answer && !n.answer.startsWith("Execution failed"))
            .map((n) => [n.id, n.answer]),
        );
        onLog(`insight filter yielded nothing at first layer; falling back to all ${Object.keys(insightBank).length} findings`);
      }
      onLog(
        `layer ${layer} done: avg ${avgTurns} turns/question, ` +
          `insight bank ${Object.keys(insightBank).length}/${nodes.length}`,
      );

      // ── Step 4: goal sufficiency 自评（决定是否早停）──
      missingAspects = [];
      if (goalSufficiencyCheck && layer >= goalSufficiencyMinLayers && layer < maxLayers) {
        const verdict = await evaluateSufficiency({
          generateJson,
          goal,
          insights: Object.values(insightBank),
          skillGuidance: sufficiencyGuidance,
        });
        if (verdict.sufficient) {
          onLog(`goal-sufficiency: findings answer the goal — stopping early at layer ${layer}`);
          stoppedEarly = true;
          break;
        }
        missingAspects = verdict.missingAspects;
        onLog(`goal-sufficiency: not yet — ${missingAspects.length} focus aspect(s)`);
      }

      // ── Step 5: thesis 生成/精炼 ──
      if (layer % thesisInterval === 0) {
        const insights = Object.values(insightBank);
        thesis = thesis
          ? await refineThesis({
              generateJson,
              topic: goal,
              dbDescription: schemaContext,
              currentThesis: thesis,
              insights,
            }).catch(() => thesis)
          : await generateThesis({
              generateJson,
              topic: goal,
              dbDescription: schemaContext,
              insights,
            }).catch(() => null);
        if (thesis) onLog(`thesis: ${thesis.title}`);
      }
    }

    // ── 最终 summary（自一致性合并）──
    const summary = await generateSummary({
      generateText,
      goal,
      nodes,
      samples: summarySamples,
    }).catch((e) => {
      onLog(`summary failed: ${e.message}`);
      return "";
    });
    onLog(`summary: ${summary.length} chars`);

    return {
      nodes,
      insightBank,
      thesis,
      summary,
      schemaContext,
      layersRun,
      stoppedEarly,
      skillPackage: useSkills
        ? { version: skillPkg.meta.version, status: skillPkg.meta.status, count: skillPkg.skills.length }
        : null,
      usage: usage.toJSON(),
    };
  } finally {
    await pool.close();
  }
}
