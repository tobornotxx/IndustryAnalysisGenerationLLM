import { randomUUID } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { delimiter, dirname, join, resolve, sep } from "node:path";
import { spawnSync } from "node:child_process";
import { Agent } from "@earendil-works/pi-agent-core";
import { Type } from "@sinclair/typebox";
import { NativeSkillRuntime, auditNativeSkillDirectory } from "./native-skills.mjs";

function toolResult(value, details = undefined) {
  return {
    content: [{ type: "text", text: typeof value === "string" ? value : JSON.stringify(value) }],
    details,
  };
}

function uniqueStrings(values) {
  return [...new Set((values ?? []).map(String).map((value) => value.trim()).filter(Boolean))];
}

function safeSkillName(name) {
  const value = String(name ?? "").trim();
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(value) || value.length > 64) {
    throw new Error(`invalid skill name: ${value}`);
  }
  return value;
}

function yamlString(value) {
  return JSON.stringify(String(value));
}

function pythonCommand() {
  const configured = process.env.PYTHON?.trim();
  if (configured) return configured;
  const local = process.platform === "win32"
    ? resolve(".venv", "Scripts", "python.exe")
    : resolve(".venv", "bin", "python");
  return existsSync(local) ? local : "python";
}

export function validateCreatorEpisodes(episodes) {
  if (!episodes?.length) throw new Error("skill creator requires source-train episodes");
  if (episodes.some((episode) => episode.split !== "source-train")) {
    throw new Error("skill creator may inspect source-train episodes only");
  }
  const invalid = episodes.filter((episode) => (
    !Number.isFinite(Number(episode.score))
    || episode.score_provenance?.status !== "valid"
    || !episode.score_provenance?.scorer_id
    || Number(episode.score_provenance?.judge_calls ?? 0) < 1
  ));
  if (invalid.length) {
    throw new Error(`skill creator requires validated per-episode scores; invalid=${invalid.length}`);
  }
  const distinct = new Set(episodes.map((episode) => Number(episode.score).toFixed(8)));
  if (episodes.length > 1 && distinct.size < 2) {
    throw new Error("skill creator requires a non-degenerate quality signal; all episode scores are equal");
  }
}

export class SkillCreatorWorkspace {
  constructor({ outputDir, episodes, maxSkills = 3, forbiddenTerms = [] }) {
    if (!Number.isInteger(maxSkills) || maxSkills < 1) {
      throw new Error("maxSkills must be a positive integer");
    }
    this.outputDir = resolve(outputDir);
    this.stagingDir = `${this.outputDir}.staging-${randomUUID()}`;
    this.episodes = new Map(episodes.map((episode) => [episode.episode_id, episode]));
    this.inspected = new Set();
    this.comparisons = new Map();
    this.skills = new Map();
    this.maxSkills = maxSkills;
    this.forbiddenTerms = uniqueStrings(forbiddenTerms);
    this.trace = [];
    this.submitted = false;
    if (existsSync(this.outputDir)) throw new Error(`refusing to overwrite skill output: ${this.outputDir}`);
    mkdirSync(this.stagingDir, { recursive: true });
  }

  cleanup() {
    if (existsSync(this.stagingDir)) rmSync(this.stagingDir, { recursive: true, force: true });
  }

  inspectEpisode(episodeId) {
    const episode = this.episodes.get(episodeId);
    if (!episode) throw new Error(`unknown episode: ${episodeId}`);
    this.inspected.add(episodeId);
    this.trace.push({ sequence: this.trace.length + 1, action: "inspect_episode", episode_id: episodeId });
    return episode;
  }

  compareEpisodes(positiveEpisodeId, negativeEpisodeId) {
    const positive = this.episodes.get(positiveEpisodeId);
    const negative = this.episodes.get(negativeEpisodeId);
    if (!positive || !negative) throw new Error("comparison contains an unknown episode");
    if (!this.inspected.has(positiveEpisodeId) || !this.inspected.has(negativeEpisodeId)) {
      throw new Error("inspect both episodes before comparing them");
    }
    if (positive.case_id !== negative.case_id) {
      throw new Error("discovery comparison requires two runs from the same case");
    }
    if (!(Number(positive.score) > Number(negative.score))) {
      throw new Error("positive episode score must exceed negative episode score");
    }
    const pos = positive.discovery_attribution;
    const neg = negative.discovery_attribution;
    if (!pos?.available || !neg?.available) {
      throw new Error("discovery comparison requires per-reference score attribution");
    }
    const threshold = Math.max(Number(pos.match_threshold ?? 0.5), Number(neg.match_threshold ?? 0.5));
    const nRefs = Math.min(pos.reference_best_match.length, neg.reference_best_match.length);
    const gainedReferenceSlots = [];
    const lostReferenceSlots = [];
    for (let index = 0; index < nRefs; index += 1) {
      if (pos.reference_best_match[index] >= threshold && neg.reference_best_match[index] < threshold) {
        gainedReferenceSlots.push(index);
      }
      if (pos.reference_best_match[index] < threshold && neg.reference_best_match[index] >= threshold) {
        lostReferenceSlots.push(index);
      }
    }
    const gainedInsights = [];
    for (let col = 0; col < (pos.match_matrix?.[0]?.length ?? 0); col += 1) {
      const matchesGainedSlot = gainedReferenceSlots.some((row) => (
        Number(pos.match_matrix?.[row]?.[col] ?? 0) >= threshold
      ));
      if (matchesGainedSlot) {
        gainedInsights.push({
          prediction_index: col,
          best_match: pos.prediction_best_match[col],
          insight: pos.prediction_insights[col],
        });
      }
    }
    const comparison = {
      case_id: positive.case_id,
      positive_episode_id: positiveEpisodeId,
      negative_episode_id: negativeEpisodeId,
      score_delta: Number(positive.score) - Number(negative.score),
      match_threshold: threshold,
      gained_reference_slots: gainedReferenceSlots,
      lost_reference_slots: lostReferenceSlots,
      gained_prediction_insights: gainedInsights,
      positive_question_path: positive.steps.map((step) => ({
        node_id: step.node_id,
        layer: step.layer,
        question: step.question,
        hypothesis: step.hypothesis,
        interpretation: step.interpretation,
        tool_names: step.tool_names,
      })),
      negative_question_path: negative.steps.map((step) => ({
        node_id: step.node_id,
        layer: step.layer,
        question: step.question,
        hypothesis: step.hypothesis,
        interpretation: step.interpretation,
        tool_names: step.tool_names,
      })),
    };
    const key = `${positiveEpisodeId}::${negativeEpisodeId}`;
    this.comparisons.set(key, comparison);
    this.trace.push({
      sequence: this.trace.length + 1,
      action: "compare_episodes",
      positive_episode_id: positiveEpisodeId,
      negative_episode_id: negativeEpisodeId,
      case_id: positive.case_id,
      gained_reference_slots: gainedReferenceSlots,
    });
    return comparison;
  }

  createSkill({
    name,
    description,
    capabilityGap,
    discoveryGain,
    instructions,
    positiveEvidenceEpisodeIds,
    negativeEvidenceEpisodeIds,
  }) {
    const skillName = safeSkillName(name);
    if (!this.skills.has(skillName) && this.skills.size >= this.maxSkills) {
      throw new Error(`skill creation limit reached: ${this.maxSkills}`);
    }
    const positiveEvidence = uniqueStrings(positiveEvidenceEpisodeIds);
    const negativeEvidence = uniqueStrings(negativeEvidenceEpisodeIds);
    const evidence = uniqueStrings([...positiveEvidence, ...negativeEvidence]);
    if (!description?.trim() || !capabilityGap?.trim() || !discoveryGain?.trim() || !instructions?.trim()) {
      throw new Error("skill requires description, capability_gap, discovery_gain, and instructions");
    }
    if (!positiveEvidence.length || !negativeEvidence.length) {
      throw new Error("discovery skill requires positive and negative evidence episodes");
    }
    const unknown = evidence.filter((id) => !this.episodes.has(id));
    if (unknown.length) throw new Error(`unknown evidence episode(s): ${unknown.join(", ")}`);
    const uninspected = evidence.filter((id) => !this.inspected.has(id));
    if (uninspected.length) throw new Error(`inspect evidence episode(s) before citing them: ${uninspected.join(", ")}`);
    const comparedCases = new Set();
    for (const positiveId of positiveEvidence) {
      for (const negativeId of negativeEvidence) {
        const positive = this.episodes.get(positiveId);
        const negative = this.episodes.get(negativeId);
        if (positive?.case_id !== negative?.case_id) continue;
        const comparison = this.comparisons.get(`${positiveId}::${negativeId}`);
        if (comparison?.gained_reference_slots?.length) comparedCases.add(positive.case_id);
      }
    }
    if (comparedCases.size < 2) {
      throw new Error("discovery skill requires reference-coverage gains in strong-vs-weak comparisons from at least two cases");
    }
    const dir = join(this.stagingDir, skillName);
    mkdirSync(dir, { recursive: true });
    const skillMd = [
      "---",
      `name: ${skillName}`,
      `description: ${yamlString(description.trim())}`,
      "---",
      instructions.trim(),
      "",
    ].join("\n");
    writeFileSync(join(dir, "SKILL.md"), skillMd, "utf8");
    const provenance = {
      schema_version: 1,
      skill_name: skillName,
      capability_type: "discovery",
      capability_gap: capabilityGap.trim(),
      discovery_gain: discoveryGain.trim(),
      evidence_episode_ids: evidence,
      positive_evidence_episode_ids: positiveEvidence,
      negative_evidence_episode_ids: negativeEvidence,
      compared_case_ids: [...comparedCases],
      evidence: evidence.map((id) => {
        const episode = this.episodes.get(id);
        return {
          episode_id: id,
          case_id: episode.case_id,
          score: episode.score,
          scorer_id: episode.score_provenance.scorer_id,
        };
      }),
    };
    writeFileSync(join(dir, "provenance.json"), `${JSON.stringify(provenance, null, 2)}\n`, "utf8");
    this.skills.set(skillName, { name: skillName, description: description.trim(), provenance });
    this.trace.push({ sequence: this.trace.length + 1, action: "create_skill", skill_name: skillName, evidence_episode_ids: evidence });
    return { name: skillName, path: dir };
  }

  writeAsset({ skillName, path, content }) {
    const name = safeSkillName(skillName);
    if (!this.skills.has(name)) throw new Error(`create SKILL.md before assets: ${name}`);
    const normalized = String(path ?? "").replaceAll("\\", "/").replace(/^\/+/, "");
    if (!/^(?:scripts|tests|references)\/[a-zA-Z0-9_.\/-]+$/.test(normalized)) {
      throw new Error(`unsupported skill asset path: ${path}`);
    }
    if (normalized.split("/").includes("..")) throw new Error(`skill asset escapes directory: ${path}`);
    const root = resolve(this.stagingDir, name);
    const target = resolve(root, normalized);
    if (!target.startsWith(`${root}${sep}`)) throw new Error(`skill asset escapes directory: ${path}`);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, String(content ?? ""), "utf8");
    this.trace.push({ sequence: this.trace.length + 1, action: "write_asset", skill_name: name, path: normalized });
    return { skill_name: name, path: normalized };
  }

  async validate() {
    if (!this.skills.size) return { passed: false, errors: ["no skills created"], skills: [] };
    const errors = [];
    let runtime;
    try {
      runtime = await NativeSkillRuntime.load([this.stagingDir]);
    } catch (error) {
      errors.push(error.message);
    }
    if (runtime && runtime.skills.length !== this.skills.size) {
      errors.push(`loader found ${runtime.skills.length}/${this.skills.size} skills`);
    }
    if (runtime && this.forbiddenTerms.length) {
      const audit = await auditNativeSkillDirectory(this.stagingDir, {
        forbiddenTerms: this.forbiddenTerms,
      });
      for (const finding of audit.findings) {
        errors.push(
          `${finding.skill_name} ${finding.kind} in ${finding.path}: ${finding.terms.join(", ")}`,
        );
      }
    }
    for (const skill of this.skills.values()) {
      if (!skill.provenance.evidence_episode_ids.every((id) => this.inspected.has(id))) {
        errors.push(`${skill.name} cites an episode that was not inspected`);
      }
      const body = readFileSync(join(this.stagingDir, skill.name, "SKILL.md"), "utf8");
      if (!/when|use this skill|applicable|trigger/i.test(body)) {
        errors.push(`${skill.name} does not explain when it applies`);
      }
      if (!/do not|avoid|not use|limitations?/i.test(body)) {
        errors.push(`${skill.name} does not explain a boundary or non-applicable condition`);
      }
      const skillDir = join(this.stagingDir, skill.name);
      const scriptDir = join(skillDir, "scripts");
      const testDir = join(skillDir, "tests");
      const scripts = existsSync(scriptDir)
        ? readdirSync(scriptDir, { recursive: true }).filter((path) => String(path).endsWith(".py"))
        : [];
      const tests = existsSync(testDir)
        ? readdirSync(testDir, { recursive: true }).filter((path) => String(path).endsWith(".py"))
        : [];
      if (scripts.length && !tests.length) {
        errors.push(`${skill.name} includes executable scripts but no Python test asset`);
      }
      if (scripts.length && tests.length) {
        const testSources = tests.map((testPath) => (
          readFileSync(join(testDir, String(testPath)), "utf8")
        )).join("\n");
        if (!/\brun\s*\(/.test(testSources)) {
          errors.push(`${skill.name} Python tests must call the run(sql_results, skill_args) entrypoint`);
        }
      }
      for (const scriptPath of scripts) {
        const source = readFileSync(join(scriptDir, String(scriptPath)), "utf8");
        if (!/^\s*def\s+run\s*\(\s*sql_results(?:\s*:[^,]+)?\s*,\s*skill_args(?:\s*:[^)]+)?\s*\)\s*(?:->\s*[^:]+)?\s*:/m.test(source)) {
          errors.push(
            `${skill.name} Python script must define run(sql_results, skill_args): ${join("scripts", String(scriptPath))}`,
          );
        }
      }
      for (const testPath of tests) {
        const relativeTest = join("tests", String(testPath));
        const result = spawnSync(pythonCommand(), [relativeTest], {
          cwd: skillDir,
          encoding: "utf8",
          env: {
            ...process.env,
            PYTHONDONTWRITEBYTECODE: "1",
            PYTHONPATH: [skillDir, process.env.PYTHONPATH].filter(Boolean).join(delimiter),
          },
          timeout: 30_000,
          maxBuffer: 1_000_000,
        });
        if (result.error || result.status !== 0) {
          const detail = [result.error?.message, result.stdout, result.stderr]
            .filter(Boolean).join("\n").trim().slice(-4_000);
          errors.push(`${skill.name} Python test failed (${relativeTest}): ${detail || `exit ${result.status}`}`);
        }
      }
    }
    return {
      passed: errors.length === 0,
      errors,
      skills: [...this.skills.keys()],
      inspected_episode_ids: [...this.inspected],
    };
  }

  async submit() {
    const validation = await this.validate();
    if (!validation.passed) return validation;
    writeFileSync(join(this.stagingDir, "creator_trajectory.jsonl"), (
      this.trace.map((item) => JSON.stringify(item)).join("\n") + "\n"
    ), "utf8");
    writeFileSync(join(this.stagingDir, "creator_manifest.json"), `${JSON.stringify({
      schema_version: 1,
      produced_by: "pi-skill-creator-agent",
      skills: [...this.skills.keys()],
      inspected_episode_ids: [...this.inspected],
      validation,
    }, null, 2)}\n`, "utf8");
    renameSync(this.stagingDir, this.outputDir);
    this.submitted = true;
    return { ...validation, output_dir: this.outputDir };
  }
}

export function createSkillCreatorTools(workspace) {
  return [
    {
      name: "inspect_episode",
      label: "Inspect training episode",
      description: "Read one scored source-train trajectory, including questions, answers, tools, and judge evidence.",
      parameters: Type.Object({ episode_id: Type.String() }),
      async execute(_id, { episode_id: episodeId }) {
        return toolResult(workspace.inspectEpisode(episodeId), { episode_id: episodeId });
      },
    },
    {
      name: "compare_episodes",
      label: "Compare strong and weak discovery trajectories",
      description:
        "Compare two inspected runs from the same case. Returns reference slots found only by the stronger run, the responsible predicted insights, and both question paths.",
      parameters: Type.Object({
        positive_episode_id: Type.String(),
        negative_episode_id: Type.String(),
      }),
      async execute(_id, input) {
        return toolResult(workspace.compareEpisodes(
          input.positive_episode_id, input.negative_episode_id,
        ));
      },
    },
    {
      name: "create_skill",
      label: "Create or revise skill",
      description:
        "Create a physical SKILL.md backed by inspected episodes. Instructions must teach an actionable analytical method, applicability, and boundaries.",
      parameters: Type.Object({
        name: Type.String(),
        description: Type.String(),
        capability_gap: Type.String(),
        discovery_gain: Type.String(),
        instructions: Type.String(),
        positive_evidence_episode_ids: Type.Array(Type.String()),
        negative_evidence_episode_ids: Type.Array(Type.String()),
      }),
      async execute(_id, input) {
        return toolResult(workspace.createSkill({
          name: input.name,
          description: input.description,
          capabilityGap: input.capability_gap,
          discoveryGain: input.discovery_gain,
          instructions: input.instructions,
          positiveEvidenceEpisodeIds: input.positive_evidence_episode_ids,
          negativeEvidenceEpisodeIds: input.negative_evidence_episode_ids,
        }));
      },
    },
    {
      name: "write_skill_asset",
      label: "Write skill asset",
      description: "Write an optional script, test, or reference file inside an already-created skill directory.",
      parameters: Type.Object({
        skill_name: Type.String(),
        path: Type.String(),
        content: Type.String(),
      }),
      async execute(_id, input) {
        return toolResult(workspace.writeAsset({
          skillName: input.skill_name,
          path: input.path,
          content: input.content,
        }));
      },
    },
    {
      name: "validate_skill_set",
      label: "Validate skill set",
      description: "Validate PI Skill loading, provenance, applicability, and boundaries before submission.",
      parameters: Type.Object({}),
      async execute() {
        return toolResult(await workspace.validate());
      },
    },
    {
      name: "submit_skill_set",
      label: "Submit skill set",
      description: "Atomically finalize the complete physical Skill directory after validation passes.",
      parameters: Type.Object({}),
      async execute() {
        return toolResult(await workspace.submit());
      },
    },
  ];
}

function creatorSystemPrompt(maxSkills) {
  return `You are a PI Skill Creator for a general-purpose PI agent that may optionally become a data-insight research agent by discovering and reading Skills.

Your job in this cycle is specifically to create DISCOVERY Skills: reusable search policies that help the agent uncover additional decision-relevant patterns that weaker runs miss. Inspect scored source-train trajectories, compare stronger and weaker runs of the SAME case, identify the question or computation where the stronger path first gained reference coverage, and create at most ${maxSkills} physical Agent Skill${maxSkills === 1 ? "" : "s"}. The Skill is optional knowledge: at runtime the agent initially sees only its name and trigger description and independently decides whether to read or execute it. Do not assume forced prompt injection or a fixed workflow.

A useful Skill must:
- teach a concrete, reusable, multi-step SEARCH method the base agent did not reliably perform;
- use its description as a precise trigger: say what observable analytical situation should make an agent read it;
- expand the hypothesis space before narrowing it: enumerate several plausible slices, interactions, regimes, text clusters, sequences, or mechanisms appropriate to the observed signal;
- rank candidate discoveries using computed evidence such as coverage, effect magnitude, stability, distinctness, and decision relevance;
- explain how the top candidates redirect the next research questions, including at least one rival explanation where relevant;
- say when it applies, when it should not be used, and what evidence would make its conclusion unsafe;
- remain independent of benchmark names, literal answers, entities, dates, values, and dataset-specific column names;
- cite only episodes you actually inspected and compared;
- include a reusable script and test/reference asset when computation can be made executable.

Evidence requirements:
- Use inspect_episode, then compare_episodes on a stronger and weaker run from the same case.
- A candidate must be supported by strong-vs-weak comparisons from at least two distinct source cases.
- In discovery_gain, state what the stronger paths found that weaker paths missed and which search transition plausibly enabled it.
- Positive evidence means the stronger run; negative evidence means the weaker same-case run. Do not substitute unrelated high- and low-scoring cases.

Selection and scope requirements:
- Make the catalog description discriminating, not universal: state the observable situation where the Skill can change a downstream decision, and do not use catchalls such as "ANY derived metric".
- State whether the Skill is a local diagnostic for one claim or a path-level research strategy. This is guidance for autonomous planning, not a fixed question count.
- Put cheap qualification checks inside the relevant analytical method. Do not require a separate research question merely to decide whether the Skill applies.
- State an early-exit condition and how a positive result should redirect later questions.

Executable Skill contract:
- A Python asset must define exactly the runtime entrypoint run(sql_results, skill_args).
- run receives the Agent-selected SQL result as a pandas DataFrame and adaptive JSON arguments chosen for the current dataset.
- One call to run must perform the complete relevant battery, including internally parallel or sequential checks and branching; do not make the Agent reconstruct the script across repeated Python calls.
- Return one JSON-serializable object with applicability, checks performed, evidence, verdict, limitations, and suggested next questions. Printing is not the interface.
- Its offline test must call run once on a representative DataFrame fixture and assert decision-relevant outputs. The Creator validator will reject scripts without this entrypoint or failing tests.

Reject candidate ideas that are merely generic reminders, final-answer formatting, token/budget management, stopping rules, tool-use etiquette, or restatements of the base research loop. Also reject a candidate whose main purpose is only to validate, audit, falsify, clean, or calibrate an already discovered claim. Those may be useful validation Skills, but they do not test the discovery-transfer hypothesis in this cycle. A discovery Skill must change what hypotheses are generated, what evidence is searched, or which branch is explored next.

Compare stronger and weaker trajectories before writing. Diagnose the discovery gap, create the smallest atomic Skill set needed, and prefer one strong Skill over several overlapping reminders. Use create_skill for SKILL.md, write_skill_asset for supporting files, validate_skill_set, fix all validation errors, then call submit_skill_set exactly once. The output must be a usable Skill directory, not JSON advice.`;
}

export async function runSkillCreatorAgent({
  episodes,
  deepseek,
  usage,
  outputDir,
  maxTurns = 20,
  maxSkills = 3,
  forbiddenTerms = [],
}) {
  validateCreatorEpisodes(episodes);
  const workspace = new SkillCreatorWorkspace({
    outputDir, episodes, maxSkills, forbiddenTerms,
  });
  const catalog = episodes.map((episode) => ({
    episode_id: episode.episode_id,
    case_id: episode.case_id,
    score: episode.score,
    scorer_id: episode.score_provenance.scorer_id,
    steps: episode.steps?.length ?? 0,
    reference_slots_covered: episode.discovery_attribution?.available
      ? episode.discovery_attribution.reference_best_match.filter(
        (value) => value >= Number(episode.discovery_attribution.match_threshold ?? 0.5),
      ).length
      : null,
  }));
  const agent = new Agent({
    initialState: {
      systemPrompt: `${creatorSystemPrompt(maxSkills)}\n\nAVAILABLE SOURCE-TRAIN EPISODES:\n${JSON.stringify(catalog)}`,
      model: deepseek.model,
      tools: createSkillCreatorTools(workspace),
    },
    streamFn: deepseek.streamFn,
  });
  let turns = 0;
  const unsubscribe = agent.subscribe((event) => {
    if (event.type === "turn_end") turns += 1;
    if (event.type === "message_end") usage?.record(event.message?.usage);
  });
  agent.shouldStopAfterTurn = () => turns >= maxTurns || workspace.submitted;
  try {
    await agent.prompt("Mine same-case discovery gains and create the smallest evidence-backed set of transferable discovery Skills.");
    if (!workspace.submitted) throw new Error("skill creator stopped without submitting a validated skill set");
    return { output_dir: workspace.outputDir, turns, skills: [...workspace.skills.keys()] };
  } catch (error) {
    workspace.cleanup();
    throw error;
  } finally {
    unsubscribe?.();
  }
}
