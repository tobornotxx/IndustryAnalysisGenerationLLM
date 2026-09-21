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

  createSkill({ name, description, capabilityGap, instructions, evidenceEpisodeIds }) {
    const skillName = safeSkillName(name);
    if (!this.skills.has(skillName) && this.skills.size >= this.maxSkills) {
      throw new Error(`skill creation limit reached: ${this.maxSkills}`);
    }
    const evidence = uniqueStrings(evidenceEpisodeIds);
    if (!description?.trim() || !capabilityGap?.trim() || !instructions?.trim()) {
      throw new Error("skill requires description, capability_gap, and instructions");
    }
    if (!evidence.length) throw new Error("skill requires evidence_episode_ids");
    const unknown = evidence.filter((id) => !this.episodes.has(id));
    if (unknown.length) throw new Error(`unknown evidence episode(s): ${unknown.join(", ")}`);
    const uninspected = evidence.filter((id) => !this.inspected.has(id));
    if (uninspected.length) throw new Error(`inspect evidence episode(s) before citing them: ${uninspected.join(", ")}`);
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
      capability_gap: capabilityGap.trim(),
      evidence_episode_ids: evidence,
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
      name: "create_skill",
      label: "Create or revise skill",
      description:
        "Create a physical SKILL.md backed by inspected episodes. Instructions must teach an actionable analytical method, applicability, and boundaries.",
      parameters: Type.Object({
        name: Type.String(),
        description: Type.String(),
        capability_gap: Type.String(),
        instructions: Type.String(),
        evidence_episode_ids: Type.Array(Type.String()),
      }),
      async execute(_id, input) {
        return toolResult(workspace.createSkill({
          name: input.name,
          description: input.description,
          capabilityGap: input.capability_gap,
          instructions: input.instructions,
          evidenceEpisodeIds: input.evidence_episode_ids,
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

Your job is to inspect scored source-train trajectories, identify a transferable DATA-INSIGHT capability bottleneck, and create at most ${maxSkills} physical Agent Skill${maxSkills === 1 ? "" : "s"}. The Skill is optional knowledge: at runtime the agent initially sees only its name and trigger description and independently decides whether to read or execute it. Do not assume forced prompt injection or a fixed workflow.

A useful Skill must:
- teach a concrete, reusable, multi-step analytical method the base agent did not reliably perform;
- use its description as a precise trigger: say what observable analytical situation should make an agent read it;
- help discover a pattern, propose competing explanations, test them against computed evidence, check plausible confounding or composition effects, and calibrate any causal language to the evidence;
- explain what calculations, comparisons, falsification checks, or sensitivity checks to perform and how their outcomes change the next question;
- say when it applies, when it should not be used, and what evidence would make its conclusion unsafe;
- remain independent of benchmark names, literal answers, entities, dates, values, and dataset-specific column names;
- cite only episodes you actually inspected;
- include a reusable script and test/reference asset when computation can be made executable.

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

Reject candidate ideas that are merely generic reminders, final-answer formatting, token/budget management, stopping rules, tool-use etiquette, or restatements of the base research loop. Those belong in the agent runtime, not in a data-analysis Skill. A Skill must add analytical capability that could change which evidence is computed or how rival explanations are distinguished.

Compare stronger and weaker trajectories before writing. Inspect evidence from more than one source case when possible. Diagnose the capability gap, create the smallest atomic Skill set needed, and prefer one strong Skill over several overlapping reminders. Use create_skill for SKILL.md, write_skill_asset for supporting files, validate_skill_set, fix all validation errors, then call submit_skill_set exactly once. The output must be a usable Skill directory, not JSON advice.`;
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
    await agent.prompt("Inspect contrasting trajectories and create the smallest evidence-backed set of transferable data-insight Skills.");
    if (!workspace.submitted) throw new Error("skill creator stopped without submitting a validated skill set");
    return { output_dir: workspace.outputDir, turns, skills: [...workspace.skills.keys()] };
  } catch (error) {
    workspace.cleanup();
    throw error;
  } finally {
    unsubscribe?.();
  }
}
