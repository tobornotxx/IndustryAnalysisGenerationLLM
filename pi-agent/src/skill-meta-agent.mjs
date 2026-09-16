import { createHash } from "node:crypto";
import { Agent } from "@earendil-works/pi-agent-core";
import { Type } from "@sinclair/typebox";

const CandidateSchema = Type.Object({
  id: Type.String(),
  trigger: Type.String(),
  trigger_terms: Type.Array(Type.String()),
  stages: Type.Array(Type.String()),
  action: Type.String(),
  rationale: Type.String(),
  implementation: Type.Optional(Type.String()),
  always: Type.Optional(Type.Boolean()),
});

export function createExtractionTools(episodes, onSubmit) {
  const byId = new Map(episodes.map((episode) => [episode.episode_id, episode]));
  return [
    {
      name: "inspect_episode",
      label: "Inspect training episode",
      description: "Read one source-train trajectory, including its questions, answers, tools, and score.",
      parameters: Type.Object({ episode_id: Type.String() }),
      async execute(_toolCallId, { episode_id: episodeId }) {
        const episode = byId.get(episodeId);
        if (!episode) throw new Error(`unknown episode: ${episodeId}`);
        return {
          content: [{ type: "text", text: JSON.stringify(episode) }],
          details: { episode_id: episodeId },
        };
      },
    },
    {
      name: "submit_skill_candidates",
      label: "Submit skill candidates",
      description: "Submit reusable procedural candidates after comparing the available training episodes.",
      parameters: Type.Object({ candidates: Type.Array(CandidateSchema) }),
      async execute(_toolCallId, { candidates }) {
        onSubmit(candidates);
        return {
          content: [{ type: "text", text: `Accepted ${candidates.length} candidates for generalization and audit.` }],
          details: { count: candidates.length },
        };
      },
    },
  ];
}

export async function runSkillExtractionAgent({ episodes, deepseek, usage, maxTurns = 8 }) {
  if (!episodes.length) throw new Error("cannot run extraction agent on an empty corpus");
  if (episodes.some((episode) => episode.split !== "source-train")) {
    throw new Error("extraction agent can inspect source-train only");
  }
  const catalog = episodes.map((episode) => ({
    episode_id: episode.episode_id,
    case_id: episode.case_id,
    score: episode.score,
    steps: episode.steps?.length ?? 0,
  }));
  const systemPrompt = [
    "You are a meta-agent that learns reusable skills for a data-analysis agent.",
    "Use inspect_episode to compare successful and weak source-train trajectories.",
    "Propose only procedural actions that can transfer to unrelated schemas.",
    "Never include a benchmark name, answer, literal column, entity, date, value, or threshold.",
    "Every candidate needs a trigger, trigger_terms, runtime stages, action, rationale, and implementation.",
    "Finish by calling submit_skill_candidates exactly once. Do not emit the final package yourself.",
    `Available episodes: ${JSON.stringify(catalog)}`,
  ].join("\n\n");
  let submitted = null;
  const agent = new Agent({
    initialState: {
      systemPrompt,
      model: deepseek.model,
      tools: createExtractionTools(episodes, (candidates) => { submitted = candidates; }),
    },
    streamFn: deepseek.streamFn,
  });
  let turns = 0;
  const unsubscribe = agent.subscribe((event) => {
    if (event.type === "turn_end") turns += 1;
    if (event.type === "message_end") usage?.record(event.message?.usage);
  });
  agent.shouldStopAfterTurn = () => turns >= maxTurns || submitted !== null;
  await agent.prompt("Inspect the training evidence, identify transferable failure/success patterns, and submit candidates.");
  unsubscribe?.();
  if (!submitted?.length) throw new Error("skill extraction agent did not submit candidates");
  return {
    candidates: submitted,
    turns,
    prompt_hash: createHash("sha256").update(systemPrompt).digest("hex"),
  };
}
