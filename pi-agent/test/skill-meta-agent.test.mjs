import test from "node:test";
import assert from "node:assert/strict";
import { createExtractionTools } from "../src/skill-meta-agent.mjs";

const episodes = [{
  episode_id: "episode-1", case_id: "flag-1", split: "source-train",
  score: 0.5, steps: [{ question: "q", answer: "a" }],
}];

test("meta-agent tools expose only requested training episodes", async () => {
  const tools = createExtractionTools(episodes, () => {});
  const inspected = await tools[0].execute("call", { episode_id: "episode-1" });
  assert.match(inspected.content[0].text, /flag-1/);
  await assert.rejects(() => tools[0].execute("call", { episode_id: "missing" }), /unknown episode/);
});

test("meta-agent candidate submission is explicit and auditable", async () => {
  let submitted = null;
  const tools = createExtractionTools(episodes, (candidates) => { submitted = candidates; });
  const candidate = {
    id: "compare-groups", trigger: "group comparison", trigger_terms: ["group"],
    stages: ["executor"], action: "Compare relevant groups.", rationale: "Aggregates hide variation.",
  };
  const result = await tools[1].execute("call", { candidates: [candidate] });
  assert.deepEqual(submitted, [candidate]);
  assert.equal(result.details.count, 1);
});
