import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { packageHash } from "./skill-lifecycle.mjs";
import { assertCaseSplit } from "./split-registry.mjs";
import { SkillPackage } from "./skills.mjs";

export function sha256Path(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function expandCases(caseIds) {
  return caseIds.flatMap((caseId) => {
    const match = /^([a-zA-Z_-]+):(\d+)-(\d+)$/.exec(caseId);
    if (!match) return [caseId];
    return Array.from(
      { length: Number(match[3]) - Number(match[2]) + 1 },
      (_, index) => `${match[1]}-${Number(match[2]) + index}`,
    );
  });
}

export function validateFormalProtocol(protocol, systemRegistry, { stage, skillPackagePath } = {}) {
  const errors = [];
  if (protocol.schema_version !== 1) errors.push("protocol schema_version must be 1");
  if (protocol.status !== "frozen-before-formal-api-runs") errors.push("protocol is not frozen");
  const generation = protocol.generation ?? {};
  if (generation.model !== "deepseek-flash") errors.push("generation model must be deepseek-flash");
  if (generation.reasoning !== "medium" || generation.thinking_mode !== true) {
    errors.push("thinking mode must be explicitly enabled at medium reasoning");
  }
  if (generation.agent_runs_per_case !== 3) errors.push("formal protocol requires three agent runs per case");
  if (protocol.scoring?.judge_runs_per_prediction !== 1) errors.push("each prediction must be judged exactly once");
  const expectedOrder = ["source-train", "source-valid", "source-test", "target-test"];
  if (JSON.stringify(protocol.required_order) !== JSON.stringify(expectedOrder)) {
    errors.push("stage order must be source-train, source-valid, source-test, target-test");
  }
  for (const [split, spec] of Object.entries(protocol.stages ?? {})) {
    for (const systemId of spec.systems ?? []) {
      if (!systemRegistry.systems?.[systemId]) errors.push(`unknown system in ${split}: ${systemId}`);
    }
    for (const caseId of expandCases(spec.case_ids ?? [])) {
      try { assertCaseSplit(spec.benchmark_id, caseId, split); }
      catch (error) { errors.push(error.message); }
    }
  }
  if (stage && !protocol.stages?.[stage]) errors.push(`stage is not in protocol: ${stage}`);

  let skillPackage = null;
  if (["source-test", "target-test"].includes(stage)) {
    if (!skillPackagePath) errors.push(`${stage} requires --skill-package`);
    else {
      try {
        const raw = JSON.parse(readFileSync(skillPackagePath, "utf8"));
        const loaded = new SkillPackage(raw, { path: skillPackagePath, requireFrozen: true });
        const calculated = packageHash(raw);
        if (raw.meta?.package_hash !== calculated) errors.push("frozen skill package content hash does not match");
        skillPackage = { version: loaded.meta.version, package_hash: calculated, skills: loaded.skills.length };
      } catch (error) { errors.push(error.message); }
    }
  }
  return { passed: errors.length === 0, errors, stage: stage ?? null, skill_package: skillPackage };
}
