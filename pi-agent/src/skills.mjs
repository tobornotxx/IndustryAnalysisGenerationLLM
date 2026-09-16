/** Skill-package loading, validation, deterministic selection, and rendering. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

export const DEFAULT_PACKAGE = fileURLToPath(
  new URL("../skills/manual-v9.json", import.meta.url),
);

const FALLBACK_CATEGORICAL = "the primary categorical columns shown in the schema above";
const ALLOWED_STAGES = new Set(["planner", "executor", "sufficiency", "insight_bank", "summary"]);

function packageHash(raw) {
  return createHash("sha256").update(JSON.stringify(raw)).digest("hex");
}

export class SkillPackage {
  constructor(raw, { path = null, requireFrozen = false } = {}) {
    this.meta = raw?.meta ?? {};
    this.skills = raw?.skills ?? [];
    this.episodes = raw?.episodes ?? [];
    this.config = raw?.config ?? {};
    this.path = path;
    this.hash = raw ? packageHash(raw) : null;
    if (requireFrozen && (!this.meta.frozen || this.meta.status !== "frozen-validated")) {
      throw new Error(`auto-skill package must be frozen and validated: ${path ?? "<memory>"}`);
    }
    for (const skill of this.skills) {
      if (!skill.id || !skill.action) throw new Error("skill requires id and action");
      if (skill.stages?.some((stage) => !ALLOWED_STAGES.has(stage))) {
        throw new Error(`skill ${skill.id} has an unsupported stage`);
      }
    }
  }

  static load(path = DEFAULT_PACKAGE, { requireFrozen = false, allowMissing = false } = {}) {
    try {
      return new SkillPackage(JSON.parse(readFileSync(path, "utf8")), { path, requireFrozen });
    } catch (error) {
      if (!allowMissing) throw error;
      console.error(`[skills] load failed (${path}): ${error.message}`);
      return new SkillPackage(null);
    }
  }

  get(id) {
    return this.skills.find((skill) => skill.id === id);
  }

  select(stage, context, { maxSkills = 4, fallbackIds = [] } = {}) {
    if (!ALLOWED_STAGES.has(stage)) throw new Error(`unsupported skill stage: ${stage}`);
    const haystack = String(context ?? "").toLowerCase();
    const candidates = this.skills
      .filter((skill) => (skill.stages ?? []).includes(stage))
      .map((skill) => ({
        skill,
        score: (skill.trigger_terms ?? []).reduce(
          (total, term) => total + (haystack.includes(String(term).toLowerCase()) ? 1 : 0), 0,
        ),
      }))
      .filter(({ skill, score }) => skill.always || score > 0)
      .sort((left, right) => right.score - left.score || left.skill.id.localeCompare(right.skill.id))
      .slice(0, maxSkills)
      .map(({ skill }) => skill);
    if (candidates.length) return candidates;
    return fallbackIds.map((id) => this.get(id)).filter(Boolean).slice(0, maxSkills);
  }

  renderSkills(skills, { header, categoricalColumns } = {}) {
    if (!skills.length) return "";
    const cols = categoricalColumns || FALLBACK_CATEGORICAL;
    const title = header ?? "ANALYTICAL GUIDANCE (generic — applies to this analysis)";
    const lines = skills.map(
      (skill) => `- [${skill.id}] ${skill.action.replaceAll("{categorical_columns}", cols)}`,
    );
    return `\n\n${title}:\n${lines.join("\n")}`;
  }

  renderGuidance(ids, options = {}) {
    return this.renderSkills(ids.map((id) => this.get(id)).filter(Boolean), options);
  }
}

export function describeCategoricalColumns(names, maxN = 4) {
  const picked = (names ?? []).filter(Boolean).slice(0, maxN);
  return picked.length ? picked.join(", ") : FALLBACK_CATEGORICAL;
}

export const EXECUTOR_SKILL_IDS = [
  "decompose-trend-by-category", "decompose-imbalance-by-group", "pick-adequate-time-bucket",
];
export const SUFFICIENCY_SKILL_IDS = ["coverage-multidim-selfcheck", "anti-premature-stop"];
