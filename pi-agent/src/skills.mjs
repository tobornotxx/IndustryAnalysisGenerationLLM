/**
 * Skill 包加载 —— 读的是 Python 侧同一份 skill_package_v9.json。
 *
 * 刻意不另建一份 TS 专用的 skill 数据：两份会漂移，而 skill 的正当性依赖
 * provenance 可追溯（见 SKILL_EXTRACTION_PLAN §1.1）。
 *
 * 占位符（如 {categorical_columns}）在运行时由当前数据集的真实 schema 填充 ——
 * 这是消除 ServiceNow 字段硬编码那次修复的核心机制，不能丢。
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

/**
 * 默认位置：本 repo 的同级目录下的 MyDataStorm。
 * 两个 repo 必须并列 clone（见 handover/HANDOVER.md §2.3）。
 * 换机器/换布局时用 SKILL_PACKAGE_PATH 覆盖，不要改这里。
 */
const DEFAULT_PACKAGE =
  process.env.SKILL_PACKAGE_PATH ??
  resolve(
    dirname(fileURLToPath(import.meta.url)),
    "../../../MyDataStorm/datastorm/skills/skill_package_v9.json",
  );

/** 占位符找不到真实列名时的兜底措辞（仍然数据无关）。 */
const FALLBACK_CATEGORICAL =
  "the primary categorical columns shown in the schema above";

export class SkillPackage {
  constructor(raw) {
    this.meta = raw?.meta ?? {};
    this.skills = raw?.skills ?? [];
    this.episodes = raw?.episodes ?? [];
    this.config = raw?.config ?? {};
  }

  static load(path = DEFAULT_PACKAGE) {
    try {
      return new SkillPackage(JSON.parse(readFileSync(path, "utf8")));
    } catch (e) {
      // 加载失败降级为「无引导」，不让主流程崩
      console.error(`[skills] load failed (${path}): ${e.message}`);
      return new SkillPackage(null);
    }
  }

  get(id) {
    return this.skills.find((s) => s.id === id);
  }

  /**
   * 把若干 skill 渲染成可注入 prompt 的引导文本块。
   * 缺失的 id 静默跳过 —— 便于 ablation 时关掉某条。
   */
  renderGuidance(ids, { header, categoricalColumns } = {}) {
    const cols = categoricalColumns || FALLBACK_CATEGORICAL;
    const selected = ids.map((id) => this.get(id)).filter(Boolean);
    if (!selected.length) return "";
    const title =
      header ?? "ANALYTICAL GUIDANCE (generic — applies to any question on this data)";
    const lines = selected.map(
      (s) => `- ${s.action.replaceAll("{categorical_columns}", cols)}`,
    );
    return `\n\n${title}:\n${lines.join("\n")}`;
  }
}

/** 把列名列表格式化成可嵌入 prompt 的字符串。 */
export function describeCategoricalColumns(names, maxN = 4) {
  const picked = (names ?? []).filter(Boolean).slice(0, maxN);
  return picked.length ? picked.join(", ") : FALLBACK_CATEGORICAL;
}

/** executor 侧消费的 skill（对应 Python 的 _BRIDGE_SKILL_IDS）。 */
export const EXECUTOR_SKILL_IDS = [
  "decompose-trend-by-category",
  "decompose-imbalance-by-group",
  "pick-adequate-time-bucket",
];

/** goal-sufficiency 自评消费的 skill（对应 Python 的 _SUFFICIENCY_SKILL_IDS）。 */
export const SUFFICIENCY_SKILL_IDS = [
  "coverage-multidim-selfcheck",
  "anti-premature-stop",
];
