import { createHash } from "node:crypto";
import {
  existsSync,
  readFileSync,
  readdirSync,
  statSync,
} from "node:fs";
import { dirname, relative, resolve, sep } from "node:path";
import {
  formatSkillInvocation,
  formatSkillsForSystemPrompt,
  loadSkills,
} from "@earendil-works/pi-agent-core";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import { Type } from "@sinclair/typebox";

function textResult(text, details = undefined) {
  return { content: [{ type: "text", text }], details };
}

function walkFiles(root, output = []) {
  if (!existsSync(root)) return output;
  for (const entry of readdirSync(root, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    const path = resolve(root, entry.name);
    if (entry.isDirectory()) walkFiles(path, output);
    else if (entry.isFile()) output.push(path);
  }
  return output;
}

function discoverSkillDirs(root, output = []) {
  if (!existsSync(root) || !statSync(root).isDirectory()) return output;
  const entries = readdirSync(root, { withFileTypes: true });
  if (entries.some((entry) => entry.isFile() && entry.name === "SKILL.md")) {
    output.push(root);
    return output;
  }
  for (const entry of entries.filter((item) => item.isDirectory()).sort((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.name.startsWith(".") && entry.name !== "node_modules") {
      discoverSkillDirs(resolve(root, entry.name), output);
    }
  }
  return output;
}

export function hashSkillDirectories(directories) {
  const hash = createHash("sha256");
  const skillDirs = [...new Set(
    [...directories].flatMap((path) => discoverSkillDirs(resolve(path))),
  )].sort((left, right) => left.localeCompare(right));
  for (const skillDir of skillDirs) {
    hash.update(`skill:${skillDir.split(/[\\/]/).pop()}\n`);
    for (const file of walkFiles(skillDir)) {
      hash.update(`file:${relative(skillDir, file).replaceAll("\\", "/")}\n`);
      hash.update(readFileSync(file));
    }
  }
  return hash.digest("hex");
}

export class NativeSkillRuntime {
  constructor({ directories, skills, diagnostics }) {
    this.directories = directories;
    this.skills = skills;
    this.diagnostics = diagnostics;
    this.byName = new Map(skills.map((skill) => [skill.name, skill]));
    this.reads = [];
    this.executions = [];
    this.hash = hashSkillDirectories(directories);
  }

  static async load(directories, { cwd = process.cwd(), requireFrozen = false } = {}) {
    const roots = [...new Set((directories ?? []).map((path) => resolve(cwd, path)))];
    if (!roots.length) return new NativeSkillRuntime({ directories: [], skills: [], diagnostics: [] });
    const env = new NodeExecutionEnv({ cwd });
    const loaded = await loadSkills(env, roots);
    await env.cleanup();
    const errors = loaded.diagnostics.map((item) => `${item.code}: ${item.path}: ${item.message}`);
    if (errors.length) throw new Error(`invalid skill directory:\n${errors.join("\n")}`);
    const duplicateNames = loaded.skills
      .map((skill) => skill.name)
      .filter((name, index, names) => names.indexOf(name) !== index);
    if (duplicateNames.length) throw new Error(`duplicate skill name(s): ${[...new Set(duplicateNames)].join(", ")}`);
    const runtime = new NativeSkillRuntime({ directories: roots, skills: loaded.skills, diagnostics: loaded.diagnostics });
    if (requireFrozen) {
      if (roots.length !== 1) throw new Error("a frozen skill set must be loaded from exactly one root directory");
      const manifestPath = resolve(roots[0], "skill-set-manifest.json");
      if (!existsSync(manifestPath)) throw new Error(`frozen skill manifest is missing: ${manifestPath}`);
      const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
      if (manifest.status !== "frozen-validated" || manifest.frozen !== true) {
        throw new Error("skill-set manifest is not frozen and validated");
      }
      if (manifest.content_hash !== runtime.hash) throw new Error("frozen skill content hash does not match");
      const expectedNames = [...(manifest.skills ?? [])].sort();
      const actualNames = runtime.skills.map((skill) => skill.name).sort();
      if (JSON.stringify(expectedNames) !== JSON.stringify(actualNames)) {
        throw new Error("frozen skill manifest names do not match loaded skills");
      }
      runtime.frozenManifest = manifest;
    }
    return runtime;
  }

  get catalogPrompt() {
    return formatSkillsForSystemPrompt(this.skills);
  }

  get(name) {
    return this.byName.get(name);
  }

  read(name) {
    const skill = this.get(name);
    if (!skill) throw new Error(`unknown skill: ${name}`);
    this.reads.push({ name, sequence: this.reads.length + 1 });
    return formatSkillInvocation(skill);
  }

  hasBeenRead(name) {
    return this.reads.some((item) => item.name === name);
  }

  resolveScript(name, relativePath) {
    const skill = this.get(name);
    if (!skill) throw new Error(`unknown skill: ${name}`);
    if (!this.hasBeenRead(name)) throw new Error(`read skill before executing its scripts: ${name}`);
    const root = resolve(dirname(skill.filePath));
    const script = resolve(root, String(relativePath ?? ""));
    if (script !== root && !script.startsWith(`${root}${sep}`)) {
      throw new Error(`skill script escapes its directory: ${relativePath}`);
    }
    if (!script.endsWith(".py")) throw new Error("skill script must be a .py file");
    if (!existsSync(script) || !statSync(script).isFile()) throw new Error(`skill script not found: ${relativePath}`);
    return script;
  }
}

export function createNativeSkillTools(runtime, { state, pool }) {
  if (!runtime.skills.length) return [];
  return [
    {
      name: "read_skill",
      label: "Read skill",
      description:
        "Read the complete instructions for one available skill when its description matches the current analytical need. " +
        "Do not read skills speculatively.",
      parameters: Type.Object({ name: Type.String() }),
      async execute(_id, { name }) {
        return textResult(runtime.read(name), { skill_name: name });
      },
    },
    {
      name: "run_skill_python",
      label: "Run skill Python script",
      description:
        "Execute a Python script supplied by a skill that has already been read. " +
        "The selected SQL result is available to the script as `sql_results`, and JSON arguments as `skill_args`.",
      parameters: Type.Object({
        skill_name: Type.String(),
        script: Type.String(),
        question_id: Type.String(),
        sql: Type.String(),
        arguments: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
      }),
      async execute(_id, {
        skill_name: skillName,
        script: relativePath,
        question_id: questionId,
        sql,
        arguments: skillArguments = {},
      }) {
        state.requireOpen(questionId);
        const scriptPath = runtime.resolveScript(skillName, relativePath);
        const source = readFileSync(scriptPath, "utf8");
        const code = `import json\nskill_args = json.loads(${JSON.stringify(JSON.stringify(skillArguments))})\n${source}`;
        const result = await pool.call("python", { sql, code });
        const execution = {
          skill_name: skillName,
          script: relative(dirname(runtime.get(skillName).filePath), scriptPath).replaceAll("\\", "/"),
          question_id: questionId,
        };
        runtime.executions.push(execution);
        state.recordToolEvidence(
          questionId,
          `run_skill_python:${skillName}/${execution.script}`,
          { sql, arguments: skillArguments },
          result.output,
        );
        return textResult(result.output, execution);
      },
    },
  ];
}
