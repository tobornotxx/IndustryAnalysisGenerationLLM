/** Refuse formal experiment execution when the frozen protocol is not reproducible. */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { sha256Path, validateFormalProtocol } from "./src/protocol.mjs";
import { splitRegistrySha256 } from "./src/split-registry.mjs";
import { writeJsonExclusive } from "./src/skill-lifecycle.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = process.env.REPO_ROOT ?? resolve(HERE, "..");
function arg(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}
const protocolPath = arg("protocol", resolve(REPO, "run_on_benchmark/protocols/skill_transfer_v1.json"));
const registryPath = arg("system-registry", resolve(REPO, "run_on_benchmark/system_registry.json"));
const stage = arg("stage");
if (!stage) throw new Error("--stage is required");
const skillPackagePath = arg("skill-package");
const protocol = JSON.parse(readFileSync(protocolPath, "utf8"));
const systemRegistry = JSON.parse(readFileSync(registryPath, "utf8"));
const validation = validateFormalProtocol(protocol, systemRegistry, { stage, skillPackagePath });

const git = (args) => spawnSync("git", args, { cwd: REPO, encoding: "utf8" });
const headResult = git(["rev-parse", "HEAD"]);
const statusResult = git(["status", "--porcelain"]);
const branchResult = git(["branch", "--show-current"]);
const gitErrors = [];
if (headResult.status !== 0) gitErrors.push("repository HEAD is unavailable");
if (statusResult.status !== 0) gitErrors.push("repository status is unavailable");
else if (statusResult.stdout.trim()) gitErrors.push("repository is dirty; commit or remove all changes before formal runs");

const report = {
  schema_version: 1,
  passed: validation.passed && gitErrors.length === 0,
  checked_at: new Date().toISOString(),
  stage,
  protocol_id: protocol.protocol_id,
  protocol_sha256: sha256Path(protocolPath),
  split_registry_sha256: splitRegistrySha256(),
  system_registry_sha256: sha256Path(registryPath),
  repository: {
    commit: headResult.stdout?.trim() || null,
    branch: branchResult.stdout?.trim() || null,
    clean: !statusResult.stdout?.trim(),
  },
  skill_package: validation.skill_package,
  errors: [...validation.errors, ...gitErrors],
};
const output = arg("output");
if (output) writeJsonExclusive(output, report);
console.log(JSON.stringify(report, null, 2));
if (!report.passed) process.exitCode = 1;
