import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";

export const MANIFEST_SCHEMA_VERSION = 1;
export const DATA_SPLITS = Object.freeze([
  "dev-contaminated", "source-train", "source-valid", "source-test", "target-test",
]);

export function validateDataSplit(split) {
  if (!DATA_SPLITS.includes(split)) {
    throw new Error(`unsupported data split: ${split}; expected one of ${DATA_SPLITS.join(", ")}`);
  }
  return split;
}

export function sha256Files(paths) {
  const hash = createHash("sha256");
  for (const path of [...paths].sort()) {
    if (!existsSync(path)) continue;
    hash.update(path.split(/[\\/]/).slice(-2).join("/"));
    hash.update("\0");
    hash.update(readFileSync(path));
    hash.update("\0");
  }
  return hash.digest("hex");
}

export function gitState(cwd) {
  const run = (args) => spawnSync("git", ["-c", `safe.directory=${cwd}`, ...args], { cwd, encoding: "utf8" });
  const head = run(["rev-parse", "HEAD"]);
  const status = run(["status", "--porcelain"]);
  return {
    commit: head.status === 0 ? head.stdout.trim() : "unknown",
    dirty: status.status === 0 ? Boolean(status.stdout.trim()) : null,
  };
}

export function buildRunDirectory({ outRoot, experimentId, systemId, caseId, agentRun }) {
  return resolve(outRoot, experimentId, systemId, caseId, `agent_run_${agentRun}`);
}

export function createRunDirectory(path) {
  if (existsSync(path)) throw new Error(`refusing to overwrite existing run: ${path}`);
  mkdirSync(path, { recursive: true });
}

export function writeJsonAtomic(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = join(dirname(path), `.${randomUUID()}.tmp`);
  writeFileSync(tmp, JSON.stringify(value, null, 2), "utf8");
  renameSync(tmp, path);
}

export function makeManifest(fields) {
  return {
    schema_version: MANIFEST_SCHEMA_VERSION,
    run_id: fields.runId ?? randomUUID(),
    created_at: fields.createdAt ?? new Date().toISOString(),
    status: fields.status ?? "planned",
    ...fields,
  };
}
