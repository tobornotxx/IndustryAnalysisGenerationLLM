import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

export const SPLIT_REGISTRY_PATH = fileURLToPath(
  new URL("../../run_on_benchmark/splits/thesis_split_v2.json", import.meta.url),
);

let cached = null;

export function loadSplitRegistry(path = SPLIT_REGISTRY_PATH) {
  if (path === SPLIT_REGISTRY_PATH && cached) return cached;
  const value = JSON.parse(readFileSync(path, "utf8"));
  if (value.schema_version !== 2) throw new Error(`unsupported split registry: ${path}`);
  if (path === SPLIT_REGISTRY_PATH) cached = value;
  return value;
}

export function splitRegistrySha256(path = SPLIT_REGISTRY_PATH) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function matches(caseId, member) {
  if (caseId === member) return true;
  const range = /^([a-zA-Z_-]+):(\d+)-(\d+)$/.exec(member);
  const item = /^([a-zA-Z_-]+)-(\d+)$/.exec(caseId);
  if (!range || !item) return false;
  return range[1].toLowerCase() === item[1].toLowerCase()
    && Number(item[2]) >= Number(range[2])
    && Number(item[2]) <= Number(range[3]);
}

export function assignedSplit(benchmarkId, caseId, registry = loadSplitRegistry()) {
  const splits = registry.benchmarks?.[benchmarkId]?.splits ?? {};
  const found = Object.entries(splits)
    .filter(([, members]) => members.some((member) => matches(caseId, member)))
    .map(([split]) => split);
  if (found.length > 1) {
    throw new Error(`case belongs to multiple splits: ${benchmarkId}/${caseId}: ${found.join(", ")}`);
  }
  return found[0] ?? null;
}

export function assertCaseSplit(benchmarkId, caseId, requestedSplit, registry) {
  const actual = assignedSplit(benchmarkId, caseId, registry);
  if (!actual) throw new Error(`case is not assigned in frozen split registry: ${benchmarkId}/${caseId}`);
  if (actual !== requestedSplit) {
    throw new Error(
      `split mismatch for ${benchmarkId}/${caseId}: requested ${requestedSplit}, frozen as ${actual}`,
    );
  }
  return actual;
}
