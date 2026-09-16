import test from "node:test";
import assert from "node:assert/strict";
import { assignedSplit, assertCaseSplit, splitRegistrySha256 } from "../src/split-registry.mjs";

test("frozen source and target memberships are enforced", () => {
  assert.equal(assignedSplit("insightbench-overhaul", "flag-1"), "source-train");
  assert.equal(assignedSplit("insightbench-overhaul", "flag-11"), "source-valid");
  assert.equal(assignedSplit("insightbench-overhaul", "flag-20"), "source-test");
  assert.equal(assignedSplit("insighteval-official", "insighteval-100"), "target-test");
});

test("mislabeled cases fail before an experiment starts", () => {
  assert.throws(
    () => assertCaseSplit("insightbench-overhaul", "flag-13", "source-valid"),
    /frozen as source-test/,
  );
});

test("split registry is content-addressed", () => {
  assert.match(splitRegistrySha256(), /^[a-f0-9]{64}$/);
});
