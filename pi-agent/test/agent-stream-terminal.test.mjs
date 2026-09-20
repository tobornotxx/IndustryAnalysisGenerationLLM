import assert from "node:assert/strict";
import test from "node:test";
import { terminalStreamMessage } from "../src/agent.mjs";

test("terminalStreamMessage reads successful terminal messages", () => {
  const message = { stopReason: "stop" };
  assert.equal(terminalStreamMessage({ type: "done", message }), message);
});

test("terminalStreamMessage preserves provider errors", () => {
  const error = { stopReason: "error", errorMessage: "authentication failed" };
  assert.equal(terminalStreamMessage({ type: "error", error }), error);
});
