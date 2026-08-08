import assert from "node:assert/strict";
import test from "node:test";

import { hearingDownloadPercent, hearingModelReady } from "../src/hearingSetup.js";

test("reports readiness only for the selected verified local model", () => {
  assert.equal(hearingModelReady({ model: "qwen", modelInstalled: true }, "qwen"), true);
  assert.equal(hearingModelReady({ model: "gigaam", modelInstalled: true }, "qwen"), false);
  assert.equal(hearingModelReady({ model: "qwen", modelInstalled: false }, "qwen"), false);
  assert.equal(hearingModelReady(null, "remote"), true);
});

test("bounds byte progress below completion until verification finishes", () => {
  assert.equal(hearingDownloadPercent({ installBytesPresent: 50, installBytesRequired: 100 }), 50);
  assert.equal(hearingDownloadPercent({ installBytesPresent: 100, installBytesRequired: 100 }), 99);
  assert.equal(hearingDownloadPercent({ installBytesPresent: 0, installBytesRequired: 100 }), 0);
});
