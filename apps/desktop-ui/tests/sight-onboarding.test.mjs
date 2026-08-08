import assert from "node:assert/strict";
import test from "node:test";
import {
  readSightOnboardingDecision,
  saveSightOnboardingDecision,
  shouldOfferSightOnboarding,
  sightDownloadPercent,
} from "../src/sightOnboarding.js";

function memoryStorage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  };
}

test("offers Qwen once to a local first-run desktop user", () => {
  const base = {
    nativeRuntime: true,
    trayView: false,
    provider: "local",
    status: { ready: false },
    decision: null,
  };

  assert.equal(shouldOfferSightOnboarding(base), true);
  assert.equal(shouldOfferSightOnboarding({ ...base, trayView: true }), false);
  assert.equal(shouldOfferSightOnboarding({ ...base, provider: "openai" }), false);
  assert.equal(shouldOfferSightOnboarding({ ...base, status: { ready: true } }), false);
  assert.equal(shouldOfferSightOnboarding({ ...base, decision: "later" }), false);
});

test("persists only known onboarding decisions", () => {
  const storage = memoryStorage();

  assert.equal(readSightOnboardingDecision(storage), null);
  assert.equal(saveSightOnboardingDecision("later", storage), true);
  assert.equal(readSightOnboardingDecision(storage), "later");
  assert.equal(saveSightOnboardingDecision("unexpected", storage), false);
  assert.equal(readSightOnboardingDecision(storage), "later");
});

test("converts partial model bytes into bounded progress", () => {
  assert.equal(sightDownloadPercent({ bytesPresent: 388, bytesRequired: 1000 }), 39);
  assert.equal(sightDownloadPercent({ bytesPresent: 0, bytesRequired: 1000 }), 0);
  assert.equal(sightDownloadPercent({ bytesPresent: 2000, bytesRequired: 1000 }), 99);
  assert.equal(sightDownloadPercent({ ready: true, bytesPresent: 1000, bytesRequired: 1000 }), 100);
});
