export const SIGHT_ONBOARDING_KEY = "sens.sightOnboarding.v1";

const SAVED_DECISIONS = new Set(["complete", "later"]);

export function readSightOnboardingDecision(storage = globalThis.localStorage) {
  try {
    const value = storage?.getItem(SIGHT_ONBOARDING_KEY);
    return SAVED_DECISIONS.has(value) ? value : null;
  } catch {
    return null;
  }
}

export function saveSightOnboardingDecision(decision, storage = globalThis.localStorage) {
  if (!SAVED_DECISIONS.has(decision)) return false;
  try {
    storage?.setItem(SIGHT_ONBOARDING_KEY, decision);
    return true;
  } catch {
    return false;
  }
}

export function shouldOfferSightOnboarding({ nativeRuntime, trayView, provider, status, decision }) {
  return Boolean(
    nativeRuntime
      && !trayView
      && provider === "local"
      && status
      && !status.ready
      && decision === null,
  );
}

export function sightDownloadPercent(status) {
  if (status?.ready) return 100;
  const required = Number(status?.bytesRequired || 0);
  const present = Number(status?.bytesPresent || 0);
  if (!Number.isFinite(required) || !Number.isFinite(present) || required <= 0 || present <= 0) return 0;
  return Math.max(0, Math.min(99, Math.round((present / required) * 100)));
}
