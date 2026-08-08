export function hearingDownloadPercent(status) {
  const required = Number(status?.installBytesRequired || 0);
  const present = Number(status?.installBytesPresent || 0);
  if (!required || present <= 0) return 0;
  return Math.max(0, Math.min(99, Math.round((present / required) * 100)));
}

export function hearingModelReady(status, model) {
  if (model === "remote") return true;
  return status?.model === model && status?.modelInstalled === true;
}
