#!/usr/bin/env node
// Extracts images pasted into the Z-Code chat from the host's session
// artifact store into qa/incoming/ so they can be reviewed with Sens Eye.
//
// Usage:
//   node scripts/extract-chat-images.mjs            # newest session with attachments
//   node scripts/extract-chat-images.mjs <sessionId> # explicit session (sess_...)
import { readdirSync, readFileSync, writeFileSync, mkdirSync, statSync } from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const artifactsRoot = path.join(os.homedir(), ".zcode", "cli", "artifacts");
const incomingDir = path.join(repoRoot, "qa", "incoming");
mkdirSync(incomingDir, { recursive: true });

const sessions = readdirSync(artifactsRoot)
  .filter((name) => name.startsWith("sess_"))
  .map((name) => path.join(artifactsRoot, name))
  .filter((dir) => readdirSync(dir).some((file) => file.startsWith("prompt-attachment-upload-")))
  .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);

const sessionArg = process.argv[2];
const sessionDir = sessionArg
  ? path.join(artifactsRoot, sessionArg)
  : sessions[0];
if (!sessionDir || !statSync(sessionDir).isDirectory()) {
  console.error("Session not found. Usage: node scripts/extract-chat-images.mjs [sessionId]");
  process.exit(1);
}

const files = readdirSync(sessionDir)
  .filter((file) => file.startsWith("prompt-attachment-upload-"))
  .sort((a, b) => statSync(path.join(sessionDir, a)).mtimeMs - statSync(path.join(sessionDir, b)).mtimeMs);

if (files.length === 0) {
  console.log(`No attachments in ${path.basename(sessionDir)}`);
  process.exit(0);
}

const sessionId = path.basename(sessionDir);
let written = 0;
for (const file of files) {
  const raw = readFileSync(path.join(sessionDir, file), "utf8").trim();
  const mime = raw.match(/^data:image\/(png|jpe?g|gif|webp);base64,/)?.[1];
  const b64 = raw.slice(raw.indexOf("base64,") + 7);
  if (!mime || !b64) {
    console.warn(`skip ${file}: not a base64 image`);
    continue;
  }
  const ext = mime === "jpeg" ? "jpg" : mime;
  const stamp = statSync(path.join(sessionDir, file)).mtime;
  const name = `${stamp.toISOString().replace(/[:.]/g, "-")}-${sessionId.slice(5, 11)}-${String(written).padStart(2, "0")}.${ext}`;
  writeFileSync(path.join(incomingDir, name), Buffer.from(b64, "base64"));
  written++;
  console.log(path.join(incomingDir, name));
}
console.log(`${written} image(s) extracted from ${sessionId}`);
