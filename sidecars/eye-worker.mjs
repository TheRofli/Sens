import path from 'node:path';
import readline from 'node:readline';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const eyeRoot = process.env.SENS_EYE_ROOT;
if (!eyeRoot) {
  process.stderr.write('SENS_EYE_ROOT is required.\n');
  process.exit(2);
}

const artifactOutputRoot = process.env.SENS_ARTIFACTS_ROOT
  ?? path.join(
    process.env.LOCALAPPDATA ?? process.env.APPDATA ?? process.cwd(),
    'Sens',
    'artifacts',
    'sight-results',
  );

const serviceUrl = pathToFileURL(path.join(eyeRoot, 'src', 'service.mjs')).href;
const { runEye, readArtifactById } = await import(serviceUrl);
const configUrl = pathToFileURL(path.join(eyeRoot, 'src', 'config.mjs')).href;
const providerUrl = pathToFileURL(path.join(eyeRoot, 'src', 'provider.mjs')).href;
const { loadConfig, resolveProvider, keyConfigured } = await import(configUrl);
const { callChat } = await import(providerUrl);

function assertSightEnabled() {
  const configPath = path.join(eyeRoot, 'config.json');
  const config = JSON.parse(readFileSync(configPath, 'utf8'));
  if (config.vision?.enabled === false) {
    throw new Error('Sight is disabled in Sens settings.');
  }
}

function assertVideoEnabled() {
  const configPath = path.join(eyeRoot, 'config.json');
  const config = JSON.parse(readFileSync(configPath, 'utf8'));
  if (config.vision?.videoEnabled !== true) {
    throw new Error('Video analysis is disabled in Sens settings.');
  }
}

function videoMimeFor(videoPath) {
  const ext = String(videoPath).split('.').pop().toLowerCase();
  return {
    mp4: 'video/mp4',
    webm: 'video/webm',
    mov: 'video/quicktime',
    mkv: 'video/x-matroska',
    avi: 'video/x-msvideo',
    m4v: 'video/x-m4v',
  }[ext] ?? 'video/mp4';
}

async function videoWatch(message) {
  assertVideoEnabled();
  const input = message.input ?? {};
  const videoPath = String(input.videoPath ?? '').trim();
  if (!videoPath) {
    throw new Error('videoPath is required');
  }
  const config = loadConfig({ cwd: eyeRoot, moduleDir: eyeRoot });
  const provider = resolveProvider(config, { model: input.model ?? null });
  if (provider.capabilities?.auth !== false && !keyConfigured(provider.apiKey)) {
    throw new Error(
      `Provider "${provider.name}" has no configured API key. Set it in config.json or EYE_API_KEY.`,
    );
  }
  const prompt = String(input.prompt ?? '').trim()
    || 'Describe this video in detail: what happens, what is visible, and any on-screen text.';
  const dataUri = `data:${videoMimeFor(videoPath)};base64,${readFileSync(videoPath).toString('base64')}`;
  const response = await callChat(
    provider,
    [{ role: 'user', content: [
      { type: 'video_url', video_url: { url: dataUri } },
      { type: 'text', text: prompt },
    ] }],
    { jsonMode: false, maxTokens: 4000 },
  );
  const text = String(response?.choices?.[0]?.message?.content ?? '').trim();
  return { text, model: provider.model, provider: provider.name, video: videoPath };
}

function buildOptions(message) {
  const input = message.input ?? {};
  const common = {
    imagePath: input.imagePath,
    artifactId: input.artifactId,
    detail: input.detail,
    prompt: input.prompt,
    noStore: message.noStore ?? input.noStore,
    maxCalls: message.maxCalls ?? input.maxCalls,
    outputDir: artifactOutputRoot,
  };
  switch (message.operation) {
    case 'see':
      return { ...common, mode: input.mode ?? 'describe' };
    case 'read':
      return { ...common, mode: 'ocr' };
    case 'locate':
      return { ...common, mode: 'locate', prompt: input.target };
    case 'inspect':
      return { ...common, mode: 'inspect', detail: input.detail ?? 'deep', region: input.region };
    case 'compare':
      return {
        ...common,
        mode: 'review',
        imagePath: input.candidatePath,
        referencePath: input.referencePath,
        heatmapPath: input.heatmapPath,
      };
    default:
      throw new Error(`Unsupported Sight operation: ${message.operation}`);
  }
}

function compact(result) {
  return {
    artifactId: result.storage.artifactId,
    mode: result.payload.mode,
    fromCache: result.fromCache,
    usage: result.payload.usage,
    data: result.payload.data,
  };
}

async function handle(message) {
  assertSightEnabled();
  if (message.operation === 'artifact_get') {
    const artifact = readArtifactById(message.input?.artifactId);
    return {
      artifactId: message.input.artifactId,
      mode: artifact.mode,
      usage: null,
      data: artifact.payload?.data ?? artifact.data ?? artifact,
    };
  }
  if (message.operation === 'watch') {
    return videoWatch(message);
  }
  return compact(await runEye(buildOptions(message)));
}

const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of input) {
  if (!line.trim()) continue;
  let requestId = null;
  try {
    const message = JSON.parse(line);
    requestId = message.requestId ?? null;
    const result = await handle(message);
    process.stdout.write(`${JSON.stringify({ ok: true, requestId, result })}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify({
      ok: false,
      requestId,
      error: { message: error instanceof Error ? error.message : String(error) },
    })}\n`);
  }
}
