#!/usr/bin/env node

import { spawn } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";

function parseArgs(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 1) {
    const name = argv[index];
    if (!name.startsWith("--")) {
      throw new Error(`Unexpected positional argument: ${name}`);
    }
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for ${name}`);
    }
    values.set(name.slice(2), value);
    index += 1;
  }
  return values;
}

function required(args, name) {
  const value = args.get(name)?.trim();
  if (!value) {
    throw new Error(`--${name} is required`);
  }
  return value;
}

function defaultCliPath() {
  const localAppData = process.env.LOCALAPPDATA;
  if (!localAppData) {
    throw new Error("LOCALAPPDATA is unavailable; pass --zcode-cli");
  }
  return join(localAppData, "Programs", "ZCode", "resources", "glm", "zcode.cjs");
}

function wait(milliseconds) {
  return new Promise((resolveWait) => setTimeout(resolveWait, milliseconds));
}

class ProtocolClient {
  constructor(child) {
    this.child = child;
    this.buffer = "";
    this.nextId = 1;
    this.pending = new Map();
    this.notifications = [];
    this.stderr = "";
    this.closedError = null;

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => this.#consume(chunk));
    child.stderr.on("data", (chunk) => {
      this.stderr = `${this.stderr}${chunk}`.slice(-16_384);
    });
    child.once("error", (error) => this.#close(error));
    child.once("exit", (code, signal) => {
      this.#close(new Error(`Z-Code app-server exited early (code=${code}, signal=${signal})`));
    });
  }

  request(method, params = {}) {
    if (this.closedError) {
      return Promise.reject(this.closedError);
    }
    const id = `sens-benchmark-${this.nextId}`;
    this.nextId += 1;
    const promise = new Promise((resolveRequest, rejectRequest) => {
      this.pending.set(id, { resolve: resolveRequest, reject: rejectRequest });
    });
    this.#send({ id, method, params });
    return promise;
  }

  #send(message) {
    this.child.stdin.write(`${JSON.stringify(message)}\n`);
  }

  #consume(chunk) {
    this.buffer += chunk;
    for (;;) {
      const newline = this.buffer.indexOf("\n");
      if (newline < 0) {
        return;
      }
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (!line) {
        continue;
      }
      let message;
      try {
        message = JSON.parse(line);
      } catch (error) {
        this.#close(new Error(`Invalid JSON from Z-Code: ${error.message}`));
        return;
      }
      this.#handle(message);
    }
  }

  #handle(message) {
    if (Object.hasOwn(message, "id") && (Object.hasOwn(message, "result") || Object.hasOwn(message, "error"))) {
      const pending = this.pending.get(String(message.id));
      if (!pending) {
        return;
      }
      this.pending.delete(String(message.id));
      if (message.error) {
        const error = new Error(`${message.error.message} (${message.error.code})`);
        error.protocolError = message.error;
        pending.reject(error);
      } else {
        pending.resolve(message.result);
      }
      return;
    }

    if (Object.hasOwn(message, "id") && message.method) {
      // Headless benchmark runs use yolo mode. This is a defensive response for
      // transports that still surface a permission request.
      if (message.method === "interaction/requestPermission") {
        this.#send({ id: message.id, result: { decision: "allow", reason: "Autonomous benchmark run" } });
        return;
      }
      this.#send({
        id: message.id,
        error: { code: -32601, message: `Unsupported benchmark interaction: ${message.method}` },
      });
      return;
    }

    if (message.method) {
      this.notifications.push({ method: message.method, params: message.params });
      if (this.notifications.length > 2_000) {
        this.notifications.shift();
      }
    }
  }

  #close(error) {
    if (this.closedError) {
      return;
    }
    this.closedError = error;
    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
  }
}

function compactMessages(messages) {
  return messages.slice(-12).map((message) => ({
    id: message.info?.id,
    role: message.info?.role,
    text: Array.isArray(message.parts)
      ? message.parts
          .filter((part) => part.type === "text")
          .map((part) => part.text)
          .join("\n")
          .slice(0, 4_000)
      : undefined,
  }));
}

async function terminate(child) {
  if (child.exitCode !== null || child.signalCode !== null) {
    return;
  }
  child.stdin.end();
  await Promise.race([
    new Promise((resolveExit) => child.once("exit", resolveExit)),
    wait(3_000),
  ]);
  if (child.exitCode === null && child.signalCode === null) {
    child.kill();
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const cwd = resolve(required(args, "cwd"));
  const promptFile = resolve(required(args, "prompt-file"));
  const resultFile = resolve(required(args, "result-file"));
  const thoughtLevel = args.get("thought-level") ?? "off";
  const requestedSessionId = args.get("session-id")?.trim();
  const timeoutMs = Number(args.get("timeout-ms") ?? 3_600_000);
  const pollMs = Number(args.get("poll-ms") ?? 1_000);
  const zcodeCli = resolve(args.get("zcode-cli") ?? defaultCliPath());

  if (!existsSync(cwd)) {
    throw new Error(`Workspace does not exist: ${cwd}`);
  }
  if (!existsSync(promptFile)) {
    throw new Error(`Prompt file does not exist: ${promptFile}`);
  }
  if (!existsSync(zcodeCli)) {
    throw new Error(`Z-Code CLI does not exist: ${zcodeCli}`);
  }
  if (!process.env.ZCODE_API_KEY) {
    throw new Error("ZCODE_API_KEY must be provided through the environment");
  }
  if (!Number.isFinite(timeoutMs) || timeoutMs < 1_000) {
    throw new Error("--timeout-ms must be at least 1000");
  }

  const prompt = readFileSync(promptFile, "utf8");
  const workspace = { workspacePath: cwd, workspaceKey: cwd };
  const child = spawn(process.execPath, [zcodeCli, "app-server"], {
    cwd,
    env: process.env,
    stdio: ["pipe", "pipe", "pipe"],
    windowsHide: true,
  });
  const protocol = new ProtocolClient(child);
  const startedAt = Date.now();
  let sessionId;
  let finalSnapshot;
  let usage;

  try {
    const workspaceState = await protocol.request("workspace/readState", {
      workspace,
      preferWorkspaceDefaults: true,
    });
    const availableThoughtLevels = workspaceState.settings?.thoughtLevel?.available?.map((item) => item.value) ?? [];
    if (!availableThoughtLevels.includes(thoughtLevel)) {
      throw new Error(
        `Thought level ${thoughtLevel} is unavailable; available=${availableThoughtLevels.join(",") || "none"}`,
      );
    }

    if (requestedSessionId) {
      sessionId = requestedSessionId;
      const modelRef = workspaceState.settings?.model?.current;
      const catalogProvider = workspaceState.modelCatalog?.providers?.find(
        (provider) => provider.providerId === modelRef?.providerId,
      );
      if (!modelRef || !catalogProvider) {
        throw new Error("Cannot resume: the workspace model is missing from the current catalog");
      }
      const { apiKeyRef, ...providerWithoutRef } = catalogProvider;
      const workspaceConfigPath = join(cwd, "zcode.json");
      const workspaceConfig = existsSync(workspaceConfigPath)
        ? JSON.parse(readFileSync(workspaceConfigPath, "utf8"))
        : {};
      const configuredProvider = workspaceConfig.provider?.[modelRef.providerId] ?? {};
      const configuredModels = Object.entries(configuredProvider.models ?? {}).map(
        ([modelId, model]) => ({
          modelId,
          contextWindow: model.limit?.context,
          maxOutputTokens: model.limit?.output,
          reasoning: model.reasoning,
          supportsImages: model.modalities?.input?.includes("image") ?? false,
          supportsTools: true,
        }),
      );
      const apiKey = apiKeyRef?.source === "session-secret"
        ? { source: "env", name: "ZCODE_API_KEY" }
        : apiKeyRef ?? { source: "env", name: "ZCODE_API_KEY" };
      const runtimeModel = {
        revision: workspaceState.modelCatalog?.providerRevision ?? String(workspaceState.modelCatalog?.revision ?? 0),
        generatedAt: Date.now(),
        model: modelRef,
        provider: {
          ...providerWithoutRef,
          kind: configuredProvider.kind ?? providerWithoutRef.kind,
          apiFormat: configuredProvider.kind === "anthropic"
            ? "anthropic-messages"
            : providerWithoutRef.apiFormat,
          source: "workspace",
          baseURL: configuredProvider.options?.baseURL ?? providerWithoutRef.baseURL,
          apiKeyRequired: configuredProvider.options?.apiKeyRequired ?? providerWithoutRef.apiKeyRequired,
          models: providerWithoutRef.models?.length ? providerWithoutRef.models : configuredModels,
          ...(apiKey ? { apiKey } : {}),
        },
        thoughtLevel,
      };
      finalSnapshot = await protocol.request("session/resume", {
        sessionId,
        workspace,
        runtimeModel,
        thoughtLevel,
      });
      if (!finalSnapshot?.session?.sessionId && !finalSnapshot?.projection) {
        throw new Error(`Z-Code session/resume could not resume ${sessionId}`);
      }
    } else {
      const created = await protocol.request("session/create", {
        workspace,
        mode: "yolo",
        thoughtLevel,
        persistence: "immediate",
        titleGenerationEnabled: false,
      });
      finalSnapshot = created;
      sessionId = created.session?.sessionId;
      if (!sessionId) {
        throw new Error("Z-Code session/create returned no session id");
      }
    }
    const currentThought = finalSnapshot.settings?.thoughtLevel?.current;
    if (currentThought !== thoughtLevel) {
      throw new Error(`Z-Code created session with thought=${currentThought ?? "missing"}, expected=${thoughtLevel}`);
    }

    await protocol.request("session/send", {
      sessionId,
      inputId: `sens-${Date.now()}`,
      content: prompt,
    });

    let sawRunning = false;
    for (;;) {
      if (Date.now() - startedAt > timeoutMs) {
        await protocol.request("session/stop", { sessionId }).catch(() => undefined);
        throw new Error(`Z-Code benchmark timed out after ${timeoutMs} ms`);
      }
      await wait(pollMs);
      const read = await protocol.request("session/read", { sessionId, messageLimit: 20 });
      finalSnapshot = read;
      const status = finalSnapshot?.projection?.status;
      const activeTurn = finalSnapshot?.runtime?.activeTurnId;
      sawRunning ||= status === "running" || Boolean(activeTurn);
      if (sawRunning && !activeTurn && ["idle", "completed", "error", "paused"].includes(status)) {
        break;
      }
    }

    usage = await protocol.request("session/usage", { sessionId }).catch(() => undefined);
  } finally {
    await terminate(child);
  }

  const result = {
    benchmark: basename(cwd),
    sessionId,
    requestedThoughtLevel: thoughtLevel,
    actualThoughtLevel: finalSnapshot?.settings?.thoughtLevel?.current,
    status: finalSnapshot?.projection?.status,
    elapsedMs: Date.now() - startedAt,
    turnCount: finalSnapshot?.projection?.turnCount,
    totalTokenCount: finalSnapshot?.projection?.totalTokenCount,
    contextUsed: finalSnapshot?.projection?.contextUsed,
    usage,
    lastError: finalSnapshot?.projection?.lastError,
    messages: compactMessages(finalSnapshot?.messages ?? []),
    notificationCount: protocol.notifications.length,
    stderrTail: protocol.stderr ? protocol.stderr.slice(-4_000) : undefined,
  };
  writeFileSync(resultFile, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  if (result.status === "error" || result.lastError) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error.message}\n`);
  process.exitCode = 1;
});
