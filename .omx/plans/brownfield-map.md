# Sens 1.3.5 brownfield map

## Scope

Fold the working `D:\Speech` dictation product into Sens, preserve the accepted
Ctrl+Win push-to-talk flow, and replace the heavyweight/duplicated ASR runtime
with one CPU-only Hearing worker owned by the Rust broker.

## Repository truth

- The Sens desktop currently starts `D:\Speech\.venv\Scripts\pythonw.exe`
  directly through `speech_runtime.rs`.
- The broker separately starts `sidecars/hearing-worker.py`, which imports the
  same Speech source and can load a second ASR model.
- Settings also default to `D:\Speech\data\settings.json`.
- `D:\Speech` local `main` and its configured `origin/main` have no merge base;
  the tested local checkout is migration input, not a Git branch to merge.
- The bundled Sight Python is an embeddable Python 3.11 runtime without Tk.

## Target boundaries

### Rust broker (`crates/sens-broker`)

- Is the only owner and supervisor of the persistent Hearing worker.
- Serves both side-effect-free file transcription and internal desktop
  dictation control over the existing broker envelope.
- Starts the worker lazily, keeps one process/model resident, and kills the
  complete process tree on timeout, shutdown, or update.
- Supplies code, data, model, artifact, and Python runtime paths explicitly.

### Hearing worker (`sidecars/speech`, `sidecars/hearing-worker.py`)

- Owns one shared engine manager used by Ctrl+Win dictation and MCP file
  transcription.
- Preserves hotkey, microphone, overlay, clipboard/paste, and history behavior
  only for the explicit user-facing dictation flow.
- Never writes clipboard, pastes, or stores history for model-originated
  `hear` requests unless the request explicitly opts into history.
- Uses local CPU inference only for local presets and emits protocol JSON on
  stdout; diagnostics go to stderr.

### Sens desktop (`apps/desktop-ui`)

- Starts/configures dictation by sending internal Hearing operations to the
  broker; it no longer owns a Speech child process.
- Reads/writes Hearing settings under the Sens local data directory.
- Shows model installation/runtime state and keeps one Sens tray/window.

### Mutable files

- Code is bundled under the Sens installation (`sidecars/speech`).
- Settings, history, lock/runtime files live under the Sens local data root.
- Downloaded ASR packs live outside the installation under the Sens model root
  and survive app updates.
- A one-time migration imports compatible settings from `D:\Speech\data` only
  when the new Sens settings do not exist. Secrets are never logged.

## Model decision

- Remove Parakeet and all NeMo/Transformers development dependencies.
- Add Qwen3-ASR 0.6B INT8 through sherpa-onnx as the balanced multilingual
  local preset (30 languages, including Russian).
- Keep GigaAM v3 through a compact sherpa-onnx export as the Russian specialist.
- Keep optimized faster-whisper as the broad-language fallback.
- Keep the existing remote provider as an optional user-selected mode.
- Use Silero VAD for utterance boundaries and long-form chunking.

## CPU policy

- Detect available logical and physical CPUs at runtime.
- Reserve interactive headroom and choose a bounded worker count instead of
  the existing fixed four threads.
- Allow an advanced override, but keep the automatic policy as the default.
- Validate the policy with mocked CPU counts and benchmark candidate thread
  counts on the current machine.

## Must-preserve behavior

- A model cannot activate the microphone. Only the visible Sens desktop flow
  arms Ctrl+Win dictation.
- Holding the configured shortcut records; releasing it transcribes and applies
  the configured clipboard/paste behavior once.
- MCP transcription remains side-effect-free by default.
- API keys and IPC secrets never appear in command-line arguments, stdout,
  logs, activity records, or diagnostics.
- `sens-mcp` stdout remains protocol-only.
- Existing Eye behavior and unrelated dirty workspace files remain untouched.

## Migration/deletion gate

`D:\Speech` may be removed only after all of the following are true:

1. No tracked Sens source, config, test, or generated release file refers to it.
2. Compatible settings and user data have an autonomous migration path.
3. Dev runtime and an installed 1.3.5 build both pass Ctrl+Win dictation.
4. MCP `sens_hear` passes without clipboard/paste/history side effects.
5. Local model install/status/load/transcribe flows pass from Sens-owned paths.
6. The release artifact is built, signed, published, and update metadata is
   reachable.

## Principal risks

- Tk and native audio/hotkey dependencies are absent from the current embedded
  runtime and must be packaged and smoke-tested on Windows.
- Qwen3-ASR support requires a sufficiently new pinned sherpa-onnx build.
- Long recordings can monopolize a single resident engine; file requests and
  dictation need explicit serialization and honest busy state.
- Model archives are large; downloads need checksums, resumable staging, and no
  partially-installed success state.
- Existing GigaAM and Whisper caches use different layouts and cannot be
  assumed compatible with new compact packs.

## Recommended path

Migrate the tested Speech package into Sens first, convert the worker into the
single process for both internal dictation control and MCP transcription, then
replace engines behind the preserved public behavior. Package one Python
runtime with Tk and the CPU inference dependencies. Delete `D:\Speech` only
after the installed-build gate, immediately before the final 1.3.5 release.
