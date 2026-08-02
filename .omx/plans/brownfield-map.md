# Sens 1.1 brownfield map

## Scope

Repair the installed Windows experience reported after Sens 1.0: interactive Speech dictation, hidden background processes, draggable frameless window, custom right-click tray panel, and an in-app update path.

## Boundaries

### Sens desktop (`apps/desktop-ui`)

- Owns the only product window and tray icon.
- Reads and saves capability settings.
- Must start and supervise Speech in managed mode.
- Owns update UI and update installation feedback.

### Speech (`D:\Speech\speech_app`)

- Remains the source of truth for the global hotkey, microphone recording, overlay, local ASR, clipboard, paste, and history behavior.
- Already exposes `speech run --managed`, which suppresses the legacy Speech tray/window while preserving dictation.
- Already exposes a localhost control API and settings model.

### Sens broker and workers (`crates/sens-broker`, `sidecars`)

- Remain the only owner of model-facing capability workers.
- Hearing file transcription stays separate from user-owned interactive dictation.
- All spawned broker/Node/Python processes must use hidden Windows process creation.

### Release/update channel

- Use the official Tauri v2 updater with signed artifacts.
- The installed 1.0.0 build has no updater, so 1.1.0 is the one unavoidable bootstrap installer. Later versions update in-app.
- A real update check requires a stable HTTPS `latest.json` endpoint and a release-signing key. The repository currently has no Git remote or release host.

## Must-preserve behavior

- A model never activates the microphone. Dictation begins only while the user holds the configured hotkey.
- API keys and updater private keys never enter source control, command-line arguments, logs, or diagnostics.
- Speech standalone mode and its existing tests remain valid.
- Sens keeps one tray icon and one main window.
- MCP stdout remains protocol-only.
- Existing Eye and Speech configuration fields not owned by Sens are preserved.

## Lifecycle

1. Sens starts.
2. Sens launches `pythonw.exe -m speech_app run --managed` without a console window.
3. Speech registers the configured global hotkey and exposes runtime state locally.
4. Holding the hotkey records; releasing it transcribes and applies the configured clipboard/paste behavior.
5. Sens settings show the hotkey and runtime status and synchronize changes to the managed Speech process.
6. Exiting Sens stops the managed Speech process; hiding the main window does not.

## Risks

- A separately running legacy Speech instance owns the single-instance lock. Sens must report that state rather than launch a duplicate.
- Saving the settings file alone does not refresh a running Speech instance; settings must also be sent through its local API or the managed process restarted.
- Tauri updates cannot be made secure without a durable private signing key and a public key embedded in the bootstrap build.
- Windows installers terminate the app while applying an update; progress and restart copy must state this clearly.

## Recommended path

Reuse Speech managed mode and its API; do not reimplement recording or hotkeys in Rust. Add a small Rust supervisor/control layer in the desktop app, expose runtime status to React, and keep broker Hearing for model-invoked audio-file transcription. Use the official signed Tauri updater once a release endpoint is selected.
