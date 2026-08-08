# Sens 1.1 recommendation

## Decision

The missing Hearing feature is an orchestration bug, not an ASR rewrite. Speech already contains the required global hotkey, microphone recorder, overlay, model loading, clipboard/paste behavior, and a Sens-managed mode. Sens 1.0 never launched it.

## Planned slices

1. Lock Speech/Sens settings and runtime contracts with regression tests.
2. Start managed Speech invisibly and expose clear runtime/hotkey state.
3. Hide every broker and worker console on Windows.
4. Make the frameless title area draggable and open the custom tray panel on right-click.
5. Add signed in-app updater UI and release tooling; activate the network channel after the release host is selected.
6. Run Rust, Speech, browser interaction, visual, and installer verification.

## External decision still required

Choose the permanent HTTPS release location. A public GitHub Releases repository is the lowest-maintenance option and lets the build publish `latest.json`, signed NSIS artifacts, and release notes. No Git remote is configured today, so inventing an endpoint would make the updater appear complete while remaining unusable.

## Sens 1.3 first-run vision recommendation

Keep the Windows installer lightweight and require one visible confirmation before downloading Qwen3-VL 2B. Bundling the GGUF files would add roughly 1.45 GiB to every installer, while a silent first-launch download would consume material bandwidth without informed consent. The chosen path is a one-time native dialog with exact CPU, RAM, and download facts; **Install** starts the existing verified downloader, **Later** permanently dismisses the prompt while preserving the manual settings action, and active `.part` sizes provide honest progress.
