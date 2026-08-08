# Sens 1.3.5 Hearing execution plan

## Slice 1 - ownership and paths

- Import the tested Speech source/tests into `sidecars/speech`.
- Split immutable code, mutable data, and model roots.
- Add autonomous legacy settings migration.
- Make the broker the only Hearing-worker owner.
- Route desktop start/status/settings through broker internal operations.

Verify:

- Imported brownfield tests pass from `D:\Sens`.
- Rust tests prove path discovery and broker control envelopes.
- One worker PID serves dictation status and file transcription.
- Repository search finds no runtime default to `D:\Speech`.

## Slice 2 - model registry and installers

- Remove Parakeet source, presets, requirements, UI options, docs, and tests.
- Add pinned sherpa-onnx engines/installers for Qwen3-ASR INT8 and GigaAM.
- Keep faster-whisper INT8 as broad-language fallback.
- Add staged, checksummed model-pack downloads and honest status reporting.

Verify:

- Registry tests expose exactly Qwen, GigaAM, Whisper, and remote.
- Interrupted/partial downloads never report installed.
- Each installed local engine loads and transcribes a smoke fixture on CPU.

## Slice 3 - VAD, long form, and resource policy

- Replace RMS-only trimming with bundled Silero VAD.
- Add speech-aware segmentation for long dictation/files.
- Return structured timing/provenance fields.
- Add adaptive CPU thread policy with override and interactive headroom.

Verify:

- Silence, speech, seam, >30-second, and cancellation fixtures pass.
- Unit tests cover 1/2/4/8/16/32+ logical CPU configurations.
- Benchmarks record cold load, warm latency, RTF, and peak RSS.

## Slice 4 - Windows runtime and UI

- Package Tk, audio/hotkey dependencies, sherpa-onnx, faster-whisper, and VAD
  in the embedded runtime without Torch/NeMo.
- Add model download/progress/status controls in Hearing settings.
- Preserve Ctrl+Win hold/release overlay and output behavior.

Verify:

- Isolated packaged-runtime import smoke passes.
- Desktop build/test and browser interaction pass.
- Real Ctrl+Win dictation works in a clean installed build.

## Slice 5 - cleanup and release

- Run full Rust, Python, UI, packaging, and security/secret gates.
- Migrate live legacy settings/data and verify rollback copy if needed.
- Remove every tracked dependency/reference to `D:\Speech`.
- Delete old `D:\Speech` only after the migration/deletion gate.
- Bump all versions and release notes to 1.3.5.
- Commit and push deliberate slices, tag `v1.3.5`, verify GitHub release assets
  and updater `latest.json`.

## Verification map

| Contract | Evidence |
|---|---|
| Broker owns one Hearing worker | Rust lifecycle tests, PID/runtime smoke |
| Model cannot open microphone | MCP operation tests and public tool schema |
| Ctrl+Win behavior preserved | hotkey unit tests and installed interaction |
| MCP has no output side effects | clipboard/paste/history spies |
| CPU-only local inference | provider configuration and runtime smoke |
| Adaptive resources | mocked topology tests plus benchmark report |
| Long Russian dictation | >30 s fixture and seam assertions |
| Multilingual fallback | RU/EN/code-switch fixture matrix |
| Safe model install | checksum/partial/atomic installer tests |
| No external Speech dependency | repository/package search and clean-machine install |
| Release 1.3.5 usable | signed installer, GitHub assets, updater fetch |
