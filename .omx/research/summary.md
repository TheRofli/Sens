# Sens 1.3.5 Hearing recommendation

## Decision

The current Hearing implementation needs both an ownership fix and an ASR
rewrite. Sens launches one Speech process for Ctrl+Win dictation while the Rust
broker launches another for MCP transcription. Both can hold mutable settings
and model state, and both depend on an external `D:\Speech` checkout.

Sens 1.3.5 will use one persistent Hearing worker owned by the broker. The
desktop sends internal start/settings/status requests to that worker; MCP uses
the same engine through side-effect-free operations.

## Local model stack

1. **Qwen3-ASR 0.6B INT8**: balanced automatic local preset for Russian,
   English, code-switching, and the model's other supported languages.
2. **GigaAM v3 compact ONNX**: fastest Russian-specialist preset.
3. **Whisper through faster-whisper INT8**: retained broad-language fallback.
4. **Remote provider**: optional and user-selected; never required for local
   dictation.

Parakeet is removed. The new local stack avoids NeMo, Torch, and a development
Transformers checkout in the shipped runtime.

## Audio pipeline

- Capture mono PCM at 16 kHz.
- Run Silero VAD locally to reject empty recordings and form utterances.
- Keep model input chunks below engine limits with speech-aware boundaries and
  bounded overlap only where a hard cut is unavoidable.
- Serialize access to one resident engine and expose measured duration,
  processing time, real-time factor, selected model, detected language where
  available, and timestamped segments.
- Publish clipboard/paste/history only in the explicit hotkey dictation flow.

## Packaging

- Bundle Speech source under `sidecars/speech`.
- Extend the verified embedded Python runtime with Tk and pinned CPU ASR/audio
  dependencies.
- Store mutable data and downloaded model packs under Sens local app data.
- Install models with staged downloads, integrity checks, and atomic promotion.
- Migrate compatible legacy settings once without printing secrets.

## Quality and performance gate

Use a checked-in manifest for representative Russian, English, code-switch,
noise/silence, short, and >30-second fixtures. Report normalized WER/CER where
reference text exists, real-time factor, peak RSS, cold load, and warm latency.
Do not declare a default winner from model reputation alone.

## Release gate

The release is complete only after Python and Rust tests, protocol/side-effect
tests, real Ctrl+Win interaction, model install/load/transcribe smoke tests,
Windows package installation, updater metadata validation, commit/push, and
published version 1.3.5 all succeed.
