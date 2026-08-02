# GigaAM 25s limit — research

Date: 2026-08-02. Question: how to handle GigaAM's hard 25-second transcription
limit in push-to-talk dictation, for a distributable (offline-first) product.

## Root cause (repo truth)

- `D:\Speech\models\gigaam\gigaam\modeling_gigaam.py:31` — `LONGFORM_THRESHOLD = 25 * SAMPLE_RATE`
- `:1251-1252` — `transcribe()` raises `ValueError("Too long wav file...")` for audio > 25 s
- `speech_app/engines/gigaam.py` previously forwarded the exception → "Transcription failed" toast

## Engine limits (repo + external truth)

| Engine | Long-audio behavior | Evidence |
|---|---|---|
| GigaAM v3 | Hard 25 s limit in `transcribe()`; `transcribe_longform()` exists but needs PyAnnote + HF_TOKEN | repo code |
| Parakeet TDT 0.6B | Up to ~20-24 min per pass; chunked inference script for longer; local-attention for >1 h | NVIDIA NGC / HF Space |
| Whisper (faster-whisper) | No practical limit — built-in Silero VAD + chunking, `vad_filter=True` | repo code, known behavior |

## External facts (web, 2026-08-02)

- **pyannote/segmentation-3.0** — gated on HF (needs token + accepting conditions), but **MIT license** → commercial redistribution allowed if weights bundled. Gating is contact-collection, not license restriction. Sources: HF model page, Wan2GP issue #494.
- **Silero VAD** — MIT, offline ONNX ~2.2 MB, no keys/telemetry, v6.x. Used by faster-whisper internally. Source: github.com/snakers4/silero-vad.
- **NeMo Parakeet** — buffered inference (buffer ~30 s, chunk ~10 s) and streaming variants exist; OOM on long audio without chunking. Sources: NVIDIA NGC, NeMo examples.

## Venv state (repo truth)

- `onnxruntime` 1.27.0 — installed (Silero ONNX would need no new pip deps)
- `pyannote_audio` 4.0.7 — installed
- `ctranslate2` — installed (faster-whisper)

## Options

| # | Option | Offline | New deps | Boundary quality | Effort | Notes |
|---|---|---|---|---|---|---|
| 1 | RMS-VAD chunking (`split_audio` in vad.py) | ✅ | none | medium | **done** | hard cut at limit if no pause |
| 2 | Silero VAD chunking (upgrade of 1) | ✅ | bundle silero_vad.onnx (~2.2 MB) | high | small | MIT; better pause detection → fewer word splits |
| 3 | PyAnnote longform | ✅ if weights bundled | pyannote.audio (installed) + ~5 MB gated weights | high | medium | HF_TOKEN needed once at build time; MIT OK to ship |
| 4 | Overlap + tail-dedupe on chunks | ✅ | none | high (no seam loss) | small | ~1 s overlap; drop duplicated suffix/prefix by text compare |
| 5 | Switch default model to Parakeet | ✅ | none | — | trivial | no limit at all; accuracy differs from GigaAM |
| 6 | Product guard: auto-stop recording at ~24 s + notice | ✅ | none | — | small | UX safety net, independent of chunking |
| 7 | Incremental/streaming transcription while holding hotkey | ✅ | none | high | large | architecture change; not for 1.0 |

## Recommendation

**Done:** option 1 (RMS-VAD chunking) + **option 4 (overlap 1.5 s + text dedupe, ≥2-word match)** —
seams now keep the cut word whole (it is heard in full by the next chunk) and the
duplicated region is dropped from the joined text. Quality at seams ≈ mid-chunk.
Limits: only memory (30 min float32 ≈ 115 MB) and CPU time (GigaAM on CPU is roughly
realtime or slower; a 30-min dictation takes tens of minutes to transcribe).

Not done, optional upgrades: option 2 (Silero VAD boundaries, MIT, offline), option 3
(PyAnnote longform, MIT but needs build-time HF_TOKEN), option 5 (Parakeet default —
no limit at all), option 6 (auto-stop UX guard).

## Open questions

- None blocking. Empirically, whether RMS cuts are good enough for real dictation
  needs the user's hold-and-speak test with pauses vs continuous speech.
