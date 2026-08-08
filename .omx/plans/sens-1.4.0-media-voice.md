# Sens 1.4.0 — Media understanding and Voice output

## Outcome

Sens can locally analyze an explicitly supplied video, song, audio file,
supported document, or public media URL on CPU, returning evidence tied to the
source timeline. A host text model can then optionally synthesize its final
answer to a local audio file. Existing Sight, Hearing, Ctrl+Win, and MCP tools
remain compatible.

## Non-goals

- Live/model-controlled microphone or screen capture.
- DRM bypass, authenticated/private-site scraping, or browser-cookie import.
- Training or shipping a new end-to-end video language model.
- Claiming complete observation of frames that were not sampled.
- Automatic playback or clipboard/paste/history writes from agent tools.

## Slice 0 — Freeze fixtures and contracts

- Record current `sens_hear`, `sens_watch`, and `sens_fetch` schemas/results.
- Add short/long, silent, speech, song, screen-recording, scene-cut, subtitle,
  vertical-video, corrupt, and oversized fixtures with redistributable sources.
- Add Russian, English, and code-switched song references for Qwen/Whisper
  evaluation; record uncertainty where lyrics are genuinely ambiguous.
- Define `MediaDocument v1` and the shared observed/measured/inferred claim
  schema before changing workers.

Gate: existing public tests pass and new contract fixtures are checked in.

## Slice 1 — Safe media preparation

- Introduce broker-owned `MediaExecutor` and `sidecars/media` preparation
  worker; share existing Sight/Hearing executor instances.
- Fix yt-dlp result identity using an explicit output template/printed final
  pathname rather than directory scanning.
- Probe captions/manual/automatic subtitles first; cap default video at 720p.
- Add allowlisted schemes, DNS/redirect private-range rejection, duration/size
  limits, cancellation, staged download, checksum/content ID, atomic promotion,
  cache quota/TTL/delete, and complete `noStore` cleanup.
- Preserve `sens_fetch` through a compatibility adapter.

Gate: distinct audio/video paths, captions, redirects, quotas, cancellation,
cleanup, and installed-runtime ffmpeg/PyAV behavior pass integration tests.

## Slice 2 — Local video timeline

- Demux audio and call the shared Hearing executor for timestamped ASR.
- Add the small sherpa-onnx INT8 audio tagger and apply it to bounded windows so
  music, singing, instruments, laughter, applause, alarms, and other non-speech
  events enter the same source timeline.
- Scan low-resolution frames for scene, motion, and OCR change; add uniform
  guards; implement quick/standard/deep budgets with deterministic selection.
- Run deterministic Sight measurements on selected frames and Qwen VLM only on
  salient frames; preserve source timestamps.
- Fuse subtitles/ASR/visual evidence into `MediaDocument v1`, including
  coverage, warnings, and focused-inspection suggestions.
- Add `sens_analyze` and `sens_media_inspect`; keep `sens_watch` compatible.

Gate: every timeline assertion has source time/evidence class; a brief event
fixture proves focused re-inspection; peak RSS, cold/warm time, and real-time
factor are recorded for all detail modes on the reference CPU.

## Slice 3 — Songs and audio mode

- Auto-detect audio-only/song inputs without pretending genre detection is
  certain; allow an explicit `contentHint`.
- Default to Qwen3-ASR; retain Whisper and GigaAM selection.
- Compare Qwen 0.6B INT8 and Whisper Small INT8 on the checked-in song corpus.
- Add optional deep dual-ASR comparison that exposes disagreements and cost.
- Keep VAD disabled for files/songs and avoid hallucinating text over
  instrumental/silent spans.

Gate: release notes publish measured corpus results and chosen default; no
engine is declared best from vendor benchmarks alone.

## Slice 4 — Supported documents

- Route TXT/Markdown/HTML/JSON through bounded text extraction.
- Add bounded PDF extraction and page rendering; scanned/selected pages go to
  Sight with page-number evidence.
- Add DOCX/PPTX/XLSX readers with explicit limits on pages/slides/sheets,
  embedded files, formulas/macros, and archive expansion.
- Treat extracted content as untrusted data and report skipped/unsupported
  portions.

Gate: native-text and scanned fixtures pass; bombs, malformed archives,
unsupported/encrypted files, and macro-bearing documents fail safely.

## Slice 5 — Voice

- Add broker-owned lazy `VoiceExecutor` using sherpa-onnx OfflineTts.
- Ship/download `vits-piper-ru_RU-denis-medium` as the Russian default.
- Offer Kokoro-82M as an optional international quality pack; do not label it
  Russian-capable without a verified Russian voice.
- Benchmark Qwen3-TTS 0.6B as an experimental higher-quality multilingual pack;
  do not add its Torch/CUDA-oriented runtime to the default installation unless
  CPU latency and peak RAM pass the release budget.
- Add `sens_speak`; write WAV by default and permit `play: true` only when the
  user's request explicitly asks for playback.
- Expose voice/model install status, sample generation, speed, output format,
  output directory, and deletion in Settings.

Gate: Russian/English smoke samples, cold/warm synthesis time, peak RSS,
artifact cleanup, cancellation, and no-autoplay tests pass in dev and installed
Windows builds.

## Slice 6 — Product and release

- Add a third Settings card: `Медиа — видео, песни, документы и ссылки`.
- Configure local cache quota/TTL, max duration/size, default detail, ASR
  preference, visual budget, and public-link permission.
- Present Voice as a separate output section/capability with honest language
  support and download sizes.
- Add first-run and update migrations without downloading optional packs unless
  the selected experience needs them.
- Document examples for text-only hosts: analyze a file/URL, ask a timestamped
  follow-up, and speak the host's final response.

Gate: Python tests; `cargo fmt --check`; `cargo clippy --workspace
--all-targets -- -D warnings`; `cargo test --workspace`; MCP stdout/secret and
side-effect tests; UI interaction; installed Windows package; updater; commit,
push, and published 1.4.0 all pass.

## Acceptance examples

1. A local screen recording returns transcript, sound/UI/OCR changes, visual
   events, timestamped evidence, sampling coverage, and a focused follow-up
   path.
2. A public YouTube URL with captions uses those captions, analyzes selected
   frames locally, and leaves no files when `noStore` is true.
3. A Russian/English song returns timestamped candidate lyrics with uncertain
   spans marked; deep mode shows Qwen/Whisper disagreement rather than guessing.
4. A scanned PDF reports per-page OCR/visual evidence and skipped pages.
5. After the host model writes a response, `sens_speak` creates a Russian WAV;
   nothing is played unless the user explicitly requested playback.
