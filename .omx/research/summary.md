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

# Sens Media and Voice research (2026-08-08)

## Product decision

The next input capability should be named **Media**. A link is only one source
of media, and video is only one of the formats. Media accepts one explicit
source (local path, public URL, or supplied text), determines its kind, and
returns a source-backed `MediaDocument`. Compatibility tools such as
`sens_hear`, `sens_watch`, and `sens_fetch` remain available while the new
entry point is introduced.

Speech synthesis is a separate **Voice** output capability. Media cannot speak
the host model's final answer because that answer is written after Media has
returned. The correct chain is:

`sens_analyze -> MediaDocument -> host model answer -> sens_speak(answer)`.

This split also makes side effects honest: `sens_speak` writes an audio
artifact by default, and playback is allowed only when the user explicitly
asks for it.

## Current repository truth

- `sens_hear` already transcribes local audio and a video's audio track. It can
  extract uniformly sampled stills, but it only returns their paths; it does
  not analyze the frames or fuse them with transcript timestamps.
- `sens_watch` is still routed through the legacy/cloud Eye worker. There is no
  complete local video-understanding path.
- `sens_fetch` uses `yt-dlp` and can place audio/video in a local cache, but the
  downloader discovers its result by scanning the destination directory. A
  second download can therefore report the first audio file as `videoPath`.
- URL ingestion has no private-network/redirect guard, caption-first path,
  staged atomic promotion, post-download quota check, or complete `noStore`
  cleanup contract.
- Sight and Hearing already have the right CPU building blocks: PyAV/OpenCV,
  OCR, local Qwen VLM, Qwen3-ASR, GigaAM, and faster-whisper. A second large
  multimodal model is unnecessary and would violate the light CPU/RAM goal.
- The bundled `sherpa-onnx` exposes offline TTS, so Voice can reuse the current
  inference runtime instead of shipping Torch or the Piper executable.

## External evidence

- Qwen3-ASR officially supports 30 languages and describes speech, singing,
  and songs with background music. The published song table covers the 1.7B
  model, not Sens's 0.6B INT8 pack, so song quality must be benchmarked rather
  than inferred: https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- Whisper is multilingual and processes long input with a sliding 30-second
  window, making it a valuable broad-language fallback:
  https://github.com/openai/whisper
- Gemini's public video interface is a useful behavior reference: it combines
  audio and visual streams, supports timestamps, defaults to sparse frame
  sampling, and accepts public YouTube URLs:
  https://ai.google.dev/gemini-api/docs/video-understanding
- Qwen3-VL advertises long-video understanding and timestamp alignment, but the
  local llama.cpp multimodal path used by Sens does not provide an equivalent
  stable native video contract. Sens must build an explicit sampled-frame and
  timeline pipeline instead of claiming native video understanding:
  https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct and
  https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- `yt-dlp` exposes manual/automatic subtitle and format-selection controls;
  captions should be tried before ASR when available:
  https://github.com/yt-dlp/yt-dlp/blob/master/README.md
- Kokoro-82M is Apache-2.0, compact, and has 8 languages/54 voices, but its
  official voice list has no Russian. It is an optional international pack,
  not the Russian default: https://huggingface.co/hexgrad/Kokoro-82M and
  https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
- Qwen3-TTS has an official 0.6B family with Russian among ten supported
  languages, including custom voice and three-second voice cloning. Its
  official runtime is currently Torch and its examples target CUDA, so it is a
  promising quality experiment but not the default lightweight CPU pack until
  Sens measures cold load, peak RAM, and real-time factor on Windows:
  https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
- sherpa-onnx supports both Kokoro and VITS/Piper-format ONNX models. The
  `ru_RU-denis-medium` voice is a small Russian model that can run through
  sherpa-onnx without bundling the Piper engine:
  https://k2-fsa.github.io/sherpa/onnx/c-api/html/tts.html and
  https://huggingface.co/csukuangfj/vits-piper-ru_RU-denis-medium
- A transcript is not full audio understanding. sherpa-onnx also supports a
  26 MB INT8 AudioSet tagger for music, singing, instruments, laughter,
  applause, alarms, animals, and other sound classes. Run it on bounded
  timestamped windows so Media can report non-speech events without another
  inference framework:
  https://k2-fsa.github.io/sherpa/onnx/audio-tagging/pretrained_models.html

## Recommended local pipeline

1. Resolve and validate the explicit source. For URLs, reject loopback,
   private, link-local, and redirects to those ranges; public unauthenticated
   sources only in the first release.
2. Probe metadata and available captions. Preserve source timestamps and
   measured duration, dimensions, codecs, and content size.
3. Demux audio and transcribe it with the selected Hearing engine. Qwen3-ASR is
   the balanced default; Whisper remains the broad fallback; GigaAM stays the
   Russian speech specialist. Apply lightweight audio tagging on timestamped
   windows for music, singing, applause, alarms, and other non-speech events.
4. Run a low-resolution scan and select frames using scene, motion, and OCR
   change, with uniform guard frames so quiet scenes are not invisible.
5. Apply deterministic OCR/layout/color measurements to all selected frames
   and invoke Qwen VLM only for the most salient frames under a clear budget.
6. Join transcript segments, captions, frames, and detected events by source
   timestamp. Return claims as observed, measured, or inferred, with coverage
   and warnings.
7. Let the host language model synthesize the final answer and request focused
   timestamp/range inspection when the first pass is insufficient.

Suggested frame budgets are 12 for quick, 32 for standard, and 96 for deep.
They are starting limits, not accuracy claims, and must be tuned against the
video fixture corpus and measured CPU/RAM.

## Song policy

Use Qwen3-ASR first for songs because it is the installed multilingual default
and explicitly targets singing, but retain Whisper as a selectable fallback.
The release gate compares both engines on Russian, English, code-switched, and
music-heavy fixtures. Deep lyrics mode may run a second engine only when the
user accepts the extra latency, returning disagreements rather than silently
merging invented words. VAD remains disabled for file/song analysis.

## Document scope

The first Media contract can route plain text, Markdown, HTML, JSON, PDF,
DOCX, PPTX, and XLSX. Text-bearing documents should be parsed directly;
scanned PDF pages and important rendered pages are handed to Sight. The UI and
result must list supported types and never imply arbitrary-file understanding.

## Principal risks

- Downloaded subtitle, OCR, transcript, and document text are untrusted source
  content, never model instructions.
- Sparse frame sampling can miss brief events. Coverage and sampling strategy
  are part of every result, and focused re-inspection is a first-class tool.
- Long video can monopolize CPU and memory. The broker enforces one heavy VLM,
  one ASR engine, bounded queues, cancellation, and explicit detail budgets.
- Cached media can be sensitive. Downloads and derived frames have quota/TTL,
  explicit deletion, and complete `noStore` cleanup.
- URL fetching is open-world network access. It is disabled until the user
  explicitly submits a URL and is never a route to cookies, DRM bypass, or
  private services.

# Vision reconstruction decision (2026-08-08)

The Summer Drive trace showed four coupled defects rather than a single weak
vision model: false UI-control heuristics on poster typography, an underspecified
model-facing reconstruction contract, silent candidate resizing in compare,
and a duplicated response payload that made every repair turn expensive. The
host then compounded them by rendering at the wrong viewport, mixing `cqw`
font sizing with a capped percentage-positioned canvas, inventing content below
the reference, and declaring a measured failure successful.

The recommended path is therefore deterministic-first rather than a larger VLM:

1. Add `profile=reconstruct` and return an implementation-ready
   `ReconstructionSpec` with exact canvas dimensions, visible-only content,
   source-coordinate boxes, text confidence, asset strategy, and a bounded
   focus plan.
   The full pass stays deterministic; Qwen is reserved for the at-most-four
   returned crops, where it can resolve text without downscaling the entire
   screenshot. A resolved regional crop has no recursive focus plan.
2. Require real closed-boundary evidence before classifying static artwork as
   controls. Text glyphs and word gaps are not interaction evidence.
3. Make compare strict-size by default and return a hard verdict plus blocking
   reasons. Aggregate similarity is supporting evidence, never the completion
   decision by itself.
4. Make the canonical compact document the default MCP response. Keep the old
   Markdown and raw legacy dump only behind explicit `response=full`.
5. Teach the host to use one coordinate system, render at the source viewport
   and DPR, preserve or trace the principal illustration instead of loosely
   redrawing it, repair the largest hot region first, and stop only on `pass`.

For the observed poster class, a credible loop should need about 10-20 host
turns. Fifteen to twenty-five minutes is a reasonable structural target; a
careful vector reconstruction of the main illustration may take 30-60 minutes.
The previous 40 minutes was not excessive by itself, but it was inefficient for
the resulting failure. The largest token reduction comes from compact results
and fewer full-scene calls, not from lowering the host model's reasoning budget.

The implemented local evidence supports this structure: Summer Drive's full
reconstruction ran in roughly 6.6-10.6 seconds, a Qwen date crop in roughly
19.5 seconds, and the compact response in roughly 25-28k JSON characters. The
strict compare rejected both a `318x628` candidate against `2557x1273` and the
same-size poor browser render; neither can set `canComplete=true`.

# Web reconstruction integrity decision (2026-08-09)

The follow-up Z-Code trace invalidated image-only completion as the final gate
for screenshot-to-web work. The winning candidate contained ten images, nine
with reference text, zero semantic controls, six unselectable live text nodes,
and roughly 69% raster coverage. A genuine `0.9625` pixel score therefore
optimized the wrong product objective. The user-visible capture later scored
`0.8926` and failed strict visual gates as well, showing exact-raster overfit.

The recommended path is an additive web representation contract plus a
browser-backed completion tool, not a higher compare threshold and not a larger
VLM. Deterministic geometry/OCR continues to describe the reference;
`targetKind=web` makes live/selectable text, semantic controls, structural CSS,
and bounded raster regions explicit. `sens_review` captures the candidate at
the immutable source viewport and combines the existing visual metrics with
DOM, accessibility, selection, control, raster, and source-slice evidence.

Image-only `sens_compare` remains compatible and reports a visual scope. It
cannot alone authorize completion of a web reconstruction. Z-Code is instructed
to keep a champion checkpoint, reject semantic regressions, and stop after the
first combined web pass. This directly addresses the 164-call trace, including
the 74 calls and 23.76M aggregate cached-input tokens spent after its first
image-only pass.

This design preserves the broker ownership invariant and keeps all perception
local/CPU-only. It does not infer destinations or business actions from a
screenshot: a measured button-like outline requires semantic control structure,
while its behavior remains explicitly unresolved without external evidence.
