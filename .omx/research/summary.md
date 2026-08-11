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

# Licensing and GitHub surface decision (2026-08-10)

Sens needs public download, inspection, modification, and noncommercial use,
while TheRofli retains the right to use the code in proprietary commercial
products and to license it commercially to others. An OSI open-source license
cannot satisfy that restriction because open-source licenses must allow
commercial use.
Creative Commons also recommends against using CC licenses for software.

The selected public license is PolyForm Noncommercial 1.0.0. It is a
software-specific, SPDX-listed source-available license that permits
noncommercial purposes and does not prevent the licensor from granting other
licenses. The exact official text remains unmodified apart from Sens's
`Required Notice` copyright line.

The boundary is prospective: releases through and including `v1.3.7` were
published under MIT and retain those permissions. The current development line
and releases beginning with `v1.3.8` use
`PolyForm-Noncommercial-1.0.0`. External code is not merged without a separate
contribution-rights agreement so future commercial use is not silently
encumbered. Future Windows packages include both `LICENSE` and `LICENSING.md`
as bundle resources.

Primary references:

- https://polyformproject.org/licenses/noncommercial/1.0.0
- https://spdx.org/licenses/PolyForm-Noncommercial-1.0.0.html
- https://opensource.org/faq
- https://creativecommons.org/faq/index.html

# URL reconstruction capture decision (2026-08-11)

Continuous source polling is counterproductive for pixel reconstruction:
animation, advertisements, personalization, and A/B variants can move the
target while the candidate is being repaired. Sens 1.3.8 therefore freezes one
source capture and keeps its screenshot hash, contract, viewport, DPR, theme,
locale, and navigation policy in broker-owned state.

Candidate observation is event-driven. The first preview and every subsequent
bounded repair receive a fresh Playwright capture through `sens_web_review`.
The previous capture is already the correct `before` state, so taking a second
pre-repair screenshot would add browser load without new evidence. A requested
final review always captures again and issues a completion receipt only when
the new screenshot passes visual and live-web checks with no blockers.

This design adds bounded memory rather than a permanent browser: at most eight
small session records with a two-hour idle TTL. Heavy screenshot and contract
files remain content-addressed artifacts on disk. CPU and RAM cost per review is
approximately the existing one-shot review cost; there is no polling process.

The first headless Slush session also proved that compactness must apply to the
combined review, not only to the initial reconstruction document. The raw
review was 62,621 bytes and the Z-Code tool bridge truncated it at 50,000 bytes,
after which the agent searched redundant match tables and improvised pixel
scripts instead of following the bounded repair hint. Sens therefore keeps the
full result internally for champion and receipt decisions, then returns a
lossless-for-action projection: pass flags, blockers, repair hints, metrics,
six hottest regions, live-web coverage, provenance, iteration state, and
before/after captures. Duplicate match arrays, visual zones, raster-element
records, and artifact inventories are not model-facing review content.

A subsequent automatic Z-Code compaction retained the champion score and
rollback decision but dropped the exact repair-hint geometry. Because prior
reviews existed only in conversational context, the model began scanning the
contract and cache. The corrective boundary is broker-owned durable review
evidence: every compact review is written to a numbered JSON report, its path
is returned in both structured content and the bounded text summary, and MCP
instructions tell the host to reread it after compaction.

# Browser-loaded source raster decision (2026-08-11)

The corrected Slush contract reached `0.8985` similarity with all live-web gates
passing, but failed one material hot-region bound. Inspection showed that the
page had already loaded a clean `Slush_Logo_3D_Blue.avif` hero layer. Sens kept
only a composited element screenshot, discarded the original response body, and
filled the removed wordmark with Telea-generated blue blobs. More model turns
cannot recover pixels that the pipeline intentionally threw away.

Sens should preserve selected browser-loaded image responses as evidence. The
recommended implementation listens to responses during the existing guarded
capture, persists only image bodies referenced by visible raster elements under
strict count/byte limits, and returns their hashes and measured geometry. It
does not refetch URLs. The broker forwards those capture-owned records to the
internal `see` call; they are not a public arbitrary-file parameter.

Sight may replace a synthetic full-canvas background only with a hash-verified,
viewport-dominant source raster that overlaps separately observed live DOM
text. The original asset is copied into the generated starter at its measured
box, while text and controls stay semantic. Unsupported or ambiguous evidence
keeps the existing protected-background fallback. This resolves the information
loss without a larger VLM, a resident browser, a new SSRF surface, or relaxed
completion gates.

# Browser-observed SVG wordmark decision (2026-08-11)

The remaining Slush display mismatch was not a font-selection problem. The
visible wordmark consists of five separate SVG letter outlines; the page's
`h1` is only a hidden accessible label. Injecting those five live vector roots
raised the deterministic starter substantially, while substituting the page's
declared Lateral font made the result worse.

The bounded solution is to treat large visible top-level SVG roots as guarded
DOM evidence rather than rasters. Capture sanitizes an allowlisted XML subset,
removes executable and external content, namespaces fragment IDs, and stores
content-addressed bytes. Sight verifies and sanitizes again. A vector sequence
is materialized only when high-confidence display OCR proves that its ordered
count and union geometry match one live word. The visible outline is
`aria-hidden`; a transparent selectable text label preserves semantics.

Unmatched decorative SVGs are not added on top of the protected screenshot
overlay because that duplicates pixels. With only the five proven wordmark
letters materialized, the direct Slush starter measures `0.8924`, passes the
pixel, foreground, layout, hot-region, and live-web gates, and misses only the
small-text OCR gate (`0.6879` against `0.7`). A real agent run must still repair
that bounded remainder and obtain a fresh completion receipt.

# Slush capture-state and OCR diagnosis (2026-08-12)

The clean headless reference/candidate audit isolated two independent sources
of noise. First, screenshot and DOM observation were not frozen as one atomic
visual state. Capture now pauses Web Animations, disables CSS motion and caret
painting, waits two frames, and records `visualFreeze` provenance before both
observations.

Second, the only failed review gate was page-wide OCR similarity. Its mismatch
was dominated by the repeated `CARD / SLUSH / WAITLIST` ticker, while all five
source SVG wordmark letters were already verified. The prior generic largest
hot-region hint sent the agent toward the correct wordmark and caused a
measured regression; champion rollback prevented it from landing. Text-only
failures now return the actual OCR strings and repeated-element geometry first,
and explicitly protect verified vector wordmarks from speculative replacement.

Finally, the source captured fractional SVG bounds but three later boundaries
rounded them to integers. Frozen browser evidence showed corresponding one-pixel
fringes around several giant letters. Vector bounds now retain three-decimal
precision from Playwright through the broker-owned reconstruction contract into
generated CSS. Targeted red/green tests cover capture persistence, hydration,
and starter rendering.

# Caldera pass-state diagnosis (2026-08-12)

The clean Caldera run passed on its first review at `0.9562`, with OCR `0.9842`
and every live-web gate green. However, the compact pass still exposed
nonblocking 1-2 px text hints. DeepSeek treated them as work despite
`requiredAction=request-fresh-final-review`, spending another review for a
small improvement to `0.9570`. A pass now suppresses repair hints and publishes
an explicit no-modification workflow before the fresh final capture.

The same run spent several post-receipt turns searching the Sens cache because
the benchmark requested ordered review IDs but persisted compact reports did
not include their own request IDs. The broker already owned those identifiers;
it now includes `reviewRequestId` in the compact result, report metadata, MCP
summary, and durable JSON. This is an observability fix with no extra model or
vision work.

# dope.security live-word geometry and materiality diagnosis (2026-08-12)

The first clean dope.security run remained an honest failure after bounded
repairs. Its best candidate passed the live-web contract but stopped at a
`0.0839` material bounding ratio against the `0.08` limit. The visible defect
was the `Secure Web` hero line: the Python capture had exact browser `Range`
boxes for each word, but `sanitize_source_text_nodes` in the Rust broker
discarded `wordBoxes` while forwarding capture evidence to Sight. The
downstream starter therefore centered full-size source-font words inside
unreliable OCR slots and made the words overlap.

The broker now forwards at most 128 validated word boxes per source text node,
with bounded text and finite coordinates. The runtime contract consequently
changed from one shared `[72,239,548,436]` box for both words to observed boxes
`Secure=[72,239,349,366]` and `Web=[366,239,548,366]`. The generated page keeps
both as selectable text using the packaged `Whyte Inktrap` source font.

That correction raised the next clean candidate to `0.9247` similarity with
pixel mismatch `0.0828`, foreground mismatch `0.0230`, text similarity
`0.8629`, and all live-web checks green. It also exposed a false-negative in
the hot-region gate: sparse glyph-edge differences across a multi-line heading
occupied `0.0324` of the canvas but their mostly empty bounding rectangle
occupied `0.1174`. Material-bounding density now requires `0.30` rather than
`0.25`. A regression test proves the accurate multi-line case is ignored while
the older overlapping-word case (`0.0291 / 0.0839`, density above `0.30`) still
blocks.

The final clean v9 run passed its first non-final review and an independent
fresh final review with score `0.9074`, pixel mismatch `0.1066`, foreground
mismatch `0.0447`, text similarity `0.8602`, and hot signal `0.0333`. It issued
completion receipt
`e714c595-4da6-4853-b6fe-61a319c53ece:8644c23c-4907-425d-9073-af25e804e89f`.
