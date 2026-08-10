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

# Media and Voice brownfield extension (2026-08-08)

## Existing contracts to preserve

| Contract | Current owner | Preserve while extending |
| --- | --- | --- |
| `sens_see` / `sens_inspect` | Rust Sight executor + Sight workers | Image/OCR/measurement behavior and shared result envelope |
| `sens_watch` | Rust Sight executor + legacy/cloud Eye | Tool schema and compatibility behavior until local Media is proven |
| `sens_hear` | Rust Hearing executor + Hearing worker | Side-effect-free file transcription and timestamped segments |
| `sens_fetch` | Rust Hearing executor + Hearing worker | Public tool response shape during migration; fix path identity first |
| Ctrl+Win dictation | Desktop -> broker-owned Hearing worker | Visible user consent and its explicit clipboard/paste flow only |

## Target ownership map

```text
sens-mcp / desktop UI
        |
        v
Rust broker (only mutable runtime owner)
  |-- shared Arc<SightExecutor>   -> local/cloud Sight workers
  |-- shared Arc<HearingExecutor> -> Hearing worker
  |-- MediaExecutor              -> lightweight media preparation worker
  |      |-- calls the same Sight executor
  |      `-- calls the same Hearing executor
  `-- VoiceExecutor              -> lazy sherpa-onnx TTS worker
```

`build_core` must construct Sight and Hearing once and pass the same `Arc`
instances to Media. Media must not spawn duplicate VLM/ASR workers or own
settings outside the broker. Fetch/demux/frame preparation moves from Hearing
to `sidecars/media`; Hearing delegates to it temporarily for compatibility.

## New public contracts

- `sens_analyze`: exactly one of `path`, `url`, or `text`; optional `detail`,
  `question`, language/model preferences, and `noStore`.
- `sens_media_inspect`: a content ID plus `at` or `range` and an optional
  question; produces a focused source-backed result.
- `sens_speak`: text, voice, speed, format, and optional explicit `play`;
  returns an audio artifact and measured synthesis metadata.
- `MediaDocument v1`: source identity, media metadata, transcript/captions,
  timestamped audio events and visual observations, timeline, chapters,
  coverage, warnings, and next actions. Each claim is observed, measured, or
  inferred.

## Dependency and side-effect boundaries

- Media preparation may read local files and explicitly submitted public URLs,
  create bounded cache entries, and delete only its own staged/derived files.
- Sight remains the only owner of vision model state; Hearing remains the only
  owner of ASR model state; Voice owns one lazy TTS model.
- `sens_analyze` never captures screen/microphone and never writes clipboard,
  paste, or history.
- `sens_speak` writes a file by default. System playback is a visible,
  separately requested side effect.
- All worker diagnostics use stderr. MCP stdout remains protocol-only. No URL
  credentials, API keys, IPC secrets, transcripts, or document content enter
  process arguments or diagnostic exports.

## Known brownfield defects to close before exposure

1. Media fetch currently determines output by scanning a shared directory, so
   audio can be misreported as video.
2. URL capture/fetch has no complete SSRF/private-network/redirect policy.
3. Fetch omits caption-first extraction, final cumulative size checks, atomic
   promotion, cache TTL/quota, and full `noStore` cleanup.
4. Extracted frames are paths without visual analysis or transcript fusion.
5. Current tests mock download success and do not prove distinct audio/video
   paths, redirect safety, cleanup, cancellation, or installed-runtime codecs.

## Compatibility strategy

Land Media additively. Internally route `sens_watch` and `sens_fetch` through
the new preparation path only after equivalence tests pass. Deprecation, if
ever useful, is a later explicit release decision. Voice is additive and must
not change Ctrl+Win dictation.

# Vision reconstruction hardening (2026-08-08)

## Failure being corrected

The first Summer Drive reconstruction proved that the current scene document is
useful for semantic orientation but is not yet a safe completion contract for
pixel-accurate work. A text-only host accepted false controls inferred from
poster lettering, mixed coordinate systems in the implementation, compared
different render sizes after an implicit resize, and treated a low aggregate
similarity as success. The repair loop then repeated the full, duplicated Sight
payload until the session reached roughly 23 million aggregate input tokens.

## Boundaries and ownership

- `sidecars/sight` continues to own deterministic image analysis, the Visual
  Scene document, reconstruction guidance, and compare metrics.
- The Rust broker remains the only owner of the Sight worker and mutable
  runtime state. It validates files and forwards additive request options; it
  does not reimplement image analysis.
- `sens-mcp` owns the model-facing tool schema and instructions. Its default
  path must be compact and unambiguous, while an explicit compatibility mode
  retains the full legacy projection for existing 1.x consumers.
- Z-Code must launch the installed `sens-mcp.exe`, not a stale repository build.
  A client restart is required after the executable/configuration changes.

## Contract changes

### Reconstruction profile

`sens_see` accepts an additive `profile` option. `profile=reconstruct` returns
a `ReconstructionSpec` inside Visual Scene v2 with exact source dimensions and
aspect ratio, a single coordinate-system recommendation, confirmed and
uncertain text, foreground regions, asset strategy, static-artwork warnings,
and a bounded first set of focus actions. It must explicitly prohibit adding
content or interactions that are not visible in the reference.

The full reconstruction pass does not call a generative model. It returns at
most four source-pixel focus regions; `sens_zoom(profile=reconstruct)` applies
the configured local Qwen pack only to that crop, exposes disagreements and a
low-confidence `preferredValue`, and returns an empty `focusPlan` so the host
cannot recursively inspect the same region. Dense screenshots auto-select this
profile when an MCP client drops the user's prompt/profile arguments.

Static artwork suppresses UI-control claims unless a candidate has measured
closed-boundary evidence. Saturated glyph columns, inter-letter gaps, or an OCR
box alone are never enough to call text a button or pill.

### Compare gate

`sens_compare` is strict by default. It never silently resamples a candidate.
The result reports both decoded sizes, aspect-ratio delta, whether resampling
was requested, foreground-weighted mismatch metrics, largest-hot-region ratio,
an explicit `pass|partial|fail` verdict, blocking reasons, and `canComplete`.
An explicit compatibility fit mode may resize, but its result remains marked
as resampled and cannot prove exact-size completion.

Initial fixture thresholds are exact dimensions, similarity at least `0.88`,
pixel mismatch at most `0.12`, foreground mismatch at most `0.18`, layout
similarity at least `0.80`, and largest hot region at most `0.05` of the
reference. They are release-policy starting points and must be calibrated
against checked-in fixtures rather than weakened to make a failing poster pass.

### Compact MCP projection

The default Sight response contains the canonical `doc`, compact summary,
artifact references, pack, and compatibility metadata. It does not duplicate
the same data as rendered Markdown plus `doc` plus the entire legacy dump.
`response=full` retains those fields for explicit legacy/debug use. Zoom uses
the same compact projection and identifies the analyzed source region.

For reconstruction, compact `doc` also omits generic claims, the composition
ASCII preview, and the generic element projection because those duplicate the
exact reconstruction contract. The Summer Drive payload measured about 25-28k
JSON characters after this projection.

## Must-preserve behavior

- Existing tool names and the shared request/result envelope remain valid.
- `response=full` exposes the existing `document`, `doc`, `somPath`, `legacy`,
  and `pack` fields while clients migrate.
- OCR and VLM text remain inferred; decoded dimensions and deterministic
  geometry remain measured.
- Source coordinates stay reversible through crop/upscale operations.
- `noStore` still leaves no cache or artifact.
- Sight stays local and CPU-only, `sens-mcp` stdout stays protocol-only, and
  unrelated Hearing/Media/Voice behavior is unchanged.

## Verification surface

1. Synthetic strict-size and threshold tests fail before compare changes and
   pass after them.
2. A poster-text fixture proves plain saturated text produces no controls while
   a genuinely outlined button remains detectable.
3. Protocol tests prove compact is the default and full compatibility is
   opt-in without losing canonical `doc` data.
4. Rust schema tests prove reconstruction/profile/response/fit options and
   completion-language instructions are visible to the host.
5. The Summer Drive reference is rendered and compared at exactly
   `2557x1273`, DPR 1, after fonts settle; a result below the gate remains a
   hard failure with an actionable largest region.

# Web reconstruction integrity (Sens 1.3.7, 2026-08-09)

## Failure being corrected

The second Summer Drive Z-Code run reached an image-only compare pass by
cutting the immutable reference into ten raster assets. Nine assets contained
text, the remaining live text was explicitly unselectable, both visible
`TICKETS` controls were images, and the continuous divider survived only as
three crop fragments. The generated DOM had no buttons, links, handlers, or
keyboard focus. Pixel convergence therefore improved while the requested web
implementation became less real.

This is a completion-contract failure, not merely a weak OCR/model result.
`sens_compare` can prove visual convergence between two bitmaps, but cannot
prove representation, interaction, selection, accessibility, or asset origin.

## Entry points and ownership

- `sidecars/sight/perception.py` owns measured structural-line segments and
  outlined-control candidates around OCR text.
- `sidecars/sight/document.py` owns the additive `targetKind=web`
  ReconstructionSpec, live-text/raster/control rules, reconciled element roles,
  and bounded completion plan.
- `sidecars/sight/capture.py` owns browser-observed DOM text, selection styles,
  semantic controls, raster elements, accessibility, and content-addressed
  screenshots.
- New `sidecars/sight/web_review.py` owns deterministic reference-to-DOM
  coverage, raster-text/source-slice detection, semantic-control coverage, and
  the combined web completion verdict.
- `sidecars/sight/compare.py` remains the image-only visual metric. Its public
  result stays compatible and is explicitly labelled `completionScope=visual`.
- `sidecars/sight/server.py` routes the additive `review` operation.
- `crates/sens-broker/src/sight.rs` remains the only worker owner, validates the
  new operation, and returns prompt backpressure instead of hiding concurrent
  CPU requests in a mutex queue until clients time out.
- `crates/sens-mcp/src/main.rs` exposes `sens_review`, publishes the web loop,
  and avoids duplicating full structured JSON in textual tool content.

## Must-preserve behavior

- Existing Sight tool names, arguments, result envelopes, and full/compact
  projections remain valid.
- `sens_compare` remains usable for pure image/visual work and compatibility
  callers; web completion moves to the additive `sens_review` tool.
- General `profile=analyze` results are unchanged except for additive fields.
- The broker remains the only owner of workers and mutable runtime state.
- Sight stays local/CPU-only; no screen capture or microphone permission is
  added; explicit URL capture remains the only browser input.
- `noStore` removes all temporary capture/review artifacts.
- Hearing, Media, Voice, desktop dictation, and unrelated user files remain
  outside this release slice.

## Core contract changes

1. `sens_see(profile=reconstruct, targetKind=web)` returns a representation
   contract: readable text must be live/selectable DOM text; raster text,
   reference slices, and raster layout structure are forbidden; raster is
   allowed only in measured illustration/photo/logo regions.
2. Skeleton output includes source-pixel line segments with endpoints,
   thickness, color, and orientation while retaining legacy center arrays.
3. Same-background rounded outlines containing OCR text become measured visual
   control candidates. Raw `sens_element` reports the reconciled reconstruction
   role and warnings instead of contradicting the scene document.
4. `sens_review(referencePath, url, exact viewport/DPR)` combines strict visual
   compare with browser-observed live-text coverage, selectable-text checks,
   semantic-control coverage, accessibility evidence, raster coverage, and
   rasterized-reference-text detection. Only its combined pass can set web
   `canComplete=true`.
5. Reconstruction instructions require checkpoint/champion promotion: a trial
   cannot replace the best candidate if visual metrics improve while web gates
   regress. The host stops after the first combined pass.
6. Compact MCP content is one canonical structured payload plus a bounded text
   summary, not two serialized copies of the same document.
7. Concurrent local VLM requests receive explicit `sight_busy` backpressure;
   focus regions are executed serially or through a bounded batch path.

## Side effects and risky branches

- Web review launches a local headless browser only for the caller-supplied
  explicit HTTP(S) URL and writes content-addressed review artifacts unless
  `noStore=true`.
- Candidate asset screenshots are derived from visible DOM raster elements;
  they must not overwrite source files or shared basenames.
- A visually button-like shape can justify semantic `<button>`/`<a>` structure,
  but Sens must not invent a destination, external action, purchase flow, or
  hidden interaction.
- OCR/VLM disagreement remains explicit. World knowledge never resolves a
  glyph conflict.

## Recommended edit boundary

Keep this release additive and bounded to Sight/MCP/broker plus tests, release
metadata, benchmark fixtures, and documentation. Do not refactor model loading,
Hearing, Media/Voice, or desktop UI while closing this failure class.
