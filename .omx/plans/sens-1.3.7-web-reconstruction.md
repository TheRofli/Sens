# Sens 1.3.7 Web reconstruction integrity

## Outcome

A text-only host can reconstruct a supplied screenshot as a real web page and
cannot complete by slicing the reference into raster text. Sens exposes the
reference structure, validates the rendered DOM and interaction surface, keeps
the best non-regressing candidate, and finishes only after visual and web gates
pass together.

## Slice 1 - measured structure and controls (TDD)

Files: `sidecars/sight/perception.py`, `sidecars/sight/ops.py`,
`sidecars/sight/document.py`, `tests/sight/test_perception.py`,
`tests/sight/test_controls.py`, `tests/sight/test_document.py`.

- Return source-pixel skeleton segments with endpoints, thickness, color, and
  orientation while preserving legacy line centers.
- Detect same-background rounded outlines containing OCR text, including the
  two Summer Drive `TICKETS` outlines, without restoring poster-glyph false
  positives.
- Reconcile raw elements with reconstruction roles and expose that role from
  `sens_element`.

Done when focused tests prove one continuous divider and two labelled control
candidates on the immutable Summer Drive fixture while existing poster safety
tests remain green.

## Slice 2 - web ReconstructionSpec (TDD)

Files: `sidecars/sight/document.py`, `sidecars/sight/ops.py`,
`tests/sight/test_document.py`, `tests/sight/test_responses.py`.

- Add optional `targetKind` resolution with a prompt-derived `web` mode.
- For web, require live/selectable text and semantic controls; prohibit raster
  text, reference slices, and raster layout structure.
- Remove the custom-font asset loophole. Font mismatch uses measured glyph
  metrics or a declared fallback, never a text crop.
- Expose allowed raster regions from measured illustration/photo/logo evidence.
- Include structural lines and a combined web-review completion action.

Done when the compact document is implementation-complete, backward compatible,
and explicitly prevents every representation failure from the observed run.

## Slice 3 - browser web review (TDD)

Files: `sidecars/sight/capture.py`, new `sidecars/sight/web_review.py`,
`sidecars/sight/server.py`, `tests/sight/test_capture.py`, new
`tests/sight/test_web_review.py`, `tests/sight/test_protocol.py`.

- Capture text-node boxes and selection styles, semantic controls and names,
  raster element boxes/screenshots, page text, assets, and accessibility.
- Match reference OCR boxes to candidate live DOM text geometrically.
- Detect unselectable content, missing controls, rasterized reference text,
  disallowed raster structure, excessive/unallowed raster regions, and
  full/sliced-reference laundering.
- Combine strict visual metrics and web checks into `visualPass`, `webPass`,
  `verdict`, `blockingReasons`, `requiredAction`, and web `canComplete`.
- Keep optional inferred semantic before/after review non-blocking and bounded.

Done when a real-text fixture passes, an image-sliced fixture fails despite a
strong visual score, and `noStore` leaves no review artifacts.

## Slice 4 - MCP and broker contract (TDD)

Files: `crates/sens-mcp/src/main.rs`, `crates/sens-broker/src/sight.rs`,
`crates/sens-protocol/src/lib.rs` only if the capability manifest needs an
additive operation.

- Add `targetKind` to see/zoom and publish `sens_review` with exact viewport,
  DPR, and optional bounded semantic review.
- Label `sens_compare` as visual-only for web work; publish the required
  `see -> focus -> build -> render -> review -> repair` loop.
- Return a bounded textual summary plus canonical structured content instead of
  duplicating the full JSON payload.
- Return prompt `sight_busy` backpressure for concurrent local requests and
  instruct hosts to execute CPU VLM focus work serially.

Done when Rust schema/router/backpressure/response tests are green and stdout
remains protocol-only.

## Slice 5 - release and runtime acceptance

- Run all Sight Python tests and required Rust format/Clippy/workspace tests.
- Build release binaries, deploy the verified sidecar/runtime, reconnect Z-Code,
  and verify capability/version/tool schemas through the real MCP protocol.
- Run the full clean headless DeepSeek v4 Flash benchmark matrix below with
  immutable inputs, fixed viewport/DPR/state, bounded turns, and live rollout
  capture. Every case follows `task -> result -> sens_review -> trace audit ->
  Sens repair -> clean rerun`; a pass on one design is not release evidence.
- Reject and repair any run that rasterizes reconstructable text/controls/
  structure, loses a salient measured element, regresses its champion, or
  fails combined web review. Repeat each affected case in a fresh workspace
  until the matrix is proven.
- Update version/release metadata and docs, review the deliberate diff, commit,
  tag `v1.3.7`, push `main` and tag, and verify GitHub release/update state.

### Required 1.3.7 reconstruction matrix

| Case | Source | Stress being tested | Required representation evidence |
| --- | --- | --- | --- |
| Summer Drive | immutable `2557x1273` screenshot | oversized display type, exact date/time, long divider, illustration, two outlined controls | all copy live/selectable; divider is independent structure; only car may be raster; two semantic Tickets controls |
| Dub Partner Program | supplied dashboard screenshot | dense small text, sidebars, cards, chart, counters, repeated controls | readable live DOM hierarchy; semantic navigation/controls; chart/decoration separated from text |
| Beyond Human Wear | supplied landing screenshot | dominant photographic hero, blur/feathering, compact nav, pills, tiny copy | hero photo may be raster; all navigation/copy/pills remain live and semantic |
| Hyperstudio ASCII hands | supplied dark screenshot | large character-built hand artwork plus normal heading/navigation/CTAs | hand artwork is reconstructed as exact selectable monospace characters, never flattened into an image |
| Hungry Tiger | `https://www.eathungrytiger.com/` | live responsive site, typography, assets, interaction and motion | frozen baseline plus live text/control/raster audit at the declared state |
| dope.security | `https://dope.security/` | live production composition and responsive behavior | frozen baseline plus live text/control/raster audit at the declared state |
| Caldera | `https://caldera.xyz/` | live production composition, graphics and motion | frozen baseline plus live text/control/raster audit at the declared state |

The three URL baselines are captured immediately before their runs and record
URL, redirect target, viewport, DPR, locale, theme, timestamp, screenshot hash,
DOM/accessibility snapshot, animation state, and external asset hashes. A live
site changing after the baseline is not silently counted as a Sens regression.

## Completion rule

1. Only measured illustration/photo/logo regions may remain raster.
2. Every reference text region is covered by live selectable DOM text.
3. Every measured visual control candidate is represented by an accessible
   semantic control without invented external behavior.
4. Structural lines are independently represented and visually converge.
5. Strict visual checks and every web representation gate pass together.
6. Seven clean Z-Code rollouts demonstrate the behavior on deployed 1.3.7 and
   stay within their recorded turn/time/token budgets or report a justified
   measured exception per case.
7. Tests, installed runtime, GitHub source/tag/release, and updater metadata all
   identify the same 1.3.7 build.
