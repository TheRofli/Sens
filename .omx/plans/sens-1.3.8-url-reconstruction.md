# Sens 1.3.8 URL reconstruction sessions

## Outcome

Make a URL reconstruction run observable and bounded for a text-only model.
The source page is frozen once. The candidate is freshly captured after each
bounded repair and once more before completion. Completion is valid only with a
broker-issued receipt tied to that final capture.

## Public workflow

1. `sens_web_start(sourceUrl, prompt, assetOutputDir, capture settings)`
   validates the source as public, captures it once, runs the existing local
   reconstruction analysis, stores the immutable reference and contract in a
   broker-owned session, and returns `sessionId` plus the starter project.
2. The model serves or copies the starter and calls
   `sens_web_review(sessionId, candidateUrl)`.
3. Each review composes the existing deterministic `review` operation with the
   stored reference, contract, viewport, DPR, theme, locale, and wait policy.
   The previous candidate capture is `beforeCapture`; the newly observed
   candidate is `afterCapture`.
4. The model applies one measured repair, preserves broker-owned champions, and
   repeats step 2. It must not re-capture the moving source.
5. `sens_web_review(sessionId, final=true)` performs another fresh capture. It
   returns `completionReceipt` only when visual, live-web, and blocking-reason
   gates all pass.

Existing `sens_capture`, `sens_see`, and `sens_review` stay compatible for
general analysis and older clients.

## Ownership and limits

- Rust broker owns session state, candidate history, TTL, capacity, champion
  policy, and completion receipts.
- Python Sight remains a stateless worker for capture, analysis, and review.
- At most eight sessions are retained; idle sessions expire after two hours.
- There is no resident browser and no polling loop. A browser exists only for a
  source start or candidate review operation.
- One review produces one candidate screenshot. Reusing it as the next
  `beforeCapture` avoids redundant CPU/RAM/browser work.

## Network boundary

- Source mode rejects URL credentials and every loopback, private, link-local,
  multicast, reserved, or unspecified address.
- Candidate mode permits loopback preview servers and rejects other non-public
  destinations.
- The same policy applies to initial navigation, redirects, and HTTP(S)
  subresources. `data:`, `blob:`, and `about:` browser resources remain local.
- Legacy explicit capture retains its existing caller-authorized behavior.

## Verification

- Python unit tests cover policy normalization, address resolution, browser
  request guarding, and candidate review policy.
- Rust unit tests cover public tool publication, broker request composition,
  bounded session retention, before/after evidence, and fresh-final receipt
  issuance.
- Installed MCP smoke must prove both tools route registry -> broker -> worker.
- Clean Z-Code runs must reconstruct Hungry Tiger, Slush, Caldera, and
  dope.security at a fixed viewport and demonstrate at least one fresh review,
  bounded repair evidence, and a final receipt or an explicit measured failure.

## Slush source-layer repair

The first corrected-contract Slush run is an honest measured failure, not an
acceptable completion: `0.8985` aggregate similarity and all live-web checks
passed, while one material region remained `0.1483` against the `0.08` bound.
The reference capture observed the clean hero AVIF but the reconstruction path
dropped it and inpainted the flattened screenshot beneath live `SLUSH` glyphs.

Implementation slice:

1. Capture bounded, already-loaded image response bodies referenced by visible
   raster elements; never refetch them.
2. Forward sanitized `sourceRasterAssets` from capture to the broker-created
   internal `see` request.
3. Validate hash/size/type/geometry in Sight and prefer one dominant source
   background only when separately observed live text overlaps it.
4. Copy the original source asset into the starter, preserve its measured box,
   and keep live text/controls above it.
5. Re-run focused unit/Rust tests, then a clean Slush session. A real final
   receipt is required before proceeding to Caldera and Dope.

## Live source-vector wordmarks

The Slush hero exposes a second source-layer boundary: the visible `SLUSH`
wordmark is five top-level SVG letter outlines, while the accessible `h1` is a
screen-reader-only label. Fitting a bundled font can preserve the word but
cannot reproduce those custom outlines.

Implementation slice:

1. Capture at most twelve viewport-material top-level SVG roots from the
   already guarded page. Inline only computed paint properties required for
   standalone rendering.
2. Parse the observed markup as XML, retain a narrow graphics allowlist,
   namespace internal IDs, and remove scripts, event attributes, foreign
   content, external references, and unsafe URL expressions before writing a
   content-addressed `.svg` artifact.
3. Forward bounded `sourceVectorAssets` only from the broker-owned capture into
   the internal `see` request. Sight rechecks path, size, hash, media type, and
   sanitizes again into its own cache.
4. Materialize vectors only when display OCR and measured geometry prove an
   exact one-vector-per-character wordmark. Keep a transparent selectable live
   label above the `aria-hidden` vector artwork.
5. Leave unmatched decorative SVGs in the protected background overlay to
   prevent double rendering.

The direct deterministic Slush starter now reaches `0.8924` aggregate
similarity with `webPass=true`; dimensions, pixel, foreground, layout, and hot
region checks pass. The remaining direct-starter failure is the OCR text score
(`0.6879` vs `0.7`), so completion still requires a clean agent repair run and
a broker receipt.

## Compact review projection

The first brokered Slush run exposed a host-facing transport failure: one
`sens_web_review` result was 62,621 bytes and Z-Code truncated it to its 50,000
byte tool-result budget. The model kept the verdict but lost part of the
ordered repair evidence, then spent many turns searching the contract and
running prohibited home-grown pixel scans.

The broker now applies a compact projection only after the raw review has
updated champion state, the previous/fresh capture snapshot, and any final
receipt. The public result preserves verdicts, blockers, ordered repair hints,
acceptance checks, metrics, at most six hot regions, coverage summaries,
observed/measured/inferred evidence, iteration policy, session captures, and
the URL workflow. It omits duplicated match tables, duplicate visual zones,
per-raster-element audit records, and artifact lists. A Rust regression test
requires an at-least-eightfold reduction for a representative verbose review
without losing the repair or receipt contract.

Context compaction is a second boundary. The broker persists every projected
review under its cache as `web-sessions/<session>/review-NNN.json`, adds the
path plus review number to `reviewReport`, and surfaces that path in the short
MCP text summary. Tool instructions require the host to retain and reread the
latest report after compaction; searching arbitrary cache files or recreating
old hints from memory is not an accepted repair path.

## Stable capture and exact vector geometry

Every reference and candidate capture freezes Web Animations, disables CSS
animations/transitions and the caret, waits two animation frames, and only then
takes the screenshot and DOM measurements. The capture records this operation
as observed `visualFreeze` evidence so a comparison cannot silently combine two
different animation states.

Captured source-vector boxes remain subpixel measurements, rounded only to
three decimals, through browser capture, the reconstruction contract, and
starter CSS. Integer rounding at any intermediate boundary is forbidden: it
produced one-pixel fringes around the giant Slush letters and degraded both
hot-region and OCR scores.

When OCR text similarity alone fails, its repair hint is authoritative before
generic hot-region advice. It includes the reference/candidate OCR strings,
repeated text groups and boxes, and verified vector-wordmark IDs. Verified
source vectors must not be replaced solely because a repeated ticker or other
small text lowered the page-wide OCR score.

## Passing-review terminal contract

A passing non-final review is a terminal champion state, not an invitation to
polish diagnostics. The broker removes `repairHints`, sets the workflow to
`ready-for-fresh-final-review`, marks iteration as non-continuable, and instructs
the host to call `sens_web_review(final=true)` without modifying the candidate.
The final pass changes that workflow to `complete`.

Every review now carries `reviewRequestId` at the top level and inside its
persisted `reviewReport` metadata. This makes ordered benchmark/audit reporting
self-contained and prevents agents from scanning caches for IDs that the prior
compact projection had discarded.
