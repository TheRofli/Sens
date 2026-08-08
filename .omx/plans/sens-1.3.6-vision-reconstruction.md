# Sens 1.3.6 Vision reconstruction hardening

## Outcome

A text-only model can use Sens to reconstruct a supplied screenshot without
inventing hidden UI, silently comparing different canvases, or completing on a
weak aggregate score. The default MCP workflow is compact enough for repeated
repair turns and remains compatible through an explicit full response.

## Slice 1 - strict compare gate

- Add exact dimension/aspect metadata and strict-by-default alignment.
- Add explicit optional fit mode, foreground metrics, hot-region area ratios,
  verdict, blocking reasons, threshold checks, and `canComplete`.
- Forward the fit option through the worker and MCP schema.

Verify: focused compare tests red/green, then full Sight tests.

## Slice 2 - static-artwork control safety

- Require closed-boundary evidence for controls.
- Suppress saturated glyph/inter-letter false positives.
- Preserve genuinely outlined buttons in deterministic fixtures.

Verify: new poster/control fixture tests red/green.

## Slice 3 - reconstruction document

- Add `profile=reconstruct` and a compact `ReconstructionSpec`.
- Include exact canvas, coordinate-system rule, visible content, text
  confidence, asset strategy, blocking uncertainties, and bounded next actions.
- Keep the full-image pass deterministic; run Qwen only on the at-most-four
  source-pixel focus regions, expose a low-confidence `preferredValue`, and
  terminate regional zooming when its `focusPlan` is empty.
- Preserve ASCII/monospace content as exact characters rather than flattening
  it into an image, and project zoomed font measurements back to source pixels.
- Add explicit no-invention and exact-render instructions.

Verify: document and protocol tests.

## Slice 4 - compact compatibility projection

- Default to canonical compact result.
- Keep full Markdown/raw dump only for `response=full`.
- Remove generic claims, ASCII preview, and element projections that duplicate
  `ReconstructionSpec` from compact reconstruction responses.
- Advertise the option in MCP schemas and server instructions.

Verify: response-shape and Rust router tests; measure representative payload.

Measured on the Summer Drive source: the cold deterministic reconstruction
completed in about 6.6-10.6 seconds during local runs, the Qwen date crop in
about 19.5 seconds, and the compact payload in about 25-28k JSON characters.
The Qwen crop returned `06.24.21` where deterministic OCR remained uncertain.

## Slice 5 - runtime acceptance

- Run all Python Sight tests and the required Rust format, Clippy, and workspace
  test gates.
- Build release broker/MCP, install/copy the verified runtime safely, and point
  Z-Code user configuration at the installed MCP executable.
- Restart the broker/MCP host, exercise status/see/compare through the real
  protocol, and render the Summer Drive candidate at `2557x1273` with Playwright.
- Commit only deliberate paths and push `main`.

## Completion rule

The slice is complete only when tests and real protocol smoke pass, Z-Code is
configured for the current installed executable, and the poster comparison
cannot report completion unless the strict verdict is `pass`.
