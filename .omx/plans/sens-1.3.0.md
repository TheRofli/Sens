# Sens 1.3.0 release plan

## Outcome

Ship `v1.3.0` as a trustworthy, CPU-only external perception runtime for text-first MCP models. The release must improve real visual reconstruction and debugging work, preserve Sight/Hearing public behavior, install from GitHub Releases, and document an honest new-user path.

## Non-negotiable contracts

- The Rust broker remains the only owner of capability workers and mutable runtime state.
- `sens-mcp` stdout contains MCP protocol output only; diagnostics use stderr.
- Images and audio stay local unless the user explicitly selects a network operation.
- GPU inference is disabled, not merely optional.
- Every claim is classified as `observed`, `measured`, or `inferred` with method/evidence.
- Source coordinates remain reversible through every crop and resize.
- `noStore` leaves no persistent cache or artifact.
- Existing MCP tool names remain compatible through 1.3.0.
- User-owned untracked files are preserved.

## Wave 1 - correctness baseline

### Slice 1.1 - deterministic truth

Likely files: `sidecars/sight/perception.py`, `sidecars/sight/ops.py`, `sidecars/sight/tree.py`, `sidecars/sight/cache.py`, `tests/sight/*`.

Must do:

- fix BGR/RGB palette conversion;
- give stored SoM artifacts content-addressed operation-aware names;
- make `noStore` persist nothing;
- carry crop-to-source transforms and source-pixel boxes;
- classify detector/OCR/VLM outputs as inferred, not measured;
- add regression fixtures for every behavior.

Exit: focused tests fail before implementation and pass after it; full Sight suite stays green.

### Slice 1.2 - shared Visual Scene v2 contract

Likely files: `sidecars/sight/document.py`, `crates/sens-protocol/src/lib.rs`, `crates/sens-broker/src/sight.rs`, `tests/sight/*`, Rust unit tests.

Must do:

- introduce a versioned scene document with source identity, coordinate spaces, claims, uncertainty, artifacts, and recommended next actions;
- populate envelope provenance and warnings;
- retain a compatibility projection for current clients.

Exit: schema fixtures round-trip; every returned claim has epistemic type and evidence/method; legacy fields remain available.

## Wave 2 - useful perception

### Slice 2.1 - exact text and ASCII

Likely files: `sidecars/sight/ascii_map.py`, new `sidecars/sight/ascii_text.py`, OCR/document modules, tests.

Must do:

- separate luminance composition maps from exact monospace text extraction;
- preserve whitespace/newlines;
- expose ambiguity instead of inventing characters;
- use DOM text directly for instrumented URL sources.

Exit: generated monospace fixtures achieve exact line/whitespace recovery at supported resolution.

### Slice 2.2 - adaptive focus

Likely files: new `sidecars/sight/focus.py`, `ops.py`, `vlm.py`, document tests.

Must do:

- derive candidate focus regions from small text, density, conflicts, and user intent;
- inspect only the regions required by the task;
- preserve source transforms and padding;
- return executable `nextActions`.

Exit: fixtures prove deterministic focus selection and stable source boxes.

### Slice 2.3 - semantic model decision

Likely files: `sidecars/sight/vlm.py`, `scripts/download-vision-models.py`, benchmark scripts and report.

Must do:

- benchmark current SmolVLM packs against official Qwen3-VL-2B GGUF on Sens-specific tasks;
- select the default from measured OCR, grounding, hallucination, latency, disk, and peak-RSS results;
- enforce CPU backend and one loaded model at a time;
- make model absence an explicit degraded state.

Exit: checked-in benchmark report justifies the default; download manifests contain hashes and licenses.

## Wave 3 - model-facing workflow

### Slice 3.1 - structured MCP surface

Likely files: `crates/sens-mcp/src/main.rs`, protocol/broker tests.

Must do:

- return MCP structured content/output schemas instead of nested JSON strings where supported;
- make descriptions task-oriented and internally consistent;
- expose a compact primary workflow while keeping legacy tools;
- add correct read-only/open-world/idempotent annotations;
- teach the host to use focus after uncertainty and compare after implementation.

Exit: MCP Inspector/client smoke proves discovery, invocation, structured output, and clean stdout.

### Slice 3.2 - instrumented URL capture

Likely files: `sidecars/sight/capture.py`, ops/server/MCP, tests.

Must do:

- accept explicit viewport, DPR, theme, locale, wait policy, and full-page mode;
- wait for fonts/hydration with bounded timeouts;
- collect DOM/a11y/computed styles, assets, CSS variables, animations, and element screenshots;
- content-address capture artifacts and isolate per request;
- keep arbitrary model-controlled screen capture out of scope.

Exit: local deterministic fixture site yields stable captures and exact DOM-backed claims.

### Slice 3.3 - reconstruction and QA loop

Likely files: compare/capture/focus modules, MCP facade, test fixture site, reports.

Must do:

- render reference and candidate with identical browser settings;
- report pixel/color/edge/text/layout deltas and top hot regions;
- return next focus calls for the largest actionable differences;
- prove at least one screenshot-to-code repair trajectory improves measured similarity.

Exit: checked-in report includes before/after metrics and artifacts.

## Wave 4 - productization

### Slice 4.1 - runtime and installer truth

Likely files: Tauri resources/build scripts, Rust runtime discovery, release workflow.

Must do:

- package all repository-owned Sight worker modules, not only wrapper scripts;
- provide a supported first-run runtime/model bootstrap or bundle the required runtime;
- verify a clean-machine-style install path without `D:\Sens` assumptions;
- preserve the external Speech bridge while documenting its current installation requirement if it cannot legally/technically be bundled;
- ensure release artifacts contain everything claimed by README.

Exit: installer smoke from an isolated directory successfully starts the broker and runs a local Sight fixture.

### Slice 4.2 - desktop/onboarding/documentation

Likely files: `README.md`, `docs/`, desktop translations and settings UI.

Must do:

- add a short new-user guide for install, model download, MCP connection, first vision request, privacy, and troubleshooting;
- replace stale/overstated performance and architecture claims with measured truth;
- document tools and recreation workflow;
- expose model/runtime readiness clearly in the UI;
- update screenshots only if the visible UI changes.

Exit: EN/RU copy is consistent and every command is exercised.

### Slice 4.3 - release v1.3.0

Must do:

- bump all versions and lockfiles;
- run Python, Rust, frontend, MCP, installer, and runtime acceptance gates;
- review diff and tracked artifacts; stage only deliberate files;
- commit coherent slices, push `main`, create/push `v1.3.0`;
- verify GitHub Actions succeeds and GitHub Release contains installer, signature, and updater manifest;
- inspect the published updater URLs and release notes.

Exit: the public `v1.3.0` release is downloadable and its current evidence satisfies `.omx/plans/sens-1.3.0-verification.md`.
