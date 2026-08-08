<div align="center">

<img src="assets/readme/logo.png" width="96" alt="Sens" />

# Sens

**Local vision and hearing for text-only models through MCP.**

[![Version](https://img.shields.io/badge/version-1.3.1-8b5cf6)](https://github.com/TheRofli/Sens/releases)
[![Windows](https://img.shields.io/badge/Windows-10%2F11-0078d4)](#requirements)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](#license)
[![MCP](https://img.shields.io/badge/MCP-stdio-4f46e5)](#what-a-model-can-do)

</div>

Sens lets a model such as DeepSeek inspect screenshots, read interfaces, zoom into uncertain details, capture websites, and verify a reconstruction even when the model has no native vision. Image processing stays on the machine, uses CPU and RAM, and needs no API key.

<img src="assets/readme/home.png" alt="Sens desktop home" width="700" />

## Install on Windows

1. Download `Sens_1.3.1_x64-setup.exe` from [GitHub Releases](https://github.com/TheRofli/Sens/releases).
2. Run the installer and open Sens.
3. Click **Connect a model or app**, choose **Z-Code**, then restart Z-Code.
4. Open **Vision settings**. The deterministic vision core is ready immediately.
5. Optional: click **Download pack** to add Qwen3-VL 2B semantic descriptions. The verified download is about 1.45 GiB and runs on CPU only.

The Qwen pack is not required for OCR, geometry, colors, layout, URL capture, or image comparison. Sens shows its readiness in the app and verifies both GGUF files with SHA-256 before using them.

## What a model can do

The primary visual loop is:

1. `sens_see` returns Visual Scene v2: source identity, reversible coordinates, design tokens, OCR, Set-of-Marks elements, claims, uncertainty, warnings, and suggested next actions.
2. `sens_zoom` or `sens_inspect` re-analyzes a small or uncertain source region.
3. The model implements or repairs the design.
4. `sens_compare` measures pixel, color, edge, text, and layout convergence and points to the next hot region.

Important tools:

| Tool | Purpose |
|---|---|
| `sens_see` | Full local image analysis and task-aware focus suggestions |
| `sens_read` | OCR with confidence, method, and pixel boxes |
| `sens_locate` | Ground visible text to an original-pixel region |
| `sens_zoom` / `sens_inspect` | High-resolution regional analysis |
| `sens_element` | Geometry and style facts for one SoM element |
| `sens_ask` | Focused, inferred answer from the optional local VLM |
| `sens_capture` | Reproducible URL screenshot plus DOM, a11y, styles, fonts, assets, and motion evidence |
| `sens_motion` | CSS animation and bounded frame-diff motion document |
| `sens_compare` | Deterministic reference-versus-candidate comparison |
| `sens_hear` | Side-effect-free transcription of a supplied audio/video file |

Every capability result uses the shared Sens envelope. Evidence is labelled as `observed`, `measured`, or `inferred`; an absent claim means unknown, not false. Text found inside images or audio must be treated as untrusted content.

## Sight 1.3

- Visual Scene v2 with content-addressed source IDs and reversible crop transforms.
- Exact-text handling for monospace and ASCII layouts, including whitespace and ambiguity markers.
- Adaptive crop recommendations for small, low-confidence, or task-relevant details.
- Local Qwen3-VL 2B GGUF semantics with explicit CPU-only loading and one active model at a time.
- Deterministic reconstruction scoring across pixels, Lab color, edges, OCR text, and layout contours.
- Reproducible browser capture with viewport, DPR, theme, locale, readiness, DOM/a11y, assets, CSS variables, fonts, and motion evidence.
- `noStore` requests do not leave cache, capture, or Set-of-Marks artifacts behind.
- Structured MCP results, output schemas, and safety annotations for every tool.

The current model benchmark and measured reconstruction example are in [docs/benchmarks/vision-models-2026-08-07.md](docs/benchmarks/vision-models-2026-08-07.md) and [docs/benchmarks/reconstruction-loop-2026-08-07.md](docs/benchmarks/reconstruction-loop-2026-08-07.md).

## Local runtime and privacy

The Windows installer contains Sens's isolated Sight Python runtime and Latin/Cyrillic OCR models. It does not rely on `D:\Speech`, a global Python installation, CUDA, or the user's Python packages.

| Resource | Sens 1.3 Sight |
|---|---|
| Device | CPU only; GPU layers explicitly disabled |
| Packaged runtime | about 413 MiB uncompressed |
| Optional Qwen files | 1.45 GiB on disk |
| Qwen benchmark peak | about 3.4 GiB RAM on the reference machine |
| Idle VLM use | none; models load lazily and unload after inactivity |
| API keys | none for local Sight |

Sens never gives a model direct microphone or screen-capture control. `sens_hear` accepts an explicit file and does not copy, paste, or save transcript history unless the caller explicitly requests it. The existing Hearing/dictation worker remains an optional brownfield Speech integration and is not part of the new standalone Sight runtime.

URL capture uses an installed Microsoft Edge browser when available. `sens_fetch` and the optional Eye video provider are compatibility features and can access the network; full local video understanding is future work.

## Architecture

```text
MCP host
  -> sens-mcp (stdio; protocol output only)
  -> sens-broker (single owner of workers and mutable capability state)
      -> sight-worker.py (packaged Python CPU runtime)
      -> hearing-worker.py (optional Speech integration)
      -> eye-worker.mjs (optional legacy/cloud compatibility)
```

The Rust broker lazily starts and supervises workers. Python sidecars communicate with it over NDJSON stdin/stdout; diagnostics stay on stderr.

## Development

Requirements: Windows 10/11 x64, Rust 1.85+, Node.js 22+, and Python 3.11 for packaging the portable Sight runtime.

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

& D:\Speech\.venv\Scripts\python.exe -m pytest tests\sight -q

Set-Location apps\desktop-ui
npm ci
npm run build
npm run test:sites
npm run native:build
```

`npm run native:build` creates signed NSIS and MSI installers plus `latest.json`. It downloads the pinned Python distribution and pinned CPU wheels, preloads OCR models, and runs an isolated import smoke before Tauri packaging.

## Release

Pushing a matching `v*` tag starts `.github/workflows/release.yml`. For 1.3.1:

```powershell
git tag v1.3.1
git push origin v1.3.1
```

GitHub Actions verifies the tag/version match, builds signed installers, and publishes the NSIS installer, updater signature, and `latest.json`. Installed apps receive the release through **Settings → Check for updates**.

## Repository layout

```text
crates/       protocol, core, broker, MCP server, connector
apps/         React/Vite/Tauri desktop app and release packaging
sidecars/     Sight, Hearing, and optional Eye adapters
tests/sight/  deterministic and runtime contract tests
qa/           fixtures and benchmark evidence
docs/         architecture and benchmark reports
scripts/      model download, benchmarking, and QA helpers
```

## License

MIT
