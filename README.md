# Sens

Sens is a local capability layer for language models. It combines visual perception from **Eye** and audio transcription from **Speech** behind one Rust broker, one MCP connection, one tray application, and one capability contract. The desktop UI is available in **Russian and English** — switch it in Settings → Interface language.

## What ships in 1.1

- `sens-desktop` — the Tauri desktop and tray experience (RU/EN interface).
- `sens-broker` — the single-owner named-pipe runtime and lazy worker supervisor. Stops itself on quit and before update installs.
- `sens-mcp` — a standard stdio MCP server with 15 Sens and Eye-compatible tools.
- `sens-connect` — reversible connector setup for Z-Code.
- `sens-protocol` and `sens-core` — versioned contracts, registry, state, policy, and execution.

### Sight (vision)

- Image analysis, OCR, element locating, region inspection, and reference comparison through Eye.
- Configurable provider (MiMo, OpenAI, or a custom OpenAI-compatible endpoint), analysis detail (quick/normal/deep), per-image call budget (up to 32), result caching, and an optional **two-pass verification** that cross-checks the answer against the image.
- Images are sent at higher resolution (max side 2048 px) and split into overlapping crops for large screenshots; seams are merged with deduplication.

### Hearing (dictation)

- Push-to-talk dictation with a global hotkey (Ctrl+Win by default) that inserts text into the active field.
- Local models: Parakeet, Whisper RU, GigaAM v3. Long recordings are chunked at silence gaps with overlap so there is no length limit in practice.
- No model access to the microphone in Sens 1.0.

### Updates

- In-app updater with signed releases: Settings → Check for updates.
- Releases are built automatically by GitHub Actions when a `v*` tag is pushed — no manual packaging.

## Build and verify

```powershell
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

Set-Location D:\Sens\apps\desktop-ui
npm install
npm run build
npm run test:sites
npm run native:build
```

The native build emits both installers:

- `target\release\bundle\nsis\Sens_<version>_x64-setup.exe`
- `target\release\bundle\msi\Sens_<version>_x64_ru-RU.msi`

## Releasing a new version

1. Bump the version in `Cargo.toml`, `apps/desktop-ui/package.json`, and `apps/desktop-ui/src-tauri/tauri.conf.json`.
2. Commit, push, then tag and push the tag:

```powershell
git tag v1.1.3
git push origin v1.1.3
```

3. The `Release` workflow builds, signs, and publishes the installers plus `latest.json` to a GitHub release.
4. Installed apps pick it up via Settings → Check for updates.

The workflow requires two repository secrets: `TAURI_SIGNING_PRIVATE_KEY` (contents of your minisign secret key) and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` (the plaintext key password).

## Connect Z-Code

```powershell
$env:SENS_EYE_ROOT = 'C:\Users\kanal\.zcode\workspace\default\eye'
$env:SENS_SPEECH_ROOT = 'D:\Speech'
D:\Sens\target\release\sens-connect.exe zcode install `
  C:\Users\kanal\.zcode\cli\config.json `
  D:\Sens\target\release\sens-mcp.exe
```

Restart Z-Code after installation so it reloads the MCP configuration. The connector preserves unrelated settings, creates a timestamped backup, and disables the legacy `eye` entry. To reverse only the Sens changes:

```powershell
D:\Sens\target\release\sens-connect.exe zcode uninstall
```

Other MCP hosts can launch `sens-mcp` over stdio with the same three root variables.

## Capability settings

The desktop app separates three different jobs:

- **Connect a model or app** configures an MCP client connection.
- **Sight** and **Hearing** open the detailed settings for that capability.
- **Future senses** is a passive roadmap area; it does not create another connection.

In the native app, capability settings are loaded from the current Eye and Speech installations. Saving updates only the supported fields, preserves API keys and unrelated settings, creates a timestamped backup, and atomically replaces the configuration file.

## Layout

- `crates/` — protocol, core, broker, MCP server, connect CLI.
- `apps/desktop-ui/` — React + Vite + Tauri frontend and native shell.
- `sidecars/` — Eye and Speech worker adapters (Node and Python inference cores are reused from the existing installations; set `SENS_EYE_ROOT` and `SENS_SPEECH_ROOT` when they are not in their default locations).
- `.github/workflows/release.yml` — automated release pipeline.
