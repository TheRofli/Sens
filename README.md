# Sens 1.0

Sens is a local capability layer for language models. It combines visual perception from Eye and audio transcription from Speech behind one Rust broker, one MCP connection, one tray application, and one capability contract.

## What ships in 1.0

- `sens-desktop` — the Tauri desktop and tray experience.
- `sens-broker` — the single-owner named-pipe runtime and lazy worker supervisor.
- `sens-mcp` — a standard stdio MCP server with 15 Sens and Eye-compatible tools.
- `sens-connect` — reversible connector setup for Z-Code.
- `sens-protocol` and `sens-core` — versioned contracts, registry, state, policy, and execution.

Eye's Node inference core and Speech's Python inference core remain isolated lazy workers. Sens bundles their adapters, but it intentionally uses the existing Eye and Speech installations in 1.0 instead of duplicating the model runtimes. Set `SENS_EYE_ROOT` and `SENS_SPEECH_ROOT` when they are not in their default locations.

Stored visual result files are written to `%LOCALAPPDATA%\Sens\artifacts\sight-results`; set `SENS_ARTIFACTS_ROOT` to override that writable location.

## Capability settings

The desktop app separates three different jobs:

- **Connect a model or app** configures an MCP client connection.
- **Sight** and **Hearing** open the detailed settings for that capability.
- **Future senses** is a passive roadmap area for later modules; it does not create another connection.

In the native app, capability settings are loaded from the current Eye and Speech installations. Saving updates only the supported fields, preserves API keys and unrelated settings, creates a timestamped backup, and atomically replaces the configuration file. The browser preview keeps changes in memory so the complete interaction can be reviewed without touching the user's live configuration.

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

- `D:\Sens\target\release\bundle\nsis\Sens_1.0.0_x64-setup.exe`
- `D:\Sens\target\release\bundle\msi\Sens_1.0.0_x64_ru-RU.msi`

## Connect Z-Code

```powershell
$env:SENS_EYE_ROOT = 'C:\Users\kanal\.zcode\workspace\default\eye'
$env:SENS_SPEECH_ROOT = 'D:\Speech'
D:\Sens\target\release\sens-connect.exe zcode install `
  C:\Users\kanal\.zcode\cli\config.json `
  D:\Sens\target\release\sens-mcp.exe
```

Restart Z-Code after installation so it reloads the MCP configuration. The connector preserves unrelated settings, creates a timestamped backup, and disables the legacy `eye` entry to prevent two servers from owning the same capability. To reverse only the Sens changes:

```powershell
D:\Sens\target\release\sens-connect.exe zcode uninstall
```

Other MCP hosts can launch `sens-mcp` over stdio with the same three root variables. Automatic connector presets for those hosts are planned after the Z-Code-first 1.0 validation pass.
