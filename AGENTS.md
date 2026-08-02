# Sens development rules

- Keep the Rust broker as the only owner of capability workers and mutable runtime state.
- Keep stdout clean in `sens-mcp`; protocol output only. Diagnostics go to stderr.
- Never pass API keys or IPC secrets through command-line arguments, logs, activity records, or diagnostic exports.
- Eye and Speech are brownfield capability workers. Preserve their public behavior and tests before refactoring internals.
- Agent audio transcription must not write to clipboard, paste text, or save history unless a user-facing flow explicitly requests it.
- A model-controlled microphone or screen capture requires explicit, visible consent and is out of scope for Sens 1.0.
- Every capability result uses the shared request/result envelope and distinguishes observed, measured, and inferred data.
- Run `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` before declaring a Rust slice complete.
- Chat screenshots: Z-Code saves pasted images as base64 data URIs in `~/.zcode/cli/artifacts/<session>/prompt-attachment-upload-*.txt`. When the user attaches a screenshot, decode it with `node scripts/extract-chat-images.mjs` into `qa/incoming/` and review it with the Sens Eye tools (`sens_see`, `sens_inspect`), not with `Read`.

