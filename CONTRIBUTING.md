# Contributing to Sens

Sens is currently owner-maintained. Reproducible bug reports, difficult visual
fixtures, performance measurements, documentation corrections, and focused
feature proposals are welcome.

## Before opening an issue

- Search existing issues and the latest release notes.
- Reproduce the problem on the latest Sens release when possible.
- Include Sens version, Windows version, CPU, RAM, selected capability/model,
  exact steps, expected result, and actual result.
- Remove API keys, IPC secrets, private file paths, confidential screenshots,
  cookies, and personal audio from logs or attachments.
- For visual reconstruction, attach both the reference and candidate plus the
  final `sens_review` result when it is safe to share them.

## Code contributions

Do not start a large pull request without opening a proposal first.

Because Sens is offered under a public noncommercial license while the
copyright holder also maintains commercial licensing rights, external code
cannot be merged casually under ambiguous terms. A code contribution requires
prior written agreement on contribution rights before it can be accepted. An
unsolicited pull request may be used for discussion but will not be merged
until that agreement exists.

This policy keeps the ownership chain clear for both community users and future
commercial products. It does not apply to issue reports or ideas that contain
no contributed code.

## Development gates

For an agreed code change, keep the Rust broker as the only owner of workers and
mutable runtime state, preserve protocol-only stdout in `sens-mcp`, and keep
secrets out of arguments and diagnostics.

Run the relevant tests and always complete the Rust gates:

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

$env:PYTHONPATH = "sidecars"
python -m pytest tests\sight -q

Set-Location apps\desktop-ui
npm run test:ui
npm run test:sites
npm run build
```

## License questions

See [LICENSING.md](LICENSING.md). For commercial licensing, contact
[TheRofli through GitHub](https://github.com/TheRofli) without posting
confidential details in a public issue.
