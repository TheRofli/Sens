# Sens 1.3.2 first-run vision onboarding

## Outcome

A new local-provider user receives one clear offer to install Qwen3-VL 2B during first launch. The transfer is never silent, can be deferred, reports real progress, and does not block deterministic vision.

## Slice

- Add a first-run Sight dialog matching the accepted Sens visual contract.
- Persist `complete` or `later` as UI-only onboarding state.
- Report active `.part` bytes through `sight_setup_status` and poll them while installing.
- Keep the existing Vision-settings download as the recovery and deferred-install path.

## Verification

- `npm run test:ui` — 3 onboarding decision/progress tests passed.
- Focused `sens-desktop` Sight setup tests — 3 passed.
- `npm run build` and `npm run test:sites` — build succeeded and 4 tests passed.
- Playwright interaction and screenshot at 1100 x 770 — Russian dialog fits, **Later** closes it, 0 console errors and 0 warnings.
- `cargo fmt --all -- --check`, workspace clippy with warnings denied, and `cargo test --workspace` — passed.
- `D:\Speech\.venv\Scripts\python.exe -m pytest tests\sight -q` — 41 passed.
- Native 1.3.2 packaging completed: signed NSIS, MSI, and `latest.json` exist under `target/release/bundle`.

## Review verdict

Approved. The one residual risk is the external 1.45 GiB transfer itself: it was not repeated during verification. The verified downloader is unchanged; new coverage validates onboarding decisions, byte-to-progress behavior, and active `.part` file reporting.
