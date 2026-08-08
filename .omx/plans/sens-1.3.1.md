# Sens 1.3.1 UI cleanup

## Outcome

Normal users see one verified semantic pack: Qwen3-VL 2B. The SmolVLM and Qwen2.5 experimental packs remain available only through the internal compatibility surface. Existing `quality` and `quality_large` settings migrate to `lite` when the desktop app loads.

## Regression contract

- The Vision settings screen contains no semantic model selector.
- The Qwen status card is either ready or offers one download action.
- Both legacy pack values persistently migrate to `lite` without dropping unrelated configuration fields.
- Deterministic vision remains usable without the optional Qwen download.

## Verification

- Red: `cargo test -p sens-desktop legacy_vision_pack_is_migrated_to_recommended_qwen --lib` failed because the migration did not exist.
- Green: the focused migration test passed for `quality` and `quality_large`.
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `D:\Speech\.venv\Scripts\python.exe -m pytest tests\sight -q` — 41 passed.
- `npm run build` and `npm run test:sites` — build succeeded and 4 tests passed.
- Real Playwright browser QA — no model selector, Qwen-only card visible, 0 console errors and 0 warnings.
- `npm run native:build` — signed NSIS/MSI and updater manifest created.
- MSI administrative extraction — exit 0 and all required payload files present.

## Verdict

Accepted for the `v1.3.1` patch release.
