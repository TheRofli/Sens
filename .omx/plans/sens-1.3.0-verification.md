# Sens 1.3.0 verification map

| Requirement | Authoritative evidence |
|---|---|
| RGB correctness | synthetic red/green/blue palette tests and live fixture output |
| No persistent writes | before/after filesystem assertion for `noStore` |
| Reversible coordinates | crop/resize transform unit and integration tests |
| Provenance truth | schema tests plus representative `sens_see` result audit |
| Exact text/ASCII | generated ground-truth fixtures with character and whitespace exact-match metrics |
| Adaptive focus | deterministic candidate-region tests and bounded-call integration run |
| Default VLM choice | Sens-specific benchmark report with quality/latency/RSS/disk measurements |
| CPU-only behavior | runtime `usage.backend`, process configuration, and no-GPU smoke |
| Structured MCP | MCP client/Inspector discovery and invocation transcript; stdout cleanliness test |
| URL capture | local fixture server capture repeated twice with stable DOM/style/pixel evidence |
| Reconstruction loop | before/after screenshot metrics and heatmaps |
| Broker ownership | code review plus worker process lifecycle tests |
| Installer completeness | isolated install/extract smoke with no repository-path dependencies |
| New-user guide | all documented commands executed on the release candidate |
| Rust quality | `cargo fmt --all -- --check`; `cargo clippy --workspace --all-targets -- -D warnings`; `cargo test --workspace` |
| Python quality | full `pytest` suite plus benchmark smoke |
| Frontend quality | `npm ci`; `npm run test:sites`; `npm run build`; browser interaction/console check when UI changes |
| Native packaging | `npm run native:build`; inspect NSIS/MSI/updater artifacts |
| Published release | remote `main`, tag `v1.3.0`, successful Release workflow, GitHub Release assets and updater manifest |

## Completion rule

An item is complete only when the named evidence exists and directly covers its full scope. A green narrow unit test cannot prove an installer, runtime, or model-facing workflow requirement.
