# Sens 1.3.7 verification map

| Requirement | Authoritative evidence | Gate |
| --- | --- | --- |
| Divider survives as one structure | Summer Drive measured skeleton fixture | Segment has source endpoints spanning the reference divider |
| Tickets are control candidates | Immutable source integration fixture | Exactly two labelled outline candidates at the visible boxes |
| Poster glyphs are not controls | Existing + extended synthetic fixtures | No false controls for plain saturated typography |
| Web contract forbids text assets | ReconstructionSpec unit test | `liveTextRequired=true`, `rasterTextAllowed=false`, no custom-font asset strategy |
| Raw element cannot contradict doc | Element operation regression | Reconciled role/warning matches ReconstructionSpec |
| Real web page is semantic | Browser fixture + `sens_review` | Live/selectable text and semantic controls pass |
| Raster clone is rejected | Browser fixture built from source slices | Web verdict fails with raster-text/source-slice blockers even if visual pass is true |
| Image compare remains compatible | Existing compare suite | Visual metrics/schema remain valid and scope is explicit |
| Compact MCP is not duplicated | Rust `CallToolResult` test + measured payload | Full JSON appears once; textual content is bounded |
| Parallel VLM calls fail fast | Rust concurrency/backpressure test | Second local request receives recoverable busy, not a hidden timeout |
| Public MCP surface is additive | Rust tool/schema tests + real MCP list | Existing tools retained; `sens_review` and `targetKind` visible |
| noStore is honest | Python capture/review storage test | No persistent screenshot/asset/review directory remains |
| Installed runtime is current | Hash/version/status after deployment | Sidecar and binaries match release build; connector enabled |
| End-to-end host behavior | Fresh Z-Code session + rollout + final DOM | No text crops; live text, two controls, divider, combined pass |
| Dense dashboard generalization | Dub screenshot + fresh Z-Code session | Small text/cards/nav/chart remain decomposed into appropriate live DOM and graphics |
| Photo-led page generalization | Beyond Human Wear screenshot + fresh Z-Code session | Hero photo stays an allowed asset while copy, nav, pills and controls remain live |
| ASCII reconstruction | Hyperstudio screenshot + fresh Z-Code session | Character art is exact selectable monospace content, not img/canvas/SVG paths |
| Live URL generalization | timestamped Hungry Tiger, dope.security and Caldera baselines | Each frozen state receives independent visual/web pass and interaction audit |
| Performance loop is bounded | Rollout parser report | Calls/tokens/time and post-pass work are recorded; no unbounded tail |
| Release is published | Git/GitHub/updater checks | main commit, `v1.3.7`, release assets, metadata agree |

## Commands

```powershell
C:\Users\kanal\AppData\Local\Sens\runtime\python\python.exe -m pytest tests/sight -q
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --release -p sens-broker -p sens-mcp -p sens-connect
```

The runtime and Z-Code commands are recorded with their exact session ID,
workspace, reference hash, prompt hash, deployed binary hashes, and rollout
path in the 1.3.7 acceptance report.
