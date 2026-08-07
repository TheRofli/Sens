# Sens reconstruction loop acceptance

Measured on 2026-08-07 with `sens-multisignal-compare-v2` against the immutable Hyperstudio reference `qa/incoming/test1b/copy3-1x.png` (2554×1271).

| Signal | Early candidate `render-v8` | Repaired candidate `render-v16` | Direction |
|---|---:|---:|---|
| Overall similarity | 0.7598 | 0.8749 | +0.1151 |
| Pixel mismatch | 0.1038 | 0.0751 | lower is better |
| Mean Lab color delta | 15.285 | 5.941 | lower is better |
| Edge mismatch | 0.0349 | 0.0209 | lower is better |
| OCR text similarity | 0.6822 | 0.8310 | higher is better |
| Layout box IoU similarity | 0.3827 | 0.6883 | higher is better |

The repaired candidate improves every measured signal. The largest remaining hot region is source-pixel box `[120,606,1084,671]`; `sens_compare` now returns it as an executable `sens_zoom` next action for the next repair iteration.

This evidence proves the intended workflow: reference scene → implementation → deterministic compare → focused repair → improved compare. The comparison does not claim aesthetic equivalence; it reports measured convergence.
