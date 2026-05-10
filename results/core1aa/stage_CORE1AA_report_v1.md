# CORE-1AA Stability and Namespace-Aware Pair Gate

This stage reuses CORE-1Y non-oracle observations and tests whether online-visible temporal stability plus namespace-aware cross-sequence negative auditing can produce a usable self-supervised pair curriculum. GT is used only for audit.

## Result

- Observations: 4367
- Best gate: A11_score070_cost030
- Best negative mode: cross_sequence
- Selected observations: 131
- Positive / negative pairs: 64 / 1048
- Positive precision eval-only: 0.9219
- Negative precision namespace-aware eval-only: 0.9084
- Negative precision local-id eval-only: 0.7557
- Namespace precision gain: 0.1527
- Passed: 1

## Interpretation

Cross-sequence synthetic instance ids can be reused, so local-id negative precision can undercount valid negatives. The namespace-aware audit treats different sequences as different physical streams, which is the correct evaluation for cross-sequence negative mining.

Next recommendation: CORE-1AB train diagnostic online encoder on CORE-1AA stability/namespace-aware curriculum
