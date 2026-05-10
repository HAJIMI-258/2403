# CORE-1D Stable Query Positive Diagnostic

CORE-1D tests whether the split-validated CORE-1C stable query positives are sufficient to train a query-memory alignment branch.
This is diagnostic only and does not integrate the encoder into main NOPS.

## Result
- Stable query-positive events: 3 (M-RE-TC-009, M-RE-TC-012, M-RE-TC-013).
- Passive baseline top1: 0.4118.
- Best online alignment ablation: A0_current_NOPS_passive, top1=0.4118.
- Frozen random top1: 0.0000.
- Passed minimum: 0.

## Decision
CORE-1E generate more real query positives; stable positives are too sparse for reliable online encoder integration
