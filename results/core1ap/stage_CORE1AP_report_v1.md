# CORE-1AP Uncertainty Split Gate

This stage validates CORE-1AN/CORE-1AO uncertainty thresholding under leave-one-event-out splits. A threshold is selected on all other events, then applied to the held-out event.

## Result

- Folds: 12
- Queries: 495
- Baseline top1: 0.9535
- Baseline false count: 23
- Split coverage: 0.8465
- Split committed top1: 0.9594
- Split false suppressed: 6
- Split unnecessary uncertain: 70
- Split gate passed: 1

Next recommendation: CORE-1AQ integrate uncertainty state with split-validated threshold
