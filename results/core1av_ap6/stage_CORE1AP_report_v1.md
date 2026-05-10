# CORE-1AP Uncertainty Split Gate

This stage validates CORE-1AN/CORE-1AO uncertainty thresholding under leave-one-event-out splits. A threshold is selected on all other events, then applied to the held-out event.

## Result

- Folds: 16
- Queries: 667
- Baseline top1: 0.9505
- Baseline false count: 33
- Split coverage: 0.8486
- Split committed top1: 0.9558
- Split false suppressed: 8
- Split unnecessary uncertain: 93
- Split gate passed: 1

Next recommendation: CORE-1AQ integrate uncertainty state with split-validated threshold
