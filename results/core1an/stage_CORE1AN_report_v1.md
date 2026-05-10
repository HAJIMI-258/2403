# CORE-1AN Uncertainty Abstention Gate

This stage does not change retrieval scoring. It tests whether online-visible baseline top1 margin can identify uncertain memory recalls and abstain instead of forcing a false old-object match.

## Result

- Eval gate: S55_C50
- Queries: 495
- Baseline top1: 0.9535
- Baseline false count: 23
- Best threshold: 0.0194
- Coverage: 0.8465
- Committed top1: 0.9594
- False retrievals avoided: 6
- Correct abstained: 70
- Random false avoided mean: 3.67
- Uncertainty gate passed: 1

Next recommendation: CORE-1AO integrate uncertainty/abstention state into object-file memory audit
