# CORE-1AN Uncertainty Abstention Gate

This stage does not change retrieval scoring. It tests whether online-visible baseline top1 margin can identify uncertain memory recalls and abstain instead of forcing a false old-object match.

## Result

- Eval gate: S55_C50
- Queries: 667
- Baseline top1: 0.9505
- Baseline false count: 33
- Best threshold: 0.0195
- Coverage: 0.8486
- Committed top1: 0.9558
- False retrievals avoided: 8
- Correct abstained: 93
- Random false avoided mean: 5.17
- Uncertainty gate passed: 1

Next recommendation: CORE-1AO integrate uncertainty/abstention state into object-file memory audit
