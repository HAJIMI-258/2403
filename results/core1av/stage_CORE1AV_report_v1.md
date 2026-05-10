# CORE-1AV Broader Sequence Regression

This stage reruns the CORE-1 uncertainty/bounded-wait policy on a broader six-sequence observation frontier. It does not change the policy threshold or scoring rule.

## Six-sequence result

- Selected sequences: 6
- Observations: 6513
- Decoupled frontier ready: 1
- Split gate passed: 1
- Queries: 667
- Baseline top1: 0.9505
- Policy precision: 0.9615
- Policy coverage: 0.9730
- False old recall reduction: 8
- Released wrong: 0
- Unresolved: 18
- Broader regression passed: 1

Next recommendation: CORE-1AW integrate bounded-wait policy into main evaluation harness behind a disabled-by-default flag
