# CORE-1AU Memory Policy End-to-End Smoke

This stage applies the package-level memory decision policy to the CORE-1 hard evaluation stream and compares forced old recall with uncertainty plus bounded wait.

## Result

- Query count: 667
- Baseline top1: 0.9505
- Baseline false old recalls: 33
- Policy coverage: 0.9730
- Policy old-recall precision: 0.9615
- Policy false old recalls: 25
- False old recall reduction: 8
- Delayed old recalls: 82
- Released wrong: 0
- Unresolved: 18
- Policy violations: 0
- End-to-end smoke passed: 1

Next recommendation: CORE-1AV run broader seed/sequence regression for bounded-wait memory policy
