# CORE-1AU Memory Policy End-to-End Smoke

This stage applies the package-level memory decision policy to the CORE-1 hard evaluation stream and compares forced old recall with uncertainty plus bounded wait.

## Result

- Query count: 495
- Baseline top1: 0.9535
- Baseline false old recalls: 23
- Policy coverage: 0.9677
- Policy old-recall precision: 0.9645
- Policy false old recalls: 17
- False old recall reduction: 6
- Delayed old recalls: 60
- Released wrong: 0
- Unresolved: 16
- Policy violations: 0
- End-to-end smoke passed: 1

Next recommendation: CORE-1AV run broader seed/sequence regression for bounded-wait memory policy
