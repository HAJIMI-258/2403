# CORE-1AS Bounded Wait Policy

This stage turns CORE-1AR's delayed resolution audit into a concrete bounded-wait policy: uncertain recalls wait up to 10 frames for a high-margin observation on the same online-visible track.

## Result

- Uncertain decisions: 76
- Resolved correct: 60
- Released wrong: 0
- Unresolved after horizon: 16
- Resolution rate: 0.7895
- False-suppressed cases resolved: 2
- Correct delayed cases resolved: 58
- Mean wait frames: 1.68
- Policy violations: 0
- Bounded wait policy passed: 1

Next recommendation: CORE-1AT write core memory decision spec and add regression tests
