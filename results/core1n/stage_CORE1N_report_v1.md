# CORE-1N Oracle Proposal Pair Upper Bound

This diagnostic uses GT boxes as oracle proposals inside selected windows. It is not a main-method training setup. Its purpose is to isolate whether CORE-1M failed because of objectness/proposal noise or because tracker pair mining is intrinsically unreliable.

## Result

- Proposal mode: oracle GT box memory-only
- Selected sequences: 2
- Selected windows: 6
- Assignment observations: 474
- Positive pairs: 450
- Negative pairs: 559
- Positive precision eval-only: 1.0000
- Negative precision eval-only: 1.0000
- Upper bound passed: 1

Next recommendation: front-end objectness/proposal repair is the blocker; oracle proposals make pair mining viable
