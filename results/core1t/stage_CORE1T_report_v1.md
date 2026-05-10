# CORE-1T Oracle Matched-Observation Upper Bound

This diagnostic uses GT only to remove unmatched observations and estimate whether matched-observation filtering would make pair mining viable. It is not a main-method training setup.

## Result

- Best oracle filter: A1_oracle_matched_iou25
- Best gate: A0_assignment_only
- Best observations: 431
- Best positive precision eval-only: 0.9857
- Best negative precision eval-only: 0.9407
- Upper bound passed: 1

Next recommendation: CORE-1U learn/design GT-free matched-observation confidence target
