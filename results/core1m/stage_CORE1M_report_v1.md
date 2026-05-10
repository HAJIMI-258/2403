# CORE-1M Assignment Pair Confidence Gate

This stage re-runs selected CORE-1J windows, mines pairs from frame assignments rather than stale active tracks, and scans online-visible confidence gates. GT is used only for pair correctness audit.

## Result

- Selected sequences: 2
- Selected windows: 6
- Assignment observations: 1043
- Matched assignment rate eval-only: 0.3866
- Best gate: A3_score050_cost_le_050
- Best positive pairs: 350
- Best negative pairs: 678
- Best positive precision eval-only: 0.6486
- Best negative precision eval-only: 0.3791
- Gate passed: 0

Next recommendation: repair objectness/tracker observation quality before encoder training
