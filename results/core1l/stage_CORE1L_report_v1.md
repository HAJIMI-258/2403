# CORE-1L Windowed Tracker Pair Mining Smoke

This stage runs the tracker only inside selected CORE-1J windows. Tracker state is cold-started per window, so this is a pair-mining feasibility test, not a long-identity evaluation.

## Result

- Selected sequences: 2
- Selected windows: 6
- Positive adjacent-track pairs: 1344
- Negative co-visible different-track pairs: 6535
- Positive precision eval-only: 0.2396
- Negative precision eval-only: 0.0678
- Mean matched observation rate: 0.2908
- Runtime seconds: 150.30
- Pair mining passed: 0

Next recommendation: repair tracker pair mining confidence gates before encoder training
