# CORE-1Y Expanded Cross-Event Negative Mining

This stage regenerates assignment observations for a larger selected-window set and re-runs cross-context negative mining. It still does not train an encoder.

## Result

- Selected sequences: 3
- Selected events: 6
- Selected windows: 12
- Assignment observations: 4367
- Positive pairs: 297
- Best negative mode: cross_sequence
- Best negative pair count: 2622
- Best positive precision eval-only: 0.8956
- Best negative precision eval-only: 0.7727
- Passed: 0
- Runtime seconds: 185.96

Next recommendation: expand beyond 3 rendered sequences or switch to oracle-proposal diagnostic encoder; current objectness-derived negatives remain too noisy
