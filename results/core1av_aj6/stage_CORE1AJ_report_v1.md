# CORE-1AJ Expanded Observation Frontier

This stage expands the CORE-1AI frontier beyond the initial three sequences. It regenerates non-oracle observations with the A3 lower-quantile proposal profile, computes stability fields and crop descriptors, then reruns the clean-pair/hard-failure frontier.

## Result

- Selected sequences: 6
- Selected events: 9
- Selected windows: 18
- Observations: 6513
- Descriptor availability: 6513
- Best gate: S60_C40_streak2
- Best selected observations: 596
- Pair precision: positive 0.8872, negative 0.8582
- Queries: 413
- Baseline top1: 0.9540
- Baseline failures: 19
- Hard eval ready: 0
- Runtime seconds: 587.81

Next recommendation: expanded sampling still lacks clean hard pool; repair proposal/observation generation or increase sequence coverage
