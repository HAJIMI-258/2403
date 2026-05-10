# CORE-1AC Raw Descriptor Memory Integration Smoke

This stage is a non-invasive integration smoke. It reads CORE-1AB non-oracle descriptors and tests whether a conservative raw descriptor cue can improve a track/recency memory baseline. It does not modify the main NOPS retrieval stack.

## Result

- Queries: 18
- Baseline top1: 1.0000
- Best variant: A0_track_recency_baseline
- Best top1: 1.0000
- Best false retrieval rate: 0.0000
- Best mean margin: 0.9091
- Improved / regressed: 0 / 0
- Descriptor controls passed: 1
- Baseline saturated: 1
- Safe for integration smoke: 0

Next recommendation: CORE-1AD broaden to medium-confidence observations; CORE-1AC high-confidence set is baseline-saturated
