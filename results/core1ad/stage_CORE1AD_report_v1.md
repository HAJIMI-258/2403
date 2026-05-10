# CORE-1AD Medium-Confidence Descriptor Gate

This stage broadens CORE-1AC beyond the high-confidence saturated set. It scans medium/high confidence observation gates and tests whether raw descriptor cue can rescue baseline retrieval failures without control failures.

## Result

- Gates scanned: 7
- Observations: 4367
- Descriptor availability: 4367
- Best gate: A9_score060_cost040_consecutive
- Best variant: A6_gated_fusion_w020_margin005
- Best gate queries: 297
- Gate baseline top1: 0.9562
- Best top1: 0.9630
- Delta vs gate baseline: 0.0067
- Improved / regressed: 2 / 0
- Controls passed: 1
- Safe for integration smoke: 1

Next recommendation: CORE-1AE run selected descriptor gate against broader internal focus/anchor regression guards
