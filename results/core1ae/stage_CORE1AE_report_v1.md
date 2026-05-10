# CORE-1AE Descriptor Gate Robustness Audit

This stage audits the CORE-1AD selected descriptor gate with paired bootstrap and shuffled/wrong/random controls. It is a decision gate, not a new model.

## Result

- Gate: A9_score060_cost040_consecutive
- Variant: A6_gated_fusion_w020_margin005
- Queries: 297
- Baseline top1: 0.9562
- Selected top1: 0.9630
- Delta top1: 0.0067
- 95% bootstrap CI: [0.0000, 0.0168]
- Improved / regressed: 2 / 0
- Best control delta: 0.0034
- CI excludes zero: 0
- Beats controls: 1
- Robustness gate passed: 0

Next recommendation: do not integrate descriptor cue; CORE-1AD gain is too small or not robust against controls
