# CORE-1AL Decoupled Online Metric Smoke

This stage trains a small random-initialized descriptor metric from online pseudo pairs mined in the clean CORE-1AK train pool, then evaluates it on the broader hard CORE-1AK eval pool. No pretrained weights or GT labels are used for online training/scoring.

## Result

- Train gate: S70_C30_streak2
- Eval gate: S55_C50
- Train pairs: 1284
- Pair precision: positive 0.9474, negative 0.9205
- Learned metric test AUC: 1.0000
- Shuffled-label metric test AUC: 0.5175
- Baseline top1: 0.9535
- Best variant: A8_learned_fusion_w020 top1 0.9596
- Best learned variant: A8_learned_fusion_w020 top1 0.9596
- Control best top1: 0.9596
- Learned metric passed: 0

Next recommendation: learned online metric does not beat baseline/controls on hard eval; inspect raw descriptor vs metric failure before integration
