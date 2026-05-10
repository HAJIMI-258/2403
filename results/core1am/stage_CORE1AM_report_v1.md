# CORE-1AM Metric Control Significance

This stage repeats the CORE-1AL hard evaluation with multiple shuffled descriptor controls. It asks whether the learned online metric beats random perturbation, not just the baseline.

## Result

- Baseline top1: 0.9535
- Learned top1: 0.9596
- Learned delta: 0.0061
- Control best top1: 0.9596
- Control mean top1: 0.9565 +/- 0.0018
- Control permutation rate: 0.1190
- Significance passed: 0
- Control significance passed: 0

Next recommendation: do not integrate learned metric; improvement is not statistically/control separated from shuffled descriptor perturbations
