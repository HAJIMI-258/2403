# CORE-1A Query-Memory Same-Space Alignment

This stage addresses the CORE-1 failure mode: bundle-bundle representation learning did not align re-entry query cues with memory embeddings.

- Query pair mining passed: `0`
- Event positive pairs: `4`
- Event positive precision eval-only: `0.25`
- Baseline top1: `0.4117647058823529`
- Alignment sim-only top1: `0.0`
- Best ablation: `A0_current_NOPS_passive`
- Best top1: `0.4117647058823529`
- Negative controls passed: `0`
- Passed minimum: `0`
- Next recommendation: `repair query-memory pair gate`
