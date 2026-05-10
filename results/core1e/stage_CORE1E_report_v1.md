# CORE-1E Pseudo-Reentry Curriculum

This stage creates self-supervised pseudo re-entry query-memory pairs from stable object-file continuity. It does not use GT identity for online training or scoring. GT fields are only used to audit pair correctness.

## Result

- Pseudo positives: 244
- Pseudo negatives: 248
- Hard negatives: 248
- Positive precision, eval-only: 0.9590
- Negative precision, eval-only: 0.9677
- Pair mining passed: 1
- Official main NOPS baseline top1: 0.4118
- Dense diagnostic baseline top1: 0.9530
- Best pseudo encoder variant: A4_NOPS_plus_pseudo_encoder_w003
- Best pseudo encoder diagnostic top1: 0.9530
- Frozen random diagnostic top1: 0.2437
- Shuffled control diagnostic top1: 0.9538
- Negative controls passed: 0
- Passed minimum: 0

## Interpretation

CORE-1E tests whether stable object-file continuity can generate enough query-memory positives to train alignment. The retrieval numbers marked diagnostic are measured on the dense internal observation cache, not merged into the official M-RE bundle retrieval path. Official focus and anchor metrics remain guarded by the baseline because this branch is not integrated unless controls pass.

Next recommendation: encoder objective/control gap failed; inspect pair weighting and descriptor input before integration
