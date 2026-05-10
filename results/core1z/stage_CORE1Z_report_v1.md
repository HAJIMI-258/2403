# CORE-1Z Oracle-Proposal Diagnostic Encoder

This stage is a diagnostic upper bound. It uses GT boxes only as oracle proposals inside selected synthetic windows, then tests whether same-space crop descriptors contain learnable pair/retrieval signal. It is not safe for main online training.

## Result

- Proposal mode: oracle GT box memory-only
- Observations: 474
- Positive / negative pairs: 450 / 559
- Pair precision eval-only: positive 1.0000, negative 1.0000
- Raw descriptor test AUC: 0.9273
- Learned diagnostic metric test AUC: 0.9702
- Shuffled-label metric test AUC: 0.6730
- Raw / learned retrieval top1: 0.9533 / 0.9467
- Raw / learned retrieval margin: 0.1771 / 0.3782
- Negative controls passed: 1
- Diagnostic upper bound passed: 1

## Interpretation

If this stage passes, clean oracle observations contain enough descriptor signal and the blocker remains non-oracle observation quality. If it fails, the descriptor/metric itself is too weak even before objectness/tracker noise.

Next recommendation: encoder/descriptor signal exists under clean oracle proposals; fix non-oracle observation quality before CORE-2
