# CORE-1AB Non-Oracle Curriculum Encoder

This stage trains a diagnostic descriptor metric on the CORE-1AA non-oracle stability/namespace-aware curriculum. It uses no oracle proposals, no pretrained weights, and GT only for audit.

## Result

- Gate: A11_score070_cost030
- Negative mode: cross_sequence
- Selected observations: 131
- Positive / negative pairs: 64 / 1048
- Pair precision eval-only: positive 0.9219, negative 0.9084
- Raw descriptor test AUC: 0.9821
- Learned metric test AUC: 0.9520
- Shuffled-label metric test AUC: 0.5480
- Raw / learned retrieval top1: 0.9444 / 0.8889
- Raw / learned retrieval margin: 0.4253 / 0.7086
- Negative controls passed: 0
- Raw descriptor signal passed: 1
- Diagnostic encoder passed: 0

Next recommendation: CORE-1AC run conservative raw-descriptor memory integration smoke; learned metric underperforms raw descriptor
