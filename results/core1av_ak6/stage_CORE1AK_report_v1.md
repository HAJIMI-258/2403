# CORE-1AK Decoupled Train/Eval Frontier

This stage tests whether CORE-1 can decouple a clean online pair-mining pool from a broader hard evaluation pool. It does not change the model and does not use GT labels for online scoring.

## Result

- Gates scanned: 75
- Train/eval combinations: 5625
- Ready combinations: 420
- Best train gate: S70_C30_streak2
- Best eval gate: S55_C50
- Train pair precision: positive 0.9588, negative 0.9375
- Train pairs: positive 97, negative 1536
- Eval queries: 667
- Eval baseline top1: 0.9505
- Eval baseline failures: 33
- Decoupled frontier ready: 1

Next recommendation: CORE-1AL train/integrate online descriptor encoder using clean train gate and hard eval gate
