# EXT-7 Frozen Embedding External Baseline

## Result

- Embedding model: `resnet18`
- Device: `cuda`
- Events: `234`
- Geometry passive top1: `0.5513`
- External branch top1: `0.5940`
- Embedding NN top1: `0.5171`
- Geometry + embedding best top1: `0.5726`
- External branch + embedding best top1: `0.5855`
- Mean embedding margin: `0.0988`
- Controls passed: `0`
- Significance passed: `0`

## Decision

do not use embedding fusion; keep as diagnostic external baseline

Frozen embeddings are recorded as an external pretrained baseline only, not as the NOPS main method.
