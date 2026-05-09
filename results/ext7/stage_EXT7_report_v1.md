# EXT-7 Frozen Embedding External Baseline

## Result

- Embedding model: `resnet18`
- Device: `cuda`
- Events: `504`
- Geometry passive top1: `0.5913`
- External branch top1: `0.6786`
- Embedding NN top1: `0.5496`
- Geometry + embedding best top1: `0.6151`
- External branch + embedding best top1: `0.6786`
- Mean embedding margin: `0.1277`
- Controls passed: `0`
- Significance passed: `1`

## Decision

do not use embedding fusion; keep as diagnostic external baseline

Frozen embeddings are recorded as an external pretrained baseline only, not as the NOPS main method.
