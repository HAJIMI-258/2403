# EXT-7 Frozen Embedding External Baseline

## Result

- Embedding model: `resnet18`
- Device: `cuda`
- Events: `406`
- Geometry passive top1: `0.5739`
- External branch top1: `0.6601`
- Embedding NN top1: `0.5369`
- Geometry + embedding best top1: `0.5985`
- External branch + embedding best top1: `0.6626`
- Mean embedding margin: `0.1136`
- Controls passed: `0`
- Significance passed: `0`

## Decision

do not use embedding fusion; keep as diagnostic external baseline

Frozen embeddings are recorded as an external pretrained baseline only, not as the NOPS main method.
