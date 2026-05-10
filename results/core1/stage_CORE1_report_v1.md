# CORE-1 Online Self-Supervised Object-File Representation Learning

This stage tests whether NOPS can learn a no-pretrain object memory representation from online object-file continuity.

## Pair Mining

- Positive pairs: `2509`
- Negative pairs: `2590`
- Positive precision eval-only: `1.0`
- Negative precision eval-only: `1.0`
- Pair mining passed: `1`

## Retrieval

- Baseline top1: `0.4117647058823529`
- Best ablation: `A0_current_NOPS_passive`
- Best top1: `0.4117647058823529`
- Frozen random top1: `0.0`
- Focus success count: `3`
- Negative controls passed: `0`

## Decision

- Passed minimum: `0`
- Next recommendation: `query-memory alignment failed; learned bundle embeddings do not retrieve re-entry cues better than random`
