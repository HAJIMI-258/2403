# Phase 3D Stage A.5 Claim Preservation Summary

- target event: `6`
- target frame: `990`
- target lineage: `2`

## Baseline

- target lineage visible: `0`
- claim drop stage: `None`
- claim drop reason: `None`
- final selected lineage: `0`

## Claim Preservation Only

- target lineage visible: `0`
- target lineage rank: `None`
- final selected lineage: `0`

## Claim Preservation + Identity Tie-Break

- target lineage visible: `0`
- target lineage rank: `None`
- final selected lineage: `0`
- identity tiebreak applied: `0`

## Direct Answers

1. Under the current baseline, target lineage 2 never enters the final claim set at frame 990.
2. The minimal preserve rule did not make target lineage 2 visible, which means the preserve input is still following the wrong runtime lineage hints rather than the expected continuity lineage.
3. The identity-aware tie-break never got a chance to act on target lineage 2 because the competition remained inside lineage 0 / 1 only.
