# Phase 3D Stage A.4 Candidate Ranking Summary

- target event: `6`
- target frame: `990`
- target lineage: `2`

## Baseline Reroute

- final selected lineage: `0`
- final selected source: `dormant`
- target-lineage matched: `0`

## Claim Comparison

- claim winner lineage: `0`
- claim winner score: `0.7239339546990934`
- target-lineage claim score: `None`

## Forced Two-Stage Probe

- final selected lineage: `1`
- final selected source: `recovery_anchor`
- target-lineage matched: `0`
- target same-track hint: `0`
- target same-prototype hint: `0`
- target-lineage visible in forced candidate set: `1`

## Direct Answers

1. Under the baseline reroute trace, the correct target lineage `2` is not visible in the consumer claim set at frame `990`.
2. Under the forced two-stage probe, lineage `2` does become visible again, but it still does not win the final recovery selection.
3. The current bottleneck is therefore split: baseline still suppresses the target-lineage claim, and even after restoring visibility the lineage-first selection still lands on lineage `1`, not the intended lineage `2`.
