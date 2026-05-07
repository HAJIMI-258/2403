# Phase 3P Recommendation

## Decision

Do not exit Phase 3.

Do not move to Track B.

Do not move to real data or a larger benchmark.

## Reason

Phase 3P improved Track C strict prototype continuity, but only by sacrificing recovery behavior.

The best ablation:

- raised `same-prototype` from `0.6471` to `0.8235`
- lowered `PFR` from `5.3333` to `4.6667`
- lowered `lineage mismatch` from `0.2500` to `0.1250`

But it also:

- reduced `same-track-after-concept` from `0.4545` to `0.2143`
- reduced `same-track` from `0.3529` to `0.1765`
- reduced `continuation access given concept recovery` from `0.7273` to `0.1429`
- collapsed Track A strict `same-prototype` from `1.0000` to `0.0000`

That tradeoff is not acceptable for leaving Phase 3.

## What This Means

The current bottleneck is no longer just lineage mismatch or head churn by itself.

The selector is now too tightly coupling:

1. head preservation
2. recovery attachment
3. head promotion

Those three decisions need to be separated.

## Recommended Next Fix

The next round should keep the Phase 3L lineage backbone and continuation access path intact, but decouple prototype-head stabilization from recovery routing.

Concretely:

- recovery inside a matched lineage should be allowed to attach to a non-head sibling or continuation-backed candidate without immediately switching the lineage head
- head promotion should require accumulated evidence over time, not a single recovery event
- birth suppression should stay secondary, because Stage A showed that Track C re-entry failures were not dominated by sibling birth in the audited event slice

## Explicit Boundary

The next round should still not:

- retune `field.py`
- retune continuation cost
- add new slot or bank variants
- move to Track B
- move to real data
- expand the benchmark

The next round should remain inside Phase 3 and target:

- decoupled recovery attach vs head promotion
- stable strict prototype continuity without starving continuation access
