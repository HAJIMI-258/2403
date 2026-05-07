# NOPS-OWR Phase 1 Go / No-Go Summary v1

Date: 2026-04-07

## Current Verdict

**Go, but only as a constrained Phase 2 continuation.**

The minimum evidence chain is now live:

`synthetic stream -> objectness -> temporal identity -> prototype memory -> metrics and visualization`

The project has cleared the minimum bar for continued investment, but it has **not** cleared the bar for broadening scope too aggressively. The hard synthetic scenario still exposes a real weakness in objectness and downstream stability.

## The 4 Required Questions

### 1. Can object-like regions be extracted?

**Yes.**

- Day 3 produced a working `frame -> spike response -> objectness heatmap -> proposals` pipeline.
- In the Day 7 `easy_single_object` scenario, `U-Recall = 1.0000`.
- In the Day 7 `multi_object_reentry` scenario, `U-Recall = 0.8930`.

### 2. Can the same object keep a stable identity over time?

**Yes.**

- In the representative Day 4 run (`seq_003`), `tracked_idsw = 18`.
- The no-temporal-identity comparison for the same run produced `naive_idsw = 915`.
- This is a very large reduction and is strong evidence that the temporal identity module is doing real work.

### 3. Do prototypes form instead of being recreated every frame?

**Partially yes.**

- In the representative Day 5 run (`seq_003`), the system ended with `3` prototypes and `memory_growth = 0.0040`.
- In the Day 7 `multi_object_reentry` scenario, `final_prototypes = 3`, `PFR = 0.0000`, and `IDSW = 0`.
- Re-entry reconnection exists, but is not yet strong enough: one representative run reached only `0.5000`, below the desired `>= 0.60` target.

### 4. Does memory stay controlled?

**Yes.**

- Day 5 representative runs stayed at `3-4` final prototypes, not anywhere close to sequence length.
- `memory_growth` stayed around `0.0040-0.0048`, comfortably below the target `< 0.05`.
- Baseline 1 grew much faster: `memory_growth = 0.2691` in the Day 6 comparison.

## Baseline Comparison

### Baseline 1: Frame Difference + Connected Components

The current minimal NOPS-OWR pipeline clearly beats this baseline on the most important long-horizon stability metrics:

- much lower `PFR`
- much lower `IDSW`
- much lower `memory_growth`

### Baseline 2: Edge Proposals + Online Clustering

This baseline remains competitive and should be treated as a real residual risk.

- It is still weaker on memory growth.
- It does not clearly lose on every metric.
- It shows that our current objectness + memory stack is promising, but not yet dominant.

## What Is Over The Line

- Objectness is real, not imaginary.
- Temporal identity is real, not cosmetic.
- Prototype memory is controllable and does not explode.
- The minimum engineering chain is now testable and inspectable end to end.

## What Is Still Not Over The Line

- Hard drift + occlusion still hurts recall badly.
- Prototype re-entry reconnection is not consistently strong enough yet.
- Baseline 2 is still too close for comfort.

## Recommended Phase 2 Constraint

Proceed to Phase 2 only under this order:

1. Improve objectness under hard drift and occlusion.
2. Strengthen prototype re-entry reconnection.
3. Re-run baseline comparisons after those fixes.
4. Only then expand to more realistic frame-stream benchmarks.

Do **not** jump to event streams, concept graphs, or full theory work yet.
