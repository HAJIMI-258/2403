# Phase 3P Final Summary

## Scope

Phase 3P only targeted strict prototype continuity inside an already-correct lineage. It did not change objectness, proposal generation, tracker active matching, continuation cost, or the lineage backbone.

## Stage A: Audit Result

The Track C re-entry event slice was not dominated by new sibling birth or archived sibling reactivation.

- `A.matched_lineage_kept_current_head = 8/17`
- `B.matched_lineage_replaced_with_active_sibling = 4/17`
- `F.no_valid_lineage_match = 4/17`
- `G.proposal_missing_or_upstream_missing = 1/17`
- `archived_sibling_reactivation_rate = 0.0000`
- `new_sibling_birth_rate_given_concept_recovery = 0.0000`

Within the matched-lineage subset:

- `head_keep_rate_given_matched_lineage = 0.6667`
- `head_replacement_rate_given_matched_lineage = 0.3333`
- `active_sibling_win_rate = 0.3333`

Lineage-level aggregates also showed very high head churn:

- `lineage 0 head churn = 752`
- `lineage 1 head churn = 851`
- `lineage 3 head churn = 106`

The audit conclusion was: Phase 3L strict prototype failure was mainly a head continuity problem, not a birth explosion inside the Track C re-entry event slice.

## Ablation Result

The best ablation selected by the current Phase 3P score was `keep_head_plus_grouped`.

### Track C: Phase 3L vs Phase 3P best

| metric | Phase 3L | Phase 3P best |
|---|---:|---:|
| `same-prototype` | 0.6471 | 0.8235 |
| `PFR` | 5.3333 | 4.6667 |
| `IDSW` | 375 | 371 |
| `same-track-after-concept` | 0.4545 | 0.2143 |
| `same-track` | 0.3529 | 0.1765 |
| `same-lineage-prototype` | 0.7059 | 0.8235 |
| `lineage mismatch rate` | 0.2500 | 0.1250 |
| `continuation access given concept recovery` | 0.7273 | 0.1429 |

### Track A: safety check

| metric | Phase 3L | Phase 3P best |
|---|---:|---:|
| `U-Recall` | 0.7878 | 0.7878 |
| `same-prototype` | 1.0000 | 0.0000 |
| `PFR` | 3.5000 | 2.5000 |
| `IDSW` | 133 | 132 |
| `memory_growth` | 0.0031 | 0.0021 |

## Decision

Phase 3P does not pass.

It succeeded on the first-priority strict prototype continuity metric for Track C:

- `same-prototype` crossed the target line: `0.8235`

It also improved:

- `same-lineage-prototype`
- `lineage mismatch`
- `PFR` slightly

But it failed the round because it broke the recovery path:

- `same-track-after-concept` regressed sharply
- `same-track` regressed sharply
- `continuation_bank_access_rate_given_concept_recovery` collapsed
- Track A strict `same-prototype` also collapsed

So the current policy is over-constrained. It stabilizes head identity by routing fewer recovery events into the continuation path.

## Final Interpretation

Phase 3P proved that stricter lineage-internal selector gating can recover strict prototype continuity, but the current implementation ties head preservation too tightly to recovery attachment. The next bottleneck is not raw head churn alone. It is the lack of a clean separation between:

- stable lineage/head identity
- temporary recovery attachment inside that lineage
- later head promotion

Until those three are decoupled, stronger head stabilization will keep trading away recovery quality.
