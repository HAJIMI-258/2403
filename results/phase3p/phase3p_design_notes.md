# Phase 3P Design Notes

## Goal

Phase 3P only targeted one problem:

- stabilize strict prototype continuity inside an already-correct lineage

It explicitly did not change:

- objectness
- proposal extraction
- tracker main loop
- continuation cost formula
- lineage backbone

## Stage A Audit

Added selector-level audit fields to the prototype assignment path so each Track C re-entry event could be classified by lineage-internal action:

- `keep_head`
- `replace_head`
- `reactivate_archived`
- `birth_sibling`
- `birth_lineage`
- `miss`

The audit also exported:

- head score
- best active sibling score
- best archived sibling score
- margin versus current head
- head churn per lineage
- action-attributed ID switch counts

## Strategy Changes

Phase 3P added only selector-side constraints.

### 1. Keep-head default

If the current head was still acceptable, it stayed the default choice.

Purpose:

- reduce unnecessary head flips
- reduce same-lineage prototype drift

### 2. Grouped lineage-internal gating

Prototype choice order was forced into groups:

1. current head
2. active siblings
3. archived siblings
4. new birth

Purpose:

- stop flat candidate competition inside a lineage
- make head replacement explicit instead of incidental

### 3. Birth suppression

Inside an already matched lineage, birth was treated as a last resort.

Purpose:

- reduce fragmentation pressure
- avoid creating extra sibling prototypes during recovery windows

### 4. Full stabilization knobs

The full Phase 3P variant additionally enabled:

- replacement consistency window
- head switch cooldown
- head continuity bonus
- archived sibling penalty
- newborn penalty

Purpose:

- suppress oscillatory head switching
- make head change require stronger evidence

## What This Round Was Trying To Repair

The intended failure targets were:

- excessive head replacement
- lineage-internal sibling churn
- unnecessary post-recovery sibling birth
- unnecessary archived sibling reactivation

## What The Round Actually Showed

The selector constraints can improve strict `same-prototype`, but they currently do it by over-constraining the recovery path:

- `keep_head + grouped gating` reached `same-prototype = 0.8235`
- but it drove `same-track-after-concept` down to `0.2143`
- and reduced `continuation_bank_access_rate_given_concept_recovery` to `0.1429`

So the current Phase 3P selector is strong enough to hold prototype identity more rigidly, but not yet calibrated to preserve recovery behavior at the same time.

## Stage A Attribution

Stage A showed that the Track C strict prototype continuity problem was not dominated by lineage-internal birth or archived sibling reuse in the re-entry event slice.

- `A.matched_lineage_kept_current_head`: `8/17` events
- `B.matched_lineage_replaced_with_active_sibling`: `4/17` events
- `F.no_valid_lineage_match`: `4/17` events
- `G.proposal_missing_or_upstream_missing`: `1/17` events
- `archived_sibling_reactivation_rate = 0.0000`
- `new_sibling_birth_rate_given_concept_recovery = 0.0000`

This means the initial hypothesis had to be narrowed. The main damage was not coming from event-level sibling birth. It was coming from unstable head identity inside already matched lineages and from no-valid-lineage-match events.

## Phase 3P Outcome

The best ablation was `keep_head_plus_grouped`.

On Track C it improved:

- `same_prototype: 0.6471 -> 0.8235`
- `same_lineage_prototype: 0.7059 -> 0.8235`
- `concept_recovered_but_lineage_mismatch_rate: 0.2500 -> 0.1250`
- `PFR: 5.3333 -> 4.6667`

But it regressed the recovery path:

- `same_track_after_concept: 0.4545 -> 0.2143`
- `same_track: 0.3529 -> 0.1765`
- `continuation_bank_access_rate_given_concept_recovery: 0.7273 -> 0.1429`

Track A also regressed in strict prototype continuity:

- `same_prototype: 1.0000 -> 0.0000`

So Phase 3P demonstrated a real tradeoff:

- stricter head preservation can stabilize strict prototype identity
- but the current implementation couples head stability too tightly to recovery attachment
- once that coupling becomes too strong, recovery traffic is starved and the system stops using the continuation path effectively
