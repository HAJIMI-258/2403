# Phase 3D Stage A.2b Target-Aligned Summary

## Target Event

- `event_id = 6`
- `target_gt_object_id = 2`
- `target_lineage_id = 2`
- `target_frame = 990`

## Target-Lineage Alignment

- baseline target-lineage rows at target frame: `0`
- baseline cross-lineage pollution rows at target frame: `2`
- observed pollution lineages: `0, 4`

## Recovery Surface At Target Frame

- `active/dormant/ghost/retired = 1/1/0/1`
- `continuation_bank_size = 1`
- `recovery_identity_anchor_count = 2`
- `temp_attach_alive = 1`
- `temp_attach_expired = 0`
- `recovery_surface_evaporated = 0`

## Forced Anchor Consume Probe

- forced rows with `restore_attempted_from_anchor = 1`: `4`
- forced rows with `anchor_success = 1`: `4`

## Exact Frame-990 Routing Read

- GT re-entry box at frame 990: `(286, 27, 313, 50)`
- best nearby proposal exists: `(286, 43, 314, 51)` with IoU `0.2881`
- target-lineage live surface at frame 990:
  - active track `200`, box `(289, 247, 320, 254)`, prototype `3`
  - dormant track `202`, box `(278, 0, 320, 23)`, prototype `3`
  - retired track `199`, box `(278, 315, 320, 320)`, prototype `3`
- non-target active match that preempts the target proposal:
  - active track `12`, lineage `0`, box `(286, 43, 314, 51)`, prototype `0`

## Main Reading

1. Target-aligned audit no longer treats other lineages' attach writes as target-lineage recovery evidence.
2. The target lineage state is read frame-locally from its own active/dormant/ghost/retired, continuation-bank, temp-attach, and recovery-anchor counts.
3. Temporary attach remains an observation state only. The explicit recovery source added in Stage A.2b is the lineage-level `RecoveryIdentityAnchor`.
4. Stage A.2b has already repaired the recovery surface. The remaining frame-990 break is earlier: the target proposal is consumed by an unrelated `active_match` before `apply_concept_gated_resurrection()` can see a `new_track` candidate.
