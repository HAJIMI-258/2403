# Phase 3D Stage A.2b Consume Trace

- target event: `6`
- target lineage: `2`
- target frame: `990`

## Baseline Surface

- `active/dormant/ghost/retired = 1/1/0/1`
- `continuation_bank_size = 1`
- `recovery_identity_anchor_count = 2`
- `temp_attach_alive = 1`

## Forced Anchor Consume Probe

- `forced_candidate_pool_size_max = 4`
- `forced_anchor_candidate_pool_size_max = 3`
- `forced_restore_attempted_from_anchor_max = 1`
- `forced_attach_state_consumed_by_tracker_max = 0`
- `forced_anchor_success_max = 1`
- `pulled_off_zero_metrics = []`

## Exact Target-Frame Break

- frame 990 is no longer `retired-only + empty bank`
- the target lineage already has:
  - active track `200`
  - dormant track `202`
  - continuation bank size `1`
  - recovery anchors `2`
- the target GT re-entry still fails because its nearby proposal `(286, 43, 314, 51)` is pre-consumed by `active_match` on track `12` from lineage `0`
- `apply_concept_gated_resurrection()` only processes `concept_recovered` cases, i.e. assignments whose source is `new_track` and whose prototype attach reused an old concept
- therefore the exact remaining break is `active_match` preemption, not missing anchor / missing bank / missing consumer wiring
