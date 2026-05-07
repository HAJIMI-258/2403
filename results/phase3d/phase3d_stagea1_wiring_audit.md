# Phase 3D Stage A.1 Wiring Audit

## Baseline

- matched_lineage_events = `1`
- attach_branch_enter_rate_given_matched_lineage = `1.0000`
- temp_attach_usage_rate = `0.0000`
- continuation_access_rate = `0.0000`
- same_track_after_attach = `0.0000`
- dropped_lineage_seed_before_memory_rate = `0.0000`
- attach_branch_not_entered_events = `0`
- attach_written_but_not_consumed_events = `1`

## Forced Temp Attach

- matched_lineage_events = `1`
- attach_branch_enter_rate_given_matched_lineage = `1.0000`
- temp_attach_usage_rate = `0.0588`
- continuation_access_rate = `0.0000`
- same_track_after_attach = `0.0000`
- forced_temp_attach_events = `1`
- attach_written_but_not_consumed_events = `1`

## Wiring Conclusion

- `attach_success_rate_given_matched_lineage = 1.0` is a small-denominator signal here, because Track C re-entry events only produce one matched-lineage event per run under the current Stage A path.
- In both baseline and forced runs, the matched-lineage event writes attach state, but `candidate_pool_size = 0`, `continuation_bank_size = 0`, and no restore attempt follows. That is a wiring/availability failure, not a successful recovery attach.
- Forced temp attach proves the temp slot can be instantiated, but it still does not drive continuation lookup or old-track restore. The remaining break is downstream of attach-state write, at candidate-pool / continuation access consumption.
