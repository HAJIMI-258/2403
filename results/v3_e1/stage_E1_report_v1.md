# Stage E1 Report

## 目标

对当前系统做 baseline forensic audit。先看失败主层分布，不做机制修补。

## 总览

- `num_events = 18`
- `SVR proxy = 0.2778`
- `same_lineage = 0.5000`
- `same_prototype = 0.4444`
- `same_track = 0.2778`
- `STAC = 0.2778`

## Track A / Track C

### track_a_bridge

- `U-Recall = 0.7878`
- `PFR = 2.0000`
- `IDSW = 139`
- `reentry_events = 1`
- `SVR proxy = 1.0000`
- `same_lineage = 1.0000`
- `same_prototype = 0.0000`
- `same_track = 0.0000`
- `STAC = 0.0000`
- `failure_layers = {'identity_attach': 1}`

### track_c_long_horizon

- `U-Recall = 0.7247`
- `PFR = 5.3333`
- `IDSW = 371`
- `reentry_events = 17`
- `SVR proxy = 0.2353`
- `same_lineage = 0.4706`
- `same_prototype = 0.4706`
- `same_track = 0.2941`
- `STAC = 0.2941`
- `failure_layers = {'governance': 2, 'source_visibility': 10, 'perception': 1, 'success': 2, 'lineage_routing': 2}`

## Top Failures

| ledger_event_id | scenario | gap | failure_layer | failure_reason | same_lineage | same_proto | same_track |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| M-LG-TC-007 | track_c_long_horizon | 119 | governance | continuity_source_missing_under_long_gap | 1 | 1 | 0 |
| M-LG-TC-001 | track_c_long_horizon | 95 | governance | continuity_source_missing_under_long_gap | 0 | 0 | 0 |
| M-RE-TC-002 | track_c_long_horizon | 27 | source_visibility | target_continuity_source_not_visible | 0 | 0 | 0 |
| M-RE-TC-004 | track_c_long_horizon | 11 | perception | proposal_missing | 0 | 0 | 0 |
| M-RE-TC-006 | track_c_long_horizon | 6 | source_visibility | target_continuity_source_not_visible | 0 | 0 | 0 |
| M-RE-TC-015 | track_c_long_horizon | 6 | source_visibility | target_continuity_source_not_visible | 1 | 1 | 0 |
| M-RE-TC-009 | track_c_long_horizon | 3 | source_visibility | target_continuity_source_not_visible | 1 | 1 | 0 |
| M-RE-TC-017 | track_c_long_horizon | 3 | source_visibility | target_continuity_source_not_visible | 0 | 0 | 0 |
| M-RE-TC-003 | track_c_long_horizon | 1 | source_visibility | target_continuity_source_not_visible | 1 | 1 | 0 |
| M-RE-TC-005 | track_c_long_horizon | 1 | source_visibility | target_continuity_source_not_visible | 1 | 1 | 0 |
