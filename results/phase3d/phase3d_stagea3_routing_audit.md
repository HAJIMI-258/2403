# Phase 3D Stage A.3 Routing Audit

## Target Event

- `event_id = 6`
- `target_gt_object_id = 2`
- `target_frame = 990`

## Baseline Audit

- tentative active lineage: `0`
- best recovery lineage: `1`
- `cross_lineage_preemption_flag = 1`
- final assignment source: `active_match_cross_lineage`
- final lineage: `0`

## Minimal Routing Policy

- `routing_arbitration_triggered = 1`
- `was_rerouted = 1`
- final assignment source: `resurrection_from_dormant_or_ghost`
- final lineage: `0`
- `restore_attempted_after_reroute = 1`
- `resurrection_candidate_seen = 1`

## Forced Probes

- forced target reroute final lineage: `0`
- forced target reroute final source: `resurrection_from_dormant_or_ghost`
- forced target reroute `restore_attempted_after_reroute = 1`
- forced target reroute `resurrection_candidate_seen = 1`
- forced visibility total flagged proposals: `11937`
- forced visibility rerouted proposals: `11937`
- forced visibility proposals that reached resurrection source: `11867`

## Reading

1. Stage A.3 treats cross-lineage active matches as tentative instead of irreversible.
2. Reroute now exposes those proposals to the existing resurrection consumer path.
3. The next gate is no longer surface evaporation; it is whether the rerouted proposal lands on the intended recovery lineage consistently enough to replace preemptive active claims.

## Direct Answers

1. Frame-990 target proposals are now explicitly visible as cross-lineage preemption cases in the routing trace, not as missing-surface cases.
2. Once those proposals are rerouted, resurrection consume is no longer stuck at zero: `restore_attempted_after_reroute = 1` and `resurrection_candidate_seen = 1`.
3. The remaining defect is downstream of visibility: reroute works, but the recovered lineage chosen after arbitration is still not stable enough to guarantee the intended target lineage / old track.
