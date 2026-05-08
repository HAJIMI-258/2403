# Stage EXT-1 Report

## Scope

Oracle-proposal memory-only external benchmark. GT boxes are used as proposals; GT identity is evaluation-only.
LaGOT annotations do not include raw pixels in this checkout, so this run is a geometry/trajectory external memory benchmark, not a full perception benchmark.
A passing EXT-1 means the external event ledger and baseline harness are usable; it does not claim NOPS is externally effective yet.

## Verdict

analyze external failure taxonomy before model optimization

## Compact

```json
{
  "stage": "EXT-1",
  "external_event_mining_passed": 1,
  "usable_datasets": [
    "lagot_annotations"
  ],
  "valid_event_count": 1213,
  "num_sequences": 414,
  "num_frames": 400,
  "num_objects": 66,
  "baselines_run": [
    "B0_tracker_iou_centroid_memory",
    "B1_template_descriptor_nn",
    "B2_support_trajectory_memory",
    "B3_nops_anchor_episodic_passive"
  ],
  "metric_consistency_passed": 1,
  "oracle_leakage_found": 0,
  "nops_passive_top1": 0.640560593569662,
  "best_baseline_top1": 0.7485572959604286,
  "nops_vs_best_baseline_delta": -0.10799670239076664,
  "external_validation_ready": 1,
  "next_recommendation": "analyze external failure taxonomy before model optimization"
}
```
