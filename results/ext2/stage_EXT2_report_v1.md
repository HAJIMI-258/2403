# Stage EXT-2 Geometry Passive Calibration

## Scope

Oracle-proposal / geometry-only external calibration on LaGOT annotations. No target ID is used in scoring; target identity is evaluation-only.

## Verdict

validate calibrated geometry scoring on held-out/full-pixel data before changing main NOPS; LaSOT pixels are needed for appearance/full-pipeline claims, not for this geometry diagnosis

## Compact

```json
{
  "stage": "EXT-2",
  "event_count": 1213,
  "a0_nops_current_top1": 0.640560593569662,
  "support_trajectory_reference_top1": 0.7485572959604286,
  "best_variant": "A3_support_trajectory_reference",
  "best_global_top1": 0.7485572959604286,
  "best_dev_top1": 0.7382113821138211,
  "best_test_top1": 0.7591973244147158,
  "best_delta_vs_a0": 0.10799670239076664,
  "best_nops_calibrated_variant": "A2_trajectory_heavy",
  "best_nops_calibrated_global_top1": 0.7279472382522671,
  "best_nops_calibrated_test_top1": 0.7324414715719063,
  "best_nops_calibrated_delta_vs_a0": 0.08738664468260515,
  "remaining_gap_to_support_reference": 0.02061005770816149,
  "recency_favors_wrong_when_trajectory_favors_target_count": 743,
  "a0_wrong_but_calibrated_pairwise_target_count": 143,
  "best_failure_counts": {
    "success": 908,
    "similar_distractor_top1_confusion": 284,
    "target_in_top5_but_wrong_top1": 11,
    "target_not_in_top5": 10
  },
  "primary_diagnosis": "current NOPS passive over-penalizes long-gap candidates through recency/weak trajectory weighting; trajectory-heavy calibrated scoring improves but still trails the pure support-trajectory reference",
  "next_recommendation": "validate calibrated geometry scoring on held-out/full-pixel data before changing main NOPS; LaSOT pixels are needed for appearance/full-pipeline claims, not for this geometry diagnosis"
}
```
