# Stage SYN-REG-1 Report

## Scope

Evaluation-only synthetic regression gate for the EXT-2 external geometry calibration. No main NOPS code is modified.

## Verdict

keep external geometry calibration isolated; do not alter main NOPS because calibration regresses synthetic focus/anchor path

## Compact

```json
{
  "stage": "SYN-REG-1",
  "external_calibration_variant": "A2_external_trajectory_heavy",
  "internal_baseline_top1": 0.4117647058823529,
  "best_internal_variant": "A0_internal_passive_baseline",
  "best_internal_top1": 0.4117647058823529,
  "best_calibration_variant": "A2_external_trajectory_heavy",
  "best_calibration_top1": 0.23529411764705882,
  "best_calibration_focus_success_count": 1,
  "best_calibration_regression_event_count": 3,
  "focus_success_count": 3,
  "regression_event_count": 0,
  "negative_controls_passed": 1,
  "synthetic_regression_passed": 0,
  "safe_external_geometry_branch": 1,
  "safe_main_merge": 0,
  "needs_lasot_pixels_for_next_stage": 1,
  "next_recommendation": "keep external geometry calibration isolated; do not alter main NOPS because calibration regresses synthetic focus/anchor path"
}
```
