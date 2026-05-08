# Stage EXT-3 Geometry Calibration Robustness

## Scope

Robustness audit for EXT-2 trajectory-heavy geometry calibration. This does not modify main NOPS and does not use target identity in scoring.

## Verdict

calibration is robust enough for an isolated external geometry branch, but do not merge into main NOPS until synthetic regression and full-pixel validation are checked

## Compact

```json
{
  "stage": "EXT-3",
  "event_count": 1213,
  "a0_top1": 0.640560593569662,
  "calibrated_variant": "A2_trajectory_heavy",
  "calibrated_top1": 0.7279472382522671,
  "support_reference_top1": 0.7485572959604286,
  "calibrated_delta_vs_a0": 0.08738664468260511,
  "remaining_gap_to_reference": 0.020610057708161583,
  "improved_count": 142,
  "regressed_count": 36,
  "unchanged_failure_count": 294,
  "regression_rate": 0.02967848309975268,
  "integration_gate_passed": 1,
  "top_regression_categories": {
    "kite": 5,
    "bottle": 3,
    "car": 3,
    "elephant": 2,
    "gametarget": 2,
    "licenseplate": 2,
    "microphone": 2,
    "pool": 2,
    "bear": 1,
    "cat": 1
  },
  "manual_lasot_pixels_needed_now": 0,
  "next_recommendation": "calibration is robust enough for an isolated external geometry branch, but do not merge into main NOPS until synthetic regression and full-pixel validation are checked"
}
```
