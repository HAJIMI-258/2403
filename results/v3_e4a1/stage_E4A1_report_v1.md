# Stage E4A.1 Report

## Verdict

same-space descriptor is weak: margin improves slightly in best active policy but positive-rate/retrieval gates fail; next E4A.2 stronger local descriptor or real active re-observation.

## Compact

```json
{
  "stage": "E4A.1",
  "best_ablation": "A0_passive_E34r_baseline",
  "best_active_ablation": "A3_neighbor_same_space_w030",
  "passed_minimum": false,
  "global_top1": 0.4117647058823529,
  "global_top3": 0.6470588235294118,
  "global_top5": 0.7647058823529411,
  "false_bundle_retrieval_rate": 0.5882352941176471,
  "focus_success_count": 3,
  "same_space_margin_positive_rate": 0.0,
  "mean_same_space_margin": -0.004707469659693101,
  "active_resolved_event_count": 0,
  "active_false_rescue_count": 0,
  "random_same_space_gain": 0.0,
  "memory_uncertainty_same_space_gain": 0.0,
  "descriptor_missing_rate": 0.0,
  "descriptor_degenerate_rate": 0.0,
  "best_active_global_top1": 0.4117647058823529,
  "best_active_false_bundle_retrieval_rate": 0.5882352941176471,
  "best_active_same_space_margin_positive_rate": 0.26666666666666666,
  "best_active_mean_same_space_margin": -0.00015783309936523438,
  "best_active_active_resolved_event_count": 0,
  "best_active_active_false_rescue_count": 0,
  "negative_controls_passed": 1,
  "main_failure_counts": {
    "target_not_in_top5": 60,
    "target_in_top3_but_lost_top1": 64,
    "target_in_top5_but_lost_top3": 30
  },
  "next_recommendation": "same-space descriptor is weak: margin improves slightly in best active policy but positive-rate/retrieval gates fail; next E4A.2 stronger local descriptor or real active re-observation."
}
```

## Interpretation

This stage tests real same-space historical/current crop descriptors. It does not use target labels for online scoring and does not enter attach or promotion.
