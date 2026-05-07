# Stage E3.2b Report

## Verdict

E3.2b 未通过：没有任何 top3 rerank 消融能安全超过 A0 baseline。当前 top3 内部判别仍不能可靠提升全局。

## Best Ablation

- `ablation_name = A0_E31_combined_baseline`
- `global_top1 = 0.35294117647058826`
- `global_top3 = 0.6470588235294118`
- `global_top5 = 0.7058823529411765`
- `false_bundle_retrieval_rate = 0.6470588235294118`
- `focus_top1_count = 3`
- `focus_success_count = 3`
- `regression_event_count = 0`
- `target_in_top3_but_lost_top1_count = 5`
- `bundle552_top1_count = 1`
- `proto0_top5_share = 0.17647058823529413`
- `selected_as_best = 1`

## Baseline Consistency

- `E31_summary_best`: top1=0.29411764705882354, false=0.7058823529411765, reason=matches the E31 published best-ablation metric family
- `E32_summary_best`: top1=0.35294117647058826, false=0.6470588235294118, reason=E32 stateful calibration wrapper reported a stronger A0; E32b uses freshly recomputed local E31 scoring as the authority
- `E32a_summary_best`: top1=0.29411764705882354, false=0.7058823529411765, reason=matches the E32a recomputed conservative baseline
- `E32b_recomputed_A0`: top1=0.35294117647058826, false=0.6470588235294118, reason=this run uses E32b local E31 scoring path; use this row as the E32b baseline

## Focus Events

### M-RE-TC-012

- `baseline_target_rank = 1`
- `target_bundle_rank_after = 1`
- `target_bundle_retrieved_top1 = 1`
- `pattern_completion_success = 1`
- `target_lost_reason = `

### M-RE-TC-013

- `baseline_target_rank = 1`
- `target_bundle_rank_after = 1`
- `target_bundle_retrieved_top1 = 1`
- `pattern_completion_success = 1`
- `target_lost_reason = `

### M-RE-TC-014

- `baseline_target_rank = 1`
- `target_bundle_rank_after = 1`
- `target_bundle_retrieved_top1 = 1`
- `pattern_completion_success = 1`
- `target_lost_reason = `


## Bundle 552

Bundle 552 appears in focus top3 22 times; generic_content_only=0; won_against_target=0. The swap gate blocks it when the challenger is generic or high-hub.
