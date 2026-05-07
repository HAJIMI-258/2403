# Stage E3.2a Report

## 结论

- 最优消融：`A0_E31_combined_baseline`
- focus 保持情况：`top1=3/3, success=3/3`
- 全局 top1：`0.29411764705882354`
- 全局 top3：`0.6470588235294118`
- 全局 false retrieval：`0.7058823529411765`
- regression_event_count：`0`

## 人话判断

E3.2a 这轮没有通过最低门槛。问题不在候选池，而在保守校准对全局 false retrieval 的改善还不够，或者引入了新的 regression。 主要残留 failure 仍然是 target 已经在 top3 里，但没有拿下 top1。

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

