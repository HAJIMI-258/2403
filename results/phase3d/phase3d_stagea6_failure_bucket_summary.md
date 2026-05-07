# Phase 3D Stage A.6 Failure Bucket Summary

## baseline_reroute

- `failure_bucket = input_formation_failure`
- `entered_preserve_input = 0`
- `entered_claim_builder = 0`
- `visible_in_claim_set = 0`
- `final_selected_lineage = 0`
- `runtime_lineage_registry_at_target_frame = {0,1}`
- `target_lineage_2_present_in_runtime = 0`

## forced_continuity_exposure

- `failure_bucket = input_formation_failure`
- `entered_preserve_input = 0`
- `entered_claim_builder = 0`
- `visible_in_claim_set = 0`
- `final_selected_lineage = 0`
- `runtime_lineage_registry_at_target_frame = {0,1}`
- `target_lineage_2_present_in_runtime = 0`

## forced_three_source_input

- `failure_bucket = input_formation_failure`
- `entered_preserve_input = 0`
- `entered_claim_builder = 0`
- `visible_in_claim_set = 0`
- `final_selected_lineage = 0`
- `runtime_lineage_registry_at_target_frame = {0,1}`
- `target_lineage_2_present_in_runtime = 0`

## Interpretation

- 当前主失败桶仍然是 `input_formation_failure`，但具体含义已经比 A.5 更早：
  不是“合法 lineage 2 候选被 preserve 输入静默 prune”，而是“frame 990 的当前运行时根本只剩 lineage 0/1，lineage 2 没有任何可枚举 source”。
- 因此本轮 forced preserve 不生效，不是 preserve 规则太弱，而是 continuity source 本身已经被重映射到 runtime lineage 0/1。
