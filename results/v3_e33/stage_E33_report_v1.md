# Stage E3.3 Report

## Verdict

E3.3 未通过：增强 signature 后没有任何消融安全超过 E32b A0 baseline，说明需要继续修写入侧 signature 或分事件类型评分。

## Best Ablation

- `ablation_name = A0_E32b_baseline`
- `global_top1 = 0.35294117647058826`
- `global_top3 = 0.6470588235294118`
- `global_top5 = 0.7058823529411765`
- `false_bundle_retrieval_rate = 0.6470588235294118`
- `focus_success_count = 3`
- `regression_event_count = 0`
- `target_in_top3_but_lost_top1_count = 5`
- `target_not_in_top5_count = 5`
- `signature_collision_count = 5`
- `wrong_bundle_overgeneric_count = 0`

## Focus Events

- `M-RE-TC-012`: rank=1, top1=1, success=1, reason=
- `M-RE-TC-013`: rank=1, top1=1, success=1, reason=
- `M-RE-TC-014`: rank=1, top1=1, success=1, reason=

## Interpretation

E3.3 只修改情景 signature / cue consistency 层，不进入 attach、promotion 或最终 identity 决策。
如果本轮仍未通过，下一步应继续修写入侧 signature 或做 event-type-conditioned signature scoring，而不是回到 top3 rerank。
