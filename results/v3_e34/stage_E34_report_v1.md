# Stage E3.4 Report

## Verdict

E3.4 最低通过：写入侧 signature v2 在保持 focus 3/3 的前提下降低了 false retrieval 或关键失败计数。

## Best Ablation

- `ablation_name = A2_support_trajectory_only`
- `global_top1 = 0.4117647058823529`
- `global_top3 = 0.6470588235294118`
- `global_top5 = 0.7647058823529411`
- `false_bundle_retrieval_rate = 0.5882352941176471`
- `focus_success_count = 3`
- `target_in_top3_but_lost_top1_count = 4`
- `target_not_in_top5_count = 4`
- `mean_signature_v2_margin = -0.009844475370996138`

## Next

可以继续 E3.4 refinement 或准备 E4 前置安全检查。
