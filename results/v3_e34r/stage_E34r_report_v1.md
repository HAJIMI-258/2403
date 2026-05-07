# Stage E3.4r Report

## Verdict

E3.4r did not pass support separability gate.

## Best Ablation

- `ablation_name = A0_E34_support_v2_baseline`
- `global_top1 = 0.4117647058823529`
- `global_top3 = 0.6470588235294118`
- `global_top5 = 0.7647058823529411`
- `false_bundle_retrieval_rate = 0.5882352941176471`
- `focus_success_count = 3`
- `target_not_in_top5_count = 4`
- `competition_removed_target_count = 4`
- `mean_support_v3_margin = -0.004707469659693101`

## Decision

E4A memory-uncertainty-guided active visual evidence acquisition; passive memory evidence insufficient; do not continue ranking/signature sweeps.
