# CORE-1AO Uncertainty-Aware Memory Policy

This stage turns CORE-1AN's margin signal into an explicit memory action: `old_recall` or `uncertain_need_more_evidence`. It does not change retrieval ranking.

## Result

- Baseline old-recall precision: 0.9535
- Baseline false old recalls: 23
- Best policy: A1_core1an_selected_margin_gate
- Best threshold: 0.0194
- Coverage: 0.8465
- Old-recall precision: 0.9594
- False old recalls after policy: 17
- False old recalls suppressed: 6
- Unnecessary uncertain decisions: 70
- Policy gate passed: 1

Next recommendation: CORE-1AP add uncertainty state to core memory API / downstream audit
