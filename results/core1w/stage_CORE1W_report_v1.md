# CORE-1W Negative Curriculum Audit

This stage classifies high-confidence co-visible negative pairs into safe, hard, and fragment-risk groups. GT is used only for audit precision.

## Result

- Positive pairs: 136
- All high-confidence negatives: 148
- Best curriculum: C0_all_high_conf_negatives
- Best positive precision eval-only: 0.9485
- Best negative precision eval-only: 0.7770
- Negative curriculum passed: 0

Next recommendation: safe negatives insufficient in same-frame windows; mine cross-window/cross-event negatives or repair fragmentation
