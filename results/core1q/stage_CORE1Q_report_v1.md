# CORE-1Q Assignment Fragmentation Audit

This stage audits why CORE-1P gated assignment pairs remain too noisy. It uses GT only for audit labels, not for online scoring or pair selection.

## Result

- Source gate: A3_score050_cost_le_050
- Positive precision eval-only: 0.7016
- Negative precision eval-only: 0.3636
- Unmatched frame rate: 0.8022
- Fragmented frame rate: 0.1875
- Switched track rate: 0.0794
- Main failure type: unmatched_assignment_in_pair
- Ready for encoder training: 0

Failure counts: {'unmatched_assignment_in_pair': 496, 'same_gt_fragmented_across_tracks': 14, 'track_identity_switched_between_frames': 4, 'overlapping_tracks_for_different_gt': 2}

Next recommendation: add matched-observation confidence gate or improve objectness/proposal localization before training
