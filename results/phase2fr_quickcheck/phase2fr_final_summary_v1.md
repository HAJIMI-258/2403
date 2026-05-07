# Phase 2F-R Final Summary v1

This update extends Phase 2F-R from Part A attribution into Part B/C proposal geometry repair.

## track_a_bridge

- bbox_tightness: 0.3620 -> 0.3839
- gt_coverage_by_support_region: 0.0791 -> 0.0367
- gt_coverage_by_refined_bbox: 0.7585 -> 0.5222
- fp_count_per_frame: 4.8908 -> 5.3782
- region_fill_ratio: 0.3620 -> 0.3839

## track_c_long_horizon

- bbox_tightness: 0.3225 -> 0.3698
- gt_coverage_by_support_region: 0.1024 -> 0.0469
- gt_coverage_by_refined_bbox: 0.6120 -> 0.4095
- fp_count_per_frame: 2.8050 -> 4.6604
- region_fill_ratio: 0.3225 -> 0.3698

## Readout

The proposal path now carries region support explicitly instead of collapsing everything into a coarse connected-component box. This is the intended direction for moving from block-like synthetic boxes toward object-aligned candidates that can later transfer to real-object detection.