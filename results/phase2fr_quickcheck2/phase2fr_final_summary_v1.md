# Phase 2F-R Final Summary v1

This update extends Phase 2F-R from Part A attribution into Part B/C proposal geometry repair.

## track_a_bridge

- bbox_tightness: 0.3620 -> 0.3963
- gt_coverage_by_support_region: 0.0791 -> 0.0491
- gt_coverage_by_refined_bbox: 0.7585 -> 0.5974
- fp_count_per_frame: 4.8908 -> 5.2605
- region_fill_ratio: 0.3620 -> 0.3963

## track_c_long_horizon

- bbox_tightness: 0.3225 -> 0.3727
- gt_coverage_by_support_region: 0.1024 -> 0.0600
- gt_coverage_by_refined_bbox: 0.6120 -> 0.4705
- fp_count_per_frame: 2.8050 -> 4.2201
- region_fill_ratio: 0.3225 -> 0.3727

## Readout

The proposal path now carries region support explicitly instead of collapsing everything into a coarse connected-component box. This is the intended direction for moving from block-like synthetic boxes toward object-aligned candidates that can later transfer to real-object detection.