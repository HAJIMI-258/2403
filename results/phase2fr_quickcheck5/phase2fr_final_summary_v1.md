# Phase 2F-R Final Summary v1

This update extends Phase 2F-R from Part A attribution into Part B/C proposal geometry repair.

## track_a_bridge

- bbox_tightness: 0.3620 -> 0.5147
- gt_coverage_by_support_region: 0.0791 -> 0.1104
- gt_coverage_by_refined_bbox: 0.7585 -> 0.7585
- fp_count_per_frame: 4.8908 -> 4.8824
- region_fill_ratio: 0.3620 -> 0.5147

## track_c_long_horizon

- bbox_tightness: 0.3126 -> 0.4402
- gt_coverage_by_support_region: 0.1021 -> 0.1474
- gt_coverage_by_refined_bbox: 0.6304 -> 0.6303
- fp_count_per_frame: 2.8212 -> 2.7542
- region_fill_ratio: 0.3126 -> 0.4402

## Readout

The proposal path now carries region support explicitly instead of collapsing everything into a coarse connected-component box. This is the intended direction for moving from block-like synthetic boxes toward object-aligned candidates that can later transfer to real-object detection.