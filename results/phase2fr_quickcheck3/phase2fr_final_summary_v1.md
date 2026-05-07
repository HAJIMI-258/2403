# Phase 2F-R Final Summary v1

This update extends Phase 2F-R from Part A attribution into Part B/C proposal geometry repair.

## track_a_bridge

- bbox_tightness: 0.3762 -> 0.3247
- gt_coverage_by_support_region: 0.0751 -> 0.0464
- gt_coverage_by_refined_bbox: 0.7326 -> 0.6016
- fp_count_per_frame: 5.1515 -> 5.4747
- region_fill_ratio: 0.3762 -> 0.3247

## track_c_long_horizon

- bbox_tightness: 0.3140 -> 0.2870
- gt_coverage_by_support_region: 0.1004 -> 0.0585
- gt_coverage_by_refined_bbox: 0.6344 -> 0.5248
- fp_count_per_frame: 2.7986 -> 3.5683
- region_fill_ratio: 0.3140 -> 0.2870

## Readout

The proposal path now carries region support explicitly instead of collapsing everything into a coarse connected-component box. This is the intended direction for moving from block-like synthetic boxes toward object-aligned candidates that can later transfer to real-object detection.