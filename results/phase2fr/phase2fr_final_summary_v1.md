# Phase 2F-R Final Summary v1

This update extends Phase 2F-R from Part A attribution into Part B/C proposal geometry repair.

## track_a_bridge

- bbox_tightness: 0.3716 -> 0.5257
- gt_coverage_by_support_region: 0.0693 -> 0.1001
- gt_coverage_by_refined_bbox: 0.7658 -> 0.7658
- fp_count_per_frame: 4.8075 -> 4.7992
- region_fill_ratio: 0.3716 -> 0.5257

## track_c_long_horizon

- bbox_tightness: 0.3353 -> 0.4720
- gt_coverage_by_support_region: 0.1239 -> 0.1989
- gt_coverage_by_refined_bbox: 0.6028 -> 0.6024
- fp_count_per_frame: 3.1003 -> 3.0780
- region_fill_ratio: 0.3353 -> 0.4720

## Readout

The proposal path now carries region support explicitly instead of collapsing everything into a coarse connected-component box. This is the intended direction for moving from block-like synthetic boxes toward object-aligned candidates that can later transfer to real-object detection.