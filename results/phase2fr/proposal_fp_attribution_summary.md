# Proposal FP Attribution Summary

## Overall Distribution

| fp_type | count | ratio |
| --- | ---: | ---: |
| upstream_noise_fp | 3093 | 0.5835 |
| background_block_boundary_fp | 894 | 0.1686 |
| static_texture_fp | 716 | 0.1351 |
| drift_region_fp | 307 | 0.0579 |
| fragmented_component_fp | 162 | 0.0306 |
| box_fitting_artifact_fp | 129 | 0.0243 |

## Source Stage Distribution

| source_stage | count | ratio |
| --- | ---: | ---: |
| field | 5010 | 0.9451 |
| component | 162 | 0.0306 |
| box_fit | 129 | 0.0243 |

## Scenario Breakdown

### track_a_bridge

| fp_type | count | ratio |
| --- | ---: | ---: |
| upstream_noise_fp | 2081 | 0.5472 |
| static_texture_fp | 648 | 0.1704 |
| background_block_boundary_fp | 593 | 0.1559 |
| drift_region_fp | 250 | 0.0657 |
| fragmented_component_fp | 125 | 0.0329 |
| box_fitting_artifact_fp | 106 | 0.0279 |

### track_c_long_horizon

| fp_type | count | ratio |
| --- | ---: | ---: |
| upstream_noise_fp | 1012 | 0.6756 |
| background_block_boundary_fp | 301 | 0.2009 |
| static_texture_fp | 68 | 0.0454 |
| drift_region_fp | 57 | 0.0381 |
| fragmented_component_fp | 37 | 0.0247 |
| box_fitting_artifact_fp | 23 | 0.0154 |

## Readout

Dominant FP class is `upstream_noise_fp` (3093/5301 = 0.5835); dominant source stage is `field` (5010/5301 = 0.9451). Top FP mix: `upstream_noise_fp` 0.5835, `background_block_boundary_fp` 0.1686, `static_texture_fp` 0.1351.

Geometry-stage tail (`fragmented_component_fp` + `box_fitting_artifact_fp` + `overgrown_component_fp`) accounts for 291/5301 = 0.0549.

`track_a_bridge` is led by `upstream_noise_fp` (0.5472); geometry-driven tail is 0.0607. `track_c_long_horizon` is led by `upstream_noise_fp` (0.6756); geometry-driven tail is 0.0401.

The main purpose of this audit is attribution, not filtering. The next Phase 2F-R step should target the dominant source stage first: field suppression if field-driven FP dominates, connected-component repair if overgrown or fragmented components dominate, or box fitting if low-fill artifacts dominate.