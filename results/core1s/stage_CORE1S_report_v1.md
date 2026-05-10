# CORE-1S Proposal Postprocess Pair Validation

This stage applies GT-free proposal postprocessing before tracker assignment and validates pair quality. It does not alter the main model.

## Result

- Best variant: A7_quality035_nms040_max12
- Best gate: A3_score050_cost_le_050
- Best matched assignment rate eval-only: 0.2548
- Best positive precision eval-only: 0.7093
- Best negative precision eval-only: 0.3809
- Postprocess pair gate passed: 0

Next recommendation: proposal postprocess insufficient; repair objectness/localization model before encoder training
