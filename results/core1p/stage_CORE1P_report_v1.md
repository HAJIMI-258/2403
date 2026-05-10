# CORE-1P Proposal Profile Pair Validation

This stage validates the CORE-1O A3 lower-quantile objectness profile through the same assignment-pair gates used in CORE-1M. It does not change the main model.

## Result

- Proposal profile: A3 lower quantile
- Assignment observations: 2230
- Matched assignment rate eval-only: 0.1938
- Best gate: A3_score050_cost_le_050
- Best positive precision eval-only: 0.7016
- Best negative precision eval-only: 0.3636
- CORE-1M best positive precision: 0.6486
- CORE-1M best negative precision: 0.3791
- Gate passed: 0

Next recommendation: proposal recall improved but pair quality still insufficient; inspect duplicate/fragmented assignments before training
