# CORE-1R Matched-Observation Proxy Gate

This stage builds GT-free matched-observation proxy scores from assignment score, match cost, hits, overlap, and short-term motion consistency. GT is used only for audit precision.

## Result

- Observations: 2230
- Best gate: A0_core1p_best
- Best matched-observation rate eval-only: 0.7020
- Best positive precision eval-only: 0.7016
- Best negative precision eval-only: 0.3636
- Proxy gate passed: 0

Next recommendation: proxy gate insufficient; objectness/proposal localization must be repaired before encoder training
