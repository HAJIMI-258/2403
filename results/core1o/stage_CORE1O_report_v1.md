# CORE-1O Objectness Proposal Recall Audit

This stage scans objectness-field proposal settings on the selected CORE-1J windows. It does not change the main model. GT boxes are used only to audit proposal coverage.

## Result

- Selected sequences: 2
- Selected windows: 6
- Baseline recall@0.25: 0.7806
- Baseline recall@0.50: 0.6160
- Best variant: A3_lower_quantile
- Best recall@0.25: 0.8565
- Best recall@0.50: 0.6962
- Best mean proposals/frame: 15.49
- Frontend profile candidate found: 1

Next recommendation: CORE-1P validate best proposal profile through tracker pair mining
