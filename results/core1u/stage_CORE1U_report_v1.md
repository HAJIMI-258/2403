# CORE-1U Matched-Observation Feature Audit

This stage audits whether online-visible assignment features can predict the oracle matched-observation target exposed by CORE-1T. GT labels are used only for audit and probe supervision; no model integration is performed.

## Result

- Observations: 2230
- Matched label rate eval-only: 0.1933
- Best single feature: score AUC=0.8964
- Logistic probe AUC: 0.9520
- Best matched-observation gate: logistic_probe_precision_090
- Best positive precision eval-only: 0.9485
- Best negative precision eval-only: 0.7770
- Feature gate passed: 0

Next recommendation: online features insufficient; add localization-quality features or repair objectness field
