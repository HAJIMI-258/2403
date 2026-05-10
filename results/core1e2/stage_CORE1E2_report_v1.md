# CORE-1E2 Curriculum Control Audit

CORE-1E produced clean pseudo-reentry pairs, but retrieval controls caught up. This stage checks whether the metric itself generalizes under random, sequence-holdout, and event-holdout splits before any integration.

## Key Results

- CORE-1E pair gate: 1
- Random pair AUC: 0.9185
- Sequence-holdout AUC: 0.9205
- Sequence shuffled AUC: 0.5114
- Event-holdout AUC: 0.9253
- Event shuffled AUC: 0.4949
- Split generalization passed: 1

Dense retrieval remains a sanity check only because the diagnostic pool is saturated.

Next recommendation: CORE-1E3 integrate only with held-out validated objective
