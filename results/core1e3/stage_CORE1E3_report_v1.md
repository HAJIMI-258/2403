# CORE-1E3 Retrieval Integration Gate

CORE-1E2 showed that the pseudo curriculum metric generalizes at pair level. This stage tests whether it can be safely used only when baseline ranking is low-confidence and the metric is high-confidence.

## Result

- Baseline top1 on dense diagnostic pool: 0.9530
- Best gate: A_gate_b002_m020_c6
- Best gate top1: 0.9457
- Best gate fires: 40
- Improvements/regressions: 3 / 12
- Best shuffled-control top1: 0.9433
- Integration gate passed: 0

Next recommendation: do not integrate pseudo metric; build a harder official re-entry eval bridge or improve descriptor input
