# NOPS-Bench v2 Prototype Protocol

## 1. Scope

This protocol defines the post-Phase-2B prototype boundary for NOPS-Bench. It supersedes the Phase 1-only framing by introducing a generic bridge synthetic regime and explicit Track A / Track C prototype targets.

## 2. Shared Rules

All tracks must obey the original project constraints:

1. `No-Pretrain`
2. `No-Offline-Data`
3. `One-Pass`
4. `Pure-SNN`
5. `Open-World Self-Bootstrapping Recognition`
6. fixed streaming budget
7. no future access
8. no label leakage for retuning

## 3. Evaluation Backbone

All official runs should go through a shared interface stack:

- `metrics/interface.py`
- `nops_owr/controller/budget.py`
- `nops_owr/evaluation/streaming_episode.py`

This keeps metrics, budget checks, and episode evaluation consistent across experiments.

## 4. Track A Prototype

Track A evaluates medium-length streaming episodes on generic bridge synthetic.

Focus:

- unknown discovery
- temporal consistency
- concept formation
- controlled memory growth

Recommended episode regime:

- `500-1000` frames
- `2-6` concurrent objects
- repeated target reappearance
- moderate-to-hard disturbance

Primary reporting:

- `U-Recall`
- `IDSW`
- `reentry_recovery_rate`
- `PFR`
- `Purity`
- `memory_growth`

## 5. Track C Prototype

Track C evaluates longer-horizon streaming behavior under budget pressure and repeated appearance.

Focus:

- longer re-entry gaps
- drift accumulation
- churn stability
- budget stability

Recommended episode regime:

- `1000-3000` frames
- fixed memory budget
- repeated reappearance events
- longer disturbance windows than Track A

Primary reporting:

- `IDSW`
- `reactivation_successes`
- `reentry_recovery_rate`
- `Churn`
- `memory_growth`
- `budget_violation_frames`
- `final_prototype_count`

## 6. Monitoring Priority

For Phase 3, monitoring priority is:

1. tracking / re-entry
2. prototype fragmentation / concept quality
3. objectness

Objectness remains necessary, but it is no longer the main optimization axis after Phase 2B.

## 7. Budget Enforcement

Every official run must declare:

- `memory_budget`
- `max_proposals`
- optional track-pressure limit if used

Runs should report:

- peak memory size
- peak proposal count
- budget violation frames
- final prototype count

## 8. Advancement Conditions

Phase 3 prototype work is considered healthy only if:

- unknown discovery on generic bridge synthetic does not fall clearly below current hard synthetic
- tracking / re-entry continues improving or at least does not regress materially
- `PFR` does not drift upward uncontrollably
- memory growth and churn stay compatible with the declared budget

## 9. Explicit Non-Goals

This protocol does not yet include:

- real-data benchmark claims
- Track B event-stream branch
- split operations
- concept graph
- theorem-tight formalization

## 10. Required Artifacts

Each recorded Phase 3 prototype run should include:

- config snapshot
- protocol version
- scenario preset
- metric summary
- budget report
- tracking / re-entry diagnostic figure
- memory / prototype diagnostic figure
- short failure notes
