# Experiment Gate v1

This document converts `docs/literature_mechanism_alignment_v1.md` into an execution gate. It is not a roadmap for running more experiments by inertia. Each stage must state the mechanism, practical task pain point, baseline family, current evidence, falsification condition, allowed next action, and forbidden next action.

| Stage | Mechanism being tested | Practical visual-memory pain point | Relevant baseline family | Current evidence | Falsification condition | Allowed next action | Forbidden next action |
|---|---|---|---|---|---|---|---|
| E2C | Raw runtime ID drift vs stable memory anchor | Same old object may reappear under a different runtime namespace | Raw tracker IDs; embedding nearest-neighbor memory; ReID template bank | Raw lineage is unreliable; canonical / anchor visibility works for runtime namespace shift cases | Canonical / anchor visibility fails under negative controls, or target anchor visibility is too loose to distinguish wrong prototypes | Keep raw/canonical/anchor metrics as separate evaluation views | Treat raw lineage ID as the only success criterion |
| E3.1 | Pattern separation / anti-hub episodic retrieval | A generic high-frequency memory attractor can suppress the correct old bundle | Embedding memory with top-k retrieval; prototype memory bank with NMS | Focus hard cases were repaired to top1; proto0 hub share dropped | Focus success only improves by case-specific suppression, or global false retrieval rises | Use anti-hub as a diagnostic and retain focus hard-case guard | Treat focus success as global retrieval success |
| E3.2 / E3.2a | Replay / reconsolidation / accessibility | Wrong memories can be strengthened if online updates are too aggressive | Online template update; memory bank refresh; tracker template adaptation | Immediate update was either inert or harmful; strong E32 regressed focus events | Any same-event replay/accessibility changes current top1 or reinforces wrong bundle | Only delayed, gated, evidence-consistent replay after better evidence exists | Use replay/accessibility to change current-event top1 |
| E3.2b | Safe top3 rerank | Correct bundle can be in top3 but lose top1 to a wrong bundle | Pairwise rerank; calibrated retrieval score; ReID score fusion | No top3 rerank ablation safely beat A0 baseline | Rerank cannot improve global top1 / false retrieval without regression | Stop top3 rerank and inspect cue/signature quality | Continue E3.2c ranking sweeps |
| E3.3 | Coarse cue reweighting | Existing content/support/motion/context/temporal/provenance scores may be too coarse to separate target/wrong | Weighted cue scoring; metric learning-style score fusion | Reweighting existing dimensions could not beat E32b A0; `mean_target_wrong_signature_margin = -0.0329` | No coarse cue ablation safely improves baseline | Move to write-side signature repair | Keep tuning existing coarse signature weights |
| E3.4 | Write-side support trajectory as object-file / episodic cue | Single-frame bundle signatures do not preserve enough object-file history | ReID template with temporal smoothing; VOS memory bank; tracker trajectory memory | `A2_support_trajectory_only` reached minimum pass: top1 `7/17`, false `10/17`, focus `3/3` | `mean_signature_v2_margin` remains negative; target/wrong still not separated | Run E3.4r support trajectory separability test | Enter E4 attach/promotion immediately |
| E3.4r | Multi-frame support trajectory pattern separation | Need to know whether richer passive support evidence can separate similar old memories | Trajectory memory baseline; support-shape nearest neighbor; embedding memory with temporal pooling | Pending | `mean_support_v3_margin <= 0` after refinement | If pass: E3.5 retrieval index / event-type scoring. If fail: E4A active visual evidence acquisition | More ranking / rerank sweeps |
| E4A | Memory-uncertainty-guided active fixation | Passive cues are insufficient; system must gather more visual evidence before identity decisions | Active vision / foveated glimpse policies; uncertainty-driven observation; adaptive computation | Pending; only entered if passive support margin fails | Active evidence cannot improve target/wrong separability or reduces stability | Build evidence acquisition before identity attach | Attach/promotion before evidence acquisition |
| E5 | Delayed replay / consolidation / accessibility | Long-term memory should stabilize useful anchors without reinforcing false recalls | Continual learning memory replay; VOS memory potentiation; template consolidation | E3.2 showed immediate replay is unsafe | Delayed replay does not improve stable anchor retrieval or increases false retrieval | Add delayed replay only after reliable evidence and anchor separation | Replay ambiguous current-event retrieval as if it were confirmed |

## Stage-Specific Rules

### E2C

- Mechanism: raw IDs drift, but stable anchors can preserve object memory.
- Current evidence: focus remap events are raw-invisible but canonical/anchor-visible.
- Forbidden: using raw lineage as sole success metric.

### E3.1

- Mechanism: pattern separation / anti-hub retrieval.
- Current evidence: focus hard cases were repaired to top1.
- Forbidden: treating focus 3/3 as global retrieval success.

### E3.2 / E3.2a

- Mechanism: replay, reconsolidation, accessibility.
- Current evidence: immediate update can reinforce the wrong bundle or do nothing.
- Forbidden: same-event accessibility / replay changing current top1.

### E3.2b

- Mechanism: safe top3 rerank.
- Current evidence: no rerank ablation safely beat A0.
- Forbidden: continuing with E3.2c ranking sweeps.

### E3.3

- Mechanism: coarse cue reweighting.
- Current evidence: existing score dimensions are not enough; no E3.3 ablation beat A0.
- Forbidden: continuing to tune existing coarse weights.

### E3.4

- Mechanism: write-side support trajectory as object-file / episodic cue.
- Current evidence: support trajectory achieved minimum pass and reduced false retrieval by one event.
- Falsification: `mean_signature_v2_margin` remains negative.
- Allowed: E3.4r support trajectory separability test.
- Forbidden: E4 attach/promotion before separability is established.

### E3.4r

- Mechanism: support trajectory pattern separation.
- Must answer: can target/wrong bundles separate in support trajectory space?
- Primary metrics:
  - `mean_support_v3_margin`
  - `competition_removed_target_count`
  - target-vs-wrong support trajectory margin
  - `target_not_in_top5_count`
  - `false_bundle_retrieval_rate`
  - `focus_success_count`
- Decision rule:

```text
if mean_support_v3_margin > 0:
    next_recommendation = E3.5 retrieval index / event-type scoring
else:
    next_recommendation = E4A memory-uncertainty-guided active visual evidence acquisition
```

### E4A

- Mechanism: active visual evidence acquisition.
- Task: acquire better object support, boundary, and context evidence when passive memory cues are insufficient.
- Forbidden: identity attach or promotion before evidence is acquired.

### E5

- Mechanism: delayed replay and consolidation.
- Current constraint: replay is not allowed to update current-event top1.
- Forbidden: online replay of ambiguous retrieval as confirmed memory.

## Global Rule

No experiment may be started only because a metric did not improve. Every experiment must be a mechanism test with a falsification condition.
