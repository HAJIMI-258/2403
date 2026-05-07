# Phase 1 Result Thresholds

This file freezes the Go / No-Go rules before major experiments begin.

## 1. Core Decision Principle

Move to Phase 2 only if at least three of the following four statements are true:

1. Objectness heatmaps visibly lift target regions and proposals cover the objects reasonably well.
2. Temporal identity materially reduces identity switching relative to a version without temporal identity.
3. Reappearing objects reconnect to an existing prototype or a nearby stable prototype instead of spawning a fresh prototype every time.
4. Prototype growth stays controlled and clearly sub-linear relative to sequence length in the tested scenarios.

If fewer than three statements hold, stop expansion and fix the weakest link before adding more modules.

## 2. Quantitative Thresholds for Phase 1 v1

These thresholds are intentionally modest. Phase 1 is a feasibility gate, not a paper leaderboard.

| Item | Threshold | Action if missed |
| --- | --- | --- |
| U-Recall | `>= 0.70` on the simplest synthetic scenario and `>= 0.55` averaged over all Phase 1 scenarios | Fix objectness and proposal extraction first |
| IDSW improvement | At least `20%` lower than the same pipeline without temporal identity, or clearly lower than Baseline 1 | Revisit matching cost and thresholds |
| Prototype formation | Re-entry should recover or merge into an existing prototype in most visible repeats; target `>= 60%` reconnection on tracked repeats | Tune birth and merge before adding split |
| Memory Growth | Final prototype count should stay well below sequence length and should not hit budget cap persistently; target average growth `< 0.05` prototypes per frame in default settings | Strengthen decay or budget gate |
| Baseline comparison | Win on at least `3/5` key indicators: U-Recall, PFR, IDSW, Churn, Memory Growth | Pause direction expansion and inspect logic |

## 3. Required End-of-Phase Artifacts

Before making a Go / No-Go call, collect:

- one objectness visualization
- one identity/prototype behavior visualization
- one aggregate metric table or comparison figure
- one-page conclusion

## 4. Rule for Changing These Thresholds

Thresholds may only be updated by editing this file and recording the reason in `experiments/exp_registry.md` before rerunning the affected experiment set.

