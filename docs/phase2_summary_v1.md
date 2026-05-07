# NOPS-OWR Phase 2 Summary v1

Date: 2026-04-07

## Scope

This round followed the Phase 2 follow-up document in the requested order:

1. Failure slicing on `hard_drift_occlusion`
2. Habituation / background suppression in objectness
3. Tracking prediction + keepalive style re-activation
4. Re-run scenario summary and baseline comparison
5. Add bridge synthetic configuration stubs

## What Was Added

- `results/phase2_failure_slicing/failure_slices_v1.csv`
- `results/phase2_objectness/hard_drift_occlusion_before_after.csv`
- `results/phase2_tracking/tracking_reentry_eval_v1.csv`
- `configs/bridge_synth_industrial.yaml`
- `configs/bridge_synth_farmland.yaml`

## Failure Slicing

The current hard scenario still fails on a non-trivial subset of frames.

- failure frames counted by the slicing script: `276`
- zero-recall frames: `81`
- frames where Baseline 2 was better: `33`
- mean `false_hot_area`: about `0.0231`
- mean `drift_strength`: about `0.0383`

Interpretation:

- the weak point is still not primarily identity collapse
- the earliest break still happens in objectness / proposal quality under hard drift and occlusion
- Baseline 2 remains competitive on a specific subset of frames rather than everywhere

## Objectness Habituation Result

Habituation / background suppression helped, but only modestly.

- hard scenario mean `U-Recall`: `0.5700 -> 0.5744`
- mean `false_hot_area`: `0.0231 -> 0.0212`

Interpretation:

- background suppression is moving in the right direction
- it is already reducing false hot regions
- it is not yet enough to push `hard_drift_occlusion` to the desired `>= 0.65`

## Tracking Prediction / Keepalive Result

The current tracking enhancement helps stability more than re-entry recovery.

### hard_drift_occlusion

- `IDSW`: `45 -> 18`
- created tracks: `62 -> 20`
- re-entry recovery: still `0 / 2`

### multi_object_reentry

- created tracks: `79 -> 35`
- `IDSW`: did not improve in this run (`41 -> 46`)
- re-entry recovery: still `0 / 3`

Interpretation:

- prediction + keepalive is successfully reducing track churn in the harder occlusion setting
- true old-ID re-entry recovery is still not solved
- this remains an open Phase 2 item

## Updated Scenario Summary

Current three-scenario summary after this round:

- `easy_single_object`: `U-Recall=1.0000`, `Purity=1.0000`, `IDSW=0`
- `multi_object_reentry`: `U-Recall=0.8731`, `Purity=0.9122`, `PFR=0.3333`, `IDSW=3`
- `hard_drift_occlusion`: `U-Recall=0.5525`, `Purity=0.7246`, `PFR=1.3333`, `IDSW=10`

Interpretation:

- the medium-difficulty re-entry scenario improved meaningfully
- the hard drift + occlusion scenario improved only slightly
- the main unresolved weakness is still hard-scenario objectness robustness

## Updated Baseline Comparison

Current aggregate comparison on the selected synthetic sequences:

- `minimal_nops_owr`: `U-Recall=0.6097`, `PFR=1.1667`, `IDSW=16.00`, `memory_growth=0.0050`
- `baseline_frame_diff_cc`: `U-Recall=0.6506`, `PFR=13.1667`, `IDSW=93.50`, `memory_growth=0.2691`
- `baseline_edge_cluster`: `U-Recall=0.6259`, `PFR=0.8333`, `IDSW=8.50`, `memory_growth=0.0060`

Interpretation:

- the main pipeline still clearly beats Baseline 1 on long-horizon fragmentation and memory control
- Baseline 2 remains a real residual competitor
- the target from the Phase 2 follow-up document, namely stable superiority over Baseline 2 on `PFR`, `Purity`, and `memory_growth`, has **not** been fully met yet

## Current Verdict

**Continue Phase 2, but stay in robustness mode.**

What is now true:

- the system still has a valid end-to-end mechanism
- objectness suppression under drift is slightly better
- tracking churn is lower in the hard scenario
- medium-difficulty multi-object re-entry is stronger than before

What is still not true:

- `hard_drift_occlusion` has not crossed `U-Recall >= 0.65`
- re-entry recovery to the old track ID is still weak
- Baseline 2 is still too close

## Recommended Next Move

The next highest-value step is still not a large module expansion.

Do next:

1. Improve hard-scenario proposal quality further, likely via stronger local suppression and better proposal splitting under occlusion.
2. Add a more explicit dormant-track reactivation rule for re-entry beyond pure motion extrapolation.
3. Only after that, start actually using the new bridge synthetic configs for industrial and farmland variants.
