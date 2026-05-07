# NOPS-Bench v1 Protocol

## 1. Purpose

This protocol defines the Phase 1 evaluation boundary for NOPS-OWR. Its purpose is to make results comparable, prevent leakage, and keep the project focused on the minimum closure experiment.

## 2. Data Regime

Phase 1 uses only synthetic frame streams.

Required properties:

- moving shapes
- short occlusion
- re-entry after leaving the frame
- background drift
- mild appearance perturbation
- stable per-frame ground truth

Each frame must expose:

- `frame`
- `boxes`
- `masks`
- `instance_id`
- `concept_id`

## 3. Streaming Rules

All experiments in Phase 1 must obey the following rules:

1. One-pass only. Each sequence is processed once in time order.
2. No future access. Methods cannot read future frames, labels, or sequence summaries.
3. No offline training set. Parameters must not be trained on a separate labeled corpus.
4. No leakage-based retuning. Thresholds may be chosen from protocol-defined presets or a small fixed development split, but not retuned after seeing test results.
5. Fixed memory budget. Every experiment must declare and honor a prototype budget cap.
6. Batch size is fixed to 1 during streaming evaluation.

## 4. Default Synthetic Setup

The default Phase 1 setup is:

- image size: `256x256`
- sequence length: `300-1000` frames
- concurrent objects: `1-3`
- concept shapes: `circle`, `square`, `triangle`
- scenarios: simple motion, occlusion + re-entry, background drift

Any change from this default must be recorded in the experiment registry.

## 5. Minimal Model Boundary

Phase 1 allows only the following core modules:

- spike encoder: frame difference + edge term + ON/OFF threshold
- objectness field: boundary term + temporal persistence + surprise term
- temporal identity: IoU + centroid + simple spike signature
- prototype memory: birth + merge + decay
- budget gate: threshold-based cap

The following are explicitly postponed:

- split operations
- concept graph
- full habituation/co-motion terms
- full theoretical gate design
- real-data transfer

## 6. Baselines

Exactly two weak but fair baselines are required:

1. Frame difference -> threshold -> connected components -> IoU tracking
2. Edge/saliency proposal -> hand-crafted feature -> online clustering

Phase 1 should not claim success unless the main pipeline beats these baselines on at least three key indicators.

## 7. Primary Metrics

Primary metrics:

- `U-Recall`
- `PFR`
- `IDSW`
- `Churn`
- `Memory Growth`

Recommended secondary metric:

- `Purity`

Metric implementations must come from a single shared source in `metrics/metrics_core.py`.

## 8. Reporting Rules

Every recorded run must include:

- experiment id
- config path or config hash
- dataset scenario
- memory budget
- result path
- metric summary
- notes on failures or abnormal behavior

Required artifacts for the end of Phase 1:

- one objectness visualization figure
- one identity/prototype behavior figure
- one aggregate comparison figure or table

## 9. Advancement Rule

Go to Phase 2 only if at least three of the following four conditions hold:

1. Objects are visibly extracted by the objectness heatmap.
2. Identity switching is meaningfully reduced by temporal identity.
3. Reappearing objects reconnect to existing or nearby prototypes.
4. Memory growth is controlled and does not scale close to linearly with sequence length.

