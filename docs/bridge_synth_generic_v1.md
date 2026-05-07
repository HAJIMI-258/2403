# Generic Bridge Synthetic v1

## Goal

This document defines the next-phase synthetic regime after Phase 2B. The purpose is not to mimic a single industry, but to move from simple controlled streams to a harder, still fully controllable bridge synthetic regime that better matches the intended NOPS-Bench prototype.

## Design Principles

- Keep the original NOPS-OWR problem definition unchanged.
- Stay fully synthetic and fully auto-labeled.
- Increase difficulty through generic structure and perturbation, not domain-specific art direction.
- Make tracking and re-entry the first-class stressors.
- Preserve one-pass streaming evaluation and fixed-budget constraints.

## Core Scene Axes

Each bridge synthetic sequence should vary along these axes:

1. Background complexity
   - repeated local texture
   - clutter patches
   - structured edges
   - moderate low-frequency illumination gradients
2. Target diversity
   - multiple shape families
   - mild intra-concept deformation
   - scale variation
   - low-contrast targets
3. Temporal disruption
   - longer occlusion
   - later re-entry
   - crossing trajectories
   - temporary disappearance without concept change
4. Sensor-style disturbance
   - camera jitter
   - local blur
   - additive noise
   - contrast drift
   - partial washout or dimming regions

## Required Difficulty Controls

The generator should expose a small, explicit difficulty surface:

- `background_repeat_density`
- `background_texture_strength`
- `illumination_drift_strength`
- `local_noise_std`
- `local_blur_probability`
- `camera_jitter_std`
- `occlusion_duration_range`
- `reentry_gap_range`
- `crossing_probability`
- `target_deformation_strength`
- `low_contrast_probability`

These controls should be monotonic enough that a future benchmark script can define easy / medium / hard presets without hidden retuning.

## Sequence Structure

Recommended default bridge synthetic episodes:

- resolution: `256x256` or `320x320`
- length: `500-1500` frames
- concurrent objects: `2-6`
- repeated appearance events per concept: at least `2`
- long-gap re-entry: required in a subset of sequences
- mixed disturbance frames: required, but not every frame

## Annotation Format

Per-frame outputs must keep the current Phase 2 format and add a small amount of benchmark metadata:

- `frame`
- `boxes`
- `masks`
- `instance_id`
- `concept_id`
- `visibility_flag`
- `occlusion_ratio`
- `drift_strength`
- `blur_level`
- `noise_level`
- `reentry_event`

Per-sequence metadata should include:

- `difficulty_preset`
- `active_concept_count`
- `planned_reentry_events`
- `planned_long_occlusion_events`
- `budget_reference`

## Phase 3 Success Checks

Bridge synthetic v1 is useful only if it directly supports Track A / Track C style evaluation:

- unknown discovery does not collapse relative to current hard synthetic
- tracking / re-entry remains the primary monitoring axis
- prototype fragmentation stays controlled
- memory growth and churn remain budget-compatible

## Explicit Non-Goals

- no real data
- no industry-specific visual branding
- no event-stream branch
- no split or concept graph
- no hidden benchmark-specific tuning loops

## Expected Next Step

The next implementation step after this document is a `generic bridge synthetic v1` generator/config set with fixed presets and fully auto-generated labels, not another round of Phase 2B parameter chasing.
