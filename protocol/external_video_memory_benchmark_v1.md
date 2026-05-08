# External Video Memory Benchmark Protocol v1

## Purpose

Synthetic bridge streams are mechanism-debug data. They are appropriate for locating broken links in memory write, source visibility, retrieval, active evidence, and metric accounting. They are not sufficient as final evidence that NOPS-OWR works in real agricultural, industrial, road, or robotics scenes.

The external benchmark gate exists to test whether persistent object memory works under public video data with object identity, occlusion, disappearance, re-entry, and similar distractors.

## Dataset Selection Criteria

- Video or temporally ordered frames.
- Stable object identity across frames.
- Boxes or masks.
- Occlusion / disappearance / re-entry events.
- Long-gap events where possible.
- Similar distractors or multiple same-category objects where possible.
- Public license or reproducible access instructions.

## Priority Datasets

1. LVOS / long-term VOS style datasets: closest to long-gap memory and re-entry.
2. LaSOT / GOT-10k / TrackingNet: long-horizon single-object tracking, useful for support trajectory and active evidence.
3. TAO / MOT / BDD100K / KITTI / nuScenes: multi-object identities and real scene complexity.
4. Agriculture / industrial inspection datasets: later transfer targets; many public sets are still single-frame and need longitudinal task construction.

## Unified Adapter Output

Each adapter must emit `FrameSampleExternal`:

- `sequence_id`
- `frame_idx`
- `frame_path`
- `boxes`
- optional `masks`
- `instance_ids`
- optional `category_ids`
- `visibility`
- `metadata`

Each adapter should derive an external event ledger:

- `sequence_id`
- `instance_id`
- `disappear_frame`
- `reappear_frame`
- `gap_length`
- `event_type`
- relevant metadata

## Evaluation Rules

- No future access.
- No test-time threshold tuning.
- No use of `target_bundle_id`, `old_track_id`, `old_prototype_id`, `instance_id`, GT box, GT mask, target anchor, or event label in online scoring.
- GT and target anchors are evaluation-only.
- Train/dev/test splits must be explicit once external full datasets are used.

## Required Baselines

- Tracker-only baseline.
- Frozen embedding + nearest-neighbor memory.
- ReID template bank.
- VOS-style memory baseline.
- NOPS passive memory.
- NOPS active evidence memory.

## Current Smoke Dataset

For initial adapter validation, the repo uses a small public HuggingFace LVOS-style point-track parquet:

- Source: `allenai/molmo2-single-object-track`, subset `lvosv1`
- Local path: `data/external/hf_lvosv1_sample/train-00000-of-00001.parquet`
- Limitation: point trajectories only; no raw video frames or masks. This is smoke-test data, not final benchmark evidence.
