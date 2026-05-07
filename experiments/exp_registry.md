# Experiment Registry

Use this file to assign a unique id to every run before launching it.

## Rules

1. Every experiment must have a unique `exp_id`.
2. Record the config path before the run starts.
3. Record the result path after the run ends.
4. If a run fails, keep the row and mark the status as `failed`.
5. Any protocol deviation must be described in the notes column.

## Suggested ID Format

`P1-SYN-XXX`

Examples:

- `P1-SYN-001`
- `P1-SYN-002`
- `P1-SYN-003`

## Registry Table

| exp_id | date | stage | scenario | method | config | budget | status | result_path | key_metrics | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1-SYN-001 | 2026-04-06 | day1-setup | n/a | scaffold | configs/synth.yaml | 32 | planned | results/ | n/a | repository scaffold and protocol freeze |
| P1-SYN-002 | 2026-04-06 | day2-dataset | synth_stream_v1 | generator_smoke_test | configs/synth.yaml | 32 | passed | results/synth_preview/ | frames=500; first_frame_boxes=1; first_mask_shape=(256,256) | preview script generated png and json artifacts successfully |
| P1-SYN-003 | 2026-04-06 | day2-dataset | synth_stream_v1_forced_features | generator_feature_check | configs/synth.yaml (runtime overrides) | 32 | passed | in-memory check | max_visible=3; reentered_ids=[1] | forced `num_objects=3`, `reentry_probability=1.0`, `occlusion_probability=1.0` to verify overlap and re-entry support |
| P1-SYN-004 | 2026-04-07 | day3-objectness | synth_stream_v1_seq_000 | minimal_objectness_preview | configs/synth.yaml | 32 | passed | results/day3_objectness/ | mean_u_recall=1.0000; best_frame=1; best_frame_u_recall=1.0000 | easy one-object sanity run for pipeline wiring |
| P1-SYN-005 | 2026-04-07 | day3-objectness | synth_stream_v1_seq_003 | minimal_objectness_preview | configs/synth.yaml | 32 | passed | results/day3_objectness/ | mean_u_recall=0.6283; best_frame=27; best_frame_u_recall=1.0000; gt=3; proposals=3 | multi-object Day 3 run used as the representative objectness figure |
| P1-SYN-006 | 2026-04-07 | day4-tracking | synth_stream_v1_seq_002 | temporal_identity_preview | configs/synth.yaml | 32 | passed | results/day4_tracking/ | mean_u_recall=0.5628; tracked_idsw=53; naive_idsw=833; unique_tracks=33 | harder 3-object run confirms temporal identity lowers ID switches sharply |
| P1-SYN-007 | 2026-04-07 | day4-tracking | synth_stream_v1_seq_003 | temporal_identity_preview | configs/synth.yaml | 32 | passed | results/day4_tracking/ | mean_u_recall=0.6283; tracked_idsw=18; naive_idsw=915; unique_tracks=20 | representative Day 4 run with stable short-term IDs and four clean tracked frames |
| P1-SYN-008 | 2026-04-07 | day5-memory | synth_stream_v1_seq_002 | prototype_memory_preview | configs/synth.yaml | 32 | passed | results/day5_memory/ | u_recall=0.5628; purity=0.9294; pfr=1.0000; idsw=15; memory_growth=0.0040; final_prototypes=4 | higher-purity memory run with controlled growth but no re-entry reconnection on this sequence |
| P1-SYN-009 | 2026-04-07 | day5-memory | synth_stream_v1_seq_003 | prototype_memory_preview | configs/synth.yaml | 32 | passed | results/day5_memory/ | u_recall=0.6283; purity=0.6993; pfr=0.6667; idsw=12; memory_growth=0.0040; final_prototypes=3; reconnect=0.5000 | representative Day 5 run showing prototype formation without memory explosion |
| P1-SYN-010 | 2026-04-07 | day6-baselines | synth_stream_v1_seq_002_003 | baseline_comparison | configs/synth.yaml | 32 | passed | results/day6_baselines/ | main vs baseline1 vs baseline2 csv/json/png generated | main pipeline strongly beats Baseline 1 on PFR, IDSW, and memory growth; Baseline 2 remains competitive and marks an honest residual gap |
| P1-SYN-011 | 2026-04-07 | day7-scenarios | easy+reentry+hard | main_pipeline_scenario_summary | configs/synth.yaml (runtime scenario overrides) | 32 | passed | results/day7_scenarios/ | easy U-Recall=1.0000; reentry U-Recall=0.8930; hard U-Recall=0.5398; memory growth stayed <=0.0048 | three-scenario summary confirms the pipeline works beyond the easy case but still weakens under strong drift + occlusion |
| P2-SYN-001 | 2026-04-07 | phase2-step1 | hard_drift_occlusion | failure_slicing | configs/synth.yaml (runtime hard scenario override) | 32 | passed | results/phase2_failure_slicing/ | failure_frames=276; baseline2_better_frames=33; zero_recall_frames=81 | failure slices confirm the main weakness still begins in proposal quality under hard drift + occlusion |
| P2-SYN-002 | 2026-04-07 | phase2-step2 | hard_drift_occlusion | habituation_before_after | configs/synth.yaml | 32 | passed | results/phase2_objectness/ | before_recall=0.5700; after_recall=0.5744; before_false_hot=0.0231; after_false_hot=0.0212 | habituation helps suppress false hot regions but has not yet solved the hard-scenario recall gap |
| P2-SYN-003 | 2026-04-07 | phase2-step3 | reentry+hard | tracking_reentry_eval | configs/synth.yaml | 32 | passed | results/phase2_tracking/ | hard IDSW 45->18; created_tracks 62->20; reentry recovery still 0 | tracking prediction + keepalive reduced churn in the hard scenario, but old-ID re-entry recovery is still weak |
| P2-SYN-004 | 2026-04-07 | phase2-step4 | easy+reentry+hard | rerun_scenarios_after_phase2 | configs/synth.yaml | 32 | passed | results/day7_scenarios/ | easy U-Recall=1.0000; reentry U-Recall=0.8731; hard U-Recall=0.5525 | rerun shows slight hard-scenario improvement and a stronger multi-object re-entry scenario |
| P2-SYN-005 | 2026-04-07 | phase2-step4 | seq_002_003 | rerun_baseline_comparison_after_phase2 | configs/synth.yaml | 32 | passed | results/day6_baselines/ | main U-Recall=0.6097; PFR=1.1667; IDSW=16.00; memory_growth=0.0050 | Baseline 1 is still clearly beaten; Baseline 2 remains a competitive residual gap |
