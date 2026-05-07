# Phase A Debug Commands

```powershell
python experiments/run_phaseA_visibility_repair.py --config configs/bridge_synth_generic_v1.yaml --output-dir results/phaseA
python experiments/run_phase3d_stage_a7_remap_trace.py --config configs/bridge_synth_generic_v1.yaml --output-dir results/phase3d
```

重点文件：

- `results/phaseA/source_visibility_trace.jsonl`
- `results/phaseA/claim_builder_source_breakdown.csv`
- `results/phaseA/attach_decisions.jsonl`
- `results/phaseA/continuity_key_overwrite_events.csv`
- `results/phaseA/phasea_result_summary.md`