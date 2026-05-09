# EXT-11 Full-pixel Expansion Instructions

Current full-pixel subset has 504 events.
Do not download the full LaSOT dataset yet. Use one of the staged plans below.

## Target 400 events

- Categories: ``
- Estimated new events: `0`
- Estimated download size: `0.00 GB`
- Projected total events: `504`

```powershell

```

## Target 500 events

- Categories: ``
- Estimated new events: `0`
- Estimated download size: `0.00 GB`
- Projected total events: `504`

```powershell

```

## Target 750 events

- Categories: `pool,basketball,monkey,electricfan,truck,swing,book,helmet`
- Estimated new events: `247`
- Estimated download size: `29.95 GB`
- Projected total events: `751`

```powershell
python scripts/download_lasot_hf_categories.py --categories pool,basketball,monkey,electricfan,truck,swing,book,helmet --execute
```

After download/extraction, rerun:

```powershell
python experiments\run_ext4_full_pixel_readiness.py
python experiments\run_ext5_multicategory_full_pixel_validation.py
python experiments\run_ext5c_appearance_control_audit.py
python experiments\run_ext6_stronger_local_descriptor_validation.py
python experiments\run_ext7_frozen_embedding_baseline.py --embedding-model resnet18 --bootstrap-samples 300
python experiments\run_ext9_event_conditioned_geometry_analysis.py
python experiments\run_ext10_geometry_routing_split_gate.py
python experiments\run_ext12_strong_descriptor_split_gate.py
python experiments\run_ext8_external_evidence_synthesis.py
python experiments\run_ext13_freeze_external_protocol.py
```
