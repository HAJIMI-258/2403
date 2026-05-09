# EXT-13 Rerun Protocol After Target-500 Download

After downloading `guitar, car, drone`, run exactly:

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

No variants or scoring rules may be added during this rerun.
If a new method is required, open EXT-14.
