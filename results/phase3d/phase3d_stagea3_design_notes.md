# Phase 3D Stage A.3 Design Notes

- `active_match` is treated as a tentative claim for routing audit.
- cross-lineage preemption is detected when the tentative active lineage conflicts with a recovery lineage that still has visible recovery surface.
- rerouted proposals are converted into `rerouted_to_resurrection` candidates before memory + resurrection consume them.
- no anchor redesign, temp-attach redesign, or promotion logic is changed in Stage A.3.