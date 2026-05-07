## Phase 2B Go / No-Go Summary

### Decision

Go to bridge synthetic, but keep the scope narrow.

Phase 2B fixed the main Phase 2A issue: the hard case is no longer primarily limited by objectness. With the final Phase 2B config, `hard_drift_occlusion` reached `U-Recall=0.6874`, which is above the target floor of `0.60`.

### Why This Is A Go

- Adaptive thresholding improved hard-case proposal quality without inflating false hot area. In the hard-case comparison, mean objectness recall moved from `0.5509` to `0.6746`, while mean false hot area moved from `0.0238` down to `0.0143`.
- On the three required scenarios, the final main method results are:
  - `easy_single_object`: `U-Recall=0.9972`, `PFR=0.0000`, `IDSW=0`, `memory_growth=0.0000`
  - `multi_object_reentry`: `U-Recall=0.9439`, `PFR=0.0000`, `IDSW=0`, `memory_growth=0.0048`
  - `hard_drift_occlusion`: `U-Recall=0.6874`, `PFR=0.6667`, `IDSW=6`, `memory_growth=0.0024`
- Against `Baseline 2` on the same three-scenario Phase 2B protocol, the main method keeps all three required advantages in the mean row:
  - `PFR`: `0.2222` vs `0.8889`
  - `IDSW`: `2.00` vs `9.33`
  - `memory_growth`: `0.0024` vs `0.0032`

### Remaining Bottleneck

Tracking is now the main residual bottleneck, not objectness.

The upgraded hard-case failure slicing gives:

- `objectness`: `0.0000`
- `tracking`: `0.4440`
- `memory`: `0.0361`
- `mixed`: `0.5199`

That means objectness is no longer the dominant failure source. The largest cleanly attributable single block is now tracking, especially under long occlusion and re-entry.

### What Helped In Phase 2B

- Final field settings kept the adaptive threshold but stayed conservative: `tau_obj=0.47`, `threshold_mode=quantile_local`, `q_obj=0.92`, `local_k=0.95`.
- Final tracking settings kept continuous matching and added dormant reactivation: `max_match_cost=0.62`, `keepalive_frames=8`, `use_dormant_reactivation=true`, `dormant_frames=20`, `reactivation_cost=0.68`, `reactivation_proto_sim=0.55`.
- Final memory settings kept the same birth / merge thresholds but decayed stale prototypes faster: `tau_birth=0.35`, `tau_merge=0.18`, `tau_sim=0.20`, `lr_proto=0.35`, `decay_rate=0.03`, `decay_patience=16`.

### Bridge Synthetic Guidance

Proceed, but monitor three things first:

- `hard_drift_occlusion` ID stability under longer occlusion chains
- re-entry recovery rate
- prototype count / memory growth on cluttered multi-object scenes

If bridge synthetic introduces stronger long-gap occlusion, the first module likely to need more work is still tracking, not objectness.
