# Phase 3R.3 Summary v1

## Selected Params

- keepalive_frames=8, dormant_frames=16, ghost_frames=80, tau_g=12.00.
- tau_res_short=0.56, tau_res_long=0.68, slot_topk_per_proto=2, slot_max_gap=64, slot_tau=0.58, slot_margin=0.05, min_track_age_for_slot=3.

## Track A

- U-Recall: 0.7878 -> 0.7878
- same-prototype: 1.0000 -> 1.0000
- memory_growth: 0.0021 -> 0.0021

## Track C

- candidate-pool-nonempty-rate: 0.2143 -> 0.2143
- same-track: 0.2353 -> 0.2353
- same-prototype: 0.8235 -> 0.8235
- same-track-after-concept: 0.2857 -> 0.2857
- slot-pool-nonempty-rate: 0.0000
- slot-resurrection-attempt-rate: 0.0000
- slot-resurrection-success-rate: 0.0000
- new-track-with-old-prototype-rate: 0.0000
- PFR: 3.0000 -> 3.0000
- IDSW: 363 -> 363

## Verdict

- status: fail
- Interpretation: this round is only successful if concept-recovered events stop failing because the candidate pool is empty.
