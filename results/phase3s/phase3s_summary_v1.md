# Phase 3S Summary v1

## Selected Params

- keepalive_frames=8, dormant_frames=16, ghost_frames=80, tau_g=12.00.
- tau_continuation=0.62, continuation_margin=0.05, continuation_topk_per_proto=2, continuation_max_gap=96, min_track_age_for_continuation=3.

## Track A

- U-Recall: 0.7878 -> 0.7878
- same-prototype: 1.0000 -> 0.0000
- memory_growth: 0.0021 -> 0.0031

## Track C

- continuation-bank-nonempty-rate: 0.0000 -> 0.1667
- candidate-pool-nonempty-rate: 0.2143 -> 0.1667
- same-track: 0.2353 -> 0.2353
- same-prototype: 0.8235 -> 0.3529
- same-track-after-concept: 0.2857 -> 0.3333
- continuation-attempt-rate: 0.0000
- continuation-success-rate: 0.0000
- new-track-with-old-prototype-rate: 0.0000
- PFR: 3.0000 -> 4.6667
- IDSW: 363 -> 366

## Verdict

- status: fail
- Interpretation: this round succeeds only if concept-recovered events stop failing because prototype-owned continuation is missing.
