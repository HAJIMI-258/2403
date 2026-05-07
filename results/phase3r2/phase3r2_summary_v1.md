# Phase 3R.2 Summary v1

## Selected Params

- keepalive_frames=8, dormant_frames=16, ghost_frames=80.
- tau_g=12.00, tau_res_short=0.56, tau_res_long=0.68.

## Track A

- U-Recall: 0.7878 -> 0.7878
- same-prototype: 0.0000 -> 1.0000
- memory_growth: 0.0010 -> 0.0021

## Track C

- same-track: 0.2353 -> 0.2353
- same-prototype: 0.8235 -> 0.8235
- same-track-after-concept: 0.2857 -> 0.2857
- prototype-gated resurrection attempt rate: 0.2143
- resurrection success | candidate exists: 1.0000
- PFR: 3.3333 -> 3.0000
- IDSW: 347 -> 363

## Verdict

- status: fail
- Interpretation: this round is only successful if old-track resurrection becomes a default path after concept recovery, not a rare exception.
