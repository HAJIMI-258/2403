# Phase 3R.3 Failure Notes v1

## Track C Core Readout

- candidate-pool-nonempty-rate: 0.2143 -> 0.2143
- same-track: 0.2353 -> 0.2353
- same-prototype: 0.8235 -> 0.8235
- same-track-after-concept: 0.2857 -> 0.2857
- slot-pool-nonempty-rate: 0.0000
- slot-resurrection-attempt-rate: 0.0000
- slot-resurrection-success-rate: 0.0000
- PFR: 3.0000 -> 3.0000
- IDSW: 363 -> 363

## Bottleneck

- If candidate-pool-nonempty-rate stays low, slot preservation is still too weak and identity is leaving the pool before concept recovery arrives.
- If candidate-pool-nonempty-rate rises but same-track-after-concept stays low, the remaining blocker is candidate selection, not pool coverage.
- If same-prototype drops below 0.80, the slot layer is polluting the concept layer and should be rolled back.
