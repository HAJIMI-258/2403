# Phase 3S Failure Notes v1

## Track C Core Readout

- continuation-bank-nonempty-rate: 0.0000 -> 0.1667
- candidate-pool-nonempty-rate: 0.2143 -> 0.1667
- same-track: 0.2353 -> 0.2353
- same-prototype: 0.8235 -> 0.3529
- same-track-after-concept: 0.2857 -> 0.3333
- continuation-attempt-rate: 0.0000
- continuation-success-rate: 0.0000
- PFR: 3.0000 -> 4.6667
- IDSW: 363 -> 366

## Bottleneck

- If continuation-bank-nonempty-rate rises but same-track-after-concept stays flat, the remaining blocker is candidate selection, not continuation storage.
- If continuation-bank-nonempty-rate stays low, prototype-owned continuation is still not being archived onto the same prototypes that later recover concept identity.
- If same-prototype drops below 0.80, continuation handling is polluting the concept layer and should be rolled back.
