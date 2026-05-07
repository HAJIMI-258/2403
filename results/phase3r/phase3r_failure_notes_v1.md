# Phase 3R Failure Notes v1

## Track C Before / After

- same-track recovery: 0.1176 -> 0.2353
- same-prototype recovery: 0.2353 -> 0.8235
- same-track-after-concept: 0.0000 -> 0.2857
- PFR: 8.3333 -> 3.3333
- IDSW: 390 -> 347

## Gap Bucket Readout

- long_gap: same-track=0.0000, same-prototype=0.0000, fragmentation_proxy=1.0000
- medium_gap: same-track=0.0000, same-prototype=0.5000, fragmentation_proxy=0.5000
- short_gap: same-track=0.3000, same-prototype=1.0000, fragmentation_proxy=0.0000
- very_long_gap: same-track=0.5000, same-prototype=1.0000, fragmentation_proxy=0.0000

## Bottleneck

- The main bottleneck is no longer concept reuse. Same-prototype recovery is already high on short / very-long buckets.
- The remaining blocker is same-track recovery through medium / long gaps, where dormant reactivation is still too weak and old track ids are not consistently recovered.
- PFR improved but remains above the acceptance line, which means concept-only attachment is helping but still not preventing enough prototype fragmentation end-to-end.
