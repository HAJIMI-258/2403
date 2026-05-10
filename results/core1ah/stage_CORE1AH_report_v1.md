# CORE-1AH Broad Hard-Pool Tradeoff

This stage scans all CORE-1AA gates, including noisy low-confidence gates, to measure the tradeoff between hard retrieval opportunities, pair quality, descriptor rescues, and controls.

## Result

- Gate/variant combinations: 140
- Best gate: A0_core1y_all
- Best variant: A1_raw_descriptor_only
- Best baseline top1: 0.9525
- Best top1: 0.9062
- Baseline failures: 41
- Clean rescues: 26
- Regressions: 69
- Pair quality passed: 0
- Hard pool found: 0

Next recommendation: no reliable hard descriptor pool; prioritize proposal/observation quality repair over descriptor integration
