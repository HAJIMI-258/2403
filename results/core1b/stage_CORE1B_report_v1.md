# CORE-1B Query Pair Gate Repair

CORE-1B scans online-visible query-positive gates after CORE-1A showed baseline-top1 pseudo labels are too noisy.

- Best scanned gate count: `7`
- Best scanned gate precision eval-only: `0.8571428571428571`
- Candidate gate selected events: `M-RE-TC-005,M-RE-TC-008,M-RE-TC-009,M-RE-TC-003,M-RE-TC-012,M-RE-TC-013,M-RE-TC-016`
- Candidate gate precision eval-only: `0.8571428571428571`
- Best ablation: `A0_current_NOPS_passive`
- Best top1: `0.4117647058823529`
- Passed minimum: `0`
- Next recommendation: `CORE-1C split-validate cue-consensus query gate and collect more query positives; do not integrate yet`
