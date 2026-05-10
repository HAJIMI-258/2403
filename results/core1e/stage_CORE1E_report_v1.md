# CORE-1E Query Positive Source Expansion

CORE-1E checks whether the project has enough high-precision query positives to continue online encoder work.

## Internal Synthetic
- Stable internal query positives: 3 at precision 1.0000.
- This is not enough to train or integrate a main CORE encoder.

## External Geometry
- Best external source: external_a0_cal_ref_all_agree.
- Positives: 880, precision 0.8057.
- Categories: 67.
- Controls passed: 0.

## Decision
CORE-1F create denser internal synthetic event ledger before training main online encoder

External positives may support an isolated diagnostic alignment experiment. They must not be used as main NOPS training data because they come from oracle-proposal external geometry evaluation.
