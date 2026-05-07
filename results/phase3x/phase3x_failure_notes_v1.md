# Phase 3X Failure Notes v1

## Track C Readout

- Phase 3S strict same-prototype: 0.3529 -> 0.1765
- lineage-aware same-prototype: 0.2353
- continuation_write_success_rate: 0.9624
- continuation_survival_until_concept_recovery_rate: 0.4667
- concept_recovered_but_lineage_mismatch_rate: 0.7500
- continuation_bank_access_rate_given_same_lineage: 0.5000
- dominant failure stage: lineage_mismatch
- primary loss stage: after_archive

## Decision Rule

- If lineage-aware recovery is much higher than strict recovery, the next branch is lineage-preserving prototype update and lineage-aware evaluation.
- If write/survival is low, the next branch is continuation lifecycle repair.
- If write and survival are fine but access is low, the next branch is concept-to-continuation binding.
