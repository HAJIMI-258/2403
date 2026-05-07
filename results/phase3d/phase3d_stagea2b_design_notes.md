# Phase 3D Stage A.2b Design Notes

## Split

- `TemporaryObservationAttach`: current observation state near a lineage/prototype; not a legal old-track restore source.
- `RecoveryIdentityAnchor`: lineage-level minimal old-identity source that survives retire/archive and can be consumed by resurrection.

## Consumer Order

1. same-lineage dormant/ghost tracks
2. same-lineage continuation bank
3. same-lineage recovery identity anchor

## Current Limit

- Stage A.2b fixes recovery-surface evaporation.
- It does not yet protect the target proposal from being consumed earlier by unrelated `active_match`.
- The next structural repair point is upstream of anchor consume: proposal-to-track assignment routing at the exact re-entry frame.
