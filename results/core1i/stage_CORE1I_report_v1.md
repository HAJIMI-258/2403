# CORE-1I Dense Feature Failure Audit

CORE-1I checks whether CORE-1H failed because of broad candidate scope or because fast-planned geometry is intrinsically weak.

## Result
- Same-sequence same-concept top1: 0.9255.
- Same-sequence top1: 0.6631.
- Same-split top1: 0.0337.
- Main failure counts: {'candidate_scope_too_broad': 564, 'large_reentry_displacement': 364, 'fast_planned_area_degenerate': 207, 'same_concept_geometry_collision': 42, 'different_concept_geometry_collision': 148}.

## Decision
CORE-1J render selected dense ledger windows and mine tracker-derived online pairs
