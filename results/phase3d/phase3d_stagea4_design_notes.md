# Phase 3D Stage A.4 Design Notes

1. Resurrection consumer now emits all-lineage recovery candidate rows for rerouted proposals.
2. Candidate comparison is split into lineage-level claims and within-lineage identity selection.
3. The Stage A.4 repair path is gated to rerouted proposals and does not touch normal same-lineage active matches.
4. Two-stage selection first selects a recovery lineage from aggregated claim evidence, then chooses an old identity inside that lineage.
5. Identity-aware tie-breaks prefer richer continuity evidence over a single locally convenient candidate.
