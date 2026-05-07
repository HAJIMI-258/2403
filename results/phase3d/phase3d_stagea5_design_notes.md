# Phase 3D Stage A.5 Design Notes

1. Rerouted proposals now carry an explicit final claim set, not just raw grouped claims.
2. Matched-lineage and recovery-hint lineages are preserved into the final claim set if they expose legal recovery evidence.
3. Identity-aware tie-break is only used after visibility is established and only on rerouted proposals.
4. Within-lineage old-identity selection stays separate from lineage-level claim visibility and ranking.
