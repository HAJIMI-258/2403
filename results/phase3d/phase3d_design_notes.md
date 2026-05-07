# Phase 3D Design Notes

## Core Change

Phase 3D Stage A separates lineage match, temporary recovery attach, and head promotion state.

## Stage A Structures

- dual score channels: `attach_score` vs `promote_score`
- lineage-local temporary attach slot
- deferred promotion state carried on the lineage backbone

## Stage A Rules

- matched lineage may attach to head, active sibling, archived sibling, or temp attach slot
- attach acceptance is looser than promotion eligibility
- temp attach does not count as sibling birth and does not change current head ownership
- promotion state is logged, but Stage A keeps promotion conservative and mostly deferred

## Intended Fix

Stage A is only meant to prove that recovery attach can remain alive without immediately forcing head replacement.