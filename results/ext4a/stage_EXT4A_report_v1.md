# EXT-4A Full-Pixel Appearance Validation

## Scope

This is oracle-proposal, memory-only validation on LaSOT dog pixels linked to LaGOT annotations.

It does not test detection or full perception.

## Result

- Events: `62`
- Geometry passive top1: `0.5000`
- External geometry branch top1: `0.5968`
- Best variant: `A4_external_trajectory_heavy`
- Best top1: `0.5968`
- Best appearance variant: `A5_external_trajectory_plus_appearance_w010`
- Best appearance top1: `0.5645`
- Appearance helped vs geometry passive: `1`
- Appearance beat external geometry branch: `0`
- Mean appearance margin: `-0.030378`
- Appearance margin positive rate: `0.5323`

## Decision

Do not merge into main NOPS.

Appearance improved the current passive geometry score on this dog subset, but it did not beat the isolated external geometry branch, and the mean target-vs-wrong appearance margin is still negative.

Next: run larger multi-category full-pixel validation and audit appearance descriptor failures before any integration.
