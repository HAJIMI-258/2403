# EXT-4B Appearance Failure Audit

## Result

Appearance is not ready as a main retrieval branch.

- Geometry passive top1: `0.5000`
- Geometry + appearance top1: `0.5323`
- External trajectory branch top1: `0.5968`
- External trajectory + appearance top1: `0.5645`
- Geometry + appearance improved events: `2`
- Geometry + appearance regressed events: `0`
- External branch + appearance improved events: `1`
- External branch + appearance regressed events: `3`
- Mean appearance margin: `-0.030378`
- Severe negative margin count: `13`

## Interpretation

Raw crop appearance helps the weaker geometry-passive baseline in a few cases, but it weakens the stronger trajectory-heavy external branch. The mean target-vs-wrong appearance margin is still negative.

The correct next step is larger multi-category full-pixel validation and descriptor failure analysis, not main NOPS integration.
