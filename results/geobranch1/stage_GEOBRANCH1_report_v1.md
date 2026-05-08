# GEO-BRANCH-1 External Geometry Branch

## Decision

The EXT-2 / EXT-3 trajectory-heavy calibration is packaged as an isolated external geometry branch.

It is not safe to merge into main NOPS.

## Evidence

- External A0 top1: `0.6406`
- External trajectory-heavy top1: `0.7279`
- External top1 delta: `0.0874`
- Support-trajectory reference top1: `0.7486`
- EXT-3 integration gate: `1`
- SYN-REG synthetic regression passed: `0`
- SYN-REG safe external branch: `1`
- SYN-REG safe main merge: `0`

## Allowed Use

- LaGOT annotation oracle-proposal geometry-only memory benchmark.
- External geometry failure analysis.
- Isolated profile comparison.

## Forbidden Use

- Main NOPS scoring merge.
- Synthetic anchor/canonical/episodic path replacement.
- Active evidence integration.
- Attach / promotion decisions.
- Full perception claims.

## Next Step

Keep this branch quarantined. Use it for external annotation geometry analysis only.

Before any main merge, run full-pixel / appearance validation and pass synthetic regression.
