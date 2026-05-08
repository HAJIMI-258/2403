# External Geometry Branch Profile v1

## Purpose

This profile preserves the EXT-2 / EXT-3 geometry calibration as an isolated external branch. It is not a main NOPS merge.

The branch exists because LaGOT annotation evaluation showed a real external failure mode:

- Current NOPS passive retrieves the target in top5 for most events.
- It loses top1 under similar distractor competition.
- A trajectory-heavy geometry score improves LaGOT oracle-proposal memory-only top1.

## Allowed Scope

- Dataset: LaGOT annotations.
- Proposal mode: oracle GT boxes.
- Evaluation type: geometry-only memory retrieval.
- Use case: external failure analysis and geometry profile comparison.

## Forbidden Scope

- Main NOPS scoring path.
- Synthetic anchor / canonical / episodic replacement.
- Active evidence integration.
- Attach / promotion decisions.
- Full perception claims.

## Evidence

EXT-3 showed:

- A0 current NOPS top1: `0.6406`
- A2 trajectory-heavy top1: `0.7279`
- Support trajectory reference top1: `0.7486`
- Regression rate: `0.0297`
- Integration gate for isolated external geometry branch: passed

SYN-REG-1 showed:

- A2 trajectory-heavy synthetic top1: `0.2353`
- Focus success: `1/3`
- Synthetic regression: failed
- Safe main merge: false

## Decision Rule

The branch can be used for external annotation geometry analysis. It cannot be merged into main NOPS until:

- synthetic regression passes,
- negative controls pass,
- full-pixel / appearance validation is available,
- and main NOPS anchor/canonical/episodic behavior is preserved.

