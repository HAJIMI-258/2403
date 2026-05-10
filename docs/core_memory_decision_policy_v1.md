# Core Memory Decision Policy v1

This document freezes the CORE-1AQ through CORE-1AS decision contract.

## Scope

The policy only decides whether a retrieval result is safe enough to expose as an old-object recall candidate.

It does not perform identity attachment, promotion, prototype head update, detector update, or final tracking decisions.

## States

`old_recall_candidate`

- Used when the online top1 retrieval margin is at or above the uncertainty threshold.
- May be passed downstream as a memory retrieval proposal.
- Does not authorize attach, promotion, or head update.

`uncertain_need_more_evidence`

- Used when the online top1 retrieval margin is below the uncertainty threshold.
- Must not update memory.
- Must not attach identity.
- Must not promote tracks.
- Must not update prototype heads.
- May enqueue active evidence acquisition or bounded delayed resolution.

## Current Parameters

- `uncertainty_margin_threshold = 0.0194`
- `bounded_wait_horizon_frames = 10`

These values come from CORE-1AP split validation and CORE-1AS bounded-wait audit.

## Evidence

CORE-1AP split gate:

- Query count: 495
- Baseline top1: 0.9535
- Split committed top1: 0.9594
- Split coverage: 0.8465
- False old recalls suppressed: 6

CORE-1AS bounded wait:

- Uncertain decisions: 76
- Resolved correct: 60
- Released wrong: 0
- Unresolved after horizon: 16
- Mean wait: 1.68 frames

## Safety Rule

Any implementation that allows `uncertain_need_more_evidence` to trigger memory update, attach, promotion, or head update violates the CORE-1 contract.
