# CORE-1 Milestone Freeze v1

## Decision

CORE-1 is frozen as a memory-decision milestone, not as a learned visual embedding milestone.

## Rejected for Integration

Online learned descriptor metric:

- CORE-1AL showed a small top1 improvement.
- CORE-1AM showed the improvement was not separated from shuffled descriptor controls.
- Decision: do not integrate learned metric.

Raw descriptor fusion:

- Earlier CORE-1AD/AE showed small gains but weak robustness.
- Decision: diagnostic only.

## Accepted as Experimental Disabled-by-default Policy

Bounded-wait uncertainty policy:

- Low top1-margin recalls become `uncertain_need_more_evidence`.
- Uncertain recalls do not update memory, attach identity, promote tracks, or update heads.
- The system waits up to 10 frames for a high-margin same-track release.

Evidence:

- CORE-1AP split gate passed.
- CORE-1AS bounded-wait audit passed.
- CORE-1AU end-to-end smoke passed.
- CORE-1AV broader 6-sequence regression passed.
- CORE-1AY parameter sensitivity passed.

## Current Best Result

On the 6-sequence CORE-1AV stream:

- Query count: 667
- Baseline top1: 0.9505
- Policy old-recall precision: 0.9615
- Policy coverage: 0.9730
- False old recall reduction: 8
- Released wrong count: 0

## Interpretation

The useful CORE-1 result is not a better object embedding. The useful result is a safer memory decision layer:

Instead of forcing every low-confidence retrieval into an old-object identity, NOPS can hold the decision as uncertain, wait briefly for better evidence, and release many cases safely.

## Next Core Objective

CORE-2 should focus on object-file consolidation and uncertainty-driven evidence handling:

- maintain bounded uncertain queues,
- learn when to wait versus request active evidence,
- prevent uncertain recalls from contaminating memory,
- consolidate only after delayed high-confidence release,
- keep attach / promotion separate.

CORE-2 should not start by adding pretrained embeddings or external geometry fusion.
