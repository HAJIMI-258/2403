# CORE-1AQ Uncertainty State Integration Audit

This stage converts the split-validated uncertainty gate into a memory decision trace and an evidence queue. It does not change ranking, attach identities, promote tracks, or update heads.

## Result

- Queries: 495
- Old recall candidates: 419
- Uncertain queue size: 76
- Coverage: 0.8465
- Old-recall precision eval-only: 0.9594
- False old before policy: 23
- False old after policy: 17
- False old suppressed: 6
- Unnecessary uncertain decisions: 70
- Policy violations: 0
- Integration passed: 1

Next recommendation: CORE-1AR connect uncertainty state to active evidence / delayed update without attach-promotion
