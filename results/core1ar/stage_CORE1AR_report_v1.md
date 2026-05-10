# CORE-1AR Delayed Uncertainty Resolution

This stage audits whether `uncertain_need_more_evidence` decisions can be resolved by waiting for a later high-margin observation on the same online-visible track. Future observations are used only for offline evaluation, not for the current decision.

## Result

- Uncertain decisions: 76
- Best horizon: 10 frames
- Resolved correct: 60
- Released wrong: 0
- Unresolved: 16
- Resolution rate: 0.7895
- False-suppressed cases resolved: 2
- Correct delayed cases resolved: 58
- Mean wait frames: 1.68
- Delayed resolution passed: 1

Next recommendation: CORE-1AS add delayed uncertainty resolution policy with bounded wait horizon
