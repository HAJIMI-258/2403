# CORE-1AW Policy Flag Harness

This stage adds a disabled-by-default evaluation harness for the memory decision policy. With the flag off, behavior is forced old recall. With the flag on, low-margin recalls use bounded wait.

## Result

- Query count: 667
- Disabled precision: 0.9505
- Enabled precision: 0.9615
- Disabled false old recalls: 33
- Enabled false old recalls: 25
- Enabled coverage: 0.9730
- Delayed old recalls: 82
- Unresolved: 18
- Policy violations: 0
- Harness passed: 1

Next recommendation: CORE-1AX run external/synthetic documentation update and mark policy experimental-disabled
