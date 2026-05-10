# CORE-1X Cross-Event Negative Mining

This stage avoids same-frame fragment pseudo-negatives by mining negatives from other windows, events, or sequences among high-confidence matched observations. GT is used only for audit.

## Result

- Positive pairs: 136
- Best negative mode: cross_event_same_sequence
- Best negative pair count: 580
- Best positive precision eval-only: 0.9485
- Best negative precision eval-only: 0.8000
- Passed: 0

Next recommendation: cross-context negatives still insufficient; expand window/sequence sample or use oracle proposals for diagnostic encoder
