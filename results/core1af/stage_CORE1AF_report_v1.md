# CORE-1AF Descriptor Opportunity Audit

This stage explains why CORE-1AE rejected descriptor integration. It audits whether the selected descriptor gate has enough clean failure-rescue opportunities.

## Result

- Gate: A9_score060_cost040_consecutive
- Selected variant: A6_gated_fusion_w020_margin005
- Queries: 297
- Baseline failures: 13
- Clean descriptor rescues: 2
- Control-confounded rescues: 0
- Descriptor opportunity rate: 0.1538
- Main class: baseline_success_no_opportunity

## Interpretation

The descriptor cue has some signal, but the current selected gate contains too few clean baseline failures. Integration would be premature because the observed gain is a small number of rescues, with controls close behind.

Next recommendation: CORE-1AG mine harder non-oracle descriptor-opportunity events before any integration
