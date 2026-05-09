# EXT-9 Event-Conditioned Geometry Failure Analysis

## Result

- Events: `234`
- Geometry passive top1: `0.5513`
- External branch top1: `0.5940`
- Delta counts: `{'both_success': 119, 'external_regressed': 10, 'external_rescued': 20, 'both_failure': 85}`
- Best diagnostic gate: `A1_all_external`
- Best diagnostic gate top1: `0.5940`

## Decision

No gate is integration-ready. These are same-subset diagnostics only.
use this as failure analysis only; next either expand full-pixel events or create train/dev split before event-conditioned geometry routing
