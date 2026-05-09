# EXT-9 Event-Conditioned Geometry Failure Analysis

## Result

- Events: `504`
- Geometry passive top1: `0.5913`
- External branch top1: `0.6786`
- Delta counts: `{'both_failure': 146, 'both_success': 282, 'external_rescued': 60, 'external_regressed': 16}`
- Best diagnostic gate: `A1_all_external`
- Best diagnostic gate top1: `0.6786`

## Decision

No gate is integration-ready. These are same-subset diagnostics only.
use this as failure analysis only; next either expand full-pixel events or create train/dev split before event-conditioned geometry routing
