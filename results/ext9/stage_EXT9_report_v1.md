# EXT-9 Event-Conditioned Geometry Failure Analysis

## Result

- Events: `406`
- Geometry passive top1: `0.5739`
- External branch top1: `0.6601`
- Delta counts: `{'both_failure': 126, 'both_success': 221, 'external_rescued': 47, 'external_regressed': 12}`
- Best diagnostic gate: `A1_all_external`
- Best diagnostic gate top1: `0.6601`

## Decision

No gate is integration-ready. These are same-subset diagnostics only.
use this as failure analysis only; next either expand full-pixel events or create train/dev split before event-conditioned geometry routing
