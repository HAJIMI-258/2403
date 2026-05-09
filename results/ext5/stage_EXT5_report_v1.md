# EXT-5 Multi-category Full-pixel Appearance Validation

## Result

- Pixel-ready events: `504`
- Categories: `12`
- Geometry passive top1: `0.5913`
- Geometry + appearance best top1: `0.6012`
- External branch top1: `0.6786`
- External branch + appearance best top1: `0.6806`
- Best variant: `A6_external_trajectory_plus_appearance_w005`
- Best top1: `0.6806`
- Mean appearance margin: `0.086332`
- Appearance margin positive rate: `0.6052`

## Decision

This stage validates multi-category full-pixel evaluation, but appearance is not safe for main NOPS merge.

Current recommendation: appearance may help external branch; require real shuffled controls and larger validation before integration
