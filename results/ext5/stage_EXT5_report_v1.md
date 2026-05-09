# EXT-5 Multi-category Full-pixel Appearance Validation

## Result

- Pixel-ready events: `406`
- Categories: `9`
- Geometry passive top1: `0.5739`
- Geometry + appearance best top1: `0.5862`
- External branch top1: `0.6601`
- External branch + appearance best top1: `0.6626`
- Best variant: `A6_external_trajectory_plus_appearance_w005`
- Best top1: `0.6626`
- Mean appearance margin: `0.071438`
- Appearance margin positive rate: `0.6108`

## Decision

This stage validates multi-category full-pixel evaluation, but appearance is not safe for main NOPS merge.

Current recommendation: appearance may help external branch; require real shuffled controls and larger validation before integration
