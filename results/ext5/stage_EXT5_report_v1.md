# EXT-5 Multi-category Full-pixel Appearance Validation

## Result

- Pixel-ready events: `234`
- Categories: `5`
- Geometry passive top1: `0.5513`
- Geometry + appearance best top1: `0.5684`
- External branch top1: `0.5940`
- External branch + appearance best top1: `0.5940`
- Best variant: `A5_external_trajectory_heavy`
- Best top1: `0.5940`
- Mean appearance margin: `0.068487`
- Appearance margin positive rate: `0.6282`

## Decision

This stage validates multi-category full-pixel evaluation, but appearance is not safe for main NOPS merge.

Current recommendation: keep appearance as auxiliary diagnostic only; do not integrate
