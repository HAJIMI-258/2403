# EXT-5C Appearance Control Audit

## Result

- Events: `406`
- Real geometry appearance gain: `0.0123`
- Shuffled geometry appearance gain: `0.0099`
- Category-shuffled geometry appearance gain: `0.0049`
- Real external-branch appearance gain: `0.0025`
- Shuffled external-branch appearance gain: `-0.0049`
- Category-shuffled external-branch appearance gain: `0.0049`
- Controls passed: `0`

## Decision

Appearance is not safe for main NOPS merge.

controls indicate appearance gains are not reliable; do not integrate
