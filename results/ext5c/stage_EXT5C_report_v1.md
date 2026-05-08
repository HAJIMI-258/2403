# EXT-5C Appearance Control Audit

## Result

- Events: `234`
- Real geometry appearance gain: `0.0171`
- Shuffled geometry appearance gain: `0.0128`
- Category-shuffled geometry appearance gain: `0.0171`
- Real external-branch appearance gain: `0.0000`
- Shuffled external-branch appearance gain: `-0.0043`
- Category-shuffled external-branch appearance gain: `0.0128`
- Controls passed: `0`

## Decision

Appearance is not safe for main NOPS merge.

controls indicate appearance gains are not reliable; do not integrate
