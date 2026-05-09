# EXT-8 External Evidence Synthesis

## Decision

- External geometry branch is valid only as an isolated external profile.
- Main NOPS merge is not allowed because SYN-REG-1 failed.
- Raw appearance and frozen embedding fusion are rejected for integration because controls/significance failed.
- Handcrafted strong descriptor also failed the split gate, so it is not allowed as external auxiliary.
- Frozen ResNet18 remains useful only as an external pretrained diagnostic baseline.

## Current Best Numbers

- Annotation external geometry branch top1: `0.7279472382522671`
- Full-pixel external geometry branch top1: `0.6600985221674877`
- Full-pixel events/categories: `406` / `9`

## Next

download target-500 categories or keep isolated all-external geometry branch; do not run more appearance/embedding fusion
