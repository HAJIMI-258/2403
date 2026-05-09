# EXT-13 Frozen External Evaluation Protocol

This protocol freezes the current full-pixel evaluation state.

Allowed:
- Rerun existing EXT-4, EXT-5, EXT-5C, EXT-6, EXT-7, EXT-9, EXT-10, EXT-12, EXT-8 scripts unchanged.
- Add more LaSOT categories only through the EXT-11/EXT-13 download plan.
- Report oracle-proposal memory-only results as oracle-proposal memory-only results.

Forbidden:
- Do not add new appearance / descriptor / embedding fusion variants inside the frozen rerun.
- Do not tune event-conditioned routing rules on the test split.
- Do not merge external geometry calibration into main NOPS.
- Do not claim full-perception results.
- Do not use target identity, GT instance id, or future frames in online scoring.

If a new method is needed, create a new stage after EXT-13 and keep it separate from the frozen protocol.
