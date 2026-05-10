# CORE-1C Query Gate Split Validation

CORE-1C audits whether the CORE-1B cue-consensus query gate generalizes under leave-one-out and deterministic 3-fold splits.
No encoder training or retrieval integration is performed in this stage.

## Key Results
- Fixed CORE-1B gate selected 7 events with precision 0.8571.
- Leave-one-out selected 6 held-out events with precision 0.5000.
- 3-fold selected 14 test events with precision 0.3571.
- Stable query positive pool size: 3.
- Gate stability passed: 0.

## Decision
CORE-1D collect more query positives / repair query gate before training integration

The split gate selection uses evaluation precision only for audit. It is not an online scoring path and must not be merged into NOPS directly.
