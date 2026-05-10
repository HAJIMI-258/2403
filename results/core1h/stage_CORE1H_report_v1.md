# CORE-1H Dense Diagnostic Encoder Upper-Bound

CORE-1H trains a small random-init encoder from CORE-1G GT-ledger diagnostic pairs.
This is an upper-bound experiment and is not safe for main NOPS integration.

## Result
- Raw geometry test top1: 0.0455.
- Diagnostic encoder test top1: 0.0000.
- Shuffled-positive control test top1: 0.0000.
- Passed minimum: 0.

## Decision
CORE-1I inspect dense diagnostic features; do not train main encoder from GT-ledger pairs
