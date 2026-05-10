# CORE-1F Dense Internal Event Ledger

CORE-1F generates a larger internal synthetic event ledger so CORE encoder work is not bottlenecked by the original 17 hand-curated re-entry events.
GT is used only to define evaluation events and audit opportunities, not online scoring.

## Result
- Sequences generated: 480.
- Real-gap events: 564.
- Train/dev/test: 320/134/110.
- Adjacent positive pair opportunities: 1598510.
- Co-visible negative pair opportunities: 2261154.
- Dense ledger ready: 1.

## Decision
CORE-1G run NOPS/core encoder pair-mining on dense internal ledger
