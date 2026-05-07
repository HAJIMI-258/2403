# Phase 3R.2 Failure Notes v1

## Track C Core Readout

- same-track: 0.2353 -> 0.2353
- same-prototype: 0.8235 -> 0.8235
- same-track-after-concept: 0.2857 -> 0.2857
- PFR: 3.3333 -> 3.0000
- IDSW: 347 -> 363

## Bottleneck

- If same-prototype stays high while same-track-after-concept stays low, the remaining problem is still identity continuation, not concept recovery.
- If candidate_exists_events grows but resurrection_success_given_candidate_exists stays low, the blocker is the resurrection cost or threshold, not lifecycle coverage.
- If candidate_exists_events remains low, then ghost coverage is still too short and old tracks are leaving the candidate pool too early.
