# Phase 3 Failure Notes v1

## track_a_bridge
- Primary failure: tracking drift remains high even when re-entry reconnects.
- Budget status: controlled. No budget-violation frames were observed.
- Headline metrics: U-Recall=0.7878, track IDSW=131, reentry=1.0000, PFR=1.0000.

## track_c_long_horizon
- Primary failure: long-gap re-entry is not reconnecting to prior track IDs reliably.
- Secondary effect: tracker instability is leaking into prototype fragmentation and PFR.
- Budget status: controlled. No budget-violation frames were observed.
- Headline metrics: U-Recall=0.6023, track IDSW=638, reentry=0.0303, PFR=4.3333.
