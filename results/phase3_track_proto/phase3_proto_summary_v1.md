# Phase 3 Prototype Summary v1

## Scope

This bundle records the first complete Phase 3 prototype pass for generic bridge synthetic under the NOPS-Bench v2 protocol.

## Scenario Summary

| scenario | U-Recall | track IDSW | reentry recovery | PFR | memory growth | budget violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| track_a_bridge | 0.7878 | 131 | 1.0000 | 1.0000 | 0.0042 | 0 |
| track_c_long_horizon | 0.6023 | 638 | 0.0303 | 4.3333 | 0.0036 | 0 |

## Protocol Check

- Hard-synthetic reference U-Recall: 0.6874
- Track A bridge keeps unknown discovery above the hard-synthetic reference: yes
- Track C long-horizon keeps unknown discovery near the hard-synthetic reference: no
- Tracking / re-entry is now the dominant bottleneck: yes
- Memory stays inside the declared budget: yes
- Prototype fragmentation remains controlled: no

## Go / No-Go

- Go for continued Phase 3 tracker work on generic bridge synthetic.
- No-Go for advancing to Track B, real-data claims, or broader benchmark expansion.
- The main blocker is long-gap tracking / re-entry, with prototype fragmentation as a secondary effect.

## Artifacts

- track_a_bridge: results\phase3_track_proto\track_a_bridge_summary_v1.json, results\phase3_track_proto\track_a_bridge_tracking_diag_v1.png, results\phase3_track_proto\track_a_bridge_memory_diag_v1.png, results\phase3_track_proto\track_a_bridge_budget_report_v1.csv
- track_c_long_horizon: results\phase3_track_proto\track_c_long_horizon_summary_v1.json, results\phase3_track_proto\track_c_long_horizon_tracking_diag_v1.png, results\phase3_track_proto\track_c_long_horizon_memory_diag_v1.png, results\phase3_track_proto\track_c_long_horizon_budget_report_v1.csv
