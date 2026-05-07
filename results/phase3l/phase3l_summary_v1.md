# Phase 3L Summary v1

## Track C

- concept_recovered_but_lineage_mismatch_rate: 0.7500 -> 0.2500
- continuation_bank_access_rate_given_concept_recovery: 0.1667 -> 0.7273
- same-lineage-prototype-reentry-recovery: 0.2353 -> 0.7059
- same-track-after-concept-recovery: 0.3333 -> 0.4545
- same-prototype-reentry-recovery: 0.3529 -> 0.6471
- PFR: 4.6667 -> 5.3333

## Config

- tracking override: {'keepalive_frames': 8, 'dormant_frames': 16, 'ghost_frames': 80, 'tau_g': 12, 'tau_res_short': 0.56, 'tau_res_long': 0.68, 'tau_continuation': 0.62, 'continuation_margin': 0.08, 'enable_identity_slots': False}
- memory override: {'protect_linked_prototypes': True, 'enable_explicit_lineage': True, 'preserve_lineage_on_archive': True, 'preserve_lineage_on_replace': True, 'preserve_lineage_on_merge': True, 'allow_alias_lineage': True, 'enable_continuation_bank': True, 'bind_continuation_to': 'lineage', 'continuation_topk_per_proto': 4, 'continuation_topk_per_lineage': 4, 'min_track_age_for_continuation': 4, 'min_hits_for_continuation': 3, 'continuation_max_gap': 96, 'continuation_decay': 0.01, 'decay_patience': 24}
