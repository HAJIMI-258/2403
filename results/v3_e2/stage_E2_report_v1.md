# Stage E2 Report

## 目标

只比较 source enumeration：runtime-owner only vs runtime+continuity dual-owner。
事件锚点使用 E1 产出的 event audit，不再用 E2 replay 自行重建 target metadata。

## Paired Summary

- `events = 18`
- `proposal_detected_events = 17`
- `source_visible_before = 14`
- `source_visible_after = 14`
- `claim_visible_before = 14`
- `claim_visible_after = 14`
- `improved_visibility_events = 0`
- `improved_claim_events = 0`
- `new_continuity_owner_visible_events = 0`
- `proposal_detected_source_visible_before = 14`
- `proposal_detected_source_visible_after = 14`

## Mode Summary

### runtime_only

- `overall = {'events': 18, 'svr': 0.7777777777777778, 'claim_visibility': 0.7777777777777778, 'continuity_owner_visible': 0.0, 'runtime_owner_visible': 0.7777777777777778}`
- `proposal_detected_only = {'events': 17, 'svr': 0.8235294117647058, 'claim_visibility': 0.8235294117647058, 'continuity_owner_visible': 0.0, 'runtime_owner_visible': 0.8235294117647058}`
- `track_a_bridge overall = {'events': 1, 'svr': 1.0, 'claim_visibility': 1.0, 'continuity_owner_visible': 0.0, 'runtime_owner_visible': 1.0}`
- `track_a_bridge proposal_detected_only = {'events': 1, 'svr': 1.0, 'claim_visibility': 1.0, 'continuity_owner_visible': 0.0, 'runtime_owner_visible': 1.0}`
- `track_c_long_horizon overall = {'events': 17, 'svr': 0.7647058823529411, 'claim_visibility': 0.7647058823529411, 'continuity_owner_visible': 0.0, 'runtime_owner_visible': 0.7647058823529411}`
- `track_c_long_horizon proposal_detected_only = {'events': 16, 'svr': 0.8125, 'claim_visibility': 0.8125, 'continuity_owner_visible': 0.0, 'runtime_owner_visible': 0.8125}`

### dual_owner

- `overall = {'events': 18, 'svr': 0.7777777777777778, 'claim_visibility': 0.7777777777777778, 'continuity_owner_visible': 0.0, 'runtime_owner_visible': 0.7777777777777778}`
- `proposal_detected_only = {'events': 17, 'svr': 0.8235294117647058, 'claim_visibility': 0.8235294117647058, 'continuity_owner_visible': 0.0, 'runtime_owner_visible': 0.8235294117647058}`
- `track_a_bridge overall = {'events': 1, 'svr': 1.0, 'claim_visibility': 1.0, 'continuity_owner_visible': 0.0, 'runtime_owner_visible': 1.0}`
- `track_a_bridge proposal_detected_only = {'events': 1, 'svr': 1.0, 'claim_visibility': 1.0, 'continuity_owner_visible': 0.0, 'runtime_owner_visible': 1.0}`
- `track_c_long_horizon overall = {'events': 17, 'svr': 0.7647058823529411, 'claim_visibility': 0.7647058823529411, 'continuity_owner_visible': 0.0, 'runtime_owner_visible': 0.7647058823529411}`
- `track_c_long_horizon proposal_detected_only = {'events': 16, 'svr': 0.8125, 'claim_visibility': 0.8125, 'continuity_owner_visible': 0.0, 'runtime_owner_visible': 0.8125}`

