# Phase A Result Summary

## 目标

阶段A只修一件事：让正确 continuity owner 真正变成可枚举、可追踪、可进入 preserve-input / claim-builder 的 recovery source。

## 最小回归

- `track_a_bridge / U-Recall = 0.7878`
- `track_a_bridge / PFR = 4.0000`
- `track_a_bridge / IDSW = 941`
- `track_c_long_horizon / U-Recall = 0.7247`
- `track_c_long_horizon / PFR = 9.3333`
- `track_c_long_horizon / IDSW = 2975`

## Target Event

- `event_id = 6`
- `frame = 990`
- `target_lineage_id = 2`
- `offline_expected_lineage_id = 2`
- `runtime_old_lineage_id = None`
- `runtime_old_continuity_lineage_id = None`
- `continuity_source_visible = 0`
- `claim_visible = 0`
- `final_selected_lineage = 0`
- `failure_class = visibility_failure`

## 解释

阶段A只看 source visibility。若 target continuity lineage 重新进入 source pool / claim-builder，则说明问题已从 visibility failure 推进到 attach/ranking failure；若它仍不可见，则说明 dual-owner 枚举还没有真正接通。

## A.7 overwrite 审计复用

- `results/phase3d/phase3d_stagea7_remap_trace.csv` 继续作为 continuity key overwrite 审计基准。
- 本轮统一 runner 只把 overwrite bug 抽取成 `continuity_key_overwrite_events.csv`，不重复跑更深的 remap trace。