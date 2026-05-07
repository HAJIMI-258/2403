# Phase 3D Stage A.7 Remap Summary

- target event: event_id=6, frame=990, gt_object_id=2
- offline target lineage metadata: 2
- target prototype: 3
- remap rows: 522
- overwrite bug rows: 391

- earliest observable overwrite:
  frame=705
  event_type=prototype_create
  object_type=prototype
  object_id=3
  code_location=_create_prototype/_allocate_lineage
  after_runtime_owner_lineage_id=0
  after_continuity_lineage_id=0

## 结论

1. 当前不是 preserve/tie-break 太弱，而是 old continuity key 在更早的 runtime rebind / create 边界已经被改写。
2. prototype 3 的最早可观察记录已经是 runtime_owner_lineage=0 且 continuity_lineage=0；因此后续 continuation/anchor 只能继续复制 0。
3. 只保 continuity key 还不够；如果 preserve-input 仍只按 runtime owner 枚举，lineage 2 依旧不可见。
4. 一旦显式拆开 runtime owner 与 continuity owner，并让 continuity source 按 continuity key 枚举，lineage 2 才重新有机会进入 preserve-input / claim-builder。