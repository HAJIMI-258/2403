# Phase 3D Stage A.2 Consume Trace

## Main Finding

Stage A.1 里的 `attach_written=1` 不能直接解释成“目标 re-entry 事件已经走进 recovery attach 主路径”。在当前 traced baseline frame 里，目标 lineage 本身没有任何可消费的 recovery 候选；CSV 里看到的 4 条 `attach_written=1` 行，实际都来自别的 lineage 的 `active_match -> temp_attach` 流量，不是目标 lineage 的 old-track recovery。

## Baseline Target Frame

目标仍按 Stage A.1 的 matched-lineage 行取：

- `frame = 990`
- `target_lineage = 2`
- `target_old_track_id = 188`
- `target_old_prototype_id = 3`

当前已经查实：

- `target-lineage assignment rows at frame 990 = 0`
- `unrelated attach-written rows at frame 990 = 4`
- 这 4 条 attach-written 行的 assignment lineage 分别落在 `3 / 3 / 0 / 5`
- `lineage 2 active/dormant/ghost/retired = 0 / 0 / 0 / 2`
- `lineage 2 continuation_bank_size = 0`
- `lineage 2 temp_attach_id = 33`
- `lineage 2 temp_attach_expired = 1`

这说明在目标 frame 上，真正的 target lineage 已经只剩 retired tracks，continuation bank 也空了，所以 pool 和 bank 同时为 0 是真实状态，不是单纯阈值问题。

## Forced Target Frame

forced trace 仍以 Stage A.1 的 forced target 为起点：

- `frame = 945`
- `target_lineage = 3`
- `target_old_track_id = 211`
- `target_old_prototype_id = 4`

当前 traced 结果是：

- `forced_target_has_assignment_row = false`
- `lineage 3 continuation_bank_size = 3`
- `lineage 3 temp_attach_present = true`
- `lineage 3 temp_attach_expired = false`

这说明 forced target frame 上虽然 lineage state 里已经有 temp attach slot，也有 continuation bank，但那个 target event 自身并没有形成一条可被当前 Stage A consumer 读取的 assignment row。也就是说，问题不只是“没读 temp slot”，而是 target event 根本没有以当前实现期待的 assignment 形态进入 consumer。

## Exact Break

当前主断点已经可以精确写成两层：

1. **Audit contamination**

   Stage A.1 把同一 frame 上别的 lineage 的 attach-written 行混进了 target re-entry 事件，所以看起来像“attach 写了但没被消费”。实际上，在 traced baseline target frame 里，target lineage 根本没有对应 assignment row。

2. **Recovery consumer mismatch**

   `apply_concept_gated_resurrection()` 只消费：

   - same-lineage dormant/ghost tracks
   - same-lineage continuation bank

   它不把 `TemporaryAttachSlot` 当作 baseline candidate source。

## Additional Structural Issue

即使后面把 temp attach slot 接进 consumer，当前 slot 结构也还不能直接恢复 old track，因为 `TemporaryAttachSlot` 里保存的是**当前 attach 的 track / prototype 状态**，不是 old-track continuity 目标本身。也就是说：

- slot 能当“临时落点状态”
- 但还不能当“旧 identity 恢复源”

这解释了为什么前面即使把 temp attach 写出来，也不会自然把 `same_track_after_attach` 拉起来。

## Stage A.2 Conclusion

Phase 3D 当前不是“promotion 还没调好”，也不是“temp attach 阈值没扫对”。当前唯一需要先修通的是：

1. 先把 target re-entry event 的 assignment/audit 对齐，避免再把非目标 lineage 的 attach 写入当成 target attach success。
2. 再决定 temp attach 到底是：
   - 要接成 consumer 可见的 fallback candidate
   - 还是只保留为 non-head temporary state
3. 若要承担 old-track restoration，temp attach 或其绑定对象必须显式持有 old identity reference，而不是只持有当前 attached track state。

## Practical Next Step

下一步不该进 Stage B。应该先做一个更小的 Stage A.2b：

- 把 target event 的 frame-local assignment/audit 改成只记录 `target_lineage` 自身的 rows
- 把 `TemporaryAttachSlot` 是否需要进入 consumer 明确成一个单独开关
- 如果进入 consumer，就必须补一个 `old_identity_ref`，否则它只能算 temporary state，不算 recovery source
