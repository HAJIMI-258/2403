# Phase 3D Stage A.7 Design Notes

本轮不改 selection/promotion，只修 identity key 保留。

## 最小结构拆分

- runtime_owner_lineage_id: 运行时当前挂在哪个 lineage 容器下，可变。
- continuity_lineage_id: old continuity 身份键，供 recovery / preserve-input 枚举使用，不应被 runtime owner 默认覆盖。
- origin_lineage_id: continuity key 的起点，用于 trace。

## 本轮补丁范围

- PrototypeState 增加 runtime/continuity/origin/valid 字段。
- IdentityContinuation 与 RecoveryIdentityAnchor 增加同样的 owner/continuity split 字段。
- _create_prototype / _update_prototype / continuation write / anchor write / trace logging 全部显式记录这两个 lineage 键。

## 目标

先证明是哪一个上游步骤第一次把 continuity key 覆盖掉，再决定下一轮是否需要把 preserve-input 从 runtime owner 改成 continuity owner 枚举。