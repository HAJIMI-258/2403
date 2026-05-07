# Phase 3D Stage A.7 Recommendation

当前不进 Stage B。

下一步只需要做一件事：把 preserve-input / claim-builder 的 continuity source 枚举从 runtime owner 扩成 runtime owner + continuity owner。

前提已经明确：

- 先保 continuity key；
- 不让 archive/replace/rebind 默认覆盖 continuity key；
- 然后再回到 A.6/A.5 的 preserve-input / claim-builder 路径验证 lineage 2 是否重新可见。