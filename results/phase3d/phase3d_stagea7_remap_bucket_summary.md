# Phase 3D Stage A.7 Remap Bucket Summary

- runtime_rebind_only: 131
- runtime_rebind_with_continuity_preserved: 0
- continuity_key_overwrite_bug: 391

判读：只要 target path 上出现 continuity_key_overwrite_bug，就说明 runtime owner 在某处被直接写回 continuity key。