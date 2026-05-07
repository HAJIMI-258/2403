# Phase 3D Stage A.6 Recommendation

- latest run: `forced_three_source_input`
- failure bucket: `input_formation_failure`
- entered preserve input: `0`
- entered claim builder: `0`
- visible in claim set: `0`
- runtime lineages at frame 990: `{0, 1}`
- target lineage `2` present in runtime: `0`
- old prototype `3` current archived owner lineage: `0`

当前不要进入 Stage B。

这轮已经说明：

1. 不是 preserve 后输掉，也不是 tie-break 不够强。
2. 是 target metadata 里的 lineage `2` 在当前运行时已经没有任何可枚举 continuity source。
3. preserve-input formation 现在只能看到 runtime lineage `0/1`，因此再加 preserve 规则也不会把 `2` 拉进来。

下一步如果继续沿 Stage A 查，应查的是：

- 哪个上游步骤把 old prototype `3` / old continuity 从 lineage `2` 重映射到了 runtime lineage `0`
- 这个重映射是否发生在 prototype lineage 更新、archive/replace，还是 continuation / anchor 写入绑定阶段

在这个点查清之前，不要进入 Stage B。
