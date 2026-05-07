# Phase 3D Stage A.6 Preserve-Input Summary

- target event: `6`
- target frame: `990`
- offline target lineage metadata: `2`

## Baseline

- entered preserve input: `0`
- entered claim-builder input: `0`
- claim visible: `0`
- final selected lineage: `0`

## Force Continuity-Lineage Exposure

- entered preserve input: `0`
- entered claim-builder input: `0`
- claim visible: `0`
- final selected lineage: `0`

## Force Runtime + Recovery + Continuity Input

- entered preserve input: `0`
- entered claim-builder input: `0`
- claim visible: `0`
- final selected lineage: `0`

## Runtime Probe At Frame 990

- tracker runtime lineages: `{0, 1}`
- memory lineage registry: `{0, 1}`
- continuation buckets: `{0, 1}`
- recovery anchor buckets: `{0, 1}`
- temp-attach buckets: `{0, 1}`
- old prototype `3` is no longer owned by lineage `2`; it is archived under runtime lineage `0`

## Direct Answers

1. 这次不是 lineage `2` 进入了 preserve input 后又被 prune 掉，而是它在当前运行时根本不再存在于任何可枚举的 lineage source 里。
2. `forced_continuity_exposure` 和 `forced_three_source_input` 仍然全 0，说明当前 preserve-input 修补没有失败在 top-k 或 tie-break，而是失败在更早的 continuity source 形成阶段。
3. A.6 当前能回答的问题是：preserve 输入没有拿到任何仍然携带 lineage `2` 的 continuity source，所以后续 claim / selection 链根本没有机会处理它。
