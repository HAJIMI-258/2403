# Phase 3D Stage A.6 Design Notes

1. preserve 输入候选理论上不应只看 runtime lineage hints，还应同时吸收 recovery surface 和 continuity evidence。
2. 本轮新增了三路输入 trace：runtime hints / recovery surface / continuity evidence，用来判断 target lineage 是在形成前缺失，还是形成后被 prune。
3. 实跑结果显示：frame 990 的当前运行时只有 lineage `0/1`，`continuation / anchor / temp-attach / memory._lineages` 都不再包含 lineage `2`。
4. 因此 A.6 这次验证到的不是“preserve 输入 prune 了 lineage 2”，而是“continuity source 本身已经不再保留 lineage 2 这个身份键”。
5. 本轮没有改 final ranking，也没有继续加强 tie-break；结论只针对 preserve 输入形成链。
