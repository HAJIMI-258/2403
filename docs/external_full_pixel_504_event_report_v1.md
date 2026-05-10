# External Full-Pixel 504-Event Report v1

Date: 2026-05-09

This report freezes the current external full-pixel evidence after expanding the LaSOT/LaGOT-linked subset to 504 oracle-proposal memory events across 12 categories.

This is not a full-perception benchmark. All full-pixel results are still oracle-proposal memory-only results: ground-truth boxes define the query/candidate crops, while target identity is used only for evaluation.

## Scope

Evaluated categories:

- bicycle
- bottle
- car
- coin
- dog
- drone
- goldfish
- guitar
- hat
- kite
- motorcycle
- volleyball

Event count: 504

The goal of this stage is to decide which external evidence branches are credible enough to keep as isolated external profiles or diagnostic baselines. It is not a merge gate for the main synthetic NOPS chain.

## Main Numbers

| Component | Top1 | Decision |
|---|---:|---|
| Geometry passive NOPS-style baseline | 0.5913 | Baseline only |
| Isolated external geometry branch | 0.6786 | Keep as isolated external branch |
| External branch + raw appearance best | 0.6806 | Diagnostic only, no integration |
| Strong handcrafted descriptor auxiliary | 0.6786 | Rejected |
| ResNet18 frozen embedding NN | 0.5496 | Diagnostic external baseline only |
| External branch + ResNet18 embedding best | 0.6786 | Rejected as auxiliary |
| EXT-10 all-external test split | 0.7120 | Isolated branch, not routing integration |
| EXT-12 selected strong split variant | 0.7120 | No gain over external branch |

## Evidence Summary

### External geometry branch

The external geometry branch remains the only stable external result.

- Full-pixel top1 improves from 0.5913 to 0.6786.
- Split evaluation selects the all-external branch, not a learned routing rule.
- SYN-REG-1 previously showed that the external trajectory-heavy calibration damages the internal synthetic anchor/focus path, so it cannot be merged into main NOPS.

Decision:

- Use as isolated external geometry profile.
- Do not merge into main NOPS.

### Raw appearance

Raw crop appearance is more plausible after the 504-event expansion, but it is still not integration-ready.

- External branch improves only from 0.6786 to 0.6806.
- EXT-5C controls pass on the refreshed subset.
- The absolute external-branch gain is only about 0.0020 top1.
- EXT-8 still blocks integration.

Decision:

- Keep as diagnostic only.
- Do not merge into main NOPS.
- Do not treat this as a stable external auxiliary yet.

### Strong handcrafted descriptor

The strong handcrafted descriptor is not reliable enough.

- External branch + strong descriptor stays at 0.6786.
- Strong external gain is 0.0.
- Strong controls fail.
- EXT-12 split gate fails.

Decision:

- Reject as external auxiliary.
- Keep only as diagnostic evidence.

### Frozen ResNet18 embedding

The frozen pretrained embedding is useful as a diagnostic baseline, but it is not a NOPS method.

- Embedding NN top1 is 0.5496, below geometry passive.
- Geometry + embedding improves weak geometry to 0.6151.
- External branch + embedding stays at 0.6786.
- Embedding controls fail.
- Significance audit passes for some comparisons, but controls block integration.

Decision:

- Keep as external pretrained baseline only.
- Do not merge into no-pretrain NOPS.
- Do not use as external branch auxiliary.

### Event-conditioned routing

Routing does not yet generalize beyond always using the external branch.

- Test geometry top1 is 0.6160.
- Test all-external top1 is 0.7120.
- Selected gate top1 is also 0.7120.
- Selected gate delta vs all-external is 0.0.
- Routing integration readiness is 0.

Decision:

- Reject routing integration for now.
- Keep all-external geometry branch isolated.

## Integration Matrix

| Component | Main NOPS | Isolated External Branch | Diagnostic Baseline |
|---|---:|---:|---:|
| External geometry branch | No | Yes | Yes |
| Raw appearance | No | No | Yes |
| Strong handcrafted descriptor | No | No | Yes |
| Frozen ResNet18 embedding | No | No | Yes |
| Event-conditioned routing | No | No | Yes |

## Scientific Interpretation

The external evidence supports a narrow but useful claim:

Object-file style geometry and trajectory continuity are the strongest current external signals for long-gap object memory under oracle-proposal evaluation.

The evidence does not yet support:

- merging external trajectory-heavy calibration into the main synthetic NOPS chain;
- using raw appearance as a general memory cue;
- using handcrafted descriptors as an auxiliary branch;
- using frozen pretrained embeddings as part of no-pretrain NOPS;
- event-conditioned routing as a general solution.

The current result is consistent with the project hypothesis that long-term visual memory is not solved by appearance similarity alone. In this subset, geometry/trajectory continuity is more reliable than raw, handcrafted, or pretrained crop appearance fusion.

## Hard Stop Rules After This Stage

Do not:

- run more appearance or embedding fusion sweeps under EXT-13;
- merge the external geometry branch into main NOPS;
- claim full-perception performance;
- claim appearance is solved;
- claim routing is solved;
- treat oracle-proposal memory-only results as detector/tracker results.

Allowed next actions:

- write a stage report or paper outline;
- keep the isolated external geometry branch as a baseline/profile;
- start a new EXT-14 only if it introduces a genuinely new hypothesis and frozen protocol;
- run full-pipeline perception only as a separate benchmark with explicit detector/proposal evaluation.

## Recommended Next Step

Stop model-rule tuning for this branch and write the stage report.

If a new experiment is needed, it should be a new stage with a new mechanism hypothesis, for example:

- EXT-14: full-pipeline proposal/detector validation;
- EXT-14A: stronger object-file geometry under fixed split protocol;
- EXT-14B: category-conditioned appearance diagnostics, not integration;
- REPORT-1: research narrative and paper skeleton.

The immediate recommendation is REPORT-1.
