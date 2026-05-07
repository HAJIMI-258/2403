# Literature-Mechanism Alignment v1

This document is a mechanism alignment memo for NOPS-OWR, not a full literature review. Its purpose is to stop the project from drifting into synthetic top-k engineering and keep every next experiment tied to a cognitive mechanism, a practical visual-memory problem, a baseline family, and a falsifiable failure mode.

## Current Project Position

NOPS-OWR should be framed as:

> A study of how a visual system forms persistent object memories from continuous input, and later re-identifies old objects after occlusion, disappearance, re-entry, drift, and runtime representation changes using episodic cues and canonical memory anchors.

It should not be framed as a direct replacement for detectors, segmenters, trackers, or large vision-language models. The value is the missing long-term object memory layer:

- Did this object appear before?
- Is this re-entering object the same old object or a new one?
- Is this anomaly the same persistent anomaly seen earlier?
- When many similar old memories exist, which one should the current cue complete?

Current experimental evidence:

- E2C established that raw runtime lineage IDs are not stable enough; canonical and anchor views are necessary.
- E3.1 fixed focus hard cases via anti-hub / pattern-separation-style retrieval.
- E3.2/E3.2a/E3.2b showed that accessibility, replay, suppression, and safe top-3 reranking are not sufficient global fixes.
- E3.3 showed that reweighting existing coarse signature dimensions is insufficient.
- E3.4 showed that write-side `support_trajectory_signature` is useful: `global_top1` improved from `6/17` to `7/17`, `false_bundle_retrieval_rate` dropped from `11/17` to `10/17`, focus stayed `3/3`, but `mean_signature_v2_margin = -0.0098`, so passive evidence still does not reliably separate target and wrong bundles.

## Experimental Gate Before Any New Runner

Every new experiment must pass four gates:

1. Mechanism gate: Which cognitive or computational memory mechanism does this test?
2. Task gate: Which real long-term visual-memory pain point does it address?
3. Baseline gate: Which conventional baseline family could explain the same result?
4. Falsification gate: What result would prove this direction is insufficient?

If an experiment cannot answer these, do not run it.

## 1. Active Vision / Foveated Perception

### Research Question

How should an agent decide where to look next when passive visual evidence is insufficient or uncertain?

Representative references:

- Mnih et al., "Recurrent Models of Visual Attention" (2014): adaptive sequences of high-resolution glimpses instead of processing every location equally. https://arxiv.org/abs/1406.6247
- Ba et al., "Multiple Object Recognition with Visual Attention" (2014): recurrent visual attention for selecting relevant regions. https://arxiv.org/abs/1412.7755
- "Active Gaze Control for Foveal Scene Exploration" (2022): next fixation selected by uncertainty-reducing information metrics. https://arxiv.org/abs/2208.11594
- "AdaGlimpse" (ECCV 2024): active visual exploration with flexible glimpse location and scale. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03128.pdf

### Mechanism Extracted

Perception is not only passive feature extraction. Under uncertainty, a visual system should gather additional evidence from task-relevant regions. Fixation should be driven by expected information gain, uncertainty, ambiguity, and memory conflict.

### Corresponding NOPS-OWR Module

Future module:

- `MemoryUncertaintyGuidedFixation`
- `ActiveEvidenceRequest`
- `FixationCuePlanner`
- `SupportRegionReinspection`

Current related signals:

- `mean_signature_v2_margin`
- `competition_removed_target`
- `target_not_in_top5`
- `target_in_top3_but_lost_top1`
- cue disagreement and high-hub memory competition

### Current Evidence

E3.4 improved retrieval with support trajectory, but `mean_signature_v2_margin` remains negative. This suggests that passive support evidence helps but still does not give enough separability.

### Missing Evidence

We do not yet have a mechanism to request additional visual evidence when memory retrieval is ambiguous.

### Next Experiment Implication

If E3.4 refinement still leaves support / signature margins <= 0, stop memory-ranking experiments and enter:

`E4A: Memory-Uncertainty-Guided Active Visual Evidence Acquisition`

This should inspect support boundaries, local context, and competing memory-specific regions before any attach or promotion decision.

## 2. Object Permanence / Object Files

### Research Question

How does a visual system maintain an object as "the same thing" across motion, occlusion, feature changes, and temporary absence?

Representative references:

- Kahneman, Treisman, and Gibbs, "The Reviewing of Object Files: Object-Specific Integration of Information" (1992): object files integrate information through object-specific preview benefits. https://citeseerx.ist.psu.edu/document?doi=35a231c981a4a5f6711eb831d0d55673c2bcb161&repid=rep1&type=pdf
- Object-file follow-up work on feature retrieval and object-specific integration: https://pmc.ncbi.nlm.nih.gov/articles/PMC3655988/

### Mechanism Extracted

Identity is not just a category label. The system maintains an object file that binds position, features, continuity, and recent history. Re-identification after occlusion depends on whether current evidence can update or retrieve the same object file.

### Corresponding NOPS-OWR Module

Mapping:

- `track` = current object file
- `prototype` = appearance / local experience entity
- `lineage` = concept or runtime continuity branch
- `anchor` = stable old-object memory index
- `episodic bundle` = stored object-experience packet

### Current Evidence

E2R-M showed raw lineage mismatch can be a runtime namespace shift rather than memory absence. E2C's `target_anchor_visible = 1` while `raw_lineage_visible = 0` matches object-file logic: the raw internal ID may drift, but the same memory object can still be recognized by anchor.

E3.4's `A2_support_trajectory_only` result supports the object-file interpretation: multi-frame object support trajectory is a useful cue for preserving object identity.

### Missing Evidence

Object-file continuity is still mostly inferred from synthetic track/prototype history. We need baselines and semi-real tasks where object-file persistence matters: long occlusion, object revisit, anomaly persistence, and similar-object competition.

### Next Experiment Implication

E3.4r should ask:

> Does object support trajectory create a more separable object-file memory than single-frame bbox/support signatures?

Primary mechanism metric:

- `mean_support_v3_margin`
- `competition_removed_target`
- target-vs-wrong support trajectory margin

## 3. Episodic Memory / Hippocampal Indexing

### Research Question

How can a partial current cue retrieve a stored episode and reconnect it to the correct old object memory?

Representative references:

- Teyler and DiScenna, hippocampal indexing theory: hippocampus as an index for reactivating distributed cortical representations. One accessible overview: https://people.whitman.edu/~herbrawt/hippocampus.pdf
- Reviews of hippocampal event / episodic memory and input pathways: https://www.sciencedirect.com/science/article/pii/S027858462300043X

### Mechanism Extracted

The hippocampal-like function is not to store a permanent raw ID. It stores an index that can reactivate distributed content, context, temporal, and object-state traces from a partial cue.

### Corresponding NOPS-OWR Module

Current modules:

- `EpisodicBundle`
- `memory_anchor_id`
- `canonical_lineage_id`
- `CueSignature`
- `PatternCompletionResult`

Needed next modules:

- `EpisodicBundleV2`
- `SupportTrajectorySignature`
- `DisappearanceBoundarySignature`
- `ContextLayoutSignature`

### Current Evidence

E3 established target bundle presence. E3.1 showed that retrieval can be improved for hard focus events. E3.4 showed that richer write-side support trajectory can improve global retrieval.

### Missing Evidence

The current bundle still does not preserve enough real episode detail. `mean_signature_v2_margin < 0` means the target bundle is not yet consistently stronger than the wrong memory under the current signature.

### Next Experiment Implication

Do not keep tuning final score. Improve the write-side episodic trace:

- multi-frame support trajectory
- disappearance boundary
- neighbor / context layout
- quality trajectory
- motion phase
- lifecycle state

Then verify target-vs-wrong separability before attaching identity.

## 4. Pattern Completion / Pattern Separation

### Research Question

How does memory retrieve a complete old episode from partial cues without collapsing similar episodes into one false memory?

Representative references:

- Rolls, "The mechanisms for pattern completion and pattern separation in the hippocampus" (2013): event memory and recall from partial cue. https://pmc.ncbi.nlm.nih.gov/articles/PMC3812781/
- Yassa and Stark, "Pattern separation in the hippocampus" (2011): similar experiences must be stored distinctly. https://pmc.ncbi.nlm.nih.gov/articles/PMC3183227/
- Reviews of pattern separation / completion behavior: https://pmc.ncbi.nlm.nih.gov/articles/PMC7819938/

### Mechanism Extracted

Pattern completion recovers old memory from partial cues. Pattern separation prevents similar old memories from becoming a generic attractor. A useful memory system must balance both.

### Corresponding NOPS-OWR Module

Current:

- `EpisodicBundle` retrieval
- anti-hub penalty
- bundle competition / NMS
- target-vs-wrong signature margin

### Current Evidence

E3.1 directly maps to this mechanism:

- proto0 / generic hub bundles acted as overgeneralized attractors.
- anti-hub and competition fixed focus events.

E3.2b showed that safe top-3 reranking cannot reliably solve global failures. E3.3 showed that reweighting coarse cues cannot solve it. E3.4 showed write-side support trajectory can help.

### Missing Evidence

We still need stronger separability evidence:

- `mean_support_v3_margin > 0`
- fewer `competition_removed_target`
- lower target/wrong cue collision count

### Next Experiment Implication

E3.4r should be a pattern-separation test, not a metric chase:

> Does multi-frame support trajectory separate target and wrong bundles better than single-frame signatures?

If no, move to active evidence acquisition rather than more ranking.

## 5. Replay / Reconsolidation / Memory Accessibility

### Research Question

When should a recalled memory be strengthened, updated, suppressed, or left unchanged?

Representative references:

- Broad replay / consolidation work shows replay is structured and task-relevant, not arbitrary rehearsal. A current computationally useful analogy is XMem's memory potentiation / long-term memory split for long videos: https://arxiv.org/abs/2207.07115
- Memory reconsolidation literature broadly supports that recall can update memory, but online updates can also distort memory if gated poorly.

### Mechanism Extracted

Memory should have accessibility states. Replay and reconsolidation should be delayed, gated, and evidence-consistent. Immediate reinforcement from a single ambiguous event can strengthen the wrong attractor.

### Corresponding NOPS-OWR Module

Attempted:

- `accessibility_score`
- replay queue
- reconsolidation trace
- competitor suppression

### Current Evidence

E3.2 and E3.2a produced an important negative result:

- Strong immediate accessibility / replay / reconsolidation hurt performance.
- Wrong bundles can get reinforced if update gates are too loose.

This is not wasted work. It establishes:

> Online visual memory should not immediately reconsolidate ambiguous retrieval results into identity memory.

### Missing Evidence

We have not yet tested delayed replay after stronger evidence accumulation.

### Next Experiment Implication

Replay should not return until:

- signature margins are positive, or
- active vision has gathered enough extra evidence.

Do not use replay as an immediate top1 boost.

## 6. Long-Term Video Object Tracking / ReID Memory Banks

### Research Question

How do conventional tracking, ReID, and memory-bank systems handle long occlusion, re-entry, template drift, and identity switches?

Representative references:

- DeepSORT uses ReID features to reduce identity switches in MOT; ByteTrack discusses association and ReID vulnerabilities under occlusion. ByteTrack paper: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf
- XMem uses sensory / working / long-term memory for long video object segmentation and explicitly addresses memory explosion and long-video decay. https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880633.pdf
- LVOS benchmark targets long-term reappearing and similar cross-temporal objects. https://arxiv.org/abs/2211.10181

### Mechanism Extracted

Memory banks can drift, overgrow, or fail under long-term reappearance. A stronger system needs memory selection, consolidation, and long-gap re-identification logic.

### Corresponding NOPS-OWR Module

Baseline families to add:

- tracker-only baseline: SORT / DeepSORT / ByteTrack / OC-SORT style association
- embedding memory baseline: DINO / CLIP / ViT embedding + nearest neighbor
- feature memory baseline: XMem-style memory bank / long-term VOS analogy
- ReID template bank baseline

### Current Evidence

NOPS-OWR currently has internal synthetic evidence only. E3.4 indicates support trajectory helps, but this does not yet prove superiority over a good embedding or ReID memory bank.

### Missing Evidence

No external baseline yet. This is a publication risk.

### Next Experiment Implication

Before any major claim, add baseline comparisons:

- `EmbeddingMemoryNN`
- `ReIDTemplateBank`
- `TrackerOnlyLongGap`
- `XMemInspiredFeatureMemory` or at least a simplified feature-memory analogue

## 7. Continual Open-World Recognition

### Research Question

How should a system detect, remember, and update objects/classes when the world is dynamic and new categories or instances appear over time?

Representative references:

- Bendale and Boult, "Towards Open World Recognition" (CVPR 2015): dynamic recognition where unseen categories appear and must be incrementally added. https://openaccess.thecvf.com/content_cvpr_2015/papers/Bendale_Towards_Open_World_2015_CVPR_paper.pdf
- "Towards Open Set Deep Networks" (CVPR 2016): open set risk / unknown handling. https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf

### Mechanism Extracted

Recognition in the real world is not closed-set classification. The system must decide when something is known, unknown, changed, merged, split, or revisited.

### Corresponding NOPS-OWR Module

Current:

- `ConceptLineage`
- `PrototypeState`
- `RecoveryAnchor`
- `EpisodicBundle`
- raw / canonical / anchor visibility metrics

### Current Evidence

E2C shows why raw IDs are insufficient. Canonical / anchor recognition is necessary for open-world continuity.

### Missing Evidence

Open-world baselines and real-world object change / revisit tasks are missing.

### Next Experiment Implication

Keep raw/canonical/anchor metrics as first-class outputs. Do not collapse success into raw lineage ID match.

## 8. Robotics / Inspection Long-Term Visual Memory

### Research Question

How can a robot or inspection system remember persistent objects, anomalies, and scene changes across time, revisits, occlusion, and viewpoint changes?

Representative references:

- Active vision in robotic systems survey: sensing uncertainty and view planning are central to robot perception. https://journals.sagepub.com/doi/abs/10.1177/0278364911410755
- Dynamic visual SLAM survey: visual systems rely on temporal feature matching, map building, and loop closure to handle drift. https://link.springer.com/article/10.1186/s42492-021-00086-w
- Long-term VOS / LVOS emphasizes reappearing objects and cross-temporal similar objects. https://arxiv.org/abs/2211.10181

### Mechanism Extracted

Real inspection is about revisiting and comparing persistent entities, not just current-frame detection. Memory must support:

- anomaly persistence
- object revisit
- change detection
- long-gap re-identification
- uncertainty-driven reinspection

### Corresponding NOPS-OWR Module

Future practical tasks:

- agricultural pest / fruit / disease spot revisit
- industrial defect persistence
- warehouse object revisit
- robot patrol anomaly memory

### Current Evidence

The synthetic bridge tasks show the mechanisms can be tested in controlled form, but they are not enough for external validity.

### Missing Evidence

No semi-real or real inspection stream yet.

### Next Experiment Implication

After E3.4/E4A mechanism validation, add a semi-real persistent-inspection benchmark:

- same object/anomaly revisited after long gaps
- similar distractors
- viewpoint/context drift
- detector false positives
- memory budget constraints

## Immediate Next Scientific Decision

E3.4 produced a useful result but not a final memory system:

- support trajectory helps retrieval,
- focus hard events remain fixed,
- false retrieval decreases by one event,
- target/wrong signature margin remains negative,
- main remaining failure is `competition_removed_target = 4`.

Therefore:

1. E3.4r is allowed only as a mechanism test of support trajectory separability, not as another ranking sweep.
2. Primary E3.4r metrics must include:
   - `mean_support_v3_margin`
   - `competition_removed_target`
   - target-vs-wrong support trajectory margin
   - target/wrong collision counts
3. If `mean_support_v3_margin <= 0`, stop passive memory scoring and enter active vision evidence acquisition.

Decision rule:

```text
if mean_support_v3_margin > 0:
    continue E3.5 retrieval index / event-type scoring
else:
    enter E4A memory-uncertainty-guided active visual evidence acquisition
```

## Baselines Required Before Publication Claims

Minimum baseline table:

1. Tracker-only long-gap association.
2. Embedding nearest-neighbor memory using frozen visual embeddings.
3. ReID template bank.
4. Feature-memory / VOS-style memory baseline.
5. NOPS-OWR raw lineage.
6. NOPS-OWR canonical lineage.
7. NOPS-OWR anchor + episodic bundle.
8. NOPS-OWR with active evidence acquisition, if E4A is reached.

## Project Rule Going Forward

Every new Codex task must follow:

```text
literature mechanism
-> practical task
-> baseline comparison
-> falsifiable experiment
-> code execution
-> mechanism-level result interpretation
```

No new runner should be created only because the previous metric did not improve.
