# Spiking Long-Term Object Memory v1

This document describes the first bounded spiking long-term memory layer for
NOPS-OWR. It is not a supervised SNN classifier and it is not a replacement for
the existing episodic memory. It is an SNN-inspired memory mechanism: sparse
spike-like signatures, fixed-size object capsules, and online plasticity updates
for object permanence under a bounded memory budget.

## Problem Definition

The system receives a one-pass visual stream. It should remember an object
without storing raw frames or unbounded per-frame descriptor history. When that
object disappears and later reappears with moderate scale, aspect-ratio,
brightness, texture, or occlusion change, the system should retrieve the same
long-term object identity while keeping memory growth controlled.

The core constraints remain:

- no offline training;
- no pretrained model;
- no raw frame storage in long-term memory;
- no unbounded episode accumulation for long-term identity;
- deterministic, bounded memory per remembered object.

## Relationship to Existing Memory

`EpisodicMemory` remains the short-term event and episode layer. It records
object-file episodes with content, support, motion, context, temporal, and
disappearance cues.

`SpikingObjectMemoryBank` is the compressed long-term identity layer. It stores
one `SpikingObjectMemoryCapsule` per remembered object. Each capsule contains
fixed-size statistics:

- sparse spike mean and variance;
- shape, appearance, topology, and deformation means and variances;
- a deterministic binary hash;
- observation/reactivation counters;
- stability, plasticity, and confidence.

The long-term memory cost is `O(number_of_capsules * descriptor_dim)`, not
`O(number_of_frames)`.

## Descriptor

`SpikingInvariantDescriptorBuilder` converts an `ObjectFile` and current
pseudo-spike encoding into a fixed-size descriptor. It pools current-gray,
edge-map, and spike-response crops into small shape, appearance, topology, and
deformation signatures. When RGB input is available, the object-file path also
adds fixed-size chromatic statistics to the descriptor; this is still bounded
metadata, not raw crop storage. A deterministic projection converts the
continuous signature into sparse spike bits and a binary hash.

This descriptor is "spiking" in the compressed sparse coding sense. It does not
simulate full LIF dynamics and it does not train a neural SNN.

## Matching

`SpikingObjectMemoryBank.match()` compares a descriptor against existing
capsules using:

- sparse spike overlap;
- identity similarity over shape, appearance, and topology;
- deformation likelihood under capsule deformation variance;
- binary hash similarity;
- capsule stability;
- conflict penalty from ambiguous top matches.

The final score combines these terms and returns ranked `SpikingMemoryMatch`
objects with explicit component scores.

## Plasticity

Capsules update with robust EMA statistics. High-confidence observations update
more strongly. Low-confidence observations update weakly. If the object appears
deformed but still has good identity evidence, deformation variance expands
carefully instead of overwriting the identity core.

Spike means are kept sparse by thresholding to the configured maximum density.
This keeps the capsule compact and prevents dense descriptors from turning the
memory into an unbounded continuous feature store.

## Budget

The bank never exceeds `max_capsules`. Eviction is deterministic and uses a
utility score based on confidence, stability, reactivation count, observation
count, and age penalty. The intent is to avoid evicting stable, recently useful
capsules first.

## Permanence Decisions

`PermanenceRecognizer` maps memory matches to:

- `same_object`;
- `familiar_but_deformed`;
- `uncertain_hold`;
- `new_object`;
- `false_resurrection_risk`.

This layer is conservative by design. Ambiguous top matches should become
`uncertain_hold` or `false_resurrection_risk`, not forced same-object matches.

## Metrics

The initial metrics are implemented in `metrics/permanence_metrics.py`:

- same-instance re-entry recall;
- false resurrection rate;
- memory growth rate;
- bytes per capsule;
- mean spike density;
- deformation tolerance curve;
- stability-plasticity score.

These metrics are data-source independent and can be used for synthetic or
external evaluations.

## Evaluation

Run the controlled morphology permanence evaluation:

```powershell
python experiments\run_spiking_morph_permanence_eval.py `
  --output-dir results\spiking_morph_permanence_eval `
  --seed 7 `
  --object-count 16 `
  --events-per-object 4 `
  --max-capsules 32 `
  --spike-dim 128 `
  --max-frames 800
```

The evaluation generates deterministic object streams with disappearance and
reappearance under scale, aspect, brightness, occlusion, and distractor changes.
It writes:

- `events.csv`;
- `summary.json`;
- `report.md`.

To audit permanence decision thresholds without changing memory capacity, run:

```powershell
python experiments\run_spiking_permanence_calibration_sweep.py `
  --output-dir results\spiking_permanence_calibration_sweep `
  --seed 7 `
  --object-count 16 `
  --events-per-object 4 `
  --max-capsules 32 `
  --spike-dim 128
```

The sweep reports the tradeoff between same-instance re-entry recall and false
resurrection. `best_safe_config` is selected from non-zero recall settings,
while `lowest_false_resurrection_config` may be a strict no-acceptance reference.
Current defaults prioritize low false resurrection over recall because a wrong
long-term identity update is more damaging than an unresolved object.

## Failure Modes

Current v1 is intentionally small. Expected failure modes include:

- sparse descriptor collision between similar objects;
- over-conservative `uncertain_hold` decisions;
- deformation variance expanding too slowly;
- false resurrection under high distractor similarity;
- weak appearance statistics when object crops are very texture-poor.

These failures should be diagnosed through component scores and permanence
metrics before increasing model complexity.
