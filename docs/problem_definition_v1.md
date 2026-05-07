# NOPS-OWR Problem Definition v1

## 1. Phase 1 Goal

Phase 1 only validates the minimum evidence chain of NOPS-OWR under a strict streaming setup:

`synthetic stream -> objectness -> temporal identity -> prototype memory -> metrics and visualization`

The question is not whether the final paper system is already complete. The question is whether the direction is worth continued investment.

## 2. Core Research Question

Under the constraints of:

- zero pretraining
- zero offline training set
- one-pass online processing
- pure SNN-oriented pipeline

can the system discover object-like regions from a continuous stream, keep identities stable over time, gradually form reusable prototypes, and prevent memory from exploding?

## 3. What Phase 1 Must Prove

Phase 1 is successful only if at least three of the following four statements are supported by evidence:

1. Object-like regions can be extracted from a continuous stream.
2. The same object can reappear without large identity instability.
3. Prototypes gradually consolidate instead of being recreated every frame.
4. Memory growth remains controlled over long sequences.

## 4. Scope Freeze

Phase 1 only includes:

- frame-stream input, not event-stream input
- synthetic streaming data, not real benchmark data
- a minimal pipeline with encoder, objectness, tracking, prototype birth/merge/decay, and budget gate
- comparison against weak but fair baselines

Phase 1 does not include:

- full NOPS-OWR theory or regret derivations
- complex concept graph logic
- split operations in prototype memory
- large-scale benchmark comparisons
- supervised SNN detection pipelines

## 5. Explicit Non-Goals

This project is **not** any of the following:

- OVD: there is no language-driven open-vocabulary supervision in Phase 1.
- anomaly detection: the target is persistent object discovery and concept formation, not rare-event flagging.
- supervised SNN detection: there is no offline labeled training stage for a detector.
- frame-wise reclustering: the system must preserve temporal continuity rather than rediscover every object from scratch at each frame.

## 6. Phase 1 Required Deliverables

By the end of Phase 1 we need:

- documentation that freezes the setup and decision rules
- runnable code for the minimum streaming pipeline and baselines
- three core figures
- a one-page Go / No-Go conclusion

## 7. Day 1 Deliverables

The first day must finish:

- repository structure
- `docs/problem_definition_v1.md`
- `protocol/nops_bench_v1.md`
- `configs/synth.yaml`
- `metrics/metrics_core.py`
- `experiments/exp_registry.md`
- `results/result_thresholds.md`

No model implementation should be started before these boundaries are written down.

