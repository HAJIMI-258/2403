"""Synthetic re-entry evaluation for the event-aware visual cognitive loop."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.synth_stream import (  # noqa: E402
    AppearancePerturbationConfig,
    BackgroundDriftConfig,
    BridgeSyntheticConfig,
    SynthDatasetConfig,
    SyntheticStreamGenerator,
)
from nops_owr.attention import AttentionGate  # noqa: E402
from nops_owr.cognition import PredictiveRecognizer  # noqa: E402
from nops_owr.cognition.visual_cognitive_loop import CognitiveFrameResult, VisualCognitiveLoop  # noqa: E402
from nops_owr.encoder.spike_encoder import MinimalSpikeEncoder  # noqa: E402
from nops_owr.evaluation.cognitive_metrics import (  # noqa: E402
    attended_object_ratio,
    false_resurrection_rate,
    long_gap_reentry_success_rate,
    memory_compression_ratio,
    prediction_error_mean,
    uncertain_hold_rate,
)
from nops_owr.memory import EpisodicMemory, MinimalPrototypeMemory  # noqa: E402
from nops_owr.objectness.field import MinimalObjectnessField  # noqa: E402
from nops_owr.tracking.temporal_identity import MinimalTemporalIdentityTracker  # noqa: E402


def run_eval(
    sequences: int = 2,
    max_frames: int = 80,
    output_dir: str | Path = "results/cognitive_reentry",
    min_gap: int = 1,
    seed: int = 41,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config = _build_config(sequences=sequences, max_frames=max_frames)
    generator = SyntheticStreamGenerator(config, seed=seed)

    frame_rows: list[dict[str, Any]] = []
    reentry_rows: list[dict[str, Any]] = []
    all_decisions: list[Any] = []
    total_object_files = 0
    total_attended = 0
    total_written = 0
    total_extended = 0
    total_reactivated = 0
    final_episode_memory_size = 0

    for sequence_id in range(sequences):
        sequence = generator.generate_sequence(sequence_id)
        frames = sequence.frames[:max_frames]
        if len(frames) < 2:
            continue
        loop = _build_loop()
        results_by_frame: dict[int, CognitiveFrameResult] = {}
        visibility_history: dict[int, list[tuple[int, bool]]] = defaultdict(list)

        for idx in range(1, len(frames)):
            prev_sample = frames[idx - 1]
            sample = frames[idx]
            result = loop.step(
                prev_sample.frame,
                sample.frame,
                sample.frame_index,
                ground_truth={
                    "boxes": sample.boxes,
                    "instance_ids": sample.instance_ids,
                    "concept_ids": sample.concept_ids,
                },
            )
            results_by_frame[sample.frame_index] = result
            all_decisions.extend(result.recognition_decisions)
            total_object_files += len(result.object_files)
            total_attended += len(result.attended_object_files)
            total_written += len([event for event in result.cognitive_events if event.event_type == "episode_started"])
            total_extended += len([event for event in result.cognitive_events if event.event_type == "episode_extended"])
            total_reactivated += len(result.reactivated_episode_ids)
            frame_rows.append(_frame_row(sequence_id, result))
            present_ids = set(int(instance_id) for instance_id in sample.instance_ids)
            for instance_id in present_ids:
                visibility_history[instance_id].append((sample.frame_index, True))
            for instance_id in set(visibility_history.keys()) - present_ids:
                visibility_history[instance_id].append((sample.frame_index, False))

        reentry_events = _discover_reentry_events(frames, min_gap=min_gap)
        for event in reentry_events:
            result = results_by_frame.get(event["reappear_frame"])
            reentry_rows.append(_reentry_row(sequence_id, event, result, loop))
        final_episode_memory_size += len(loop.episodic_memory)

    _write_csv(output_path / "frame_metrics.csv", frame_rows)
    _write_csv(output_path / "reentry_events.csv", reentry_rows)
    summary = _summary(
        sequences=sequences,
        frame_count=sequences * max(0, min(max_frames, config.sequence_length) - 1),
        object_file_count=total_object_files,
        attended_count=total_attended,
        episode_write_count=total_written,
        episode_extend_count=total_extended,
        episode_reactivation_count=total_reactivated,
        episodic_memory_size=final_episode_memory_size,
        reentry_rows=reentry_rows,
        decisions=all_decisions,
    )
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_path / "report.md").write_text(_report(summary, reentry_rows), encoding="utf-8")
    return summary


def _build_config(sequences: int, max_frames: int) -> SynthDatasetConfig:
    return SynthDatasetConfig(
        name="cognitive_reentry_eval",
        resolution=(128, 128),
        sequence_length=max_frames,
        num_sequences=sequences,
        num_objects_range=(3, 4),
        shapes=("circle", "square", "triangle"),
        object_scale_range=(9, 18),
        velocity_range=(1.0, 2.5),
        spawn_margin=18,
        occlusion_probability=0.55,
        reentry_probability=0.85,
        background_drift=BackgroundDriftConfig(
            enabled=True,
            brightness_amplitude=0.06,
            texture_noise_std=0.015,
        ),
        appearance_perturbation=AppearancePerturbationConfig(
            enabled=True,
            scale_jitter=0.08,
            intensity_jitter=0.08,
            edge_blur_probability=0.05,
        ),
        bridge_synthetic=BridgeSyntheticConfig(
            enabled=True,
            difficulty_preset="cognitive_reentry",
            background_repeat_density=0.15,
            background_texture_strength=0.08,
            illumination_drift_strength=0.10,
            local_noise_std=0.01,
            local_blur_probability=0.03,
            camera_jitter_std=0.20,
            occlusion_duration_range=(8, 16),
            reentry_gap_range=(8, 18),
            crossing_probability=0.20,
            target_deformation_strength=0.05,
            low_contrast_probability=0.10,
        ),
        outputs=("frame", "boxes", "masks", "instance_id", "concept_id"),
    )


def _build_loop() -> VisualCognitiveLoop:
    return VisualCognitiveLoop(
        encoder=MinimalSpikeEncoder(),
        objectness_field=MinimalObjectnessField(
            tau_obj=0.35,
            threshold_mode="fixed",
            min_area=16,
            max_proposals=6,
        ),
        tracker=MinimalTemporalIdentityTracker(),
        prototype_memory=MinimalPrototypeMemory(memory_budget=48),
        attention_gate=AttentionGate(max_attended_objects=4),
        episodic_memory=EpisodicMemory(memory_budget=128),
        recognizer=PredictiveRecognizer(),
    )


def _discover_reentry_events(frames: list[Any], min_gap: int) -> list[dict[str, Any]]:
    visible_by_frame: dict[int, set[int]] = {
        sample.frame_index: set(int(instance_id) for instance_id in sample.instance_ids) for sample in frames
    }
    all_ids = sorted(set().union(*visible_by_frame.values())) if visible_by_frame else []
    events: list[dict[str, Any]] = []
    for instance_id in all_ids:
        last_visible: int | None = None
        first_absent: int | None = None
        absent_count = 0
        for sample in frames:
            visible = instance_id in visible_by_frame[sample.frame_index]
            if visible:
                if first_absent is not None and last_visible is not None and absent_count >= min_gap:
                    events.append(
                        {
                            "instance_id": instance_id,
                            "disappear_frame": last_visible,
                            "reappear_frame": sample.frame_index,
                            "gap_length": absent_count,
                        }
                    )
                last_visible = sample.frame_index
                first_absent = None
                absent_count = 0
            elif last_visible is not None:
                first_absent = sample.frame_index if first_absent is None else first_absent
                absent_count += 1
    return events


def _reentry_row(
    sequence_id: int,
    event: dict[str, Any],
    result: CognitiveFrameResult | None,
    loop: VisualCognitiveLoop,
) -> dict[str, Any]:
    instance_id = int(event["instance_id"])
    matched_object = None
    if result is not None:
        for object_file in result.object_files:
            if object_file.metadata.get("gt_instance_id") == instance_id:
                matched_object = object_file
                break
    decision = None
    retrievals = []
    if result is not None and matched_object is not None:
        retrievals = result.episodic_retrievals.get(matched_object.object_file_id, [])
        for row in result.recognition_decisions:
            if row.object_file_id == matched_object.object_file_id:
                decision = row
                break
    top1 = retrievals[0] if retrievals else None
    linked_episode_id = None if decision is None else decision.linked_episode_id
    linked_episode = loop.episodic_memory.get_episode(linked_episode_id)
    top_episode = None if top1 is None else top1.bundle
    linked_gt = None if linked_episode is None else linked_episode.metadata.get("gt_instance_id")
    top_gt = None if top_episode is None else top_episode.metadata.get("gt_instance_id")
    success_same_instance = int((linked_gt == instance_id) or (top_gt == instance_id))
    false_resurrection = int(
        decision is not None
        and decision.decision_type == "same_instance"
        and linked_gt is not None
        and linked_gt != instance_id
    )
    return {
        "sequence_id": sequence_id,
        "instance_id": instance_id,
        "disappear_frame": event["disappear_frame"],
        "reappear_frame": event["reappear_frame"],
        "gap_length": event["gap_length"],
        "matched_object_file_id": "" if matched_object is None else matched_object.object_file_id,
        "decision_type": "" if decision is None else decision.decision_type,
        "linked_episode_id": "" if linked_episode_id is None else linked_episode_id,
        "retrieved_top1_episode_id": "" if top_episode is None else top_episode.episode_id,
        "retrieved_top1_score": 0.0 if top1 is None else top1.score,
        "retrieved_top1_gt_instance_id": "" if top_gt is None else top_gt,
        "success_same_instance": success_same_instance,
        "false_resurrection": false_resurrection,
        "prediction_error": 0.0 if decision is None else decision.prediction_error,
        "familiarity_score": 0.0 if decision is None else decision.familiarity_score,
        "novelty_score": 0.0 if decision is None else decision.novelty_score,
    }


def _frame_row(sequence_id: int, result: CognitiveFrameResult) -> dict[str, Any]:
    return {
        "sequence_id": sequence_id,
        "frame_index": result.frame_index,
        "object_file_count": len(result.object_files),
        "attended_object_count": len(result.attended_object_files),
        "decision_count": len(result.recognition_decisions),
        "written_episode_count": len([e for e in result.cognitive_events if e.event_type == "episode_started"]),
        "extended_episode_count": len([e for e in result.cognitive_events if e.event_type == "episode_extended"]),
        "reactivated_episode_count": len(result.reactivated_episode_ids),
        "active_episode_count": result.active_episode_count,
        "memory_context_used": int(result.memory_context_used),
        "prediction_error_mean": result.metrics_snapshot.get("prediction_error_mean", 0.0),
    }


def _summary(
    sequences: int,
    frame_count: int,
    object_file_count: int,
    attended_count: int,
    episode_write_count: int,
    episode_extend_count: int,
    episode_reactivation_count: int,
    episodic_memory_size: int,
    reentry_rows: list[dict[str, Any]],
    decisions: list[Any],
) -> dict[str, Any]:
    reentry_count = len(reentry_rows)
    decision_counts = Counter(getattr(decision, "decision_type", "unknown") for decision in decisions)
    success_rate = (
        sum(int(row["success_same_instance"]) for row in reentry_rows) / reentry_count if reentry_count else 0.0
    )
    false_rate = false_resurrection_rate(reentry_rows)
    return {
        "sequence_count": sequences,
        "frame_count": frame_count,
        "object_file_count": object_file_count,
        "attended_object_ratio_mean": attended_object_ratio(attended_count, object_file_count),
        "episodic_memory_size": episodic_memory_size,
        "episode_write_count": episode_write_count,
        "episode_extend_count": episode_extend_count,
        "episode_reactivation_count": episode_reactivation_count,
        "reentry_event_count": reentry_count,
        "long_gap_reentry_success_rate": success_rate,
        "false_resurrection_rate": false_rate,
        "uncertain_hold_rate": uncertain_hold_rate(decisions),
        "same_instance_rate": decision_counts.get("same_instance", 0) / len(decisions) if decisions else 0.0,
        "same_concept_rate": decision_counts.get("same_concept", 0) / len(decisions) if decisions else 0.0,
        "new_concept_rate": decision_counts.get("new_concept", 0) / len(decisions) if decisions else 0.0,
        "mean_prediction_error": prediction_error_mean(decisions),
        "memory_compression_ratio": memory_compression_ratio(object_file_count, episode_write_count + episode_extend_count),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _report(summary: dict[str, Any], reentry_rows: list[dict[str, Any]]) -> str:
    failures = Counter()
    for row in reentry_rows:
        if int(row["success_same_instance"]):
            continue
        if not row["matched_object_file_id"]:
            failures["no_matched_object_file"] += 1
        elif not row["retrieved_top1_episode_id"]:
            failures["no_episode_retrieved"] += 1
        elif int(row["false_resurrection"]):
            failures["false_resurrection"] += 1
        else:
            failures["retrieved_wrong_or_unresolved"] += 1
    return (
        "# Cognitive Re-entry Evaluation\n\n"
        "This is a synthetic mechanism evaluation for event-aware visual memory. "
        "GT is used only for matching/evaluation and episode metadata audit, not for online scoring.\n\n"
        f"- reentry_event_count: {summary['reentry_event_count']}\n"
        f"- long_gap_reentry_success_rate: {summary['long_gap_reentry_success_rate']:.4f}\n"
        f"- false_resurrection_rate: {summary['false_resurrection_rate']:.4f}\n"
        f"- uncertain_hold_rate: {summary['uncertain_hold_rate']:.4f}\n"
        f"- mean_prediction_error: {summary['mean_prediction_error']:.4f}\n\n"
        f"Failure buckets: {dict(failures)}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--output-dir", type=str, default="results/cognitive_reentry")
    parser.add_argument("--min-gap", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    summary = run_eval(
        sequences=args.sequences,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
        min_gap=args.min_gap,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
