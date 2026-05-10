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

REENTRY_FIELDNAMES = [
    "sequence_id",
    "instance_id",
    "disappear_frame",
    "reappear_frame",
    "gap_length",
    "gap_bucket",
    "gt_box_present",
    "matched_object_iou",
    "proposal_or_object_missing",
    "matched_object_file_id",
    "object_attended",
    "attention_failure",
    "target_episode_exists",
    "target_episode_id",
    "target_episode_closed",
    "target_episode_observation_count",
    "target_episode_frame_start",
    "target_episode_frame_end",
    "target_episode_rank",
    "target_episode_score",
    "top1_episode_id",
    "top1_score",
    "top1_margin",
    "top1_gt_instance_id",
    "top1_active_conflict",
    "top1_bundle_closed",
    "top1_reentry_gap",
    "topk_contains_target",
    "decision_type",
    "rejection_reason",
    "linked_episode_id",
    "retrieved_top1_episode_id",
    "retrieved_top1_score",
    "retrieved_top1_gt_instance_id",
    "success_same_instance",
    "false_resurrection",
    "unresolved_but_target_in_topk",
    "prediction_error",
    "familiarity_score",
    "novelty_score",
    "failure_bucket",
]

FRAME_FIELDNAMES = [
    "sequence_id",
    "frame_index",
    "object_file_count",
    "attended_object_count",
    "decision_count",
    "written_episode_count",
    "extended_episode_count",
    "reactivated_episode_count",
    "active_episode_count",
    "memory_context_used",
    "prediction_error_mean",
]


def run_eval(
    sequences: int = 2,
    max_frames: int = 80,
    output_dir: str | Path = "results/cognitive_reentry",
    min_gap: int = 8,
    seed: int = 41,
    seeds: list[int] | None = None,
    strict_min_iou: float = 0.25,
    top_k_audit: int = 5,
    recognizer_kwargs: dict[str, Any] | None = None,
    force_reentry_events: bool = True,
    guaranteed_reentry_count: int = 2,
    reentry_visibility_mode: str = "hard_hide",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    seed_values = seeds if seeds is not None else [seed]
    config = _build_config(
        sequences=sequences,
        max_frames=max_frames,
        force_reentry_events=force_reentry_events,
        guaranteed_reentry_count=guaranteed_reentry_count,
        reentry_visibility_mode=reentry_visibility_mode,
    )

    frame_rows: list[dict[str, Any]] = []
    reentry_rows: list[dict[str, Any]] = []
    all_decisions: list[Any] = []
    total_object_files = 0
    total_attended = 0
    total_written = 0
    total_extended = 0
    total_reactivated = 0
    final_episode_memory_size = 0
    planned_reentry_event_count = 0
    actual_reentry_event_count = 0
    discovered_reentry_event_count = 0
    mismatch_count = 0
    benchmark_valid_sequences = 0
    benchmark_invalid_reasons: Counter[str] = Counter()

    for seed_value in seed_values:
        generator = SyntheticStreamGenerator(config, seed=seed_value)
        for sequence_id in range(sequences):
            output_sequence_id = f"{seed_value}:{sequence_id}" if len(seed_values) > 1 else str(sequence_id)
            sequence = generator.generate_sequence(sequence_id)
            frames = sequence.frames[:max_frames]
            if len(frames) < 2:
                continue
            loop = _build_loop(recognizer_kwargs=recognizer_kwargs)
            results_by_frame: dict[int, CognitiveFrameResult] = {}
            planned_events = list(sequence.metadata.get("planned_reentry_event_rows", []) or [])
            actual_events = [
                event for event in list(sequence.metadata.get("actual_reentry_events", []) or [])
                if int(event.get("gap_length", 0)) >= int(min_gap)
            ]
            discovered_events = _discover_reentry_events(frames, min_gap=min_gap)
            planned_reentry_event_count += len(planned_events)
            actual_reentry_event_count += len(actual_events)
            discovered_reentry_event_count += len(discovered_events)
            mismatch_count += _event_mismatch_count(actual_events, discovered_events)
            if bool(sequence.metadata.get("benchmark_valid", False)) and actual_events:
                benchmark_valid_sequences += 1
            else:
                reason = str(sequence.metadata.get("benchmark_invalid_reason", "invalid_no_reentry_events") or "invalid_no_reentry_events")
                benchmark_invalid_reasons[reason] += 1
            events_by_reappear_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for event in actual_events:
                events_by_reappear_frame[int(event["reappear_frame"])].append(dict(event))

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
                frame_rows.append(_frame_row(output_sequence_id, result))
                for event in events_by_reappear_frame.get(int(sample.frame_index), []):
                    reentry_rows.append(
                        _reentry_row(
                            output_sequence_id,
                            event,
                            result,
                            loop,
                            sample,
                            strict_min_iou=strict_min_iou,
                            top_k_audit=top_k_audit,
                        )
                    )
            final_episode_memory_size += len(loop.episodic_memory)

    _write_csv(output_path / "frame_metrics.csv", frame_rows, FRAME_FIELDNAMES)
    _write_csv(output_path / "reentry_events.csv", reentry_rows, REENTRY_FIELDNAMES)
    summary = _summary(
        sequences=sequences,
        frame_count=len(seed_values) * sequences * max(0, min(max_frames, config.sequence_length) - 1),
        object_file_count=total_object_files,
        attended_count=total_attended,
        episode_write_count=total_written,
        episode_extend_count=total_extended,
        episode_reactivation_count=total_reactivated,
        episodic_memory_size=final_episode_memory_size,
        reentry_rows=reentry_rows,
        decisions=all_decisions,
        planned_reentry_event_count=planned_reentry_event_count,
        actual_reentry_event_count=actual_reentry_event_count,
        discovered_reentry_event_count=discovered_reentry_event_count,
        event_ledger_discovery_mismatch_count=mismatch_count,
        benchmark_valid_sequence_count=benchmark_valid_sequences,
        benchmark_invalid_reasons=dict(benchmark_invalid_reasons),
        total_sequence_count=len(seed_values) * sequences,
    )
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_path / "report.md").write_text(_report(summary, reentry_rows), encoding="utf-8")
    return summary


def _build_config(
    sequences: int,
    max_frames: int,
    force_reentry_events: bool = True,
    guaranteed_reentry_count: int = 2,
    reentry_visibility_mode: str = "hard_hide",
) -> SynthDatasetConfig:
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
            force_reentry_events=force_reentry_events,
            guaranteed_reentry_count=guaranteed_reentry_count,
            reentry_visibility_mode=reentry_visibility_mode,
            min_pre_visible_frames=8,
            min_post_visible_frames=8,
            actual_reentry_gap_range=(8, 18),
        ),
        outputs=("frame", "boxes", "masks", "instance_id", "concept_id"),
    )


def _build_loop(recognizer_kwargs: dict[str, Any] | None = None) -> VisualCognitiveLoop:
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
        recognizer=PredictiveRecognizer(**(recognizer_kwargs or {})),
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


def _event_mismatch_count(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    left_keys = {
        (int(event["instance_id"]), int(event["disappear_frame"]), int(event["reappear_frame"]))
        for event in left
    }
    right_keys = {
        (int(event["instance_id"]), int(event["disappear_frame"]), int(event["reappear_frame"]))
        for event in right
    }
    return len(left_keys.symmetric_difference(right_keys))


def _reentry_row(
    sequence_id: int,
    event: dict[str, Any],
    result: CognitiveFrameResult | None,
    loop: VisualCognitiveLoop,
    sample: Any | None,
    *,
    strict_min_iou: float,
    top_k_audit: int,
) -> dict[str, Any]:
    instance_id = int(event["instance_id"])
    gt_box_present = int(sample is not None and instance_id in [int(v) for v in getattr(sample, "instance_ids", [])])
    matched_object = None
    matched_iou = 0.0
    if result is not None:
        for object_file in result.object_files:
            if object_file.metadata.get("gt_instance_id") == instance_id:
                iou = float(object_file.metadata.get("gt_iou_eval_only", 0.0))
                if iou < matched_iou:
                    continue
                matched_iou = iou
                matched_object = object_file
        if matched_iou < strict_min_iou:
            matched_object = None
    decision = None
    retrievals = []
    object_attended = 0
    if result is not None and matched_object is not None:
        object_attended = int(any(row.object_file_id == matched_object.object_file_id for row in result.attended_object_files))
        retrievals = result.episodic_retrievals.get(matched_object.object_file_id, [])[:top_k_audit]
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
    target_episode = _find_target_episode(loop, instance_id, int(event["reappear_frame"]))
    target_episode_id = None if target_episode is None else target_episode.episode_id
    target_rank = 0
    target_score = 0.0
    for candidate in retrievals:
        if target_episode_id is not None and candidate.bundle.episode_id == target_episode_id:
            target_rank = int(candidate.rank)
            target_score = float(candidate.score)
            break
    topk_contains_target = int(target_rank > 0)
    decision_type = "" if decision is None else decision.decision_type
    rejection_reason = "" if decision is None else str(decision.metadata.get("rejection_reason", ""))
    success_same_instance = int(decision_type == "same_instance" and ((linked_gt == instance_id) or (top_gt == instance_id)))
    false_resurrection = int(
        decision is not None
        and decision.decision_type == "same_instance"
        and ((linked_gt is not None and linked_gt != instance_id) or (linked_gt is None and top_gt is not None and top_gt != instance_id))
    )
    unresolved_but_target_in_topk = int(topk_contains_target and not success_same_instance)
    failure_bucket = _failure_bucket(
        gt_box_present=bool(gt_box_present),
        matched_object=matched_object is not None,
        object_attended=bool(object_attended),
        target_episode=target_episode is not None,
        target_rank=target_rank,
        decision_type=decision_type,
        rejection_reason=rejection_reason,
        success=bool(success_same_instance),
        false_resurrection=bool(false_resurrection),
    )
    return {
        "sequence_id": sequence_id,
        "instance_id": instance_id,
        "disappear_frame": event["disappear_frame"],
        "reappear_frame": event["reappear_frame"],
        "gap_length": event["gap_length"],
        "gap_bucket": _gap_bucket(int(event["gap_length"])),
        "gt_box_present": gt_box_present,
        "matched_object_iou": matched_iou,
        "proposal_or_object_missing": int(matched_object is None),
        "matched_object_file_id": "" if matched_object is None else matched_object.object_file_id,
        "object_attended": object_attended,
        "attention_failure": int(matched_object is not None and not object_attended),
        "target_episode_exists": int(target_episode is not None),
        "target_episode_id": "" if target_episode is None else target_episode.episode_id,
        "target_episode_closed": 0 if target_episode is None else int(target_episode.closed),
        "target_episode_observation_count": 0 if target_episode is None else target_episode.observation_count,
        "target_episode_frame_start": "" if target_episode is None else target_episode.frame_start,
        "target_episode_frame_end": "" if target_episode is None else target_episode.frame_end,
        "target_episode_rank": target_rank,
        "target_episode_score": target_score,
        "top1_episode_id": "" if top_episode is None else top_episode.episode_id,
        "top1_score": 0.0 if top1 is None else top1.score,
        "top1_margin": 0.0 if top1 is None else top1.margin_to_next,
        "top1_gt_instance_id": "" if top_gt is None else top_gt,
        "top1_active_conflict": 0 if top1 is None else int(top1.active_conflict),
        "top1_bundle_closed": 0 if top_episode is None else int(top_episode.closed),
        "top1_reentry_gap": 0 if top1 is None else top1.reentry_gap,
        "topk_contains_target": topk_contains_target,
        "decision_type": decision_type,
        "rejection_reason": rejection_reason,
        "linked_episode_id": "" if linked_episode_id is None else linked_episode_id,
        "retrieved_top1_episode_id": "" if top_episode is None else top_episode.episode_id,
        "retrieved_top1_score": 0.0 if top1 is None else top1.score,
        "retrieved_top1_gt_instance_id": "" if top_gt is None else top_gt,
        "success_same_instance": success_same_instance,
        "false_resurrection": false_resurrection,
        "unresolved_but_target_in_topk": unresolved_but_target_in_topk,
        "prediction_error": 0.0 if decision is None else decision.prediction_error,
        "familiarity_score": 0.0 if decision is None else decision.familiarity_score,
        "novelty_score": 0.0 if decision is None else decision.novelty_score,
        "failure_bucket": failure_bucket,
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


def _find_target_episode(loop: VisualCognitiveLoop, instance_id: int, reappear_frame: int):
    candidates = [
        bundle
        for bundle in loop.episodic_memory.bundles
        if bundle.metadata.get("gt_instance_id") == instance_id
        and int(bundle.frame_start) < int(reappear_frame)
        and int(bundle.last_observed_frame if bundle.last_observed_frame is not None else bundle.frame_end) < int(reappear_frame)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda bundle: (
            int(bundle.closed),
            int(bundle.last_observed_frame if bundle.last_observed_frame is not None else bundle.frame_end),
            int(bundle.observation_count),
        ),
    )


def _gap_bucket(gap_length: int) -> str:
    if gap_length <= 3:
        return "gap_1_3"
    if gap_length <= 7:
        return "gap_4_7"
    if gap_length <= 15:
        return "gap_8_15"
    return "gap_16_plus"


def _failure_bucket(
    *,
    gt_box_present: bool,
    matched_object: bool,
    object_attended: bool,
    target_episode: bool,
    target_rank: int,
    decision_type: str,
    rejection_reason: str,
    success: bool,
    false_resurrection: bool,
) -> str:
    if success:
        return "success"
    if not gt_box_present:
        return "no_gt_box_at_reentry"
    if not matched_object:
        return "no_object_file_matched"
    if not object_attended:
        return "attention_missed_object"
    if not target_episode:
        return "target_episode_missing"
    if target_rank == 0:
        return "target_not_in_topk"
    if target_rank > 1:
        return "target_in_topk_but_low_rank"
    if rejection_reason == "low_retrieval_margin":
        return "low_margin_uncertain"
    if rejection_reason == "active_conflict":
        return "active_conflict_blocked"
    if false_resurrection:
        return "false_resurrection"
    if decision_type != "same_instance":
        return "decision_underconfident"
    return "target_in_topk_but_low_rank"


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
    planned_reentry_event_count: int,
    actual_reentry_event_count: int,
    discovered_reentry_event_count: int,
    event_ledger_discovery_mismatch_count: int,
    benchmark_valid_sequence_count: int,
    benchmark_invalid_reasons: dict[str, int],
    total_sequence_count: int,
) -> dict[str, Any]:
    reentry_count = len(reentry_rows)
    decision_counts = Counter(getattr(decision, "decision_type", "unknown") for decision in decisions)
    success_count = sum(int(row["success_same_instance"]) for row in reentry_rows)
    same_instance_reentry_count = sum(int(row["decision_type"] == "same_instance") for row in reentry_rows)
    false_count = sum(int(row["false_resurrection"]) for row in reentry_rows)
    target_present_count = sum(int(row["target_episode_exists"]) for row in reentry_rows)
    target_top1_count = sum(int(row["target_episode_rank"] == 1) for row in reentry_rows)
    target_top3_count = sum(int(1 <= int(row["target_episode_rank"]) <= 3) for row in reentry_rows)
    target_top5_count = sum(int(1 <= int(row["target_episode_rank"]) <= 5) for row in reentry_rows)
    unresolved_target_count = sum(int(row["unresolved_but_target_in_topk"]) for row in reentry_rows)
    matched_count = sum(int(row["matched_object_file_id"] != "") for row in reentry_rows)
    attended_count_reentry = sum(int(row["object_attended"]) for row in reentry_rows)
    top1_margins = [float(row["top1_margin"]) for row in reentry_rows if row["top1_episode_id"] != ""]
    target_ranks = [int(row["target_episode_rank"]) for row in reentry_rows if int(row["target_episode_rank"]) > 0]
    failure_buckets = Counter(str(row["failure_bucket"]) for row in reentry_rows)
    gap_bucket_metrics = _gap_bucket_metrics(reentry_rows)
    success_rate = success_count / reentry_count if reentry_count else 0.0
    false_rate = false_count / reentry_count if reentry_count else 0.0
    memory_items = episodic_memory_size
    episode_update_ratio = (
        (episode_write_count + episode_extend_count) / object_file_count if object_file_count else 0.0
    )
    benchmark_valid = actual_reentry_event_count > 0 and event_ledger_discovery_mismatch_count == 0
    benchmark_status = "valid" if benchmark_valid else "invalid_no_reentry_events"
    if actual_reentry_event_count > 0 and event_ledger_discovery_mismatch_count:
        benchmark_status = "invalid_event_ledger_mismatch"
    return {
        "sequence_count": total_sequence_count,
        "frame_count": frame_count,
        "benchmark_valid": bool(benchmark_valid),
        "benchmark_status": benchmark_status,
        "planned_reentry_event_count": int(planned_reentry_event_count),
        "actual_reentry_event_count": int(actual_reentry_event_count),
        "discovered_reentry_event_count": int(discovered_reentry_event_count),
        "event_ledger_discovery_mismatch_count": int(event_ledger_discovery_mismatch_count),
        "benchmark_valid_rate": benchmark_valid_sequence_count / total_sequence_count if total_sequence_count else 0.0,
        "benchmark_invalid_reason": "" if benchmark_valid else json.dumps(benchmark_invalid_reasons, sort_keys=True),
        "object_file_count": object_file_count,
        "attended_object_ratio_mean": attended_object_ratio(attended_count, object_file_count),
        "episodic_memory_size": episodic_memory_size,
        "episode_write_count": episode_write_count,
        "episode_extend_count": episode_extend_count,
        "episode_reactivation_count": episode_reactivation_count,
        "reentry_event_count": reentry_count,
        "long_gap_reentry_success_rate": success_rate,
        "false_resurrection_rate": false_rate,
        "proposal_recall_at_reentry": matched_count / reentry_count if reentry_count else 0.0,
        "attention_recall_at_reentry": attended_count_reentry / reentry_count if reentry_count else 0.0,
        "target_episode_presence_rate": target_present_count / reentry_count if reentry_count else 0.0,
        "target_episode_top1_rate": target_top1_count / reentry_count if reentry_count else 0.0,
        "target_episode_top3_rate": target_top3_count / reentry_count if reentry_count else 0.0,
        "target_episode_top5_rate": target_top5_count / reentry_count if reentry_count else 0.0,
        "unresolved_but_target_in_topk_rate": unresolved_target_count / reentry_count if reentry_count else 0.0,
        "same_instance_precision_at_reentry": success_count / same_instance_reentry_count if same_instance_reentry_count else 0.0,
        "same_instance_recall_at_reentry": success_count / reentry_count if reentry_count else 0.0,
        "false_resurrection_rate_at_reentry": false_rate,
        "mean_top1_margin": float(sum(top1_margins) / len(top1_margins)) if top1_margins else 0.0,
        "mean_target_rank": float(sum(target_ranks) / len(target_ranks)) if target_ranks else 0.0,
        "gap_bucket_metrics": gap_bucket_metrics,
        "failure_buckets": dict(failure_buckets),
        "uncertain_hold_rate": uncertain_hold_rate(decisions),
        "same_instance_rate": decision_counts.get("same_instance", 0) / len(decisions) if decisions else 0.0,
        "continuous_same_instance_rate": decision_counts.get("same_instance", 0) / len(decisions) if decisions else 0.0,
        "reentry_same_instance_rate": same_instance_reentry_count / reentry_count if reentry_count else 0.0,
        "same_concept_rate": decision_counts.get("same_concept", 0) / len(decisions) if decisions else 0.0,
        "new_concept_rate": decision_counts.get("new_concept", 0) / len(decisions) if decisions else 0.0,
        "mean_prediction_error": prediction_error_mean(decisions),
        "memory_compression_ratio": memory_compression_ratio(object_file_count, memory_items),
        "episode_update_ratio": episode_update_ratio,
    }


def _gap_bucket_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for bucket in ("gap_1_3", "gap_4_7", "gap_8_15", "gap_16_plus"):
        bucket_rows = [row for row in rows if row["gap_bucket"] == bucket]
        count = len(bucket_rows)
        output[bucket] = {
            "count": float(count),
            "success_rate": (
                sum(int(row["success_same_instance"]) for row in bucket_rows) / count if count else 0.0
            ),
            "false_resurrection_rate": (
                sum(int(row["false_resurrection"]) for row in bucket_rows) / count if count else 0.0
            ),
            "target_top5_rate": (
                sum(int(row["topk_contains_target"]) for row in bucket_rows) / count if count else 0.0
            ),
        }
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    resolved_fieldnames = fieldnames or sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _report(summary: dict[str, Any], reentry_rows: list[dict[str, Any]]) -> str:
    buckets = Counter(str(row["failure_bucket"]) for row in reentry_rows)
    total = max(1, len(reentry_rows))
    table = ["| bucket | count | rate |", "|---|---:|---:|"]
    for bucket, count in buckets.most_common():
        table.append(f"| {bucket} | {count} | {count / total:.4f} |")
    top_bucket = buckets.most_common(1)[0][0] if buckets else "none"
    false_examples = [row for row in reentry_rows if int(row["false_resurrection"])][:5]
    examples_text = "\n".join(
        f"- sequence={row['sequence_id']} instance={row['instance_id']} "
        f"top1_episode={row['top1_episode_id']} top1_gt={row['top1_gt_instance_id']} "
        f"margin={float(row['top1_margin']):.4f}"
        for row in false_examples
    ) or "- none"
    return (
        "# Cognitive Re-entry Evaluation\n\n"
        "This is a synthetic mechanism evaluation for event-aware visual memory. "
        "GT is used only for matching/evaluation and episode metadata audit, not for online scoring.\n\n"
        f"- benchmark_status: {summary['benchmark_status']}\n"
        f"- actual_reentry_event_count: {summary['actual_reentry_event_count']}\n"
        f"- discovered_reentry_event_count: {summary['discovered_reentry_event_count']}\n"
        f"- event_ledger_discovery_mismatch_count: {summary['event_ledger_discovery_mismatch_count']}\n"
        f"- reentry_event_count: {summary['reentry_event_count']}\n"
        f"- long_gap_reentry_success_rate: {summary['long_gap_reentry_success_rate']:.4f}\n"
        f"- false_resurrection_rate_at_reentry: {summary['false_resurrection_rate_at_reentry']:.4f}\n"
        f"- target_episode_top5_rate: {summary['target_episode_top5_rate']:.4f}\n"
        f"- same_instance_precision_at_reentry: {summary['same_instance_precision_at_reentry']:.4f}\n"
        f"- uncertain_hold_rate: {summary['uncertain_hold_rate']:.4f}\n"
        f"- mean_prediction_error: {summary['mean_prediction_error']:.4f}\n\n"
        "## Failure Buckets\n\n"
        + "\n".join(table)
        + "\n\n"
        f"Top failure bucket: `{top_bucket}`.\n\n"
        + (
            "No valid re-entry events were found. This benchmark run cannot be interpreted as an algorithm "
            "recovery failure.\n\n"
            if summary["benchmark_status"] != "valid"
            else ""
        )
        + "Interpretation guide: perception failures appear as `no_object_file_matched`, "
        "attention failures as `attention_missed_object`, memory-write failures as "
        "`target_episode_missing`, retrieval failures as `target_not_in_topk` or "
        "`target_in_topk_but_low_rank`, and decision failures as `low_margin_uncertain`, "
        "`active_conflict_blocked`, or `decision_underconfident`.\n\n"
        "## False Resurrection Examples\n\n"
        + examples_text
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--output-dir", type=str, default="results/cognitive_reentry")
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--strict-min-iou", type=float, default=0.25)
    parser.add_argument("--top-k-audit", type=int, default=5)
    parser.add_argument("--same-instance-margin-threshold", type=float, default=0.08)
    parser.add_argument("--reentry-same-instance-threshold", type=float, default=0.76)
    parser.add_argument("--force-reentry-events", type=int, default=1)
    parser.add_argument("--guaranteed-reentry-count", type=int, default=2)
    parser.add_argument("--reentry-visibility-mode", type=str, default="hard_hide")
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()] or None
    summary = run_eval(
        sequences=args.sequences,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
        min_gap=args.min_gap,
        seed=args.seed,
        seeds=seeds,
        strict_min_iou=args.strict_min_iou,
        top_k_audit=args.top_k_audit,
        recognizer_kwargs={
            "same_instance_margin_threshold": args.same_instance_margin_threshold,
            "reentry_same_instance_threshold": args.reentry_same_instance_threshold,
        },
        force_reentry_events=bool(args.force_reentry_events),
        guaranteed_reentry_count=args.guaranteed_reentry_count,
        reentry_visibility_mode=args.reentry_visibility_mode,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
