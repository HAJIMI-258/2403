"""Unified streaming evaluator for NOPS-OWR episodes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from metrics import build_episode_metric_bundle
from metrics.metrics_core import greedy_match_boxes, identity_switches, purity, u_recall
from nops_owr.controller import BudgetEnforcer, BudgetReport
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.memory import MinimalPrototypeMemory
from nops_owr.objectness import MinimalObjectnessField
from nops_owr.tracking import MinimalTemporalIdentityTracker


@dataclass(slots=True)
class StreamingEpisodeFrameRecord:
    frame_index: int
    frame: np.ndarray | None
    gt_boxes: list[tuple[int, int, int, int]]
    masks: list[np.ndarray]
    instance_ids: list[int]
    concept_ids: list[int]
    objectness_output: Any
    tracking_output: Any
    memory_output: Any
    predicted_boxes: list[tuple[int, int, int, int]]
    predicted_ids: list[int]
    matches: list[tuple[int, int, float]]
    recall_hit: float
    frame_purity: float
    objectness_recall: float
    false_hot_area: float


@dataclass(slots=True)
class ObjectnessFrameSnapshot:
    proposals: list[Any]


@dataclass(slots=True)
class TrackingFrameSnapshot:
    assignments: list[Any]
    active_track_count: int
    dormant_track_count: int
    ghost_track_count: int
    retired_track_count: int
    reactivation_attempts: int
    resurrection_attempts: int
    resurrection_successes: int
    continuation_resurrection_attempts: int
    continuation_resurrection_successes: int


@dataclass(slots=True)
class MemoryFrameSnapshot:
    assignments: list[Any]
    continuation_bank_count: int
    continuation_archive_events: int


@dataclass(slots=True)
class StreamingEpisodeResult:
    summary: Any
    audit: Any
    frame_metrics: list[Any]
    budget_report: BudgetReport
    action_counter: dict[str, int]
    decay_count: int
    primary_monitoring: dict[str, float | int]
    secondary_monitoring: dict[str, float | int]
    frame_records: list[StreamingEpisodeFrameRecord]


class StreamingEpisodeEvaluator:
    """Run the minimal NOPS-OWR pipeline with shared metrics and budget checks."""

    def __init__(
        self,
        config_payload: dict,
        *,
        field_override: dict | None = None,
        tracking_override: dict | None = None,
        memory_override: dict | None = None,
    ) -> None:
        self.config_payload = config_payload
        self.field_config = _get_field_config(config_payload)
        if field_override:
            self.field_config.update(field_override)
        self.tracking_config = dict(config_payload["tracking"])
        if tracking_override:
            self.tracking_config.update(tracking_override)
        self.memory_config = dict(config_payload["memory"])
        if memory_override:
            self.memory_config.update(memory_override)
        self.reentry_window = int(config_payload.get("evaluation", {}).get("reentry_window", 8))

    def evaluate(
        self,
        sequence,
        *,
        collect_frames: bool = False,
        frame_record_mode: str = "full",
    ) -> StreamingEpisodeResult:
        encoder = MinimalSpikeEncoder(**self.config_payload["model"]["spike_encoder"])
        objectness = MinimalObjectnessField(**self.field_config)
        tracker = MinimalTemporalIdentityTracker(**self.tracking_config)
        memory = MinimalPrototypeMemory(**self.memory_config)
        budget = BudgetEnforcer(
            memory_budget=self.memory_config["memory_budget"],
            proposal_budget=self.field_config.get("max_proposals"),
        )

        gt_boxes_per_frame = []
        pred_boxes_per_frame = []
        prototype_concept_assignments = []
        instance_assignment_history = []
        active_sets = []
        memory_sizes = []
        tracking_assignment_history = []
        action_counter: Counter[str] = Counter()
        decay_count = 0
        frame_records: list[StreamingEpisodeFrameRecord] = []
        unique_track_ids: set[int] = set()
        reactivation_successes = 0
        concept_only_recoveries = 0
        objectness_recalls: list[float] = []
        false_hot_areas: list[float] = []
        unmatched_tracks: list[int] = []
        dormant_tracks: list[int] = []
        ghost_tracks: list[int] = []
        retired_tracks: list[int] = []

        for frame_offset in range(1, len(sequence.frames)):
            prev_frame = sequence.frames[frame_offset - 1]
            current_frame = sequence.frames[frame_offset]

            encoding = encoder.encode(prev_frame.frame, current_frame.frame)
            objectness_output = objectness.compute(encoding)
            tracking_output = tracker.update(
                proposals=objectness_output.proposals,
                encoding=encoding,
                heatmap=objectness_output.heatmap,
                current_frame=current_frame.frame,
                frame_index=current_frame.frame_index,
            )
            memory_output = memory.update(
                tracking_output.assignments,
                frame_index=current_frame.frame_index,
                track_states=(
                    tracking_output.active_tracks
                    + tracking_output.dormant_tracks
                    + tracking_output.ghost_tracks
                    + tracking_output.retired_tracks
                ),
            )
            tracker.apply_concept_gated_resurrection(
                tracking_output,
                memory_output,
                frame_index=current_frame.frame_index,
                frame_shape=objectness_output.heatmap.shape,
            )
            tracker.bind_prototypes(memory_output.assignments)

            predicted_boxes = [assignment.box for assignment in memory_output.assignments]
            predicted_ids = [assignment.prototype_id for assignment in memory_output.assignments]
            tracking_boxes = [assignment.box for assignment in tracking_output.assignments]
            tracking_ids = [assignment.track_id for assignment in tracking_output.assignments]
            unique_track_ids.update(tracking_ids)
            reactivation_successes += len(tracking_output.reactivated_track_ids)
            concept_only_recoveries += sum(int(assignment.concept_only_recovery) for assignment in memory_output.assignments)

            gt_boxes_per_frame.append(list(current_frame.boxes))
            pred_boxes_per_frame.append(predicted_boxes)
            active_sets.append(list(memory_output.active_prototype_ids))
            memory_sizes.append(int(memory_output.total_prototypes))

            history = {instance_id: None for instance_id in current_frame.instance_ids}
            matches = greedy_match_boxes(current_frame.boxes, predicted_boxes, iou_threshold=0.5)
            matched_pairs: list[tuple[int, int]] = []
            for gt_index, pred_index, _ in matches:
                history[current_frame.instance_ids[gt_index]] = predicted_ids[pred_index]
                pair = (predicted_ids[pred_index], current_frame.concept_ids[gt_index])
                prototype_concept_assignments.append(pair)
                matched_pairs.append(pair)
            instance_assignment_history.append(history)

            tracking_history = {instance_id: None for instance_id in current_frame.instance_ids}
            tracking_matches = greedy_match_boxes(current_frame.boxes, tracking_boxes, iou_threshold=0.5)
            for gt_index, pred_index, _ in tracking_matches:
                tracking_history[current_frame.instance_ids[gt_index]] = tracking_ids[pred_index]
            tracking_assignment_history.append(tracking_history)

            for assignment in memory_output.assignments:
                action_counter[assignment.action] += 1
            decay_count += max(0, len(memory_output.retired_prototype_ids) - len(memory_output.budget_pruned_ids))

            objectness_recall = u_recall(
                current_frame.boxes,
                [proposal.box for proposal in objectness_output.proposals],
                iou_threshold=0.5,
            )
            false_hot_area = _false_hot_area(objectness_output.binary_mask, current_frame.masks)
            objectness_recalls.append(float(objectness_recall))
            false_hot_areas.append(float(false_hot_area))
            unmatched_tracks.append(int(tracking_output.unmatched_track_count))
            dormant_tracks.append(int(tracking_output.dormant_track_count))
            ghost_tracks.append(int(tracking_output.ghost_track_count))
            retired_tracks.append(int(tracking_output.retired_track_count))

            budget.observe(
                frame_index=current_frame.frame_index,
                proposals=len(objectness_output.proposals),
                active_tracks=len(tracking_output.active_tracks),
                dormant_tracks=tracking_output.dormant_track_count,
                memory_size=memory_output.total_prototypes,
            )

            if collect_frames:
                if frame_record_mode == "lite":
                    objectness_record = ObjectnessFrameSnapshot(proposals=list(objectness_output.proposals))
                    tracking_record = TrackingFrameSnapshot(
                        assignments=list(tracking_output.assignments),
                        active_track_count=int(tracking_output.active_track_count),
                        dormant_track_count=int(tracking_output.dormant_track_count),
                        ghost_track_count=int(tracking_output.ghost_track_count),
                        retired_track_count=int(tracking_output.retired_track_count),
                        reactivation_attempts=int(tracking_output.reactivation_attempts),
                        resurrection_attempts=int(tracking_output.resurrection_attempts),
                        resurrection_successes=int(tracking_output.resurrection_successes),
                        continuation_resurrection_attempts=int(getattr(tracking_output, "continuation_resurrection_attempts", 0)),
                        continuation_resurrection_successes=int(getattr(tracking_output, "continuation_resurrection_successes", 0)),
                    )
                    memory_record = MemoryFrameSnapshot(
                        assignments=list(memory_output.assignments),
                        continuation_bank_count=int(getattr(memory_output, "continuation_bank_count", 0)),
                        continuation_archive_events=int(getattr(memory_output, "continuation_archive_events", 0)),
                    )
                    frame_data = None
                    mask_data: list[np.ndarray] = []
                else:
                    objectness_record = objectness_output
                    tracking_record = tracking_output
                    memory_record = memory_output
                    frame_data = current_frame.frame
                    mask_data = list(current_frame.masks)
                frame_records.append(
                    StreamingEpisodeFrameRecord(
                        frame_index=current_frame.frame_index,
                        frame=frame_data,
                        gt_boxes=list(current_frame.boxes),
                        masks=mask_data,
                        instance_ids=list(current_frame.instance_ids),
                        concept_ids=list(current_frame.concept_ids),
                        objectness_output=objectness_record,
                        tracking_output=tracking_record,
                        memory_output=memory_record,
                        predicted_boxes=predicted_boxes,
                        predicted_ids=predicted_ids,
                        matches=matches,
                        recall_hit=len(matches) / len(current_frame.boxes) if current_frame.boxes else 1.0,
                        frame_purity=purity(matched_pairs),
                        objectness_recall=float(objectness_recall),
                        false_hot_area=float(false_hot_area),
                    )
                )

        metrics = build_episode_metric_bundle(
            gt_boxes_per_frame=gt_boxes_per_frame,
            pred_boxes_per_frame=pred_boxes_per_frame,
            prototype_concept_assignments=prototype_concept_assignments,
            instance_assignment_history=instance_assignment_history,
            active_prototype_sets=active_sets,
            memory_sizes=memory_sizes,
            birth_count=action_counter.get("birth", 0),
            merge_count=action_counter.get("merge", 0),
            decay_count=decay_count,
            iou_threshold=0.5,
        )
        same_track_reentry = _compute_reentry_recovery(
            sequence,
            tracking_assignment_history,
            recovery_window=self.reentry_window,
        )
        same_prototype_reentry = _compute_reentry_recovery(
            sequence,
            instance_assignment_history,
            recovery_window=self.reentry_window,
        )

        budget_report = budget.finalize()

        primary_monitoring = {
            "track_idsw": int(identity_switches(tracking_assignment_history)),
            "prototype_idsw": int(metrics.summary.idsw),
            "reentry_recovery_rate": float(same_track_reentry["rate"]),
            "reentry_events": int(same_track_reentry["events"]),
            "same_track_reentry_recovery": float(same_track_reentry["rate"]),
            "same_track_reentry_events": int(same_track_reentry["events"]),
            "same_prototype_reentry_recovery": float(same_prototype_reentry["rate"]),
            "same_prototype_reentry_events": int(same_prototype_reentry["events"]),
            "reactivation_successes": int(reactivation_successes),
            "concept_only_recoveries": int(concept_only_recoveries),
            "created_tracks": int(len(unique_track_ids)),
            "mean_unmatched_tracks": _mean(unmatched_tracks),
            "mean_dormant_tracks": _mean(dormant_tracks),
            "max_dormant_tracks": int(max(dormant_tracks, default=0)),
            "mean_ghost_tracks": _mean(ghost_tracks),
            "max_ghost_tracks": int(max(ghost_tracks, default=0)),
            "mean_retired_tracks": _mean(retired_tracks),
            "max_retired_tracks": int(max(retired_tracks, default=0)),
        }
        secondary_monitoring = {
            "objectness_recall": _mean(objectness_recalls),
            "false_hot_area": _mean(false_hot_areas),
            "peak_false_hot_area": float(max(false_hot_areas, default=0.0)),
            "budget_violation_frames": int(budget_report.violation_frames),
        }

        return StreamingEpisodeResult(
            summary=metrics.summary,
            audit=metrics.audit,
            frame_metrics=metrics.frame_metrics,
            budget_report=budget_report,
            action_counter=dict(action_counter),
            decay_count=decay_count,
            primary_monitoring=primary_monitoring,
            secondary_monitoring=secondary_monitoring,
            frame_records=frame_records,
        )


def _mean(values: list[int] | list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _false_hot_area(binary_mask: np.ndarray, masks: list[np.ndarray]) -> float:
    gt_mask = np.zeros_like(binary_mask, dtype=bool)
    for mask in masks:
        gt_mask |= mask.astype(bool)
    false_hot = binary_mask.astype(bool) & ~gt_mask
    return float(false_hot.sum() / max(false_hot.size, 1))


def _compute_reentry_recovery(
    sequence,
    assignment_history: list[dict[int, int | None]],
    recovery_window: int = 1,
) -> dict[str, float | int]:
    visible_history: dict[int, list[bool]] = {}
    assigned_history: dict[int, list[int | None]] = {}
    for frame_index in range(1, len(sequence.frames)):
        frame = sequence.frames[frame_index]
        assignment_map = assignment_history[frame_index - 1]
        visible_ids = set(frame.instance_ids)
        all_ids = set(visible_ids) | set(visible_history.keys()) | set(assignment_map.keys())
        for instance_id in all_ids:
            visible_history.setdefault(instance_id, []).append(instance_id in visible_ids)
            assigned_history.setdefault(instance_id, []).append(assignment_map.get(instance_id))

    events = 0
    recovered = 0
    for instance_id, visibility in visible_history.items():
        ids = assigned_history[instance_id]
        seen_ids = set()
        if visibility and visibility[0] and ids[0] is not None:
            seen_ids.add(ids[0])
        for index in range(1, len(visibility)):
            if visibility[index] and not visibility[index - 1]:
                events += 1
                window_ids = [
                    value
                    for value in ids[index : min(len(ids), index + max(1, recovery_window))]
                    if value is not None
                ]
                if any(value in seen_ids for value in window_ids):
                    recovered += 1
            if visibility[index] and ids[index] is not None:
                seen_ids.add(ids[index])
    return {"events": events, "recovered": recovered, "rate": recovered / events if events else 0.0}


def _get_field_config(config_payload: dict) -> dict:
    if "field" in config_payload:
        return dict(config_payload["field"])
    return dict(config_payload["model"]["objectness"])
