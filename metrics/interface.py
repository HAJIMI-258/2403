"""Unified metric interface for streaming NOPS-OWR episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Hashable, Mapping, Sequence

from .metric_audit import MetricAuditSummary, build_metric_audit
from .metrics_core import Box, MetricSummary, greedy_match_boxes, summarize_phase1_metrics


@dataclass(slots=True)
class FrameMetricRecord:
    frame_index: int
    num_gt: int
    num_pred: int
    matched_objects: int
    u_recall: float
    active_prototypes: int
    memory_size: int


@dataclass(slots=True)
class EpisodeMetricBundle:
    summary: MetricSummary
    audit: MetricAuditSummary
    frame_metrics: list[FrameMetricRecord]


def build_frame_metric_records(
    *,
    gt_boxes_per_frame: Sequence[Sequence[Box]],
    pred_boxes_per_frame: Sequence[Sequence[Box]],
    active_prototype_sets: Sequence[Collection[Hashable]],
    memory_sizes: Sequence[int],
    iou_threshold: float = 0.5,
) -> list[FrameMetricRecord]:
    records: list[FrameMetricRecord] = []
    for frame_index, (gt_boxes, pred_boxes) in enumerate(zip(gt_boxes_per_frame, pred_boxes_per_frame), start=1):
        matches = greedy_match_boxes(gt_boxes, pred_boxes, iou_threshold=iou_threshold)
        records.append(
            FrameMetricRecord(
                frame_index=frame_index,
                num_gt=len(gt_boxes),
                num_pred=len(pred_boxes),
                matched_objects=len(matches),
                u_recall=(len(matches) / len(gt_boxes)) if gt_boxes else 1.0,
                active_prototypes=len(active_prototype_sets[frame_index - 1]) if frame_index - 1 < len(active_prototype_sets) else 0,
                memory_size=int(memory_sizes[frame_index - 1]) if frame_index - 1 < len(memory_sizes) else 0,
            )
        )
    return records


def build_episode_metric_bundle(
    *,
    gt_boxes_per_frame: Sequence[Sequence[Box]],
    pred_boxes_per_frame: Sequence[Sequence[Box]],
    prototype_concept_assignments: Sequence[tuple[Hashable, Hashable]],
    instance_assignment_history: Sequence[Mapping[Hashable, Hashable | None]],
    active_prototype_sets: Sequence[Collection[Hashable]],
    memory_sizes: Sequence[int],
    birth_count: int,
    merge_count: int,
    decay_count: int,
    iou_threshold: float = 0.5,
) -> EpisodeMetricBundle:
    summary = summarize_phase1_metrics(
        gt_boxes_per_frame=gt_boxes_per_frame,
        pred_boxes_per_frame=pred_boxes_per_frame,
        prototype_concept_assignments=prototype_concept_assignments,
        instance_assignment_history=instance_assignment_history,
        active_prototype_sets=active_prototype_sets,
        memory_sizes=memory_sizes,
        iou_threshold=iou_threshold,
    )
    audit = build_metric_audit(
        gt_boxes_per_frame=gt_boxes_per_frame,
        pred_boxes_per_frame=pred_boxes_per_frame,
        prototype_concept_assignments=prototype_concept_assignments,
        instance_assignment_history=instance_assignment_history,
        memory_sizes=memory_sizes,
        birth_count=birth_count,
        merge_count=merge_count,
        decay_count=decay_count,
        iou_threshold=iou_threshold,
    )
    frame_metrics = build_frame_metric_records(
        gt_boxes_per_frame=gt_boxes_per_frame,
        pred_boxes_per_frame=pred_boxes_per_frame,
        active_prototype_sets=active_prototype_sets,
        memory_sizes=memory_sizes,
        iou_threshold=iou_threshold,
    )
    return EpisodeMetricBundle(summary=summary, audit=audit, frame_metrics=frame_metrics)
