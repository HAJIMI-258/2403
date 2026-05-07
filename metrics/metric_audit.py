"""Raw-count audit helpers for Phase 2B metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

from .metrics_core import Box, greedy_match_boxes, identity_switches


@dataclass(slots=True)
class MetricAuditSummary:
    total_frames: int
    total_gt_objects: int
    total_matched_objects: int
    u_recall: float
    u_recall_frame_sum: float
    u_recall_frame_denominator: int
    object_recall: float
    object_recall_numerator: int
    object_recall_denominator: int
    fragmented_concepts: int
    extra_prototypes: int
    total_concepts: int
    pfr: float
    pfr_numerator: int
    pfr_denominator: int
    initial_proto_count: int
    final_proto_count: int
    net_proto_growth: int
    memory_growth: float
    memory_growth_numerator: int
    memory_growth_denominator: int
    net_birth_count: int
    net_merge_count: int
    net_decay_count: int
    total_id_switches: int
    idsw: int

    def to_row(self) -> dict[str, float | int]:
        return {
            "total_frames": self.total_frames,
            "total_gt_objects": self.total_gt_objects,
            "total_matched_objects": self.total_matched_objects,
            "u_recall": self.u_recall,
            "u_recall_frame_sum": self.u_recall_frame_sum,
            "u_recall_frame_denominator": self.u_recall_frame_denominator,
            "object_recall": self.object_recall,
            "object_recall_numerator": self.object_recall_numerator,
            "object_recall_denominator": self.object_recall_denominator,
            "fragmented_concepts": self.fragmented_concepts,
            "extra_prototypes": self.extra_prototypes,
            "total_concepts": self.total_concepts,
            "pfr": self.pfr,
            "pfr_numerator": self.pfr_numerator,
            "pfr_denominator": self.pfr_denominator,
            "initial_proto_count": self.initial_proto_count,
            "final_proto_count": self.final_proto_count,
            "net_proto_growth": self.net_proto_growth,
            "memory_growth": self.memory_growth,
            "memory_growth_numerator": self.memory_growth_numerator,
            "memory_growth_denominator": self.memory_growth_denominator,
            "net_birth_count": self.net_birth_count,
            "net_merge_count": self.net_merge_count,
            "net_decay_count": self.net_decay_count,
            "total_id_switches": self.total_id_switches,
            "idsw": self.idsw,
        }


def build_metric_audit(
    gt_boxes_per_frame: Sequence[Sequence[Box]],
    pred_boxes_per_frame: Sequence[Sequence[Box]],
    prototype_concept_assignments: Sequence[tuple[Hashable, Hashable]],
    instance_assignment_history: Sequence[Mapping[Hashable, Hashable | None]],
    memory_sizes: Sequence[int],
    birth_count: int,
    merge_count: int,
    decay_count: int,
    iou_threshold: float = 0.5,
) -> MetricAuditSummary:
    frame_recalls: list[float] = []
    total_gt_objects = 0
    total_matched_objects = 0

    for gt_boxes, pred_boxes in zip(gt_boxes_per_frame, pred_boxes_per_frame):
        matches = greedy_match_boxes(gt_boxes, pred_boxes, iou_threshold=iou_threshold)
        total_gt_objects += len(gt_boxes)
        total_matched_objects += len(matches)
        frame_recalls.append(len(matches) / len(gt_boxes) if gt_boxes else 1.0)

    fragmented_concepts, extra_prototypes, total_concepts = fragmentation_counts(prototype_concept_assignments)
    frame_denominator = len(frame_recalls)
    frame_sum = float(sum(frame_recalls))
    u_recall = frame_sum / frame_denominator if frame_denominator else 0.0
    object_recall = total_matched_objects / total_gt_objects if total_gt_objects else 0.0

    initial_proto_count = int(memory_sizes[0]) if memory_sizes else 0
    final_proto_count = int(memory_sizes[-1]) if memory_sizes else 0
    net_proto_growth = final_proto_count - initial_proto_count
    growth_denominator = max(len(memory_sizes) - 1, 0)
    memory_growth = net_proto_growth / growth_denominator if growth_denominator else 0.0
    switches = identity_switches(instance_assignment_history)

    return MetricAuditSummary(
        total_frames=frame_denominator,
        total_gt_objects=total_gt_objects,
        total_matched_objects=total_matched_objects,
        u_recall=u_recall,
        u_recall_frame_sum=frame_sum,
        u_recall_frame_denominator=frame_denominator,
        object_recall=object_recall,
        object_recall_numerator=total_matched_objects,
        object_recall_denominator=total_gt_objects,
        fragmented_concepts=fragmented_concepts,
        extra_prototypes=extra_prototypes,
        total_concepts=total_concepts,
        pfr=extra_prototypes / total_concepts if total_concepts else 0.0,
        pfr_numerator=extra_prototypes,
        pfr_denominator=total_concepts,
        initial_proto_count=initial_proto_count,
        final_proto_count=final_proto_count,
        net_proto_growth=net_proto_growth,
        memory_growth=memory_growth,
        memory_growth_numerator=net_proto_growth,
        memory_growth_denominator=growth_denominator,
        net_birth_count=int(birth_count),
        net_merge_count=int(merge_count),
        net_decay_count=int(decay_count),
        total_id_switches=switches,
        idsw=switches,
    )


def fragmentation_counts(
    assignments: Sequence[tuple[Hashable, Hashable]],
) -> tuple[int, int, int]:
    concept_to_prototypes: dict[Hashable, set[Hashable]] = {}
    for prototype_id, concept_id in assignments:
        concept_to_prototypes.setdefault(concept_id, set()).add(prototype_id)

    fragmented_concepts = 0
    extra_prototypes = 0
    for prototypes in concept_to_prototypes.values():
        if len(prototypes) > 1:
            fragmented_concepts += 1
        extra_prototypes += max(0, len(prototypes) - 1)

    return fragmented_concepts, extra_prototypes, len(concept_to_prototypes)
