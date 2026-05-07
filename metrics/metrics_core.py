"""Core metrics for Phase 1 streaming experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Collection, Hashable, Mapping, Sequence

Box = tuple[float, float, float, float]
Assignment = tuple[Hashable, Hashable]


@dataclass(slots=True)
class MetricSummary:
    u_recall: float
    purity: float
    pfr: float
    idsw: int
    churn: float
    memory_growth: float


def bbox_iou(box_a: Box, box_b: Box) -> float:
    """Return IoU for two boxes in xyxy format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


def greedy_match_boxes(
    gt_boxes: Sequence[Box],
    pred_boxes: Sequence[Box],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Greedily match predictions to ground truth by IoU."""
    candidates: list[tuple[float, int, int]] = []
    for gt_idx, gt_box in enumerate(gt_boxes):
        for pred_idx, pred_box in enumerate(pred_boxes):
            iou = bbox_iou(gt_box, pred_box)
            if iou >= iou_threshold:
                candidates.append((iou, gt_idx, pred_idx))

    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for iou, gt_idx, pred_idx in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, iou))

    return matches


def u_recall(
    gt_boxes: Sequence[Box],
    pred_boxes: Sequence[Box],
    iou_threshold: float = 0.5,
) -> float:
    """Unsupervised recall: matched GT objects over total GT objects."""
    if not gt_boxes:
        return 1.0
    matches = greedy_match_boxes(gt_boxes, pred_boxes, iou_threshold=iou_threshold)
    return len(matches) / len(gt_boxes)


def purity(assignments: Sequence[Assignment]) -> float:
    """Cluster purity over (prototype_id, concept_id) assignments."""
    if not assignments:
        return 0.0

    prototype_to_concepts: dict[Hashable, Counter] = defaultdict(Counter)
    for prototype_id, concept_id in assignments:
        prototype_to_concepts[prototype_id][concept_id] += 1

    correct = sum(max(counter.values()) for counter in prototype_to_concepts.values())
    return correct / len(assignments)


def prototype_fragmentation_rate(assignments: Sequence[Assignment]) -> float:
    """Average number of extra prototypes used per concept."""
    if not assignments:
        return 0.0

    concept_to_prototypes: dict[Hashable, set[Hashable]] = defaultdict(set)
    for prototype_id, concept_id in assignments:
        concept_to_prototypes[concept_id].add(prototype_id)

    fragments = [max(0, len(prototypes) - 1) for prototypes in concept_to_prototypes.values()]
    return sum(fragments) / len(fragments)


def identity_switches(history: Sequence[Mapping[Hashable, Hashable | None]]) -> int:
    """Count assignment switches for persistent instances across time."""
    last_assignment: dict[Hashable, Hashable] = {}
    switches = 0

    for frame_assignments in history:
        for instance_id, assignment in frame_assignments.items():
            if assignment is None:
                continue
            if instance_id in last_assignment and last_assignment[instance_id] != assignment:
                switches += 1
            last_assignment[instance_id] = assignment

    return switches


def churn(active_sets: Sequence[Collection[Hashable]]) -> float:
    """Average frame-to-frame prototype set volatility."""
    if len(active_sets) < 2:
        return 0.0

    total = 0.0
    pairs = 0

    for previous, current in zip(active_sets, active_sets[1:]):
        prev_set = set(previous)
        curr_set = set(current)
        union = prev_set | curr_set
        if not union:
            total += 0.0
        else:
            total += len(prev_set ^ curr_set) / len(union)
        pairs += 1

    return total / pairs


def memory_growth(memory_sizes: Sequence[int]) -> float:
    """Average prototype count increase per frame."""
    if len(memory_sizes) < 2:
        return 0.0
    return (memory_sizes[-1] - memory_sizes[0]) / (len(memory_sizes) - 1)


def summarize_phase1_metrics(
    gt_boxes_per_frame: Sequence[Sequence[Box]],
    pred_boxes_per_frame: Sequence[Sequence[Box]],
    prototype_concept_assignments: Sequence[Assignment],
    instance_assignment_history: Sequence[Mapping[Hashable, Hashable | None]],
    active_prototype_sets: Sequence[Collection[Hashable]],
    memory_sizes: Sequence[int],
    iou_threshold: float = 0.5,
) -> MetricSummary:
    """Aggregate the Phase 1 core metrics into a single summary."""
    frame_recalls = [
        u_recall(gt_boxes, pred_boxes, iou_threshold=iou_threshold)
        for gt_boxes, pred_boxes in zip(gt_boxes_per_frame, pred_boxes_per_frame)
    ]
    mean_recall = sum(frame_recalls) / len(frame_recalls) if frame_recalls else 0.0

    return MetricSummary(
        u_recall=mean_recall,
        purity=purity(prototype_concept_assignments),
        pfr=prototype_fragmentation_rate(prototype_concept_assignments),
        idsw=identity_switches(instance_assignment_history),
        churn=churn(active_prototype_sets),
        memory_growth=memory_growth(memory_sizes),
    )

