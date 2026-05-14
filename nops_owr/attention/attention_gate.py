"""Sparse attention gate for object-file processing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nops_owr.cognition.object_file import ObjectFile


@dataclass(slots=True)
class AttentionGateConfig:
    max_attended_objects: int = 4
    min_quality: float = 0.05
    quality_weight: float = 0.30
    novelty_weight: float = 0.20
    surprise_weight: float = 0.15
    prediction_error_weight: float = 0.20
    motion_weight: float = 0.10
    low_familiarity_weight: float = 0.05
    task_salience_weight: float = 0.0
    proposal_source_score_weight: float = 0.0
    component_source_bonus: float = 0.0
    saliency_window_source_bonus: float = 0.0
    memory_guided_window_source_bonus: float = 0.0
    memory_template_window_source_bonus: float = 0.0
    diversity_iou_penalty: float = 0.0
    diversity_center_penalty: float = 0.0
    diversity_center_scale: float = 80.0


class AttentionGate:
    """Select high-value object files for expensive memory recognition."""

    def __init__(
        self,
        max_attended_objects: int = 4,
        min_quality: float = 0.05,
        config: AttentionGateConfig | None = None,
    ) -> None:
        self.config = config or AttentionGateConfig(
            max_attended_objects=max_attended_objects,
            min_quality=min_quality,
        )

    @property
    def max_attended_objects(self) -> int:
        return self.config.max_attended_objects

    def score(self, object_file: ObjectFile, task_salience: float = 0.0) -> float:
        motion_salience = _motion_salience(object_file.motion_signature)
        score = (
            self.config.quality_weight * object_file.quality_score
            + self.config.novelty_weight * object_file.novelty_score
            + self.config.surprise_weight * object_file.score
            + self.config.prediction_error_weight * object_file.prediction_error
            + self.config.motion_weight * motion_salience
            + self.config.low_familiarity_weight * (1.0 - object_file.familiarity_score)
            + self.config.task_salience_weight * task_salience
            + self.config.proposal_source_score_weight * float(np.clip(object_file.proposal_source_score, 0.0, 1.0))
            + _source_bonus(self.config, object_file.proposal_source)
        )
        return float(np.clip(score, 0.0, 1.0))

    def select(self, object_files: list[ObjectFile], task_salience: dict[str, float] | None = None) -> list[ObjectFile]:
        if not object_files:
            return []
        scored: list[tuple[float, ObjectFile]] = []
        for object_file in object_files:
            if object_file.quality_score < self.config.min_quality:
                continue
            task_score = 0.0 if task_salience is None else float(task_salience.get(object_file.object_file_id, 0.0))
            scored.append((self.score(object_file, task_score), object_file))
        if self.config.diversity_iou_penalty > 0.0 or self.config.diversity_center_penalty > 0.0:
            return _select_diverse(scored, self.config)
        scored.sort(key=lambda row: row[0], reverse=True)
        return [object_file for _, object_file in scored[: self.config.max_attended_objects]]


def _motion_salience(signature: np.ndarray) -> float:
    if signature.size == 0:
        return 0.0
    return float(np.clip(np.linalg.norm(signature), 0.0, 1.0))


def _source_bonus(config: AttentionGateConfig, proposal_source: str) -> float:
    if proposal_source == "component":
        return float(config.component_source_bonus)
    if proposal_source == "saliency_window":
        return float(config.saliency_window_source_bonus)
    if proposal_source == "memory_guided_window":
        return float(config.memory_guided_window_source_bonus)
    if proposal_source == "memory_template_window":
        return float(config.memory_template_window_source_bonus)
    return 0.0


def _select_diverse(scored: list[tuple[float, ObjectFile]], config: AttentionGateConfig) -> list[ObjectFile]:
    remaining = list(scored)
    selected: list[tuple[float, ObjectFile]] = []
    while remaining and len(selected) < config.max_attended_objects:
        best_index = 0
        best_adjusted = -1e9
        for index, (base_score, object_file) in enumerate(remaining):
            penalty = 0.0
            if selected:
                penalty += float(config.diversity_iou_penalty) * max(
                    _box_iou(object_file.box, chosen.box) for _, chosen in selected
                )
                penalty += float(config.diversity_center_penalty) * max(
                    _center_affinity(object_file, chosen, scale=config.diversity_center_scale) for _, chosen in selected
                )
            adjusted = float(base_score) - penalty
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_index = index
        selected.append(remaining.pop(best_index))
    return [object_file for _, object_file in selected]


def _box_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx1, ly1, lx2, ly2 = [float(v) for v in left]
    rx1, ry1, rx2, ry2 = [float(v) for v in right]
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def _center_affinity(left: ObjectFile, right: ObjectFile, *, scale: float) -> float:
    scale = max(float(scale), 1.0)
    dx = float(left.centroid[0]) - float(right.centroid[0])
    dy = float(left.centroid[1]) - float(right.centroid[1])
    distance = float(np.sqrt(dx * dx + dy * dy))
    return float(np.exp(-distance / scale))
