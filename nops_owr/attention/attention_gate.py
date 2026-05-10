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
        scored.sort(key=lambda row: row[0], reverse=True)
        return [object_file for _, object_file in scored[: self.config.max_attended_objects]]


def _motion_salience(signature: np.ndarray) -> float:
    if signature.size == 0:
        return 0.0
    return float(np.clip(np.linalg.norm(signature), 0.0, 1.0))
