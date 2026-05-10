"""Predictive recognition as memory retrieval plus evidence checking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nops_owr.cognition.object_file import ObjectFile
from nops_owr.memory.episodic_memory import RetrievedEpisode


@dataclass(slots=True)
class RecognitionDecision:
    object_file_id: str
    decision_type: str
    linked_episode_id: int | None
    linked_track_id: int | None
    linked_prototype_id: int | None
    linked_concept_id: int | None
    familiarity_score: float
    novelty_score: float
    prediction_error: float
    confidence: float
    evidence_breakdown: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class PredictiveRecognizer:
    """Classify recognition state without treating recognition as class labeling."""

    def __init__(
        self,
        same_instance_threshold: float = 0.72,
        same_concept_threshold: float = 0.50,
        conflict_threshold: float = 0.28,
        novelty_threshold: float = 0.55,
    ) -> None:
        self.same_instance_threshold = float(same_instance_threshold)
        self.same_concept_threshold = float(same_concept_threshold)
        self.conflict_threshold = float(conflict_threshold)
        self.novelty_threshold = float(novelty_threshold)

    def recognize(
        self,
        object_file: ObjectFile,
        tracking_assignment: Any | None = None,
        episodic_candidates: list[RetrievedEpisode] | None = None,
        prototype_result: Any | None = None,
    ) -> RecognitionDecision:
        del prototype_result
        candidates = episodic_candidates or []
        if not candidates:
            return self._new_or_uncertain(object_file, tracking_assignment, "no_episodic_candidates")

        top = candidates[0]
        evidence = dict(top.evidence_breakdown)
        content = evidence.get("content", 0.0)
        support = evidence.get("support", 0.0)
        context = evidence.get("context", 0.0)
        temporal = evidence.get("temporal", 0.0)
        min_core = min(content, support, context)
        conflict = min_core < self.conflict_threshold

        object_file.familiarity_score = float(np.clip(top.score, 0.0, 1.0))
        object_file.novelty_score = float(np.clip(1.0 - object_file.familiarity_score, 0.0, 1.0))
        object_file.prediction_error = float(np.clip(1.0 - min_core, 0.0, 1.0))

        linked_track_id = getattr(tracking_assignment, "track_id", object_file.linked_track_id)
        linked_prototype_id = getattr(tracking_assignment, "linked_prototype_id", object_file.linked_prototype_id)

        if top.score >= self.same_instance_threshold and not conflict and temporal >= 0.25:
            decision_type = "same_instance"
            confidence = top.score
            object_file.state = "confirmed_same_instance"
        elif top.score >= self.same_concept_threshold and content >= 0.45 and support >= 0.40:
            decision_type = "same_concept"
            confidence = 0.5 * top.score + 0.25 * content + 0.25 * support
            object_file.state = "confirmed_concept"
        elif conflict and top.score >= self.same_concept_threshold:
            decision_type = "familiar_but_unresolved"
            confidence = top.score * 0.65
            object_file.state = "familiar_unknown"
        elif object_file.novelty_score >= self.novelty_threshold and top.score < self.same_concept_threshold:
            decision_type = "new_concept"
            confidence = object_file.novelty_score
            object_file.state = "new_concept"
        else:
            decision_type = "uncertain_hold"
            confidence = max(0.1, top.score * 0.5)
            object_file.state = "uncertain_hold"

        return RecognitionDecision(
            object_file_id=object_file.object_file_id,
            decision_type=decision_type,
            linked_episode_id=top.bundle.episode_id,
            linked_track_id=linked_track_id,
            linked_prototype_id=linked_prototype_id or top.bundle.prototype_id,
            linked_concept_id=top.bundle.concept_id,
            familiarity_score=object_file.familiarity_score,
            novelty_score=object_file.novelty_score,
            prediction_error=object_file.prediction_error,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            evidence_breakdown=evidence,
            metadata={"candidate_score": top.score, "conflict": conflict},
        )

    def _new_or_uncertain(
        self,
        object_file: ObjectFile,
        tracking_assignment: Any | None,
        reason: str,
    ) -> RecognitionDecision:
        decision_type = "new_concept" if object_file.novelty_score >= self.novelty_threshold else "uncertain_hold"
        object_file.state = decision_type
        return RecognitionDecision(
            object_file_id=object_file.object_file_id,
            decision_type=decision_type,
            linked_episode_id=None,
            linked_track_id=getattr(tracking_assignment, "track_id", object_file.linked_track_id),
            linked_prototype_id=getattr(tracking_assignment, "linked_prototype_id", object_file.linked_prototype_id),
            linked_concept_id=object_file.linked_concept_id,
            familiarity_score=object_file.familiarity_score,
            novelty_score=object_file.novelty_score,
            prediction_error=object_file.prediction_error,
            confidence=float(np.clip(max(object_file.confidence, object_file.novelty_score), 0.0, 1.0)),
            evidence_breakdown={},
            metadata={"reason": reason},
        )
