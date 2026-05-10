"""Episodic memory module for object-centric visual experience.

This module extracts the episodic-bundle idea from experiment scripts into a
small reusable runtime component. Retrieval is cue-based and multi-dimensional;
appearance/content similarity alone is intentionally insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nops_owr.cognition.object_file import ObjectFile


@dataclass(slots=True)
class EpisodicBundle:
    episode_id: int
    object_file_id: str
    track_id: int | None
    prototype_id: int | None
    concept_id: int | None
    frame_start: int
    frame_end: int
    created_frame: int
    last_reactivated_frame: int | None
    content_signature: np.ndarray
    support_signature: np.ndarray
    motion_signature: np.ndarray
    context_signature: np.ndarray
    temporal_signature: np.ndarray
    disappearance_signature: np.ndarray
    accessibility_score: float = 0.5
    stability_level: str = "candidate"
    replay_priority: float = 0.0
    reactivation_count: int = 0
    source_state: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedEpisode:
    bundle: EpisodicBundle
    score: float
    evidence_breakdown: dict[str, float]


class EpisodicMemory:
    """Bounded episodic bundle store with cue-triggered retrieval."""

    def __init__(self, memory_budget: int = 256) -> None:
        self.memory_budget = int(memory_budget)
        self._bundles: dict[int, EpisodicBundle] = {}
        self._next_episode_id = 1

    @property
    def bundles(self) -> list[EpisodicBundle]:
        return list(self._bundles.values())

    def __len__(self) -> int:
        return len(self._bundles)

    def write_episode(
        self,
        object_file: ObjectFile,
        frame_index: int | None = None,
        track_id: int | None = None,
        prototype_id: int | None = None,
        concept_id: int | None = None,
        source_state: str = "active",
        frame_start: int | None = None,
        frame_end: int | None = None,
        disappearance_signature: np.ndarray | None = None,
    ) -> int:
        created_frame = int(object_file.frame_index if frame_index is None else frame_index)
        episode_id = self._next_episode_id
        self._next_episode_id += 1
        bundle = EpisodicBundle(
            episode_id=episode_id,
            object_file_id=object_file.object_file_id,
            track_id=object_file.linked_track_id if track_id is None else track_id,
            prototype_id=object_file.linked_prototype_id if prototype_id is None else prototype_id,
            concept_id=object_file.linked_concept_id if concept_id is None else concept_id,
            frame_start=created_frame if frame_start is None else int(frame_start),
            frame_end=created_frame if frame_end is None else int(frame_end),
            created_frame=created_frame,
            last_reactivated_frame=None,
            content_signature=object_file.appearance_signature.astype(np.float32, copy=True),
            support_signature=object_file.shape_signature.astype(np.float32, copy=True),
            motion_signature=object_file.motion_signature.astype(np.float32, copy=True),
            context_signature=object_file.context_signature.astype(np.float32, copy=True),
            temporal_signature=np.asarray([float(created_frame), 0.0, 1.0], dtype=np.float32),
            disappearance_signature=(
                np.zeros(0, dtype=np.float32)
                if disappearance_signature is None
                else disappearance_signature.astype(np.float32, copy=True)
            ),
            accessibility_score=float(np.clip(0.45 + 0.35 * object_file.confidence, 0.0, 1.0)),
            stability_level="candidate",
            replay_priority=float(np.clip(object_file.prediction_error + object_file.novelty_score, 0.0, 1.0)),
            reactivation_count=0,
            source_state=source_state,
        )
        self._bundles[episode_id] = bundle
        object_file.linked_episode_ids.append(episode_id)
        self._enforce_budget()
        return episode_id

    def retrieve(self, cue: ObjectFile | dict[str, np.ndarray], top_k: int = 5) -> list[RetrievedEpisode]:
        if not self._bundles:
            return []
        cue_signatures = _cue_signatures(cue)
        results: list[RetrievedEpisode] = []
        for bundle in self._bundles.values():
            breakdown = {
                "content": _cosine(cue_signatures["content"], bundle.content_signature),
                "support": _cosine(cue_signatures["support"], bundle.support_signature),
                "context": _cosine(cue_signatures["context"], bundle.context_signature),
                "motion": _cosine(cue_signatures["motion"], bundle.motion_signature),
                "temporal": _temporal_similarity(cue_signatures["temporal"], bundle.temporal_signature),
                "disappearance": _cosine(cue_signatures["disappearance"], bundle.disappearance_signature),
                "accessibility": float(bundle.accessibility_score),
            }
            score = (
                0.28 * breakdown["content"]
                + 0.22 * breakdown["support"]
                + 0.18 * breakdown["context"]
                + 0.10 * breakdown["motion"]
                + 0.10 * breakdown["temporal"]
                + 0.07 * breakdown["disappearance"]
                + 0.05 * breakdown["accessibility"]
            )
            results.append(RetrievedEpisode(bundle=bundle, score=float(score), evidence_breakdown=breakdown))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[: int(top_k)]

    def update_reactivation(self, episode_id: int, frame_index: int, score_delta: float = 0.03) -> None:
        bundle = self._bundles.get(int(episode_id))
        if bundle is None:
            return
        bundle.reactivation_count += 1
        bundle.last_reactivated_frame = int(frame_index)
        bundle.accessibility_score = float(np.clip(bundle.accessibility_score + score_delta, 0.0, 1.0))
        if bundle.reactivation_count >= 3:
            bundle.stability_level = "stable"
        elif bundle.reactivation_count >= 1:
            bundle.stability_level = "stabilizing"

    def consolidate_candidates(self, frame_index: int | None = None) -> None:
        del frame_index
        for bundle in self._bundles.values():
            age_penalty = 0.02 if bundle.stability_level == "candidate" else 0.005
            bundle.replay_priority = float(np.clip(bundle.replay_priority - age_penalty, 0.0, 1.0))
            if bundle.reactivation_count == 0 and bundle.accessibility_score < 0.20:
                bundle.stability_level = "latent"

    def _enforce_budget(self) -> None:
        while len(self._bundles) > self.memory_budget:
            victim = min(
                self._bundles.values(),
                key=lambda b: (b.accessibility_score + 0.05 * b.reactivation_count, b.created_frame),
            )
            del self._bundles[victim.episode_id]


def _cue_signatures(cue: ObjectFile | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if isinstance(cue, ObjectFile):
        return {
            "content": cue.appearance_signature,
            "support": cue.shape_signature,
            "context": cue.context_signature,
            "motion": cue.motion_signature,
            "temporal": np.asarray([float(cue.frame_index), 0.0, 1.0], dtype=np.float32),
            "disappearance": np.zeros(0, dtype=np.float32),
        }
    return {
        "content": np.asarray(cue.get("content_signature", cue.get("content", [])), dtype=np.float32),
        "support": np.asarray(cue.get("support_signature", cue.get("support", [])), dtype=np.float32),
        "context": np.asarray(cue.get("context_signature", cue.get("context", [])), dtype=np.float32),
        "motion": np.asarray(cue.get("motion_signature", cue.get("motion", [])), dtype=np.float32),
        "temporal": np.asarray(cue.get("temporal_signature", cue.get("temporal", [])), dtype=np.float32),
        "disappearance": np.asarray(
            cue.get("disappearance_signature", cue.get("disappearance", [])), dtype=np.float32
        ),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    dim = min(left.size, right.size)
    a = left.reshape(-1)[:dim].astype(np.float32, copy=False)
    b = right.reshape(-1)[:dim].astype(np.float32, copy=False)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip((float(np.dot(a, b)) / denom + 1.0) * 0.5, 0.0, 1.0))


def _temporal_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.5
    gap = abs(float(left.reshape(-1)[0]) - float(right.reshape(-1)[0]))
    return float(np.exp(-gap / 48.0))
