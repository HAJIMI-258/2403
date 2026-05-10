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
from nops_owr.memory.retrieval_context import RetrievalContext


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
    last_observed_frame: int | None = None
    observation_count: int = 1
    closed: bool = False
    close_reason: str = ""
    evidence_quality: float = 0.0
    prediction_error_mean: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedEpisode:
    bundle: EpisodicBundle
    score: float
    evidence_breakdown: dict[str, float]
    rank: int = 0
    margin_to_next: float = 0.0
    retrieval_mode: str = "general"
    active_conflict: bool = False
    closed_bonus: float = 0.0
    reentry_gap: int = 0
    status_penalty: float = 0.0


class EpisodicMemory:
    """Bounded episodic bundle store with cue-triggered retrieval."""

    def __init__(self, memory_budget: int = 256) -> None:
        self.memory_budget = int(memory_budget)
        self._bundles: dict[int, EpisodicBundle] = {}
        self._active_episode_by_track: dict[int, int] = {}
        self._next_episode_id = 1

    @property
    def bundles(self) -> list[EpisodicBundle]:
        return list(self._bundles.values())

    def __len__(self) -> int:
        return len(self._bundles)

    @property
    def active_episode_by_track(self) -> dict[int, int]:
        return dict(self._active_episode_by_track)

    def get_episode(self, episode_id: int | None) -> EpisodicBundle | None:
        if episode_id is None:
            return None
        return self._bundles.get(int(episode_id))

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
        metadata: dict[str, Any] | None = None,
    ) -> int:
        return self.begin_episode(
            object_file=object_file,
            frame_index=frame_index,
            track_id=track_id,
            prototype_id=prototype_id,
            concept_id=concept_id,
            source_state=source_state,
            frame_start=frame_start,
            frame_end=frame_end,
            disappearance_signature=disappearance_signature,
            metadata=metadata,
        )

    def begin_episode(
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
        metadata: dict[str, Any] | None = None,
    ) -> int:
        created_frame = int(object_file.frame_index if frame_index is None else frame_index)
        episode_id = self._next_episode_id
        self._next_episode_id += 1
        resolved_track_id = object_file.linked_track_id if track_id is None else track_id
        resolved_prototype_id = object_file.linked_prototype_id if prototype_id is None else prototype_id
        resolved_concept_id = object_file.linked_concept_id if concept_id is None else concept_id
        merged_metadata = dict(object_file.metadata)
        if metadata:
            merged_metadata.update(metadata)
        bundle = EpisodicBundle(
            episode_id=episode_id,
            object_file_id=object_file.object_file_id,
            track_id=resolved_track_id,
            prototype_id=resolved_prototype_id,
            concept_id=resolved_concept_id,
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
            last_observed_frame=created_frame,
            observation_count=1,
            closed=False,
            close_reason="",
            evidence_quality=float(object_file.quality_score),
            prediction_error_mean=float(object_file.prediction_error),
            metadata=merged_metadata,
        )
        self._bundles[episode_id] = bundle
        if resolved_track_id is not None:
            self._active_episode_by_track[int(resolved_track_id)] = episode_id
        object_file.linked_episode_ids.append(episode_id)
        self._enforce_budget()
        return episode_id

    def extend_episode(
        self,
        episode_id: int,
        object_file: ObjectFile,
        frame_index: int,
        update_signatures: bool = True,
        alpha: float = 0.85,
    ) -> None:
        bundle = self._bundles.get(int(episode_id))
        if bundle is None:
            return
        bundle.object_file_id = object_file.object_file_id
        bundle.frame_end = int(frame_index)
        bundle.last_observed_frame = int(frame_index)
        bundle.observation_count += 1
        bundle.closed = False
        bundle.close_reason = ""
        bundle.source_state = object_file.state
        bundle.evidence_quality = _running_mean(
            bundle.evidence_quality,
            float(object_file.quality_score),
            bundle.observation_count,
        )
        bundle.prediction_error_mean = _running_mean(
            bundle.prediction_error_mean,
            float(object_file.prediction_error),
            bundle.observation_count,
        )
        bundle.metadata.update({key: value for key, value in object_file.metadata.items() if value is not None})
        if update_signatures:
            bundle.content_signature = _ema_signature(bundle.content_signature, object_file.appearance_signature, alpha)
            bundle.support_signature = _ema_signature(bundle.support_signature, object_file.shape_signature, alpha)
            bundle.context_signature = _ema_signature(bundle.context_signature, object_file.context_signature, alpha)
            bundle.motion_signature = _ema_signature(bundle.motion_signature, object_file.motion_signature, alpha)
        bundle.temporal_signature = np.asarray(
            [
                float(bundle.created_frame),
                float(bundle.frame_end - bundle.frame_start),
                float(bundle.observation_count),
            ],
            dtype=np.float32,
        )
        bundle.accessibility_score = float(
            np.clip(bundle.accessibility_score + 0.005 * object_file.confidence, 0.0, 1.0)
        )
        if bundle.observation_count >= 6 and bundle.stability_level == "candidate":
            bundle.stability_level = "stabilizing"
        if bundle.track_id is not None:
            self._active_episode_by_track[int(bundle.track_id)] = int(bundle.episode_id)

    def close_episode(
        self,
        episode_id: int,
        frame_index: int,
        close_reason: str = "unknown",
        disappearance_signature: np.ndarray | None = None,
    ) -> None:
        bundle = self._bundles.get(int(episode_id))
        if bundle is None or bundle.closed:
            return
        bundle.closed = True
        bundle.close_reason = str(close_reason)
        bundle.frame_end = max(bundle.frame_end, int(frame_index))
        bundle.last_observed_frame = bundle.last_observed_frame or int(frame_index)
        if disappearance_signature is not None:
            bundle.disappearance_signature = disappearance_signature.astype(np.float32, copy=True)
        elif bundle.motion_signature.size:
            bundle.disappearance_signature = bundle.motion_signature.astype(np.float32, copy=True)
        if bundle.track_id is not None and self._active_episode_by_track.get(int(bundle.track_id)) == episode_id:
            del self._active_episode_by_track[int(bundle.track_id)]

    def write_or_extend_episode(
        self,
        object_file: ObjectFile,
        frame_index: int,
        track_id: int | None = None,
        prototype_id: int | None = None,
        concept_id: int | None = None,
        source_state: str = "active",
        active_episode_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        resolved_track_id = object_file.linked_track_id if track_id is None else track_id
        candidate_episode_id = active_episode_id
        if candidate_episode_id is None and resolved_track_id is not None:
            candidate_episode_id = self._active_episode_by_track.get(int(resolved_track_id))
        if candidate_episode_id is not None:
            bundle = self._bundles.get(int(candidate_episode_id))
            if bundle is not None and not bundle.closed:
                self.extend_episode(int(candidate_episode_id), object_file, frame_index)
                object_file.linked_episode_ids.append(int(candidate_episode_id))
                return int(candidate_episode_id)
        return self.begin_episode(
            object_file=object_file,
            frame_index=frame_index,
            track_id=resolved_track_id,
            prototype_id=prototype_id,
            concept_id=concept_id,
            source_state=source_state,
            metadata=metadata,
        )

    def retrieve(
        self,
        cue: ObjectFile | dict[str, np.ndarray],
        top_k: int = 5,
        context: RetrievalContext | None = None,
    ) -> list[RetrievedEpisode]:
        if not self._bundles:
            return []
        cue_signatures = _cue_signatures(cue)
        results: list[RetrievedEpisode] = []
        retrieval_mode = "general" if context is None else context.mode
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
            base_score = (
                0.28 * breakdown["content"]
                + 0.22 * breakdown["support"]
                + 0.18 * breakdown["context"]
                + 0.10 * breakdown["motion"]
                + 0.10 * breakdown["temporal"]
                + 0.07 * breakdown["disappearance"]
                + 0.05 * breakdown["accessibility"]
            )
            adjusted_score, context_breakdown = _context_adjustment(base_score, bundle, context)
            breakdown.update(context_breakdown)
            results.append(
                RetrievedEpisode(
                    bundle=bundle,
                    score=float(adjusted_score),
                    evidence_breakdown=breakdown,
                    retrieval_mode=retrieval_mode,
                    active_conflict=bool(context_breakdown["active_conflict"]),
                    closed_bonus=float(context_breakdown["closed_bonus"]),
                    reentry_gap=int(context_breakdown["reentry_gap"]),
                    status_penalty=float(context_breakdown["status_penalty"]),
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        for index, item in enumerate(results):
            item.rank = index + 1
            item.margin_to_next = float(item.score - results[index + 1].score) if index + 1 < len(results) else float(item.score)
            item.evidence_breakdown["rank"] = float(item.rank)
            item.evidence_breakdown["margin_to_next"] = float(item.margin_to_next)
        return results[: int(top_k)]

    def update_reactivation(self, episode_id: int, frame_index: int, score_delta: float = 0.03) -> None:
        bundle = self._bundles.get(int(episode_id))
        if bundle is None:
            return
        bundle.reactivation_count += 1
        bundle.last_reactivated_frame = int(frame_index)
        bundle.accessibility_score = float(np.clip(bundle.accessibility_score + score_delta, 0.0, 1.0))
        bundle.replay_priority = float(np.clip(bundle.replay_priority + 0.01, 0.0, 1.0))
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
                key=_budget_score,
            )
            del self._bundles[victim.episode_id]
            if victim.track_id is not None and self._active_episode_by_track.get(int(victim.track_id)) == victim.episode_id:
                del self._active_episode_by_track[int(victim.track_id)]


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


def _context_adjustment(
    base_score: float,
    bundle: EpisodicBundle,
    context: RetrievalContext | None,
) -> tuple[float, dict[str, float | bool]]:
    if context is None:
        return float(base_score), {
            "base_score": float(base_score),
            "adjusted_score": float(base_score),
            "closed_bonus": 0.0,
            "reentry_gap_bonus": 0.0,
            "active_conflict_penalty": 0.0,
            "status_penalty": 0.0,
            "active_conflict": False,
            "bundle_closed": bool(bundle.closed),
            "reentry_gap": 0,
        }

    last_observed = bundle.last_observed_frame if bundle.last_observed_frame is not None else bundle.frame_end
    reentry_gap = max(0, int(context.frame_index) - int(last_observed))
    active_conflict = (
        context.suppress_active_conflicts
        and bundle.track_id is not None
        and int(bundle.track_id) in context.active_track_ids
        and (context.query_track_id is None or int(bundle.track_id) != int(context.query_track_id))
    )
    closed_bonus = 0.0
    reentry_gap_bonus = 0.0
    active_conflict_penalty = 0.0
    status_penalty = 0.0

    if context.mode == "reentry":
        if context.prefer_closed_episodes and bundle.closed:
            closed_bonus = 0.06
        if reentry_gap >= int(context.min_reentry_gap):
            reentry_gap_bonus = 0.04
        if active_conflict:
            active_conflict_penalty = -0.18
        if context.query_track_id is not None and bundle.track_id == context.query_track_id:
            status_penalty = -0.05
        if not bundle.closed and not active_conflict:
            status_penalty += -0.02
    elif context.mode == "continuous":
        if context.query_track_id is not None and bundle.track_id == context.query_track_id and not bundle.closed:
            closed_bonus = 0.04
        if bundle.closed:
            status_penalty = -0.04
        if active_conflict:
            active_conflict_penalty = -0.08
    elif active_conflict:
        active_conflict_penalty = -0.10

    adjusted = float(base_score + closed_bonus + reentry_gap_bonus + active_conflict_penalty + status_penalty)
    adjusted = float(np.clip(adjusted, 0.0, 1.0))
    return adjusted, {
        "base_score": float(base_score),
        "adjusted_score": adjusted,
        "closed_bonus": float(closed_bonus),
        "reentry_gap_bonus": float(reentry_gap_bonus),
        "active_conflict_penalty": float(active_conflict_penalty),
        "status_penalty": float(status_penalty),
        "active_conflict": bool(active_conflict),
        "bundle_closed": bool(bundle.closed),
        "reentry_gap": int(reentry_gap),
    }


def _ema_signature(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    if old.size == 0:
        return new.astype(np.float32, copy=True)
    if new.size == 0:
        return old.astype(np.float32, copy=True)
    dim = min(old.size, new.size)
    out = old.astype(np.float32, copy=True)
    out.reshape(-1)[:dim] = alpha * out.reshape(-1)[:dim] + (1.0 - alpha) * new.reshape(-1)[:dim]
    return out


def _running_mean(previous_mean: float, new_value: float, count: int) -> float:
    if count <= 1:
        return float(new_value)
    return float(previous_mean + (new_value - previous_mean) / float(count))


def _budget_score(bundle: EpisodicBundle) -> tuple[float, int]:
    stability_bonus = {"stable": 0.35, "stabilizing": 0.20, "candidate": 0.05, "latent": 0.0}.get(
        bundle.stability_level,
        0.0,
    )
    closed_penalty = -0.10 if bundle.closed else 0.05
    score = (
        bundle.accessibility_score
        + 0.07 * bundle.reactivation_count
        + stability_bonus
        + 0.02 * min(bundle.observation_count, 10)
        + 0.05 * bundle.replay_priority
        + closed_penalty
    )
    return (float(score), int(bundle.created_frame))
