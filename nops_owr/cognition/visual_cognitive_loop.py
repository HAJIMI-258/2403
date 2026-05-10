"""End-to-end visual cognitive loop skeleton.

The loop composes existing minimal modules with explicit object-file,
attention, episodic retrieval, and predictive recognition layers. It is a
runtime skeleton, not a replacement for the existing experiment runners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nops_owr.attention.attention_gate import AttentionGate
from nops_owr.cognition.cognitive_event import CognitiveEvent
from nops_owr.cognition.object_file import ObjectFile, ObjectFileBuilder
from nops_owr.cognition.predictive_recognition import PredictiveRecognizer, RecognitionDecision
from nops_owr.evaluation.cognitive_metrics import (
    attended_object_ratio,
    memory_write_rate,
    prediction_error_mean,
)
from nops_owr.memory.episodic_memory import EpisodicMemory, RetrievedEpisode
from nops_owr.memory.retrieval_context import RetrievalContext


@dataclass(slots=True)
class CognitiveFrameResult:
    frame_index: int
    encoding: Any
    objectness_output: Any
    object_files: list[ObjectFile]
    attended_object_files: list[ObjectFile]
    tracking_output: Any
    memory_output: Any
    recognition_decisions: list[RecognitionDecision]
    written_episode_ids: list[int]
    metrics_snapshot: dict[str, float] = field(default_factory=dict)
    episodic_retrievals: dict[str, list[RetrievedEpisode]] = field(default_factory=dict)
    cognitive_events: list[CognitiveEvent] = field(default_factory=list)
    reactivated_episode_ids: list[int] = field(default_factory=list)
    closed_episode_ids: list[int] = field(default_factory=list)
    active_episode_count: int = 0
    memory_context_used: bool = False


class VisualCognitiveLoop:
    def __init__(
        self,
        encoder: Any,
        objectness_field: Any,
        tracker: Any,
        prototype_memory: Any,
        attention_gate: AttentionGate,
        episodic_memory: EpisodicMemory,
        recognizer: PredictiveRecognizer,
        object_file_builder: ObjectFileBuilder | None = None,
    ) -> None:
        self.encoder = encoder
        self.objectness_field = objectness_field
        self.tracker = tracker
        self.prototype_memory = prototype_memory
        self.attention_gate = attention_gate
        self.episodic_memory = episodic_memory
        self.recognizer = recognizer
        self.object_file_builder = object_file_builder or ObjectFileBuilder()
        self._prev_memory_output: Any | None = None
        self._active_episode_by_track: dict[int, int] = {}
        self._last_decision_by_track: dict[int, RecognitionDecision] = {}
        self._last_seen_track_frame: dict[int, int] = {}
        self._event_counter = 0

    def step(
        self,
        prev_frame: np.ndarray,
        current_frame: np.ndarray,
        frame_index: int,
        frame_metadata: dict[str, Any] | None = None,
        ground_truth: dict[str, Any] | None = None,
    ) -> CognitiveFrameResult:
        del frame_metadata

        encoding = self.encoder.encode(prev_frame, current_frame)
        objectness_output = self.objectness_field.compute(encoding)
        object_files = self.object_file_builder.build(
            objectness_output=objectness_output,
            encoding=encoding,
            frame_index=frame_index,
            current_frame=current_frame,
        )
        if ground_truth:
            self._attach_ground_truth_metadata(object_files, ground_truth)
        attended_object_files = self.attention_gate.select(object_files)

        memory_context_used = self._prev_memory_output is not None
        tracking_output = self.tracker.update(
            objectness_output.proposals,
            encoding,
            objectness_output.heatmap,
            current_frame,
            frame_index,
            memory_context=self._prev_memory_output,
        )
        all_track_states = self._collect_track_states(tracking_output)
        memory_output = self.prototype_memory.update(
            tracking_output.assignments,
            frame_index,
            track_states=all_track_states,
        )
        if hasattr(self.tracker, "apply_concept_gated_resurrection"):
            self.tracker.apply_concept_gated_resurrection(
                tracking_output,
                memory_output,
                frame_index=frame_index,
                frame_shape=objectness_output.heatmap.shape,
            )
        if hasattr(self.tracker, "bind_prototypes"):
            self.tracker.bind_prototypes(memory_output.assignments)

        self._link_runtime_ids(object_files, tracking_output, memory_output)

        recognition_decisions: list[RecognitionDecision] = []
        written_episode_ids: list[int] = []
        reactivated_episode_ids: list[int] = []
        cognitive_events: list[CognitiveEvent] = []
        episodic_retrievals: dict[str, list[RetrievedEpisode]] = {}
        retrieval_contexts: dict[str, RetrievalContext] = {}
        assignment_by_proposal = {
            int(assignment.proposal_index): assignment for assignment in getattr(tracking_output, "assignments", [])
        }
        if memory_context_used:
            cognitive_events.append(self._event(frame_index, "memory_context_used"))
        for object_file in attended_object_files:
            cognitive_events.append(
                self._event(
                    frame_index,
                    "object_attended",
                    object_file=object_file,
                    metadata={"quality_score": object_file.quality_score},
                )
            )

        for object_file in attended_object_files:
            retrieval_context = self._determine_retrieval_context(object_file, tracking_output, frame_index)
            retrieval_contexts[object_file.object_file_id] = retrieval_context
            candidates = self.episodic_memory.retrieve(object_file, top_k=5, context=retrieval_context)
            episodic_retrievals[object_file.object_file_id] = candidates
            decision = self.recognizer.recognize(
                object_file,
                tracking_assignment=assignment_by_proposal.get(object_file.proposal_index),
                episodic_candidates=candidates,
                prototype_result=memory_output,
                retrieval_mode=retrieval_context.mode,
            )
            recognition_decisions.append(decision)
            if decision.linked_track_id is not None:
                self._last_decision_by_track[int(decision.linked_track_id)] = decision
                self._last_seen_track_frame[int(decision.linked_track_id)] = int(frame_index)
            cognitive_events.append(self._decision_event(frame_index, object_file, decision, candidates))
            if self._should_write_episode(object_file, decision):
                active_episode_id = (
                    self._active_episode_by_track.get(int(decision.linked_track_id))
                    if decision.linked_track_id is not None
                    else None
                )
                already_known = active_episode_id is not None
                episode_id = self.episodic_memory.write_or_extend_episode(
                    object_file=object_file,
                    frame_index=frame_index,
                    track_id=decision.linked_track_id,
                    prototype_id=decision.linked_prototype_id,
                    concept_id=decision.linked_concept_id,
                    source_state=object_file.state,
                    active_episode_id=active_episode_id,
                    metadata=object_file.metadata,
                )
                if decision.linked_track_id is not None:
                    self._active_episode_by_track[int(decision.linked_track_id)] = int(episode_id)
                written_episode_ids.append(episode_id)
                cognitive_events.append(
                    self._event(
                        frame_index,
                        "episode_extended" if already_known else "episode_started",
                        object_file=object_file,
                        decision=decision,
                        episode_id=episode_id,
                    )
                )
            elif decision.linked_episode_id is not None and decision.decision_type == "same_instance":
                self.episodic_memory.update_reactivation(decision.linked_episode_id, frame_index)
                reactivated_episode_ids.append(int(decision.linked_episode_id))
                cognitive_events.append(
                    self._event(
                        frame_index,
                        "episode_reactivated",
                        object_file=object_file,
                        decision=decision,
                        episode_id=decision.linked_episode_id,
                    )
                )

        closed_episode_ids = self._close_stale_episodes(frame_index)
        for episode_id in closed_episode_ids:
            cognitive_events.append(self._event(frame_index, "episode_closed", episode_id=episode_id))
        self.episodic_memory.consolidate_candidates(frame_index)
        metrics_snapshot = {
            "attended_object_ratio": attended_object_ratio(len(attended_object_files), len(object_files)),
            "memory_write_rate": memory_write_rate(len(written_episode_ids), len(object_files)),
            "prediction_error_mean": prediction_error_mean(recognition_decisions),
            "episodic_memory_size": float(len(self.episodic_memory)),
            "active_episode_count": float(len(self._active_episode_by_track)),
            "memory_context_available": float(memory_context_used),
            "episode_reactivation_count": float(len(reactivated_episode_ids)),
            "continuous_decision_count": float(
                sum(1 for ctx in retrieval_contexts.values() if ctx.mode == "continuous")
            ),
            "reentry_decision_count": float(sum(1 for ctx in retrieval_contexts.values() if ctx.mode == "reentry")),
            "general_decision_count": float(sum(1 for ctx in retrieval_contexts.values() if ctx.mode == "general")),
            "active_conflict_count": float(
                sum(1 for rows in episodic_retrievals.values() if rows and rows[0].active_conflict)
            ),
            "low_margin_uncertain_count": float(
                sum(
                    1
                    for decision in recognition_decisions
                    if decision.decision_type == "uncertain_hold"
                    and decision.metadata.get("rejection_reason") == "low_retrieval_margin"
                )
            ),
        }
        self._prev_memory_output = memory_output
        return CognitiveFrameResult(
            frame_index=int(frame_index),
            encoding=encoding,
            objectness_output=objectness_output,
            object_files=object_files,
            attended_object_files=attended_object_files,
            tracking_output=tracking_output,
            memory_output=memory_output,
            recognition_decisions=recognition_decisions,
            written_episode_ids=written_episode_ids,
            metrics_snapshot=metrics_snapshot,
            episodic_retrievals=episodic_retrievals,
            cognitive_events=cognitive_events,
            reactivated_episode_ids=reactivated_episode_ids,
            closed_episode_ids=closed_episode_ids,
            active_episode_count=len(self._active_episode_by_track),
            memory_context_used=memory_context_used,
        )

    @property
    def active_episode_by_track(self) -> dict[int, int]:
        return dict(self._active_episode_by_track)

    def _link_runtime_ids(self, object_files: list[ObjectFile], tracking_output: Any, memory_output: Any) -> None:
        assignment_by_proposal = {
            int(assignment.proposal_index): assignment for assignment in getattr(tracking_output, "assignments", [])
        }
        prototype_by_track = {
            int(assignment.track_id): assignment for assignment in getattr(memory_output, "assignments", [])
        }
        for object_file in object_files:
            track_assignment = assignment_by_proposal.get(object_file.proposal_index)
            if track_assignment is None:
                continue
            object_file.linked_track_id = int(track_assignment.track_id)
            object_file.motion_signature = np.asarray(getattr(track_assignment, "signature", []), dtype=np.float32)
            prototype_assignment = prototype_by_track.get(int(track_assignment.track_id))
            if prototype_assignment is not None:
                object_file.linked_prototype_id = int(prototype_assignment.prototype_id)
                object_file.linked_concept_id = int(prototype_assignment.lineage_id)
            else:
                object_file.linked_prototype_id = getattr(track_assignment, "linked_prototype_id", None)
                object_file.linked_concept_id = getattr(track_assignment, "linked_lineage_id", None)

    def _should_write_episode(self, object_file: ObjectFile, decision: RecognitionDecision) -> bool:
        if object_file.quality_score < 0.05:
            return False
        if decision.linked_track_id is not None and int(decision.linked_track_id) in self._active_episode_by_track:
            return True
        if decision.decision_type in {"new_concept", "uncertain_hold", "familiar_but_unresolved"}:
            return True
        if decision.decision_type == "same_instance" and decision.prediction_error >= 0.45:
            return True
        return False

    def _collect_track_states(self, tracking_output: Any) -> list[Any]:
        states: list[Any] = []
        for field_name in ("active_tracks", "dormant_tracks", "ghost_tracks", "retired_tracks"):
            states.extend(list(getattr(tracking_output, field_name, []) or []))
        return states

    def _determine_retrieval_context(
        self,
        object_file: ObjectFile,
        tracking_output: Any,
        frame_index: int,
    ) -> RetrievalContext:
        active_track_ids = {
            int(track.track_id)
            for track in list(getattr(tracking_output, "active_tracks", []) or [])
            if getattr(track, "track_id", None) is not None
        }
        track_id = object_file.linked_track_id
        if track_id is not None and int(track_id) in self._active_episode_by_track:
            mode = "continuous"
            prefer_closed = False
        elif track_id is not None:
            mode = "reentry"
            prefer_closed = True
        else:
            mode = "general"
            prefer_closed = False
        return RetrievalContext(
            frame_index=int(frame_index),
            query_track_id=None if track_id is None else int(track_id),
            query_prototype_id=object_file.linked_prototype_id,
            query_concept_id=object_file.linked_concept_id,
            active_track_ids=active_track_ids,
            mode=mode,
            min_reentry_gap=8,
            prefer_closed_episodes=prefer_closed,
            suppress_active_conflicts=True,
        )

    def _attach_ground_truth_metadata(self, object_files: list[ObjectFile], ground_truth: dict[str, Any]) -> None:
        boxes = list(ground_truth.get("boxes", []) or [])
        instance_ids = list(ground_truth.get("instance_ids", []) or [])
        concept_ids = list(ground_truth.get("concept_ids", []) or [])
        for object_file in object_files:
            best_iou = 0.0
            best_index = -1
            for index, gt_box in enumerate(boxes):
                iou = _bbox_iou(object_file.box, tuple(int(v) for v in gt_box))
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_index >= 0 and best_iou >= 0.10:
                object_file.metadata["gt_iou_eval_only"] = float(best_iou)
                if best_index < len(instance_ids):
                    object_file.metadata["gt_instance_id"] = int(instance_ids[best_index])
                if best_index < len(concept_ids):
                    object_file.metadata["gt_concept_id"] = int(concept_ids[best_index])

    def _close_stale_episodes(self, frame_index: int, close_gap: int = 2) -> list[int]:
        closed: list[int] = []
        for track_id, episode_id in list(self._active_episode_by_track.items()):
            last_seen = self._last_seen_track_frame.get(int(track_id), -10**9)
            if int(frame_index) - int(last_seen) <= close_gap:
                continue
            self.episodic_memory.close_episode(episode_id, frame_index, close_reason="track_not_observed")
            closed.append(int(episode_id))
            del self._active_episode_by_track[int(track_id)]
        return closed

    def _decision_event(
        self,
        frame_index: int,
        object_file: ObjectFile,
        decision: RecognitionDecision,
        candidates: list[RetrievedEpisode] | None = None,
    ) -> CognitiveEvent:
        mapping = {
            "same_instance": "same_instance_recognized",
            "same_concept": "same_concept_recognized",
            "new_concept": "new_concept_created",
            "uncertain_hold": "uncertain_hold",
            "familiar_but_unresolved": "familiar_unresolved",
        }
        top1 = candidates[0] if candidates else None
        metadata = {
            "retrieval_mode": decision.metadata.get("retrieval_mode", "general"),
            "top1_episode_id": None if top1 is None else top1.bundle.episode_id,
            "top1_score": 0.0 if top1 is None else top1.score,
            "top1_margin": 0.0 if top1 is None else top1.margin_to_next,
            "active_conflict": False if top1 is None else top1.active_conflict,
            "rejection_reason": decision.metadata.get("rejection_reason", ""),
        }
        return self._event(
            frame_index,
            mapping.get(decision.decision_type, "uncertain_hold"),
            object_file,
            decision,
            metadata=metadata,
        )

    def _event(
        self,
        frame_index: int,
        event_type: str,
        object_file: ObjectFile | None = None,
        decision: RecognitionDecision | None = None,
        episode_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CognitiveEvent:
        self._event_counter += 1
        return CognitiveEvent(
            event_id=f"ce:{frame_index}:{self._event_counter}",
            frame_index=int(frame_index),
            event_type=event_type,
            object_file_id=None if object_file is None else object_file.object_file_id,
            track_id=None if decision is None else decision.linked_track_id,
            prototype_id=None if decision is None else decision.linked_prototype_id,
            concept_id=None if decision is None else decision.linked_concept_id,
            episode_id=episode_id,
            decision_type=None if decision is None else decision.decision_type,
            confidence=0.0 if decision is None else float(decision.confidence),
            prediction_error=0.0 if decision is None else float(decision.prediction_error),
            familiarity_score=0.0 if decision is None else float(decision.familiarity_score),
            novelty_score=0.0 if decision is None else float(decision.novelty_score),
            metadata=metadata or {},
        )


def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    left_area = float(max(0, lx2 - lx1) * max(0, ly2 - ly1))
    right_area = float(max(0, rx2 - rx1) * max(0, ry2 - ry1))
    union = left_area + right_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union
