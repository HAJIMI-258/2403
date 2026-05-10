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
from nops_owr.cognition.object_file import ObjectFile, ObjectFileBuilder
from nops_owr.cognition.predictive_recognition import PredictiveRecognizer, RecognitionDecision
from nops_owr.evaluation.cognitive_metrics import (
    attended_object_ratio,
    memory_write_rate,
    prediction_error_mean,
)
from nops_owr.memory.episodic_memory import EpisodicMemory, RetrievedEpisode


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

    def step(self, prev_frame: np.ndarray, current_frame: np.ndarray, frame_index: int) -> CognitiveFrameResult:
        encoding = self.encoder.encode(prev_frame, current_frame)
        objectness_output = self.objectness_field.compute(encoding)
        object_files = self.object_file_builder.build(
            objectness_output=objectness_output,
            encoding=encoding,
            frame_index=frame_index,
            current_frame=current_frame,
        )
        attended_object_files = self.attention_gate.select(object_files)

        tracking_output = self.tracker.update(
            objectness_output.proposals,
            encoding,
            objectness_output.heatmap,
            current_frame,
            frame_index,
            memory_context=None,
        )
        memory_output = self.prototype_memory.update(
            tracking_output.assignments,
            frame_index,
            track_states=tracking_output.active_tracks,
        )
        if hasattr(self.tracker, "bind_prototypes"):
            self.tracker.bind_prototypes(memory_output.assignments)

        self._link_runtime_ids(object_files, tracking_output, memory_output)

        recognition_decisions: list[RecognitionDecision] = []
        written_episode_ids: list[int] = []
        episodic_retrievals: dict[str, list[RetrievedEpisode]] = {}
        assignment_by_proposal = {
            int(assignment.proposal_index): assignment for assignment in getattr(tracking_output, "assignments", [])
        }

        for object_file in attended_object_files:
            candidates = self.episodic_memory.retrieve(object_file, top_k=5)
            episodic_retrievals[object_file.object_file_id] = candidates
            decision = self.recognizer.recognize(
                object_file,
                tracking_assignment=assignment_by_proposal.get(object_file.proposal_index),
                episodic_candidates=candidates,
                prototype_result=memory_output,
            )
            recognition_decisions.append(decision)
            if self._should_write_episode(object_file, decision):
                episode_id = self.episodic_memory.write_episode(
                    object_file=object_file,
                    frame_index=frame_index,
                    track_id=decision.linked_track_id,
                    prototype_id=decision.linked_prototype_id,
                    concept_id=decision.linked_concept_id,
                    source_state=object_file.state,
                )
                written_episode_ids.append(episode_id)
            elif decision.linked_episode_id is not None and decision.decision_type == "same_instance":
                self.episodic_memory.update_reactivation(decision.linked_episode_id, frame_index)

        self.episodic_memory.consolidate_candidates(frame_index)
        metrics_snapshot = {
            "attended_object_ratio": attended_object_ratio(len(attended_object_files), len(object_files)),
            "memory_write_rate": memory_write_rate(len(written_episode_ids), len(object_files)),
            "prediction_error_mean": prediction_error_mean(recognition_decisions),
            "episodic_memory_size": float(len(self.episodic_memory)),
        }
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
        )

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
        if decision.decision_type in {"new_concept", "uncertain_hold", "familiar_but_unresolved"}:
            return True
        if decision.decision_type == "same_instance" and decision.prediction_error >= 0.45:
            return True
        return False
