"""Gap-aware temporal identity tracker for Phase 3R."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from metrics.metrics_core import bbox_iou
from nops_owr.encoder.spike_encoder import SpikeEncoding
from nops_owr.objectness.field import Proposal

Box = tuple[int, int, int, int]


@dataclass(slots=True)
class ProposalFeature:
    proposal_index: int
    box: Box
    centroid: tuple[float, float]
    score: float
    signature: np.ndarray


@dataclass(slots=True)
class PrototypeRecoveryHint:
    prototype_id: int
    source_track_id: int
    lineage_id: int | None
    distance: float
    similarity: float
    position_error: float
    gap_length: int


@dataclass(slots=True)
class IdentitySlot:
    slot_id: int
    source_track_id: int
    prototype_id: int
    disappear_frame: int
    last_center: tuple[float, float]
    last_bbox: Box
    velocity: np.ndarray
    feature_ema: np.ndarray
    shape_signature: np.ndarray
    last_objectness: float
    hit_count: int
    track_age: int
    age_since_disappear: int
    slot_confidence: float
    was_occluded_before_disappear: bool


@dataclass(slots=True)
class TrackState:
    track_id: int
    state: str
    box: Box
    centroid: tuple[float, float]
    last_bbox: Box
    last_center: tuple[float, float]
    signature: np.ndarray
    anchor_signature: np.ndarray
    last_feature: np.ndarray
    prototype_id: int | None
    lineage_id: int | None
    prototype_signature: np.ndarray
    velocity: np.ndarray
    score: float
    hits: int
    hit_count: int
    age: int
    missed_frames: int
    miss_count: int
    last_seen_frame: int
    gap_length: int
    active: bool
    dormant: bool
    ghost: bool
    retired: bool
    continuity_lineage_id: int | None = None


@dataclass(slots=True)
class TrackAssignment:
    proposal_index: int
    track_id: int
    box: Box
    centroid: tuple[float, float]
    score: float
    signature: np.ndarray
    match_cost: float
    assignment_source: str
    tentative_assignment_source: str
    final_assignment_source: str
    tentative_lineage_id: int | None
    final_lineage_id: int | None
    was_rerouted: bool
    preempting_active_track_id: int | None
    preempting_active_lineage_id: int | None
    best_recovery_lineage_id: int | None
    best_recovery_source_type: str | None
    routing_arbitration_triggered: bool
    cross_lineage_preemption_flag: bool
    preemption_reason: str | None
    active_claim_confidence: float | None
    recovery_claim_confidence: float | None
    routing_margin: float | None
    linked_prototype_id: int | None
    linked_lineage_id: int | None
    pre_memory_linked_lineage_id: int | None
    prototype_hint_id: int | None
    prototype_hint_lineage_id: int | None
    prototype_hint_distance: float
    prototype_hint_similarity: float
    reactivation_attempted: bool
    reactivation_cost: float
    prototype_similarity: float
    position_error: float
    objectness_score: float
    gap_length: int
    previous_state: str
    concept_recovered: bool
    candidate_pool_size: int
    resurrection_attempted: bool
    resurrection_success: bool
    resurrection_cost_best: float | None
    best_candidate_state: str | None
    best_candidate_gap: int | None
    live_candidate_pool_size: int
    slot_candidate_pool_size: int
    continuation_bank_size: int
    prototype_matched_continuation_count: int
    lineage_matched_continuation_count: int
    continuation_bank_exists: bool
    candidate_pool_nonempty: bool
    continuation_attempted: bool
    continuation_success: bool
    best_continuation_cost: float | None
    best_continuation_gap: int | None
    best_continuation_age: int | None
    resurrected_from_continuation: bool
    slot_attempted: bool
    slot_success: bool
    best_slot_cost: float | None
    best_slot_gap: int | None
    best_slot_age: int | None
    resurrected_from_slot: bool
    attach_state_consumed_by_tracker: bool
    attach_state_consumed_by_continuation: bool
    restore_attempted_from_attach: bool
    anchor_candidate_pool_size: int
    restore_attempted_from_anchor: bool
    anchor_success: bool
    best_anchor_uid: str | None
    best_anchor_gap: int | None
    promotion_pending_created: bool
    promotion_step_executed: bool
    consumer_candidate_count_seen: int
    selected_recovery_lineage_id: int | None
    selected_recovery_source_type: str | None
    selected_recovery_track_id: int | None
    selected_recovery_prototype_id: int | None
    identity_tiebreak_applied: bool
    identity_tiebreak_reason: str | None
    continuity_lineage_id: int | None = None


@dataclass(slots=True)
class MatchCostStats:
    min_cost: float
    mean_cost: float
    max_cost: float
    candidate_pairs: int
    accepted_matches: int


@dataclass(slots=True)
class FrameTrackingResult:
    frame_index: int
    assignments: list[TrackAssignment]
    new_track_ids: list[int]
    reactivated_track_ids: list[int]
    lost_track_ids: list[int]
    active_tracks: list[TrackState]
    dormant_tracks: list[TrackState]
    ghost_tracks: list[TrackState]
    retired_tracks: list[TrackState]
    track_order: list[int]
    proposal_order: list[int]
    cost_matrix: np.ndarray
    cost_stats: MatchCostStats
    unmatched_track_count: int
    unmatched_proposal_count: int
    active_track_count: int
    dormant_track_count: int
    ghost_track_count: int
    retired_track_count: int
    identity_slot_count: int
    reactivation_attempts: int
    resurrection_attempts: int
    resurrection_successes: int
    slot_archive_events: int
    slot_resurrection_attempts: int
    slot_resurrection_successes: int
    continuation_resurrection_attempts: int
    continuation_resurrection_successes: int
    routing_debug_rows: list[dict[str, object]]
    preserve_input_rows: list[dict[str, object]]
    recovery_candidate_rows: list[dict[str, object]]
    lineage_claim_rows: list[dict[str, object]]
    recovery_selection_rows: list[dict[str, object]]


class MinimalTemporalIdentityTracker:
    def __init__(
        self,
        beta_iou: float | None = None,
        beta_center: float | None = None,
        beta_feat: float | None = None,
        max_match_cost: float = 0.72,
        keepalive_frames: int = 12,
        use_linear_prediction: bool = True,
        signature_momentum: float = 0.70,
        velocity_momentum: float = 0.65,
        prediction_steps_cap: int = 3,
        use_dormant_reactivation: bool = True,
        dormant_frames: int = 20,
        ghost_frames: int = 64,
        reactivation_cost: float = 0.68,
        reactivation_proto_sim: float = 0.55,
        anchor_momentum: float = 0.90,
        min_match_similarity: float = 0.0,
        missed_match_similarity_boost: float = 0.0,
        use_gap_aware_matching: bool = True,
        tau_g: float = 12.0,
        tau_react_short: float = 0.62,
        tau_react_long: float = 0.74,
        tau_res_short: float | None = None,
        tau_res_long: float | None = None,
        tau_continuation: float = 0.62,
        continuation_margin: float = 0.08,
        tau_proto_attach: float = 0.35,
        tau_obj_attach: float = 0.50,
        enable_identity_slots: bool = True,
        slot_topk_per_proto: int = 4,
        min_track_age_for_slot: int = 4,
        min_hits_for_slot: int = 3,
        slot_max_gap: int = 96,
        slot_tau: float = 0.62,
        slot_margin: float = 0.08,
        slot_decay: float = 0.01,
        debug_inherit_lineage_from_hint: bool = False,
        debug_force_attach_consume: bool = False,
        debug_force_anchor_consume: bool = False,
        enable_phase3d_routing_repair: bool = False,
        enable_phase3d_target_selection_trace: bool = False,
        enable_phase3d_target_selection_repair: bool = False,
        enable_phase3d_claim_preservation_repair: bool = False,
        enable_phase3d_identity_preference_tiebreak: bool = False,
        enable_phase3d_preserve_input_trace: bool = False,
        enable_phase3d_continuity_lineage_repair: bool = False,
        enable_phase3d_three_source_preserve_input: bool = False,
        enable_phasea_dual_owner_source_enumeration: bool = False,
        routing_recovery_max_distance: float = 0.65,
        routing_recovery_min_confidence: float = 0.30,
        routing_active_claim_override_margin: float = 0.20,
        routing_topk: int = 3,
        claim_preserve_min_score: float = 0.25,
        identity_preference_margin: float = 0.08,
        continuity_hint_min_score: float = 0.15,
        debug_force_reroute_frame: int | None = None,
        debug_force_reroute_lineage: int | None = None,
        debug_force_visibility_for_all_cross_lineage: bool = False,
        iou_weight: float | None = None,
        center_weight: float | None = None,
        signature_weight: float | None = None,
        max_missed_frames: int | None = None,
        use_motion_prediction: bool | None = None,
        max_center_distance: float | None = None,
        **_: object,
    ) -> None:
        self.beta_iou = float(beta_iou if beta_iou is not None else (iou_weight if iou_weight is not None else 0.45))
        self.beta_center = float(
            beta_center if beta_center is not None else (center_weight if center_weight is not None else 0.25)
        )
        self.beta_feat = float(
            beta_feat if beta_feat is not None else (signature_weight if signature_weight is not None else 0.30)
        )
        self.max_match_cost = float(max_match_cost)
        self.keepalive_frames = int(keepalive_frames if max_missed_frames is None else max_missed_frames)
        self.use_linear_prediction = bool(
            use_linear_prediction if use_motion_prediction is None else use_motion_prediction
        )
        self.signature_momentum = float(signature_momentum)
        self.velocity_momentum = float(velocity_momentum)
        self.prediction_steps_cap = int(max(1, prediction_steps_cap))
        self.use_dormant_reactivation = bool(use_dormant_reactivation)
        self.dormant_frames = int(max(self.keepalive_frames, dormant_frames))
        self.ghost_frames = int(max(self.dormant_frames, ghost_frames))
        self.reactivation_cost = float(reactivation_cost)
        self.reactivation_proto_sim = float(reactivation_proto_sim)
        self.anchor_momentum = float(anchor_momentum)
        self.min_match_similarity = float(min_match_similarity)
        self.missed_match_similarity_boost = float(missed_match_similarity_boost)
        self.use_gap_aware_matching = bool(use_gap_aware_matching)
        self.tau_g = float(max(1e-6, tau_g))
        self.tau_react_short = float(tau_react_short)
        self.tau_react_long = float(tau_react_long)
        self.tau_res_short = float(tau_react_short if tau_res_short is None else tau_res_short)
        self.tau_res_long = float(tau_react_long if tau_res_long is None else tau_res_long)
        self.tau_continuation = float(tau_continuation)
        self.continuation_margin = float(max(0.0, continuation_margin))
        self.tau_proto_attach = float(tau_proto_attach)
        self.tau_obj_attach = float(tau_obj_attach)
        self.enable_identity_slots = bool(enable_identity_slots)
        self.slot_topk_per_proto = int(max(1, slot_topk_per_proto))
        self.min_track_age_for_slot = int(max(1, min_track_age_for_slot))
        self.min_hits_for_slot = int(max(1, min_hits_for_slot))
        self.slot_max_gap = int(max(1, slot_max_gap))
        self.slot_tau = float(slot_tau)
        self.slot_margin = float(max(0.0, slot_margin))
        self.slot_decay = float(max(0.0, slot_decay))
        self.debug_inherit_lineage_from_hint = bool(debug_inherit_lineage_from_hint)
        self.debug_force_attach_consume = bool(debug_force_attach_consume)
        self.debug_force_anchor_consume = bool(debug_force_anchor_consume)
        self.enable_phase3d_routing_repair = bool(enable_phase3d_routing_repair)
        self.enable_phase3d_target_selection_trace = bool(enable_phase3d_target_selection_trace)
        self.enable_phase3d_target_selection_repair = bool(enable_phase3d_target_selection_repair)
        self.enable_phase3d_claim_preservation_repair = bool(enable_phase3d_claim_preservation_repair)
        self.enable_phase3d_identity_preference_tiebreak = bool(enable_phase3d_identity_preference_tiebreak)
        self.enable_phase3d_preserve_input_trace = bool(enable_phase3d_preserve_input_trace)
        self.enable_phase3d_continuity_lineage_repair = bool(enable_phase3d_continuity_lineage_repair)
        self.enable_phase3d_three_source_preserve_input = bool(enable_phase3d_three_source_preserve_input)
        self.enable_phasea_dual_owner_source_enumeration = bool(enable_phasea_dual_owner_source_enumeration)
        self.routing_recovery_max_distance = float(max(0.0, routing_recovery_max_distance))
        self.routing_recovery_min_confidence = float(max(0.0, routing_recovery_min_confidence))
        self.routing_active_claim_override_margin = float(max(0.0, routing_active_claim_override_margin))
        self.routing_topk = int(max(1, routing_topk))
        self.claim_preserve_min_score = float(max(0.0, claim_preserve_min_score))
        self.identity_preference_margin = float(max(0.0, identity_preference_margin))
        self.continuity_hint_min_score = float(max(0.0, continuity_hint_min_score))
        self.debug_force_reroute_frame = (
            None if debug_force_reroute_frame is None else int(debug_force_reroute_frame)
        )
        self.debug_force_reroute_lineage = (
            None if debug_force_reroute_lineage is None else int(debug_force_reroute_lineage)
        )
        self.debug_force_visibility_for_all_cross_lineage = bool(debug_force_visibility_for_all_cross_lineage)
        self.max_center_distance = None if max_center_distance is None else float(max_center_distance)

        self._tracks: dict[int, TrackState] = {}
        self._next_track_id = 0
        self._identity_slots_by_prototype: dict[int, deque[IdentitySlot]] = defaultdict(deque)
        self._next_slot_id = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 0
        self._identity_slots_by_prototype.clear()
        self._next_slot_id = 0

    def snapshot(self, include_retired: bool = True) -> list[TrackState]:
        return [
            _clone_track(track)
            for track in sorted(self._tracks.values(), key=lambda item: item.track_id)
            if include_retired or not track.retired
        ]

    def identity_slot_count(self) -> int:
        return int(sum(len(slots) for slots in self._identity_slots_by_prototype.values()))

    def bind_prototypes(self, prototype_assignments) -> None:
        for assignment in prototype_assignments:
            track = self._tracks.get(assignment.track_id)
            if track is None:
                continue
            track.prototype_id = int(assignment.prototype_id)
            track.lineage_id = None if getattr(assignment, "lineage_id", None) is None else int(assignment.lineage_id)
            track.continuity_lineage_id = (
                track.lineage_id
                if getattr(assignment, "continuity_lineage_id", None) is None
                else int(assignment.continuity_lineage_id)
            )
            track.prototype_signature = assignment.prototype_signature.copy()

    def update(
        self,
        proposals: list[Proposal],
        encoding: SpikeEncoding,
        heatmap: np.ndarray,
        current_frame: np.ndarray,
        frame_index: int,
        memory_context=None,
    ) -> FrameTrackingResult:
        self._refresh_identity_slots(frame_index)
        proposal_features = [
            _build_proposal_feature(index, proposal, encoding, heatmap, current_frame)
            for index, proposal in enumerate(proposals)
        ]
        active_tracks = self._tracks_by_state("active")
        dormant_tracks = self._tracks_by_state("dormant")
        ghost_tracks = self._tracks_by_state("ghost")

        projected_states = [
            _project_track_state(
                track,
                frame_shape=heatmap.shape,
                use_linear_prediction=self.use_linear_prediction,
                prediction_steps_cap=self.prediction_steps_cap,
            )
            for track in active_tracks
        ]
        cost_matrix = _build_cost_matrix(
            tracks=active_tracks,
            projected_states=projected_states,
            proposals=proposal_features,
            frame_shape=heatmap.shape,
            beta_iou=self.beta_iou,
            beta_center=self.beta_center,
            beta_feat=self.beta_feat,
            max_center_distance=self.max_center_distance,
        )
        active_matches = _greedy_accept_matches(
            cost_matrix,
            active_tracks,
            proposal_features,
            self.max_match_cost,
            min_feature_similarity=self.min_match_similarity,
            missed_similarity_boost=self.missed_match_similarity_boost,
        )
        recovery_surface_lookup = self._build_routing_surface_lookup(
            dormant_tracks=dormant_tracks,
            ghost_tracks=ghost_tracks,
            memory_context=memory_context,
        )
        recovery_candidate_lists = self._build_routing_recovery_candidates(
            proposals=proposal_features,
            dormant_tracks=dormant_tracks,
            ghost_tracks=ghost_tracks,
            memory_context=memory_context,
        )
        active_matches, routing_meta_by_proposal, routing_debug_rows = self._arbitrate_active_matches(
            active_matches=active_matches,
            active_tracks=active_tracks,
            projected_states=projected_states,
            proposal_features=proposal_features,
            cost_matrix=cost_matrix,
            recovery_candidate_lists=recovery_candidate_lists,
            recovery_surface_lookup=recovery_surface_lookup,
            frame_index=frame_index,
        )
        matched_proposal_indices = {proposal_index for _, proposal_index, _ in active_matches}
        matched_active_ids = {track_id for track_id, _, _ in active_matches}

        retired_track_ids: list[int] = []
        slot_archive_events = 0
        for track in active_tracks:
            if track.track_id not in matched_active_ids:
                retired_ids, archived = self._advance_unmatched_track(track, frame_index, heatmap.shape)
                retired_track_ids.extend(retired_ids)
                slot_archive_events += archived
        for track in dormant_tracks:
            retired_ids, archived = self._advance_unmatched_track(track, frame_index, heatmap.shape)
            retired_track_ids.extend(retired_ids)
            slot_archive_events += archived
        for track in ghost_tracks:
            retired_ids, archived = self._advance_unmatched_track(track, frame_index, heatmap.shape)
            retired_track_ids.extend(retired_ids)
            slot_archive_events += archived

        assignments: list[TrackAssignment] = []
        new_track_ids: list[int] = []
        reactivated_track_ids: list[int] = []

        for track_id, proposal_index, match_cost in active_matches:
            proposal = proposal_features[proposal_index]
            track = self._tracks[track_id]
            routing_meta = routing_meta_by_proposal.get(int(proposal.proposal_index), {})
            tracked_assignment = self._apply_match(
                track=track,
                proposal=proposal,
                frame_index=frame_index,
                assignment_source="active_match",
                match_cost=float(match_cost),
                reactivation_cost=float(match_cost),
                gap_length=int(track.gap_length),
                position_error=_centroid_distance(track.centroid, proposal.centroid),
                prototype_similarity=_prototype_similarity(track, proposal),
                reactivation_attempted=False,
                candidate_pool_size=0,
                resurrection_attempted=False,
                resurrection_success=False,
                resurrection_cost_best=None,
                best_candidate_state=None,
                best_candidate_gap=None,
                live_candidate_pool_size=0,
                slot_candidate_pool_size=0,
                continuation_bank_size=0,
                candidate_pool_nonempty=False,
                continuation_attempted=False,
                continuation_success=False,
                best_continuation_cost=None,
                best_continuation_gap=None,
                best_continuation_age=None,
                resurrected_from_continuation=False,
                slot_attempted=False,
                slot_success=False,
                best_slot_cost=None,
                best_slot_gap=None,
                best_slot_age=None,
                resurrected_from_slot=False,
            )
            tracked_assignment.tentative_assignment_source = str(
                routing_meta.get("tentative_assignment_source", tracked_assignment.tentative_assignment_source)
            )
            tracked_assignment.final_assignment_source = str(
                routing_meta.get("final_assignment_source", tracked_assignment.final_assignment_source)
            )
            tracked_assignment.tentative_lineage_id = routing_meta.get("tentative_lineage_id")
            tracked_assignment.final_lineage_id = routing_meta.get("final_lineage_id")
            tracked_assignment.was_rerouted = bool(routing_meta.get("was_rerouted", False))
            tracked_assignment.preempting_active_track_id = routing_meta.get("preempting_active_track_id")
            tracked_assignment.preempting_active_lineage_id = routing_meta.get("preempting_active_lineage_id")
            tracked_assignment.best_recovery_lineage_id = routing_meta.get("best_recovery_lineage_id")
            tracked_assignment.best_recovery_source_type = routing_meta.get("best_recovery_source_type")
            tracked_assignment.routing_arbitration_triggered = bool(
                routing_meta.get("routing_arbitration_triggered", False)
            )
            tracked_assignment.cross_lineage_preemption_flag = bool(
                routing_meta.get("cross_lineage_preemption_flag", False)
            )
            tracked_assignment.preemption_reason = routing_meta.get("preemption_reason")
            tracked_assignment.active_claim_confidence = routing_meta.get("active_claim_confidence")
            tracked_assignment.recovery_claim_confidence = routing_meta.get("recovery_claim_confidence")
            tracked_assignment.routing_margin = routing_meta.get("routing_margin")
            assignments.append(tracked_assignment)

        concept_tracks = [
            track for track in self._tracks.values() if track.prototype_id is not None and (track.dormant or track.ghost)
        ]
        concept_hints = self._build_concept_hints(concept_tracks, proposal_features, matched_proposal_indices)

        for proposal in proposal_features:
            if proposal.proposal_index in matched_proposal_indices:
                continue
            routing_meta = routing_meta_by_proposal.get(int(proposal.proposal_index), {})
            hint = concept_hints.get(proposal.proposal_index)
            hinted_lineage_id = None if hint is None or hint.lineage_id is None else int(hint.lineage_id)
            if (
                hinted_lineage_id is None
                and bool(routing_meta.get("was_rerouted", False))
                and routing_meta.get("best_recovery_lineage_id") is not None
            ):
                hinted_lineage_id = int(routing_meta["best_recovery_lineage_id"])
            linked_lineage_id = (
                hinted_lineage_id
                if (
                    (self.debug_inherit_lineage_from_hint and hinted_lineage_id is not None)
                    or bool(routing_meta.get("was_rerouted", False))
                )
                else None
            )
            hinted_prototype_id = None if hint is None else int(hint.prototype_id)
            if (
                hinted_prototype_id is None
                and bool(routing_meta.get("was_rerouted", False))
                and routing_meta.get("proposal_proto_hint") is not None
            ):
                hinted_prototype_id = int(routing_meta["proposal_proto_hint"])
            track = self._create_track(proposal, frame_index)
            new_track_ids.append(track.track_id)
            assignments.append(
                TrackAssignment(
                    proposal_index=proposal.proposal_index,
                    track_id=track.track_id,
                    box=proposal.box,
                    centroid=proposal.centroid,
                    score=proposal.score,
                    signature=proposal.signature.copy(),
                    match_cost=float(self.max_match_cost),
                    assignment_source="rerouted_to_resurrection"
                    if bool(routing_meta.get("was_rerouted", False))
                    else "new_track",
                    tentative_assignment_source=str(
                        routing_meta.get("tentative_assignment_source", "new_track")
                    ),
                    final_assignment_source="rerouted_to_resurrection"
                    if bool(routing_meta.get("was_rerouted", False))
                    else "new_track",
                    tentative_lineage_id=routing_meta.get("tentative_lineage_id"),
                    final_lineage_id=linked_lineage_id,
                    was_rerouted=bool(routing_meta.get("was_rerouted", False)),
                    preempting_active_track_id=routing_meta.get("preempting_active_track_id"),
                    preempting_active_lineage_id=routing_meta.get("preempting_active_lineage_id"),
                    best_recovery_lineage_id=routing_meta.get("best_recovery_lineage_id"),
                    best_recovery_source_type=routing_meta.get("best_recovery_source_type"),
                    routing_arbitration_triggered=bool(
                        routing_meta.get("routing_arbitration_triggered", False)
                    ),
                    cross_lineage_preemption_flag=bool(
                        routing_meta.get("cross_lineage_preemption_flag", False)
                    ),
                    preemption_reason=routing_meta.get("preemption_reason"),
                    active_claim_confidence=routing_meta.get("active_claim_confidence"),
                    recovery_claim_confidence=routing_meta.get("recovery_claim_confidence"),
                    routing_margin=routing_meta.get("routing_margin"),
                    linked_prototype_id=None,
                    linked_lineage_id=linked_lineage_id,
                    pre_memory_linked_lineage_id=linked_lineage_id,
                    prototype_hint_id=hinted_prototype_id,
                    prototype_hint_lineage_id=hinted_lineage_id,
                    prototype_hint_distance=(
                        float(routing_meta.get("best_recovery_distance", 1.0))
                        if hint is None and bool(routing_meta.get("was_rerouted", False))
                        else (1.0 if hint is None else float(hint.distance))
                    ),
                    prototype_hint_similarity=(
                        0.0
                        if hint is None and not bool(routing_meta.get("was_rerouted", False))
                        else (
                            float(
                                0.0
                                if routing_meta.get("recovery_claim_confidence") is None
                                else routing_meta.get("recovery_claim_confidence", 0.0)
                            )
                            if hint is None
                            else float(hint.similarity)
                        )
                    ),
                    reactivation_attempted=False,
                    reactivation_cost=np.inf,
                    prototype_similarity=(
                        float(
                            0.0
                            if routing_meta.get("recovery_claim_confidence") is None
                            else routing_meta.get("recovery_claim_confidence", 0.0)
                        )
                        if hint is None and bool(routing_meta.get("was_rerouted", False))
                        else 0.0
                        if hint is None
                        else float(hint.similarity)
                    ),
                    position_error=np.inf,
                    objectness_score=float(proposal.score),
                    gap_length=0,
                    previous_state="new",
                    concept_recovered=hint is not None or bool(routing_meta.get("was_rerouted", False)),
                    candidate_pool_size=0,
                    resurrection_attempted=False,
                    resurrection_success=False,
                    resurrection_cost_best=None,
                    best_candidate_state=None,
                    best_candidate_gap=None,
                    live_candidate_pool_size=0,
                    slot_candidate_pool_size=0,
                    continuation_bank_size=0,
                    prototype_matched_continuation_count=0,
                    lineage_matched_continuation_count=0,
                    continuation_bank_exists=False,
                    candidate_pool_nonempty=False,
                    continuation_attempted=False,
                    continuation_success=False,
                    best_continuation_cost=None,
                    best_continuation_gap=None,
                    best_continuation_age=None,
                    resurrected_from_continuation=False,
                    slot_attempted=False,
                    slot_success=False,
                    best_slot_cost=None,
                    best_slot_gap=None,
                    best_slot_age=None,
                    resurrected_from_slot=False,
                    attach_state_consumed_by_tracker=False,
                    attach_state_consumed_by_continuation=False,
                    restore_attempted_from_attach=False,
                    anchor_candidate_pool_size=0,
                    restore_attempted_from_anchor=False,
                    anchor_success=False,
                    best_anchor_uid=None,
                    best_anchor_gap=None,
                    promotion_pending_created=False,
                    promotion_step_executed=False,
                    consumer_candidate_count_seen=0,
                    selected_recovery_lineage_id=None,
                    selected_recovery_source_type=None,
                    selected_recovery_track_id=None,
                    selected_recovery_prototype_id=None,
                    identity_tiebreak_applied=False,
                    identity_tiebreak_reason=None,
                    continuity_lineage_id=linked_lineage_id,
                )
            )

        assignments.sort(key=lambda assignment: assignment.proposal_index)
        active_snapshot = [_clone_track(track) for track in self._tracks_by_state("active")]
        dormant_snapshot = [_clone_track(track) for track in self._tracks_by_state("dormant")]
        ghost_snapshot = [_clone_track(track) for track in self._tracks_by_state("ghost")]
        retired_snapshot = [_clone_track(track) for track in self._tracks_by_state("retired")]
        cost_stats = _summarize_cost_matrix(cost_matrix, accepted_matches=len(active_matches))

        return FrameTrackingResult(
            frame_index=frame_index,
            assignments=assignments,
            new_track_ids=new_track_ids,
            reactivated_track_ids=sorted(set(reactivated_track_ids)),
            lost_track_ids=sorted(set(retired_track_ids)),
            active_tracks=active_snapshot,
            dormant_tracks=dormant_snapshot,
            ghost_tracks=ghost_snapshot,
            retired_tracks=retired_snapshot,
            track_order=[track.track_id for track in active_tracks],
            proposal_order=[proposal.proposal_index for proposal in proposal_features],
            cost_matrix=cost_matrix,
            cost_stats=cost_stats,
            unmatched_track_count=max(0, len(active_tracks) - len(active_matches)),
            unmatched_proposal_count=max(0, len(proposal_features) - len(assignments)),
            active_track_count=len(active_snapshot),
            dormant_track_count=len(dormant_snapshot),
            ghost_track_count=len(ghost_snapshot),
            retired_track_count=len(retired_snapshot),
            identity_slot_count=self.identity_slot_count(),
            reactivation_attempts=0,
            resurrection_attempts=0,
            resurrection_successes=0,
            slot_archive_events=int(slot_archive_events),
            slot_resurrection_attempts=0,
            slot_resurrection_successes=0,
            continuation_resurrection_attempts=0,
            continuation_resurrection_successes=0,
            routing_debug_rows=routing_debug_rows,
            preserve_input_rows=[],
            recovery_candidate_rows=[],
            lineage_claim_rows=[],
            recovery_selection_rows=[],
        )

    def _tracks_by_state(self, state: str) -> list[TrackState]:
        return sorted(
            (track for track in self._tracks.values() if track.state == state),
            key=lambda track: track.track_id,
        )

    def _advance_unmatched_track(
        self,
        track: TrackState,
        frame_index: int,
        frame_shape: tuple[int, int],
    ) -> tuple[list[int], int]:
        track.missed_frames += 1
        track.miss_count += 1
        track.gap_length = max(1, frame_index - track.last_seen_frame)
        track.age += 1
        if self.use_linear_prediction:
            predicted_box, predicted_centroid = _predict_track_state(
                track,
                frame_shape=frame_shape,
                prediction_steps_cap=self.prediction_steps_cap,
            )
            track.box = predicted_box
            track.centroid = predicted_centroid
        if track.gap_length > self.ghost_frames:
            archived = self._archive_identity_slot(track, frame_index)
            _set_track_state(track, "retired")
            return [track.track_id], archived
        if track.gap_length > self.dormant_frames:
            _set_track_state(track, "ghost")
        elif track.gap_length > self.keepalive_frames:
            _set_track_state(track, "dormant")
        else:
            _set_track_state(track, "active")
        return [], 0

    def _apply_match(
        self,
        *,
        track: TrackState,
        proposal: ProposalFeature,
        frame_index: int,
        assignment_source: str,
        match_cost: float,
        reactivation_cost: float,
        gap_length: int,
        position_error: float,
        prototype_similarity: float,
        reactivation_attempted: bool,
        concept_recovered: bool = False,
        candidate_pool_size: int = 0,
        resurrection_attempted: bool = False,
        resurrection_success: bool = False,
        resurrection_cost_best: float | None = None,
        best_candidate_state: str | None = None,
        best_candidate_gap: int | None = None,
        live_candidate_pool_size: int = 0,
        slot_candidate_pool_size: int = 0,
        continuation_bank_size: int = 0,
        prototype_matched_continuation_count: int = 0,
        lineage_matched_continuation_count: int = 0,
        continuation_bank_exists: bool = False,
        candidate_pool_nonempty: bool = False,
        continuation_attempted: bool = False,
        continuation_success: bool = False,
        best_continuation_cost: float | None = None,
        best_continuation_gap: int | None = None,
        best_continuation_age: int | None = None,
        resurrected_from_continuation: bool = False,
        anchor_candidate_pool_size: int = 0,
        restore_attempted_from_anchor: bool = False,
        anchor_success: bool = False,
        best_anchor_uid: str | None = None,
        best_anchor_gap: int | None = None,
        slot_attempted: bool = False,
        slot_success: bool = False,
        best_slot_cost: float | None = None,
        best_slot_gap: int | None = None,
        best_slot_age: int | None = None,
        resurrected_from_slot: bool = False,
    ) -> TrackAssignment:
        previous_centroid = track.centroid
        previous_state = track.state
        linked_prototype_id = track.prototype_id
        track.box = proposal.box
        track.centroid = proposal.centroid
        track.last_bbox = proposal.box
        track.last_center = proposal.centroid
        track.last_feature = proposal.signature.copy()
        track.score = proposal.score
        track.signature = (
            self.signature_momentum * track.signature + (1.0 - self.signature_momentum) * proposal.signature
        ).astype(np.float32)
        track.signature = _normalize_signature(track.signature)
        track.anchor_signature = (
            self.anchor_momentum * track.anchor_signature + (1.0 - self.anchor_momentum) * proposal.signature
        ).astype(np.float32)
        track.anchor_signature = _normalize_signature(track.anchor_signature)
        instantaneous_velocity = np.array(
            [proposal.centroid[0] - previous_centroid[0], proposal.centroid[1] - previous_centroid[1]],
            dtype=np.float32,
        )
        track.velocity = (
            self.velocity_momentum * track.velocity + (1.0 - self.velocity_momentum) * instantaneous_velocity
        ).astype(np.float32)
        track.hits += 1
        track.hit_count += 1
        track.age += 1
        track.missed_frames = 0
        track.miss_count = 0
        track.last_seen_frame = frame_index
        track.gap_length = 0
        _set_track_state(track, "active")

        return TrackAssignment(
            proposal_index=proposal.proposal_index,
            track_id=track.track_id,
            box=proposal.box,
            centroid=proposal.centroid,
            score=proposal.score,
            signature=proposal.signature.copy(),
            match_cost=float(match_cost),
            assignment_source=assignment_source,
            tentative_assignment_source="active_match_same_lineage"
            if track.lineage_id is not None
            else "active_match_ambiguous",
            final_assignment_source="active_match_same_lineage"
            if track.lineage_id is not None
            else "active_match_ambiguous",
            tentative_lineage_id=track.lineage_id,
            final_lineage_id=track.lineage_id,
            was_rerouted=False,
            preempting_active_track_id=None,
            preempting_active_lineage_id=None,
            best_recovery_lineage_id=None,
            best_recovery_source_type=None,
            routing_arbitration_triggered=False,
            cross_lineage_preemption_flag=False,
            preemption_reason=None,
            active_claim_confidence=self._active_claim_confidence(match_cost),
            recovery_claim_confidence=None,
            routing_margin=None,
            linked_prototype_id=linked_prototype_id,
            linked_lineage_id=track.lineage_id,
            pre_memory_linked_lineage_id=track.lineage_id,
            prototype_hint_id=linked_prototype_id,
            prototype_hint_lineage_id=track.lineage_id,
            prototype_hint_distance=max(0.0, 1.0 - prototype_similarity),
            prototype_hint_similarity=float(prototype_similarity),
            reactivation_attempted=bool(reactivation_attempted),
            reactivation_cost=float(reactivation_cost),
            prototype_similarity=float(prototype_similarity),
            position_error=float(position_error),
            objectness_score=float(proposal.score),
            gap_length=int(gap_length),
            previous_state=previous_state,
            concept_recovered=bool(concept_recovered or linked_prototype_id is not None),
            candidate_pool_size=int(candidate_pool_size),
            resurrection_attempted=bool(resurrection_attempted),
            resurrection_success=bool(resurrection_success),
            resurrection_cost_best=None if resurrection_cost_best is None else float(resurrection_cost_best),
            best_candidate_state=best_candidate_state,
            best_candidate_gap=None if best_candidate_gap is None else int(best_candidate_gap),
            live_candidate_pool_size=int(live_candidate_pool_size),
            slot_candidate_pool_size=int(slot_candidate_pool_size),
            continuation_bank_size=int(continuation_bank_size),
            prototype_matched_continuation_count=int(prototype_matched_continuation_count),
            lineage_matched_continuation_count=int(lineage_matched_continuation_count),
            continuation_bank_exists=bool(continuation_bank_exists),
            candidate_pool_nonempty=bool(candidate_pool_nonempty),
            continuation_attempted=bool(continuation_attempted),
            continuation_success=bool(continuation_success),
            best_continuation_cost=None if best_continuation_cost is None else float(best_continuation_cost),
            best_continuation_gap=None if best_continuation_gap is None else int(best_continuation_gap),
            best_continuation_age=None if best_continuation_age is None else int(best_continuation_age),
            resurrected_from_continuation=bool(resurrected_from_continuation),
            slot_attempted=bool(slot_attempted),
            slot_success=bool(slot_success),
            best_slot_cost=None if best_slot_cost is None else float(best_slot_cost),
            best_slot_gap=None if best_slot_gap is None else int(best_slot_gap),
            best_slot_age=None if best_slot_age is None else int(best_slot_age),
            resurrected_from_slot=bool(resurrected_from_slot),
            attach_state_consumed_by_tracker=False,
            attach_state_consumed_by_continuation=False,
            restore_attempted_from_attach=False,
            anchor_candidate_pool_size=int(anchor_candidate_pool_size),
            restore_attempted_from_anchor=bool(restore_attempted_from_anchor),
            anchor_success=bool(anchor_success),
            best_anchor_uid=best_anchor_uid,
            best_anchor_gap=None if best_anchor_gap is None else int(best_anchor_gap),
            promotion_pending_created=False,
            promotion_step_executed=False,
            consumer_candidate_count_seen=0,
            selected_recovery_lineage_id=None,
            selected_recovery_source_type=None,
            selected_recovery_track_id=None,
            selected_recovery_prototype_id=None,
            identity_tiebreak_applied=False,
            identity_tiebreak_reason=None,
            continuity_lineage_id=(
                None
                if getattr(track, "continuity_lineage_id", None) is None
                else int(track.continuity_lineage_id)
            ),
        )

    def _effective_dormant_limit(self, track: TrackState) -> int:
        return int(self.dormant_frames)

    def _create_track(self, proposal: ProposalFeature, frame_index: int) -> TrackState:
        track_id = self._next_track_id
        self._next_track_id += 1
        track = TrackState(
            track_id=track_id,
            state="active",
            box=proposal.box,
            centroid=proposal.centroid,
            last_bbox=proposal.box,
            last_center=proposal.centroid,
            signature=proposal.signature.copy(),
            anchor_signature=proposal.signature.copy(),
            last_feature=proposal.signature.copy(),
            prototype_id=None,
            lineage_id=None,
            prototype_signature=np.zeros_like(proposal.signature, dtype=np.float32),
            velocity=np.zeros(2, dtype=np.float32),
            score=proposal.score,
            hits=1,
            hit_count=1,
            age=1,
            missed_frames=0,
            miss_count=0,
            last_seen_frame=frame_index,
            gap_length=0,
            active=True,
            dormant=False,
            ghost=False,
            retired=False,
            continuity_lineage_id=None,
        )
        self._tracks[track_id] = track
        return track

    def _build_reactivation_candidates(
        self,
        dormant_tracks: list[TrackState],
        proposals: list[ProposalFeature],
        frame_shape: tuple[int, int],
        frame_index: int,
    ) -> tuple[list[dict[str, float | int]], dict[int, dict[str, float | int]]]:
        candidates: list[dict[str, float | int]] = []
        best_by_proposal: dict[int, dict[str, float | int]] = {}
        if not dormant_tracks or not proposals or not self.use_dormant_reactivation:
            return candidates, best_by_proposal

        frame_diagonal = float(np.hypot(frame_shape[1], frame_shape[0]))
        for track in dormant_tracks:
            predicted_box, predicted_centroid = _project_track_state(
                track,
                frame_shape=frame_shape,
                use_linear_prediction=self.use_linear_prediction,
                prediction_steps_cap=self.prediction_steps_cap,
            )
            gap_length = max(1, frame_index - track.last_seen_frame)
            gap_decay = float(np.exp(-gap_length / self.tau_g))
            similarity_floor = 0.0 if self.use_gap_aware_matching else self.reactivation_proto_sim
            threshold = (
                self.tau_react_short + (1.0 - gap_decay) * (self.tau_react_long - self.tau_react_short)
                if self.use_gap_aware_matching
                else self.reactivation_cost
            )
            weights = _reactivation_weights(gap_decay) if self.use_gap_aware_matching else _legacy_reactivation_weights()
            center_norm = max(frame_diagonal, 1e-6)
            for proposal in proposals:
                d_pos = min(1.0, _centroid_distance(predicted_centroid, proposal.centroid) / center_norm)
                d_feat = _feature_distance(track.anchor_signature, proposal.signature)
                if track.prototype_id is not None and np.linalg.norm(track.prototype_signature) > 0.0:
                    d_proto = _feature_distance(track.prototype_signature, proposal.signature)
                else:
                    d_proto = d_feat
                d_obj = _object_consistency_distance(track, proposal)
                cost = (
                    weights["pos"] * d_pos
                    + weights["feat"] * d_feat
                    + weights["proto"] * d_proto
                    + weights["obj"] * d_obj
                )
                candidate = {
                    "track_id": int(track.track_id),
                    "proposal_index": int(proposal.proposal_index),
                    "cost": float(cost),
                    "threshold": float(threshold),
                    "position_error": float(_centroid_distance(predicted_centroid, proposal.centroid)),
                    "prototype_similarity": float(max(0.0, 1.0 - d_proto)),
                    "gap_length": int(gap_length),
                }
                if float(candidate["prototype_similarity"]) < similarity_floor:
                    continue
                candidates.append(candidate)
                current = best_by_proposal.get(proposal.proposal_index)
                if current is None or float(cost) < float(current["cost"]):
                    best_by_proposal[proposal.proposal_index] = candidate
        return candidates, best_by_proposal

    def _build_concept_hints(
        self,
        tracks: list[TrackState],
        proposals: list[ProposalFeature],
        matched_proposal_indices: set[int],
    ) -> dict[int, PrototypeRecoveryHint]:
        hints: dict[int, PrototypeRecoveryHint] = {}
        if not tracks:
            return hints

        score_floor = max(0.0, self.tau_obj_attach - 0.10)
        for proposal in proposals:
            if proposal.proposal_index in matched_proposal_indices or proposal.score < score_floor:
                continue
            best_hint: PrototypeRecoveryHint | None = None
            for track in tracks:
                if track.prototype_id is None or np.linalg.norm(track.prototype_signature) < 1e-6:
                    continue
                distance = _feature_distance(track.prototype_signature, proposal.signature)
                if distance > self.tau_proto_attach:
                    continue
                hint = PrototypeRecoveryHint(
                    prototype_id=int(track.prototype_id),
                    source_track_id=int(track.track_id),
                    lineage_id=None if track.lineage_id is None else int(track.lineage_id),
                    distance=float(distance),
                    similarity=float(max(0.0, 1.0 - distance)),
                    position_error=float(_centroid_distance(track.centroid, proposal.centroid)),
                    gap_length=int(track.gap_length),
                )
                if best_hint is None or hint.distance < best_hint.distance:
                    best_hint = hint
            if best_hint is not None:
                hints[proposal.proposal_index] = best_hint
        return hints

    def _build_routing_surface_lookup(
        self,
        *,
        dormant_tracks: list[TrackState],
        ghost_tracks: list[TrackState],
        memory_context=None,
    ) -> dict[int, dict[str, int | bool]]:
        surface: dict[int, dict[str, int | bool]] = {}

        def _ensure(lineage_id: int) -> dict[str, int | bool]:
            row = surface.get(int(lineage_id))
            if row is None:
                row = {
                    "dormant_count": 0,
                    "ghost_count": 0,
                    "continuation_bank_size": 0,
                    "recovery_anchor_count": 0,
                    "surface_nonempty": False,
                }
                surface[int(lineage_id)] = row
            return row

        for track in dormant_tracks:
            if track.lineage_id is None or int(track.lineage_id) < 0:
                continue
            row = _ensure(int(track.lineage_id))
            row["dormant_count"] = int(row["dormant_count"]) + 1
        for track in ghost_tracks:
            if track.lineage_id is None or int(track.lineage_id) < 0:
                continue
            row = _ensure(int(track.lineage_id))
            row["ghost_count"] = int(row["ghost_count"]) + 1

        if memory_context is not None:
            continuation_lookup = getattr(memory_context, "continuation_lineage_lookup", {})
            for lineage_id, continuations in continuation_lookup.items():
                row = _ensure(int(lineage_id))
                row["continuation_bank_size"] = int(len(continuations))
            recovery_anchor_lookup = getattr(memory_context, "recovery_anchor_lookup", {})
            for lineage_id, anchors in recovery_anchor_lookup.items():
                row = _ensure(int(lineage_id))
                row["recovery_anchor_count"] = int(len(anchors))

        for lineage_id, row in surface.items():
            row["surface_nonempty"] = bool(
                int(row["dormant_count"]) > 0
                or int(row["ghost_count"]) > 0
                or int(row["continuation_bank_size"]) > 0
                or int(row["recovery_anchor_count"]) > 0
            )
        return surface

    def _build_routing_recovery_candidates(
        self,
        *,
        proposals: list[ProposalFeature],
        dormant_tracks: list[TrackState],
        ghost_tracks: list[TrackState],
        memory_context=None,
    ) -> dict[int, list[dict[str, object]]]:
        candidates_by_proposal: dict[int, list[dict[str, object]]] = defaultdict(list)
        track_candidates = dormant_tracks + ghost_tracks

        for proposal in proposals:
            for track in track_candidates:
                if track.lineage_id is None or int(track.lineage_id) < 0:
                    continue
                signature_distances: list[float] = []
                if np.linalg.norm(track.prototype_signature) > 1e-6:
                    signature_distances.append(_feature_distance(track.prototype_signature, proposal.signature))
                if np.linalg.norm(track.anchor_signature) > 1e-6:
                    signature_distances.append(_feature_distance(track.anchor_signature, proposal.signature))
                if not signature_distances:
                    continue
                distance = float(min(signature_distances))
                if distance > self.routing_recovery_max_distance:
                    continue
                candidates_by_proposal[int(proposal.proposal_index)].append(
                    {
                        "lineage_id": int(track.lineage_id),
                        "prototype_id": None if track.prototype_id is None else int(track.prototype_id),
                        "source_track_id": int(track.track_id),
                        "source_type": "dormant_or_ghost",
                        "temporal_state": str(track.state),
                        "distance": distance,
                        "candidate_confidence": float(max(0.0, 1.0 - distance)),
                        "restore_eligible": True,
                        "gap_length": int(track.gap_length),
                    }
                )

        if memory_context is not None:
            continuation_lookup = getattr(memory_context, "continuation_lineage_lookup", {})
            recovery_anchor_lookup = getattr(memory_context, "recovery_anchor_lookup", {})
            for proposal in proposals:
                proposal_key = int(proposal.proposal_index)
                for lineage_id, continuations in continuation_lookup.items():
                    for continuation in continuations:
                        distance = float(_feature_distance(continuation.feature_ema, proposal.signature))
                        if distance > self.routing_recovery_max_distance:
                            continue
                        candidates_by_proposal[proposal_key].append(
                            {
                                "lineage_id": int(lineage_id),
                                "prototype_id": int(getattr(continuation, "prototype_id", -1)),
                                "source_track_id": int(getattr(continuation, "track_id", -1)),
                                "source_type": "continuation_bank",
                                "temporal_state": "continuation",
                                "distance": distance,
                                "candidate_confidence": float(max(0.0, 1.0 - distance)),
                                "restore_eligible": True,
                                "gap_length": int(getattr(continuation, "age_since_last_seen", 0)),
                            }
                        )
                for lineage_id, anchors in recovery_anchor_lookup.items():
                    for anchor in anchors:
                        distance = float(_feature_distance(anchor["feature_ema"], proposal.signature))
                        if distance > self.routing_recovery_max_distance:
                            continue
                        candidates_by_proposal[proposal_key].append(
                            {
                                "lineage_id": int(lineage_id),
                                "prototype_id": int(anchor.get("old_prototype_id", -1)),
                                "source_track_id": int(anchor.get("old_track_id", -1)),
                                "source_type": "recovery_anchor",
                                "temporal_state": str(anchor.get("anchor_state", "anchor")),
                                "distance": distance,
                                "candidate_confidence": float(max(0.0, 1.0 - distance)),
                                "restore_eligible": bool(anchor.get("anchor_state", "alive") != "expired"),
                                "gap_length": int(anchor.get("age_since_last_seen", 0)),
                            }
                        )

        for proposal_index, items in candidates_by_proposal.items():
            candidates_by_proposal[proposal_index] = sorted(
                items,
                key=lambda item: (-float(item["candidate_confidence"]), float(item["distance"]), int(item["gap_length"])),
            )
        return candidates_by_proposal

    def _build_active_candidate_rows(
        self,
        *,
        proposal_index: int,
        active_tracks: list[TrackState],
        projected_states: list[TrackState],
        cost_matrix: np.ndarray,
        proposal: ProposalFeature,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        if proposal_index >= cost_matrix.shape[1]:
            return rows
        for track_idx, track in enumerate(active_tracks):
            projected_box, _projected_centroid = projected_states[track_idx]
            match_cost = float(cost_matrix[track_idx, proposal_index])
            rows.append(
                {
                    "track_id": int(track.track_id),
                    "lineage_id": None if track.lineage_id is None else int(track.lineage_id),
                    "prototype_id": None if track.prototype_id is None else int(track.prototype_id),
                    "match_cost": match_cost,
                    "match_confidence": self._active_claim_confidence(match_cost),
                    "iou": float(bbox_iou(projected_box, proposal.box)),
                    "temporal_state": str(track.state),
                }
            )
        return sorted(rows, key=lambda item: float(item["match_cost"]))[: self.routing_topk]

    def _active_claim_confidence(self, match_cost: float) -> float:
        normalized = float(np.clip(match_cost / max(self.max_match_cost, 1e-6), 0.0, 1.0))
        return float(max(0.0, 1.0 - normalized))

    def _arbitrate_active_matches(
        self,
        *,
        active_matches: list[tuple[int, int, float]],
        active_tracks: list[TrackState],
        projected_states: list[TrackState],
        proposal_features: list[ProposalFeature],
        cost_matrix: np.ndarray,
        recovery_candidate_lists: dict[int, list[dict[str, object]]],
        recovery_surface_lookup: dict[int, dict[str, int | bool]],
        frame_index: int,
    ) -> tuple[
        list[tuple[int, int, float]],
        dict[int, dict[str, object]],
        list[dict[str, object]],
    ]:
        active_tracks_by_id = {int(track.track_id): track for track in active_tracks}
        proposal_features_by_id = {int(proposal.proposal_index): proposal for proposal in proposal_features}
        retained_matches: list[tuple[int, int, float]] = []
        routing_meta_by_proposal: dict[int, dict[str, object]] = {}
        routing_rows: list[dict[str, object]] = []
        matched_proposal_lookup = {int(proposal_index): (int(track_id), float(match_cost)) for track_id, proposal_index, match_cost in active_matches}

        for proposal in proposal_features:
            proposal_index = int(proposal.proposal_index)
            active_candidates = self._build_active_candidate_rows(
                proposal_index=proposal_index,
                active_tracks=active_tracks,
                projected_states=projected_states,
                cost_matrix=cost_matrix,
                proposal=proposal,
            )
            recovery_candidates = recovery_candidate_lists.get(proposal_index, [])
            best_active = active_candidates[0] if active_candidates else None
            matched_active = matched_proposal_lookup.get(proposal_index)
            tentative_track_id = None if matched_active is None else int(matched_active[0])
            tentative_cost = None if matched_active is None else float(matched_active[1])
            tentative_track = None if tentative_track_id is None else active_tracks_by_id.get(int(tentative_track_id))
            tentative_lineage = None if tentative_track is None or tentative_track.lineage_id is None else int(tentative_track.lineage_id)
            best_recovery = None
            best_cross_lineage_recovery = None
            for candidate in recovery_candidates:
                candidate_lineage = candidate.get("lineage_id")
                if candidate_lineage is None or int(candidate_lineage) < 0:
                    continue
                surface_row = recovery_surface_lookup.get(int(candidate_lineage), {})
                if not bool(surface_row.get("surface_nonempty", False)):
                    continue
                if best_recovery is None:
                    best_recovery = candidate
                if (
                    tentative_lineage is not None
                    and int(candidate_lineage) != int(tentative_lineage)
                    and best_cross_lineage_recovery is None
                ):
                    best_cross_lineage_recovery = candidate
                if best_recovery is not None and (
                    tentative_lineage is None or best_cross_lineage_recovery is not None
                ):
                    break
            if tentative_lineage is not None and best_cross_lineage_recovery is not None:
                best_recovery = best_cross_lineage_recovery
            best_recovery_lineage = None if best_recovery is None else int(best_recovery["lineage_id"])
            active_claim_confidence = None if tentative_cost is None else self._active_claim_confidence(float(tentative_cost))
            recovery_claim_confidence = (
                None if best_recovery is None else float(best_recovery.get("candidate_confidence", 0.0))
            )
            routing_margin = (
                None
                if active_claim_confidence is None or recovery_claim_confidence is None
                else float(active_claim_confidence - recovery_claim_confidence)
            )
            tentative_source = "new_track"
            cross_lineage_preemption = False
            preemption_reason = "no_active_claim"
            if tentative_track is not None:
                if tentative_lineage is None:
                    tentative_source = "active_match_ambiguous"
                else:
                    tentative_source = "active_match_same_lineage"
                if (
                    best_recovery_lineage is not None
                    and tentative_lineage is not None
                    and int(best_recovery_lineage) != int(tentative_lineage)
                ):
                    tentative_source = "active_match_cross_lineage"
                    cross_lineage_preemption = True
                    preemption_reason = "cross_lineage_recovery_conflict"

            reroute_triggered = False
            was_rerouted = False
            if tentative_track is not None and cross_lineage_preemption:
                forced_target_reroute = bool(
                    self.debug_force_reroute_frame is not None
                    and int(frame_index) == int(self.debug_force_reroute_frame)
                    and (
                        self.debug_force_reroute_lineage is None
                        or (
                            best_recovery_lineage is not None
                            and int(best_recovery_lineage) == int(self.debug_force_reroute_lineage)
                        )
                    )
                )
                forced_visibility = bool(self.debug_force_visibility_for_all_cross_lineage)
                active_overwhelming = bool(
                    active_claim_confidence is not None
                    and recovery_claim_confidence is not None
                    and active_claim_confidence
                    >= recovery_claim_confidence + self.routing_active_claim_override_margin
                )
                reroute_triggered = bool(
                    self.enable_phase3d_routing_repair
                    and (
                        forced_target_reroute
                        or forced_visibility
                        or (
                            recovery_claim_confidence is not None
                            and recovery_claim_confidence >= self.routing_recovery_min_confidence
                            and not active_overwhelming
                        )
                    )
                )
                if forced_target_reroute:
                    preemption_reason = "forced_target_reroute"
                elif forced_visibility:
                    preemption_reason = "forced_cross_lineage_visibility"
                elif reroute_triggered:
                    preemption_reason = "cross_lineage_preemption"
                was_rerouted = bool(reroute_triggered)

            routing_meta_by_proposal[proposal_index] = {
                "tentative_assignment_source": tentative_source,
                "tentative_lineage_id": tentative_lineage,
                "final_assignment_source": "rerouted_to_resurrection" if was_rerouted else tentative_source,
                "final_lineage_id": best_recovery_lineage if was_rerouted else tentative_lineage,
                "was_rerouted": was_rerouted,
                "preempting_active_track_id": tentative_track_id,
                "preempting_active_lineage_id": tentative_lineage,
                "best_recovery_lineage_id": best_recovery_lineage,
                "best_recovery_source_type": None if best_recovery is None else str(best_recovery.get("source_type")),
                "best_recovery_distance": None if best_recovery is None else float(best_recovery.get("distance", 1.0)),
                "routing_arbitration_triggered": reroute_triggered,
                "cross_lineage_preemption_flag": cross_lineage_preemption,
                "preemption_reason": preemption_reason,
                "active_claim_confidence": active_claim_confidence,
                "recovery_claim_confidence": recovery_claim_confidence,
                "routing_margin": routing_margin,
                "proposal_box": tuple(int(value) for value in proposal.box),
                "proposal_score": float(proposal.score),
                "proposal_proto_hint": None if best_recovery is None else best_recovery.get("prototype_id"),
                "proposal_lineage_hint_topk": [
                    {
                        "lineage_id": None if item.get("lineage_id") is None else int(item["lineage_id"]),
                        "source_type": str(item.get("source_type", "")),
                        "candidate_confidence": float(item.get("candidate_confidence", 0.0)),
                        "distance": float(item.get("distance", 1.0)),
                    }
                    for item in recovery_candidates[: self.routing_topk]
                ],
                "active_candidates_topk": active_candidates,
                "recovery_candidates_topk": recovery_candidates[: self.routing_topk],
            }
            routing_rows.append(
                {
                    "frame_id": int(frame_index),
                    "proposal_id": int(proposal_index),
                    **routing_meta_by_proposal[proposal_index],
                }
            )

        for track_id, proposal_index, match_cost in active_matches:
            meta = routing_meta_by_proposal.get(int(proposal_index), {})
            if bool(meta.get("was_rerouted", False)):
                continue
            retained_matches.append((track_id, proposal_index, match_cost))
        return retained_matches, routing_meta_by_proposal, routing_rows

    def _recovery_lineage_consistency_score(
        self,
        *,
        candidate_lineage_id: int,
        tracking_assignment: TrackAssignment,
        prototype_assignment,
    ) -> float:
        score = 0.0
        for hint_lineage in (
            tracking_assignment.best_recovery_lineage_id,
            tracking_assignment.prototype_hint_lineage_id,
            tracking_assignment.linked_lineage_id,
            tracking_assignment.pre_memory_linked_lineage_id,
            getattr(prototype_assignment, "matched_lineage_id", None),
            getattr(prototype_assignment, "lineage_id", None),
        ):
            if hint_lineage is None or int(hint_lineage) < 0:
                continue
            if int(candidate_lineage_id) == int(hint_lineage):
                if hint_lineage == tracking_assignment.best_recovery_lineage_id:
                    score += 0.40
                elif hint_lineage == getattr(prototype_assignment, "matched_lineage_id", None):
                    score += 0.25
                else:
                    score += 0.15
        return float(np.clip(score, 0.0, 1.0))

    def _score_recovery_geometry(
        self,
        *,
        proposal_track: TrackState,
        last_center: tuple[float, float],
        velocity: np.ndarray,
        gap: int,
        frame_diagonal: float,
    ) -> float:
        predicted_center = np.asarray(
            (
                float(last_center[0] + velocity[0] * max(1, gap)),
                float(last_center[1] + velocity[1] * max(1, gap)),
            ),
            dtype=np.float32,
        )
        proposal_center = np.asarray(proposal_track.centroid, dtype=np.float32)
        distance = float(np.linalg.norm(predicted_center - proposal_center) / max(frame_diagonal, 1e-6))
        return float(np.clip(1.0 - distance, 0.0, 1.0))

    def _phase3d_runtime_hint_lineage_ids(
        self,
        *,
        tracking_assignment: TrackAssignment,
        prototype_assignment,
    ) -> list[int]:
        hint_ids: set[int] = set()
        for candidate in (
            getattr(prototype_assignment, "matched_lineage_id", None),
            getattr(prototype_assignment, "lineage_id", None),
            tracking_assignment.best_recovery_lineage_id,
            tracking_assignment.linked_lineage_id,
            tracking_assignment.final_lineage_id,
            tracking_assignment.tentative_lineage_id,
            tracking_assignment.pre_memory_linked_lineage_id,
        ):
            if candidate is None:
                continue
            try:
                lineage_id = int(candidate)
            except (TypeError, ValueError):
                continue
            if lineage_id >= 0:
                hint_ids.add(lineage_id)
        return sorted(hint_ids)

    def _phase3d_hint_lineage_ids(
        self,
        *,
        tracking_assignment: TrackAssignment,
        prototype_assignment,
    ) -> list[int]:
        return self._phase3d_runtime_hint_lineage_ids(
            tracking_assignment=tracking_assignment,
            prototype_assignment=prototype_assignment,
        )

    def _phase3d_recovery_surface_summary(
        self,
        *,
        proposal_track: TrackState,
        memory_output,
    ) -> dict[int, dict[str, object]]:
        summary: dict[int, dict[str, object]] = {}

        def _row(lineage_id: int) -> dict[str, object]:
            lineage_id = int(lineage_id)
            if lineage_id not in summary:
                summary[lineage_id] = {
                    "lineage_id": lineage_id,
                    "active_count": 0,
                    "dormant_count": 0,
                    "ghost_count": 0,
                    "continuation_bank_size": 0,
                    "recovery_anchor_count": 0,
                    "runtime_owner_continuation_bank_size": 0,
                    "continuity_owner_continuation_bank_size": 0,
                    "runtime_owner_recovery_anchor_count": 0,
                    "continuity_owner_recovery_anchor_count": 0,
                    "has_temp_attach": 0,
                }
            return summary[lineage_id]

        for track in self._tracks.values():
            if int(track.track_id) == int(proposal_track.track_id):
                continue
            if track.lineage_id is None or int(track.lineage_id) < 0:
                continue
            row = _row(int(track.lineage_id))
            if track.active:
                row["active_count"] = int(row["active_count"]) + 1
            if track.dormant:
                row["dormant_count"] = int(row["dormant_count"]) + 1
            if track.ghost:
                row["ghost_count"] = int(row["ghost_count"]) + 1

        for lineage_id, continuations in getattr(memory_output, "continuation_lineage_lookup", {}).items():
            row = _row(int(lineage_id))
            row["runtime_owner_continuation_bank_size"] = int(len(continuations))
            row["continuation_bank_size"] = max(
                int(row["continuation_bank_size"]),
                int(len(continuations)),
            )

        for lineage_id, continuations in getattr(memory_output, "continuation_continuity_lookup", {}).items():
            row = _row(int(lineage_id))
            row["continuity_owner_continuation_bank_size"] = int(len(continuations))
            row["continuation_bank_size"] = max(
                int(row["continuation_bank_size"]),
                int(len(continuations)),
            )

        for lineage_id, anchors in getattr(memory_output, "recovery_anchor_lookup", {}).items():
            row = _row(int(lineage_id))
            row["runtime_owner_recovery_anchor_count"] = int(len(anchors))
            row["recovery_anchor_count"] = max(
                int(row["recovery_anchor_count"]),
                int(len(anchors)),
            )

        for lineage_id, anchors in getattr(memory_output, "recovery_anchor_continuity_lookup", {}).items():
            row = _row(int(lineage_id))
            row["continuity_owner_recovery_anchor_count"] = int(len(anchors))
            row["recovery_anchor_count"] = max(
                int(row["recovery_anchor_count"]),
                int(len(anchors)),
            )

        for lineage_id, temp_slot in getattr(memory_output, "temp_attach_lookup", {}).items():
            if not temp_slot or bool(temp_slot.get("expired", False)):
                continue
            row = _row(int(lineage_id))
            row["has_temp_attach"] = 1

        for row in summary.values():
            active_count = int(row["active_count"])
            dormant_count = int(row["dormant_count"])
            ghost_count = int(row["ghost_count"])
            continuation_bank_size = int(row["continuation_bank_size"])
            recovery_anchor_count = int(row["recovery_anchor_count"])
            has_temp_attach = int(row["has_temp_attach"])
            row["has_active_surface"] = int(active_count > 0)
            row["has_dormant_surface"] = int(dormant_count > 0)
            row["has_ghost_surface"] = int(ghost_count > 0)
            row["has_continuation_bank"] = int(continuation_bank_size > 0)
            row["has_recovery_anchor"] = int(recovery_anchor_count > 0)
            row["has_runtime_owner_continuation_bank"] = int(
                int(row["runtime_owner_continuation_bank_size"]) > 0
            )
            row["has_continuity_owner_continuation_bank"] = int(
                int(row["continuity_owner_continuation_bank_size"]) > 0
            )
            row["has_runtime_owner_recovery_anchor"] = int(
                int(row["runtime_owner_recovery_anchor_count"]) > 0
            )
            row["has_continuity_owner_recovery_anchor"] = int(
                int(row["continuity_owner_recovery_anchor_count"]) > 0
            )
            row["surface_nonempty"] = int(
                active_count > 0
                or dormant_count > 0
                or ghost_count > 0
                or continuation_bank_size > 0
                or recovery_anchor_count > 0
                or has_temp_attach > 0
            )
            row["legal_recovery_surface"] = int(
                dormant_count > 0
                or ghost_count > 0
                or continuation_bank_size > 0
                or recovery_anchor_count > 0
            )
            row["surface_hint_score"] = float(
                np.clip(
                    0.15 * int(active_count > 0)
                    + 0.20 * int(dormant_count > 0)
                    + 0.15 * int(ghost_count > 0)
                    + 0.25 * int(continuation_bank_size > 0)
                    + 0.20 * int(recovery_anchor_count > 0)
                    + 0.05 * has_temp_attach,
                    0.0,
                    1.0,
                )
            )
        return summary

    def _phase3d_continuity_lineage_summary(
        self,
        *,
        candidate_rows: list[dict[str, object]],
        surface_summary: dict[int, dict[str, object]],
    ) -> dict[int, dict[str, object]]:
        summary: dict[int, dict[str, object]] = {}

        def _row(lineage_id: int) -> dict[str, object]:
            lineage_id = int(lineage_id)
            if lineage_id not in summary:
                summary[lineage_id] = {
                    "lineage_id": lineage_id,
                    "continuation_identity_hint_score": 0.0,
                    "anchor_identity_hint_score": 0.0,
                    "old_identity_ref_count": 0,
                    "same_track_hint": 0,
                    "same_prototype_hint": 0,
                    "continuity_lineage_hint_score": 0.0,
                    "continuity_candidate_eligible": 0,
                    "enumerated_candidate_count": 0,
                }
            return summary[lineage_id]

        for candidate in candidate_rows:
            lineage_id = int(candidate["candidate_lineage_id"])
            row = _row(lineage_id)
            row["enumerated_candidate_count"] = int(row["enumerated_candidate_count"]) + 1
            source_type = str(candidate["source_type"])
            if source_type == "continuation_bank":
                row["continuation_identity_hint_score"] = max(
                    float(row["continuation_identity_hint_score"]),
                    float(candidate["recovery_score_total"]),
                )
            elif source_type == "recovery_anchor":
                row["anchor_identity_hint_score"] = max(
                    float(row["anchor_identity_hint_score"]),
                    float(candidate["recovery_score_total"]),
                )
            row["old_identity_ref_count"] = int(row["old_identity_ref_count"]) + int(
                candidate.get("old_identity_ref_valid", 0)
            )
            row["same_track_hint"] = max(int(row["same_track_hint"]), int(candidate.get("same_track_hint", 0)))
            row["same_prototype_hint"] = max(
                int(row["same_prototype_hint"]),
                int(candidate.get("same_prototype_hint", 0)),
            )

        for lineage_id in set(surface_summary.keys()) | set(summary.keys()):
            row = _row(int(lineage_id))
            surface_row = surface_summary.get(int(lineage_id), {})
            continuity_score = float(
                np.clip(
                    0.35 * float(row["continuation_identity_hint_score"])
                    + 0.35 * float(row["anchor_identity_hint_score"])
                    + 0.10 * int(row["same_track_hint"])
                    + 0.10 * int(row["same_prototype_hint"])
                    + 0.10 * min(1.0, float(row["old_identity_ref_count"]) / 3.0),
                    0.0,
                    1.0,
                )
            )
            row["continuity_lineage_hint_score"] = continuity_score
            row["continuity_candidate_eligible"] = int(
                bool(surface_row.get("legal_recovery_surface", 0))
                and (
                    float(row["continuation_identity_hint_score"]) >= self.continuity_hint_min_score
                    or float(row["anchor_identity_hint_score"]) >= self.continuity_hint_min_score
                    or int(row["old_identity_ref_count"]) > 0
                    or int(row["same_track_hint"]) == 1
                    or int(row["same_prototype_hint"]) == 1
                )
            )
        return summary

    def _phase3d_preserve_input_trace(
        self,
        *,
        proposal_track: TrackState,
        tracking_assignment: TrackAssignment,
        prototype_assignment,
        memory_output,
        candidate_rows: list[dict[str, object]],
    ) -> tuple[list[dict[str, object]], list[int], list[int]]:
        runtime_hint_lineages = self._phase3d_runtime_hint_lineage_ids(
            tracking_assignment=tracking_assignment,
            prototype_assignment=prototype_assignment,
        )
        surface_summary = self._phase3d_recovery_surface_summary(
            proposal_track=proposal_track,
            memory_output=memory_output,
        )
        continuity_summary = self._phase3d_continuity_lineage_summary(
            candidate_rows=candidate_rows,
            surface_summary=surface_summary,
        )

        before_prune: set[int] = set(int(lineage_id) for lineage_id in runtime_hint_lineages)
        if self.enable_phase3d_continuity_lineage_repair and bool(tracking_assignment.was_rerouted):
            before_prune.update(
                int(lineage_id)
                for lineage_id, row in continuity_summary.items()
                if int(row.get("continuity_candidate_eligible", 0)) == 1
            )
        if self.enable_phase3d_three_source_preserve_input and bool(tracking_assignment.was_rerouted):
            before_prune.update(
                int(lineage_id)
                for lineage_id, row in surface_summary.items()
                if int(row.get("surface_nonempty", 0)) == 1
            )

        after_prune: set[int] = set()
        for lineage_id in before_prune:
            surface_row = surface_summary.get(int(lineage_id), {})
            continuity_row = continuity_summary.get(int(lineage_id), {})
            if int(lineage_id) in runtime_hint_lineages:
                after_prune.add(int(lineage_id))
            elif int(continuity_row.get("continuity_candidate_eligible", 0)) == 1:
                after_prune.add(int(lineage_id))
            elif self.enable_phase3d_three_source_preserve_input and int(surface_row.get("surface_nonempty", 0)) == 1:
                after_prune.add(int(lineage_id))

        trace_lineages = set(runtime_hint_lineages)
        trace_lineages.update(int(lineage_id) for lineage_id in surface_summary.keys())
        trace_lineages.update(int(lineage_id) for lineage_id in continuity_summary.keys())
        trace_lineages.update(int(lineage_id) for lineage_id in before_prune)
        trace_lineages.update(int(lineage_id) for lineage_id in after_prune)
        before_text = "|".join(str(lineage_id) for lineage_id in sorted(before_prune))
        after_text = "|".join(str(lineage_id) for lineage_id in sorted(after_prune))
        runtime_text = "|".join(str(lineage_id) for lineage_id in runtime_hint_lineages)
        rows: list[dict[str, object]] = []
        for lineage_id in sorted(trace_lineages):
            surface_row = surface_summary.get(
                int(lineage_id),
                {
                    "active_count": 0,
                    "dormant_count": 0,
                    "ghost_count": 0,
                    "continuation_bank_size": 0,
                    "recovery_anchor_count": 0,
                    "runtime_owner_continuation_bank_size": 0,
                    "continuity_owner_continuation_bank_size": 0,
                    "runtime_owner_recovery_anchor_count": 0,
                    "continuity_owner_recovery_anchor_count": 0,
                    "has_temp_attach": 0,
                    "has_active_surface": 0,
                    "has_dormant_surface": 0,
                    "has_ghost_surface": 0,
                    "has_continuation_bank": 0,
                    "has_recovery_anchor": 0,
                    "has_runtime_owner_continuation_bank": 0,
                    "has_continuity_owner_continuation_bank": 0,
                    "has_runtime_owner_recovery_anchor": 0,
                    "has_continuity_owner_recovery_anchor": 0,
                    "surface_nonempty": 0,
                    "legal_recovery_surface": 0,
                    "surface_hint_score": 0.0,
                },
            )
            continuity_row = continuity_summary.get(
                int(lineage_id),
                {
                    "continuation_identity_hint_score": 0.0,
                    "anchor_identity_hint_score": 0.0,
                    "old_identity_ref_count": 0,
                    "same_track_hint": 0,
                    "same_prototype_hint": 0,
                    "continuity_lineage_hint_score": 0.0,
                    "continuity_candidate_eligible": 0,
                    "enumerated_candidate_count": 0,
                },
            )
            entered_before = int(int(lineage_id) in before_prune)
            entered_after = int(int(lineage_id) in after_prune)
            valid_continuity_dropped = int(
                int(surface_row.get("legal_recovery_surface", 0)) == 1
                and int(continuity_row.get("continuity_candidate_eligible", 0)) == 1
                and entered_before == 0
            )
            if valid_continuity_dropped:
                drop_stage = "input_formation"
                drop_reason = "valid_continuity_lineage_dropped_before_claim"
            elif entered_before == 1 and entered_after == 0:
                drop_stage = "preserve_prune"
                drop_reason = "entered_before_prune_but_removed_after_prune"
            elif entered_after == 1:
                drop_stage = "claim_builder_entry"
                drop_reason = "preserve_input_exposed_to_claim_builder"
            else:
                drop_stage = "not_eligible"
                drop_reason = "no_runtime_or_continuity_preserve_signal"

            rows.append(
                {
                    "candidate_lineage_id": int(lineage_id),
                    "proposal_proto_hint": tracking_assignment.prototype_hint_id,
                    "proposal_lineage_hint_topk": runtime_text,
                    "tentative_active_lineage": tracking_assignment.tentative_lineage_id,
                    "preempting_active_track_id": tracking_assignment.preempting_active_track_id,
                    "preemption_lineage_hint_score": None
                    if tracking_assignment.active_claim_confidence is None
                    else float(tracking_assignment.active_claim_confidence),
                    "runtime_hint_match": int(int(lineage_id) in runtime_hint_lineages),
                    "has_active_surface": int(surface_row.get("has_active_surface", 0)),
                    "has_dormant_surface": int(surface_row.get("has_dormant_surface", 0)),
                    "has_ghost_surface": int(surface_row.get("has_ghost_surface", 0)),
                    "has_continuation_bank": int(surface_row.get("has_continuation_bank", 0)),
                    "has_recovery_anchor": int(surface_row.get("has_recovery_anchor", 0)),
                    "runtime_owner_continuation_bank_size": int(
                        surface_row.get("runtime_owner_continuation_bank_size", 0)
                    ),
                    "continuity_owner_continuation_bank_size": int(
                        surface_row.get("continuity_owner_continuation_bank_size", 0)
                    ),
                    "runtime_owner_recovery_anchor_count": int(
                        surface_row.get("runtime_owner_recovery_anchor_count", 0)
                    ),
                    "continuity_owner_recovery_anchor_count": int(
                        surface_row.get("continuity_owner_recovery_anchor_count", 0)
                    ),
                    "has_runtime_owner_continuation_bank": int(
                        surface_row.get("has_runtime_owner_continuation_bank", 0)
                    ),
                    "has_continuity_owner_continuation_bank": int(
                        surface_row.get("has_continuity_owner_continuation_bank", 0)
                    ),
                    "has_runtime_owner_recovery_anchor": int(
                        surface_row.get("has_runtime_owner_recovery_anchor", 0)
                    ),
                    "has_continuity_owner_recovery_anchor": int(
                        surface_row.get("has_continuity_owner_recovery_anchor", 0)
                    ),
                    "has_temp_attach": int(surface_row.get("has_temp_attach", 0)),
                    "surface_hint_score": float(surface_row.get("surface_hint_score", 0.0)),
                    "continuation_identity_hint_score": float(
                        continuity_row.get("continuation_identity_hint_score", 0.0)
                    ),
                    "anchor_identity_hint_score": float(
                        continuity_row.get("anchor_identity_hint_score", 0.0)
                    ),
                    "old_identity_ref_count": int(continuity_row.get("old_identity_ref_count", 0)),
                    "same_track_hint": int(continuity_row.get("same_track_hint", 0)),
                    "same_prototype_hint": int(continuity_row.get("same_prototype_hint", 0)),
                    "continuity_lineage_hint_score": float(
                        continuity_row.get("continuity_lineage_hint_score", 0.0)
                    ),
                    "continuity_candidate_eligible": int(
                        continuity_row.get("continuity_candidate_eligible", 0)
                    ),
                    "preserve_candidate_lineages_before_prune": before_text,
                    "preserve_candidate_lineages_after_prune": after_text,
                    "entered_preserve_input": entered_before,
                    "entered_claim_builder_input": entered_after,
                    "preserve_input_drop_stage": drop_stage,
                    "preserve_input_drop_reason": drop_reason,
                    "valid_continuity_lineage_dropped_before_claim": valid_continuity_dropped,
                }
            )
        return rows, sorted(int(lineage_id) for lineage_id in before_prune), sorted(
            int(lineage_id) for lineage_id in after_prune
        )

    def _build_recovery_track_candidate_row(
        self,
        *,
        proposal_track: TrackState,
        tracking_assignment: TrackAssignment,
        prototype_assignment,
        candidate_track: TrackState,
        frame_index: int,
        frame_shape: tuple[int, int],
        frame_diagonal: float,
    ) -> dict[str, object]:
        runtime_owner_lineage_id = (
            None
            if candidate_track.lineage_id is None or int(candidate_track.lineage_id) < 0
            else int(candidate_track.lineage_id)
        )
        continuity_owner_lineage_id = runtime_owner_lineage_id
        gap = max(1, int(candidate_track.gap_length))
        raw_cost = float(
            self._concept_gated_resurrection_cost(
                proposal_track=proposal_track,
                candidate_track=candidate_track,
                frame_index=frame_index,
                frame_shape=frame_shape,
                frame_diagonal=frame_diagonal,
            )
        )
        gap_decay = float(np.exp(-gap / self.tau_g))
        threshold = float(self.tau_res_short + (1.0 - gap_decay) * (self.tau_res_long - self.tau_res_short))
        geometry_score = self._score_recovery_geometry(
            proposal_track=proposal_track,
            last_center=candidate_track.last_center,
            velocity=candidate_track.velocity,
            gap=gap,
            frame_diagonal=frame_diagonal,
        )
        signature = (
            candidate_track.prototype_signature
            if np.linalg.norm(candidate_track.prototype_signature) > 1e-6
            else candidate_track.anchor_signature
        )
        prototype_similarity_score = float(_feature_similarity(signature, proposal_track.signature))
        lineage_consistency_score = self._recovery_lineage_consistency_score(
            candidate_lineage_id=int(candidate_track.lineage_id),
            tracking_assignment=tracking_assignment,
            prototype_assignment=prototype_assignment,
        )
        temporal_gap_score = float(np.exp(-gap / max(self.tau_g, 1e-6)))
        recency_score = float(1.0 / (1.0 + gap / max(self.tau_g, 1.0)))
        recovery_score_total = float(
            np.clip(
                0.30 * geometry_score
                + 0.25 * prototype_similarity_score
                + 0.15 * lineage_consistency_score
                + 0.15 * temporal_gap_score
                + 0.15 * recency_score,
                0.0,
                1.0,
            )
        )
        candidate_prototype_id = None if candidate_track.prototype_id is None else int(candidate_track.prototype_id)
        return {
            "_candidate_obj": candidate_track,
            "candidate_lineage_id": int(candidate_track.lineage_id),
            "source_type": str(candidate_track.state),
            "source_kind": "track_state",
            "source_runtime_owner_id": runtime_owner_lineage_id,
            "source_continuity_owner_id": continuity_owner_lineage_id,
            "source_owner_mode": "runtime_owner",
            "candidate_track_id": int(candidate_track.track_id),
            "candidate_prototype_id": candidate_prototype_id,
            "candidate_state": str(candidate_track.state),
            "raw_cost": raw_cost,
            "accept_threshold": threshold,
            "raw_visibility_score": recovery_score_total,
            "filtered_flag": 0,
            "filter_stage": "",
            "filter_reason": "",
            "geometry_score": geometry_score,
            "prototype_similarity_score": prototype_similarity_score,
            "continuation_score": 0.0,
            "anchor_identity_score": 0.0,
            "lineage_consistency_score": lineage_consistency_score,
            "temporal_gap_score": temporal_gap_score,
            "recency_score": recency_score,
            "recovery_score_total": recovery_score_total,
            "restore_eligibility": bool(raw_cost <= threshold),
            "same_prototype_hint": int(
                candidate_prototype_id is not None
                and tracking_assignment.prototype_hint_id is not None
                and int(candidate_prototype_id) == int(tracking_assignment.prototype_hint_id)
            ),
            "same_track_hint": 0,
            "old_identity_ref_valid": 1,
        }

    def _build_recovery_continuation_candidate_row(
        self,
        *,
        proposal_track: TrackState,
        tracking_assignment: TrackAssignment,
        prototype_assignment,
        continuation,
        lineage_id: int,
        candidate_lineage_id: int | None = None,
        source_owner_mode: str = "runtime_owner",
        frame_index: int,
        frame_shape: tuple[int, int],
        frame_diagonal: float,
    ) -> dict[str, object]:
        runtime_owner_lineage_id = getattr(continuation, "runtime_owner_lineage_id", None)
        continuity_owner_lineage_id = getattr(continuation, "continuity_lineage_id", None)
        if runtime_owner_lineage_id is None:
            runtime_owner_lineage_id = int(lineage_id)
        if continuity_owner_lineage_id is None:
            continuity_owner_lineage_id = int(lineage_id)
        if candidate_lineage_id is None:
            candidate_lineage_id = int(lineage_id)
        gap = max(1, int(frame_index - continuation.last_seen_frame))
        raw_cost = float(
            self._continuation_resurrection_cost(
                continuation=continuation,
                proposal_track=proposal_track,
                frame_index=frame_index,
                frame_shape=frame_shape,
                frame_diagonal=frame_diagonal,
            )
        )
        geometry_score = self._score_recovery_geometry(
            proposal_track=proposal_track,
            last_center=continuation.last_center,
            velocity=continuation.velocity,
            gap=gap,
            frame_diagonal=frame_diagonal,
        )
        prototype_similarity_score = float(_feature_similarity(continuation.feature_ema, proposal_track.signature))
        lineage_consistency_score = self._recovery_lineage_consistency_score(
            candidate_lineage_id=int(candidate_lineage_id),
            tracking_assignment=tracking_assignment,
            prototype_assignment=prototype_assignment,
        )
        temporal_gap_score = float(np.exp(-gap / max(self.tau_g, 1e-6)))
        recency_score = float(1.0 / (1.0 + gap / max(self.tau_g, 1.0)))
        recovery_score_total = float(
            np.clip(
                0.20 * geometry_score
                + 0.25 * prototype_similarity_score
                + 0.15 * lineage_consistency_score
                + 0.15 * temporal_gap_score
                + 0.10 * recency_score
                + 0.15,
                0.0,
                1.0,
            )
        )
        candidate_prototype_id = int(getattr(continuation, "prototype_id", -1))
        return {
            "_candidate_obj": continuation,
            "candidate_lineage_id": int(candidate_lineage_id),
            "source_type": "continuation_bank",
            "source_kind": "continuation_bank",
            "source_runtime_owner_id": None
            if runtime_owner_lineage_id is None
            else int(runtime_owner_lineage_id),
            "source_continuity_owner_id": None
            if continuity_owner_lineage_id is None
            else int(continuity_owner_lineage_id),
            "source_owner_mode": str(source_owner_mode),
            "candidate_track_id": int(getattr(continuation, "track_id", -1)),
            "candidate_prototype_id": candidate_prototype_id,
            "candidate_state": "continuation",
            "raw_cost": raw_cost,
            "accept_threshold": float(self.tau_continuation),
            "raw_visibility_score": recovery_score_total,
            "filtered_flag": 0,
            "filter_stage": "",
            "filter_reason": "",
            "geometry_score": geometry_score,
            "prototype_similarity_score": prototype_similarity_score,
            "continuation_score": 1.0,
            "anchor_identity_score": 0.0,
            "lineage_consistency_score": lineage_consistency_score,
            "temporal_gap_score": temporal_gap_score,
            "recency_score": recency_score,
            "recovery_score_total": recovery_score_total,
            "restore_eligibility": True,
            "same_prototype_hint": int(
                candidate_prototype_id >= 0
                and tracking_assignment.prototype_hint_id is not None
                and int(candidate_prototype_id) == int(tracking_assignment.prototype_hint_id)
            ),
            "same_track_hint": 0,
            "old_identity_ref_valid": int(int(getattr(continuation, "track_id", -1)) >= 0),
        }

    def _build_recovery_anchor_candidate_row(
        self,
        *,
        proposal_track: TrackState,
        tracking_assignment: TrackAssignment,
        prototype_assignment,
        anchor: dict[str, object],
        lineage_id: int,
        candidate_lineage_id: int | None = None,
        source_owner_mode: str = "runtime_owner",
        frame_index: int,
        frame_diagonal: float,
    ) -> dict[str, object]:
        runtime_owner_lineage_id = anchor.get("runtime_owner_lineage_id")
        continuity_owner_lineage_id = anchor.get("continuity_lineage_id")
        if runtime_owner_lineage_id is None:
            runtime_owner_lineage_id = int(lineage_id)
        if continuity_owner_lineage_id is None:
            continuity_owner_lineage_id = int(anchor.get("old_lineage_id", lineage_id))
        if candidate_lineage_id is None:
            candidate_lineage_id = int(lineage_id)
        gap = max(1, int(anchor.get("age_since_last_seen", 0)))
        raw_cost = float(
            self._recovery_anchor_cost(
                anchor=anchor,
                proposal_track=proposal_track,
                frame_index=frame_index,
                frame_diagonal=frame_diagonal,
            )
        )
        geometry_score = self._score_recovery_geometry(
            proposal_track=proposal_track,
            last_center=tuple(anchor.get("last_center", proposal_track.centroid)),
            velocity=np.asarray(anchor.get("velocity", np.zeros(2, dtype=np.float32)), dtype=np.float32),
            gap=gap,
            frame_diagonal=frame_diagonal,
        )
        feature_ema = np.asarray(anchor.get("feature_ema", proposal_track.signature), dtype=np.float32)
        prototype_similarity_score = float(_feature_similarity(feature_ema, proposal_track.signature))
        lineage_consistency_score = self._recovery_lineage_consistency_score(
            candidate_lineage_id=int(candidate_lineage_id),
            tracking_assignment=tracking_assignment,
            prototype_assignment=prototype_assignment,
        )
        temporal_gap_score = float(np.exp(-gap / max(self.tau_g, 1e-6)))
        recency_score = float(1.0 / (1.0 + gap / max(self.tau_g, 1.0)))
        recovery_score_total = float(
            np.clip(
                0.18 * geometry_score
                + 0.22 * prototype_similarity_score
                + 0.15 * lineage_consistency_score
                + 0.15 * temporal_gap_score
                + 0.10 * recency_score
                + 0.20,
                0.0,
                1.0,
            )
        )
        candidate_prototype_id = int(anchor.get("old_prototype_id", -1))
        return {
            "_candidate_obj": anchor,
            "candidate_lineage_id": int(candidate_lineage_id),
            "source_type": "recovery_anchor",
            "source_kind": "recovery_anchor",
            "source_runtime_owner_id": None
            if runtime_owner_lineage_id is None
            else int(runtime_owner_lineage_id),
            "source_continuity_owner_id": None
            if continuity_owner_lineage_id is None
            else int(continuity_owner_lineage_id),
            "source_owner_mode": str(source_owner_mode),
            "candidate_track_id": int(anchor.get("old_track_id", -1)),
            "candidate_prototype_id": candidate_prototype_id,
            "candidate_state": str(anchor.get("anchor_state", "alive")),
            "raw_cost": raw_cost,
            "accept_threshold": float(self.tau_continuation),
            "raw_visibility_score": recovery_score_total,
            "filtered_flag": 0,
            "filter_stage": "",
            "filter_reason": "",
            "geometry_score": geometry_score,
            "prototype_similarity_score": prototype_similarity_score,
            "continuation_score": 0.0,
            "anchor_identity_score": 1.0,
            "lineage_consistency_score": lineage_consistency_score,
            "temporal_gap_score": temporal_gap_score,
            "recency_score": recency_score,
            "recovery_score_total": recovery_score_total,
            "restore_eligibility": bool(anchor.get("anchor_state", "alive") != "expired"),
            "same_prototype_hint": int(
                candidate_prototype_id >= 0
                and tracking_assignment.prototype_hint_id is not None
                and int(candidate_prototype_id) == int(tracking_assignment.prototype_hint_id)
            ),
            "same_track_hint": 0,
            "old_identity_ref_valid": int(anchor.get("old_track_id", -1) is not None),
        }

    def _append_preserved_lineage_candidates(
        self,
        *,
        rows: list[dict[str, object]],
        seen_keys: set[tuple[str, int, int]],
        preserved_lineage_ids: list[int],
        proposal_track: TrackState,
        tracking_assignment: TrackAssignment,
        prototype_assignment,
        dormant_ghost_tracks: list[TrackState],
        continuation_lineage_lookup: dict[int, list[object]],
        continuation_continuity_lookup: dict[int, list[object]],
        recovery_anchor_lookup: dict[int, list[dict[str, object]]],
        recovery_anchor_continuity_lookup: dict[int, list[dict[str, object]]],
        frame_index: int,
        frame_shape: tuple[int, int],
        frame_diagonal: float,
    ) -> None:
        if not preserved_lineage_ids:
            return
        for lineage_id in preserved_lineage_ids:
            existing_lineage_rows = [
                row for row in rows if int(row["candidate_lineage_id"]) == int(lineage_id)
            ]
            if any(
                bool(row["restore_eligibility"]) and float(row["recovery_score_total"]) >= self.claim_preserve_min_score
                for row in existing_lineage_rows
            ):
                continue

            for candidate_track in dormant_ghost_tracks:
                if candidate_track.lineage_id is None or int(candidate_track.lineage_id) != int(lineage_id):
                    continue
                key = (str(candidate_track.state), int(candidate_track.track_id), int(lineage_id))
                if key in seen_keys:
                    continue
                row = self._build_recovery_track_candidate_row(
                    proposal_track=proposal_track,
                    tracking_assignment=tracking_assignment,
                    prototype_assignment=prototype_assignment,
                    candidate_track=candidate_track,
                    frame_index=frame_index,
                    frame_shape=frame_shape,
                    frame_diagonal=frame_diagonal,
                )
                seen_keys.add(key)
                rows.append(row)

            for continuation in continuation_lineage_lookup.get(int(lineage_id), []):
                key = ("continuation_bank", int(getattr(continuation, "track_id", -1)), int(lineage_id))
                if key in seen_keys:
                    continue
                row = self._build_recovery_continuation_candidate_row(
                    proposal_track=proposal_track,
                    tracking_assignment=tracking_assignment,
                    prototype_assignment=prototype_assignment,
                    continuation=continuation,
                    lineage_id=int(lineage_id),
                    candidate_lineage_id=int(lineage_id),
                    source_owner_mode="runtime_owner",
                    frame_index=frame_index,
                    frame_shape=frame_shape,
                    frame_diagonal=frame_diagonal,
                )
                seen_keys.add(key)
                rows.append(row)

            if self.enable_phasea_dual_owner_source_enumeration:
                for continuation in continuation_continuity_lookup.get(int(lineage_id), []):
                    key = ("continuation_bank_continuity", int(getattr(continuation, "track_id", -1)), int(lineage_id))
                    if key in seen_keys:
                        continue
                    row = self._build_recovery_continuation_candidate_row(
                        proposal_track=proposal_track,
                        tracking_assignment=tracking_assignment,
                        prototype_assignment=prototype_assignment,
                        continuation=continuation,
                        lineage_id=int(lineage_id),
                        candidate_lineage_id=int(lineage_id),
                        source_owner_mode="continuity_owner",
                        frame_index=frame_index,
                        frame_shape=frame_shape,
                        frame_diagonal=frame_diagonal,
                    )
                    seen_keys.add(key)
                    rows.append(row)

            for anchor in recovery_anchor_lookup.get(int(lineage_id), []):
                key = ("recovery_anchor", int(anchor.get("old_track_id", -1)), int(lineage_id))
                if key in seen_keys:
                    continue
                row = self._build_recovery_anchor_candidate_row(
                    proposal_track=proposal_track,
                    tracking_assignment=tracking_assignment,
                    prototype_assignment=prototype_assignment,
                    anchor=anchor,
                    lineage_id=int(lineage_id),
                    candidate_lineage_id=int(lineage_id),
                    source_owner_mode="runtime_owner",
                    frame_index=frame_index,
                    frame_diagonal=frame_diagonal,
                )
                seen_keys.add(key)
                rows.append(row)

            if self.enable_phasea_dual_owner_source_enumeration:
                for anchor in recovery_anchor_continuity_lookup.get(int(lineage_id), []):
                    key = ("recovery_anchor_continuity", int(anchor.get("old_track_id", -1)), int(lineage_id))
                    if key in seen_keys:
                        continue
                    row = self._build_recovery_anchor_candidate_row(
                        proposal_track=proposal_track,
                        tracking_assignment=tracking_assignment,
                        prototype_assignment=prototype_assignment,
                        anchor=anchor,
                        lineage_id=int(lineage_id),
                        candidate_lineage_id=int(lineage_id),
                        source_owner_mode="continuity_owner",
                        frame_index=frame_index,
                        frame_diagonal=frame_diagonal,
                    )
                    seen_keys.add(key)
                    rows.append(row)

    def _build_stagea4_candidate_rows(
        self,
        *,
        proposal_track: TrackState,
        tracking_assignment: TrackAssignment,
        prototype_assignment,
        tracking_output,
        memory_output,
        frame_index: int,
        frame_shape: tuple[int, int],
        frame_diagonal: float,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[int], list[int]]:
        rows: list[dict[str, object]] = []
        seen_keys: set[tuple[str, int, int]] = set()
        dormant_ghost_tracks = [
            track
            for track in self._tracks.values()
            if track.track_id != proposal_track.track_id and (track.dormant or track.ghost)
        ]
        continuation_lineage_lookup = getattr(memory_output, "continuation_lineage_lookup", {})
        continuation_continuity_lookup = getattr(memory_output, "continuation_continuity_lookup", {})
        recovery_anchor_lookup = getattr(memory_output, "recovery_anchor_lookup", {})
        recovery_anchor_continuity_lookup = getattr(memory_output, "recovery_anchor_continuity_lookup", {})

        for candidate_track in dormant_ghost_tracks:
            if candidate_track.lineage_id is None or int(candidate_track.lineage_id) < 0:
                continue
            key = (str(candidate_track.state), int(candidate_track.track_id), int(candidate_track.lineage_id))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rows.append(
                self._build_recovery_track_candidate_row(
                    proposal_track=proposal_track,
                    tracking_assignment=tracking_assignment,
                    prototype_assignment=prototype_assignment,
                    candidate_track=candidate_track,
                    frame_index=frame_index,
                    frame_shape=frame_shape,
                    frame_diagonal=frame_diagonal,
                )
            )

        for lineage_id, continuations in continuation_lineage_lookup.items():
            for continuation in continuations:
                key = ("continuation_bank", int(getattr(continuation, "track_id", -1)), int(lineage_id))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                rows.append(
                    self._build_recovery_continuation_candidate_row(
                        proposal_track=proposal_track,
                        tracking_assignment=tracking_assignment,
                        prototype_assignment=prototype_assignment,
                        continuation=continuation,
                        lineage_id=int(lineage_id),
                        candidate_lineage_id=int(lineage_id),
                        source_owner_mode="runtime_owner",
                        frame_index=frame_index,
                        frame_shape=frame_shape,
                        frame_diagonal=frame_diagonal,
                    )
                )

        if self.enable_phasea_dual_owner_source_enumeration:
            for lineage_id, continuations in continuation_continuity_lookup.items():
                for continuation in continuations:
                    key = ("continuation_bank_continuity", int(getattr(continuation, "track_id", -1)), int(lineage_id))
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    rows.append(
                        self._build_recovery_continuation_candidate_row(
                            proposal_track=proposal_track,
                            tracking_assignment=tracking_assignment,
                            prototype_assignment=prototype_assignment,
                            continuation=continuation,
                            lineage_id=int(lineage_id),
                            candidate_lineage_id=int(lineage_id),
                            source_owner_mode="continuity_owner",
                            frame_index=frame_index,
                            frame_shape=frame_shape,
                            frame_diagonal=frame_diagonal,
                        )
                    )

        for lineage_id, anchors in recovery_anchor_lookup.items():
            for anchor in anchors:
                key = ("recovery_anchor", int(anchor.get("old_track_id", -1)), int(lineage_id))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                rows.append(
                    self._build_recovery_anchor_candidate_row(
                        proposal_track=proposal_track,
                        tracking_assignment=tracking_assignment,
                        prototype_assignment=prototype_assignment,
                        anchor=anchor,
                        lineage_id=int(lineage_id),
                        candidate_lineage_id=int(lineage_id),
                        source_owner_mode="runtime_owner",
                        frame_index=frame_index,
                        frame_diagonal=frame_diagonal,
                    )
                )

        if self.enable_phasea_dual_owner_source_enumeration:
            for lineage_id, anchors in recovery_anchor_continuity_lookup.items():
                for anchor in anchors:
                    key = ("recovery_anchor_continuity", int(anchor.get("old_track_id", -1)), int(lineage_id))
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    rows.append(
                        self._build_recovery_anchor_candidate_row(
                            proposal_track=proposal_track,
                            tracking_assignment=tracking_assignment,
                            prototype_assignment=prototype_assignment,
                            anchor=anchor,
                            lineage_id=int(lineage_id),
                            candidate_lineage_id=int(lineage_id),
                            source_owner_mode="continuity_owner",
                            frame_index=frame_index,
                            frame_diagonal=frame_diagonal,
                        )
                    )

        preserve_rows: list[dict[str, object]] = []
        preserve_before_ids: list[int] = []
        preserve_after_ids: list[int] = []
        if (
            self.enable_phase3d_preserve_input_trace
            or self.enable_phase3d_continuity_lineage_repair
            or self.enable_phase3d_three_source_preserve_input
        ):
            preserve_rows, preserve_before_ids, preserve_after_ids = self._phase3d_preserve_input_trace(
                proposal_track=proposal_track,
                tracking_assignment=tracking_assignment,
                prototype_assignment=prototype_assignment,
                memory_output=memory_output,
                candidate_rows=rows,
            )

        if self.enable_phase3d_claim_preservation_repair and bool(tracking_assignment.was_rerouted):
            preserved_lineage_ids = (
                preserve_after_ids
                if (
                    self.enable_phase3d_continuity_lineage_repair
                    or self.enable_phase3d_three_source_preserve_input
                )
                else self._phase3d_hint_lineage_ids(
                    tracking_assignment=tracking_assignment,
                    prototype_assignment=prototype_assignment,
                )
            )
            self._append_preserved_lineage_candidates(
                rows=rows,
                seen_keys=seen_keys,
                preserved_lineage_ids=preserved_lineage_ids,
                proposal_track=proposal_track,
                tracking_assignment=tracking_assignment,
                prototype_assignment=prototype_assignment,
                dormant_ghost_tracks=dormant_ghost_tracks,
                continuation_lineage_lookup=continuation_lineage_lookup,
                continuation_continuity_lookup=continuation_continuity_lookup,
                recovery_anchor_lookup=recovery_anchor_lookup,
                recovery_anchor_continuity_lookup=recovery_anchor_continuity_lookup,
                frame_index=frame_index,
                frame_shape=frame_shape,
                frame_diagonal=frame_diagonal,
            )
        return rows, preserve_rows, preserve_before_ids, preserve_after_ids

    def _build_lineage_recovery_claims(
        self,
        *,
        candidate_rows: list[dict[str, object]],
        tracking_assignment: TrackAssignment,
        prototype_assignment,
        memory_output,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        hint_lineage_ids = set(
            self._phase3d_hint_lineage_ids(
                tracking_assignment=tracking_assignment,
                prototype_assignment=prototype_assignment,
            )
        )
        grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in candidate_rows:
            grouped[int(row["candidate_lineage_id"])].append(row)
        claim_rows: list[dict[str, object]] = []
        for lineage_id, rows in grouped.items():
            live_scores = [float(row["recovery_score_total"]) for row in rows if str(row["source_type"]) in {"dormant", "ghost"}]
            continuation_scores = [float(row["recovery_score_total"]) for row in rows if str(row["source_type"]) == "continuation_bank"]
            anchor_scores = [float(row["recovery_score_total"]) for row in rows if str(row["source_type"]) == "recovery_anchor"]
            best_live = max(live_scores) if live_scores else 0.0
            best_continuation = max(continuation_scores) if continuation_scores else 0.0
            best_anchor = max(anchor_scores) if anchor_scores else 0.0
            source_count = int(sum(score > 0.0 for score in (best_live, best_continuation, best_anchor)))
            prototype_similarity_aggregate = max(float(row["prototype_similarity_score"]) for row in rows)
            temporal_consistency_aggregate = max(float(row["temporal_gap_score"]) for row in rows)
            lineage_continuity_prior = max(float(row["lineage_consistency_score"]) for row in rows)
            same_prototype_support = max(int(row["same_prototype_hint"]) for row in rows)
            same_track_support = max(int(row["same_track_hint"]) for row in rows)
            restore_eligible_count = int(sum(1 for row in rows if bool(row["restore_eligibility"])))
            strongest_source_score = max(best_live, best_continuation, best_anchor)
            mean_source_score = float(
                (best_live + best_continuation + best_anchor) / max(source_count, 1)
            )
            continuity_priority_score = float(
                np.clip(
                    0.30 * best_continuation
                    + 0.25 * best_anchor
                    + 0.15 * best_live
                    + 0.10 * same_prototype_support
                    + 0.10 * same_track_support
                    + 0.10 * lineage_continuity_prior,
                    0.0,
                    1.0,
                )
            )
            geometry_priority_score = float(
                np.clip(
                    0.60 * prototype_similarity_aggregate
                    + 0.40 * temporal_consistency_aggregate,
                    0.0,
                    1.0,
                )
            )
            claim_score_total = float(
                np.clip(
                    0.30 * strongest_source_score
                    + 0.20 * mean_source_score
                    + 0.15 * prototype_similarity_aggregate
                    + 0.15 * temporal_consistency_aggregate
                    + 0.15 * lineage_continuity_prior
                    + 0.05 * min(1.0, source_count / 3.0),
                    0.0,
                    1.0,
                )
            )
            claim_rows.append(
                {
                    "candidate_lineage_id": int(lineage_id),
                    "claim_score_total": claim_score_total,
                    "best_live_score": float(best_live),
                    "best_continuation_score": float(best_continuation),
                    "best_anchor_score": float(best_anchor),
                    "prototype_similarity_aggregate": float(prototype_similarity_aggregate),
                    "temporal_consistency_aggregate": float(temporal_consistency_aggregate),
                    "lineage_continuity_prior": float(lineage_continuity_prior),
                    "source_type_count": source_count,
                    "same_prototype_support": int(same_prototype_support),
                    "same_track_support": int(same_track_support),
                    "restore_eligible_count": int(restore_eligible_count),
                    "continuity_priority_score": continuity_priority_score,
                    "geometry_priority_score": geometry_priority_score,
                    "hint_lineage_match": int(int(lineage_id) in hint_lineage_ids),
                    "claim_built": 1,
                }
            )
        ordered_claim_rows = sorted(
            claim_rows,
            key=lambda row: (
                -float(row["claim_score_total"]),
                -float(row["continuity_priority_score"]),
                -int(row["source_type_count"]),
                -float(row["lineage_continuity_prior"]),
            ),
        )

        final_claim_rows = list(ordered_claim_rows[: self.routing_topk])
        if self.enable_phase3d_claim_preservation_repair and bool(tracking_assignment.was_rerouted):
            visible_lineages = {
                int(row["candidate_lineage_id"]) for row in final_claim_rows
            }
            preserved_rows = [
                row
                for row in ordered_claim_rows
                if int(row["hint_lineage_match"]) == 1
                and int(row["restore_eligible_count"]) > 0
                and max(
                    float(row["best_live_score"]),
                    float(row["best_continuation_score"]),
                    float(row["best_anchor_score"]),
                ) >= self.claim_preserve_min_score
                and int(row["candidate_lineage_id"]) not in visible_lineages
            ]
            for preserved in preserved_rows:
                replace_index = None
                if len(final_claim_rows) < self.routing_topk:
                    final_claim_rows.append(preserved)
                else:
                    for index in range(len(final_claim_rows) - 1, -1, -1):
                        current = final_claim_rows[index]
                        if int(current["hint_lineage_match"]) == 0:
                            replace_index = index
                            break
                    if replace_index is not None:
                        final_claim_rows[replace_index] = preserved
                visible_lineages = {
                    int(row["candidate_lineage_id"]) for row in final_claim_rows
                }
            final_claim_rows = sorted(
                final_claim_rows,
                key=lambda row: (
                    -float(row["claim_score_total"]),
                    -float(row["continuity_priority_score"]),
                    -int(row["source_type_count"]),
                ),
            )

        actual_surface_lineages: set[int] = set(
            int(lineage_id)
            for lineage_id in getattr(memory_output, "continuation_lineage_lookup", {}).keys()
        )
        actual_surface_lineages.update(
            int(lineage_id)
            for lineage_id in getattr(memory_output, "continuation_continuity_lookup", {}).keys()
        )
        actual_surface_lineages.update(
            int(lineage_id) for lineage_id in getattr(memory_output, "recovery_anchor_lookup", {}).keys()
        )
        actual_surface_lineages.update(
            int(lineage_id)
            for lineage_id in getattr(memory_output, "recovery_anchor_continuity_lookup", {}).keys()
        )
        for track in self._tracks.values():
            if track.track_id == tracking_assignment.track_id:
                continue
            if track.lineage_id is None or int(track.lineage_id) < 0:
                continue
            if track.dormant or track.ghost:
                actual_surface_lineages.add(int(track.lineage_id))
        trace_lineages = set(actual_surface_lineages)
        trace_lineages.update(hint_lineage_ids)

        claim_lookup = {
            int(row["candidate_lineage_id"]): dict(row) for row in ordered_claim_rows
        }
        final_visible_lineages = {
            int(row["candidate_lineage_id"]) for row in final_claim_rows
        }
        trace_rows: list[dict[str, object]] = []
        for lineage_id in sorted(trace_lineages):
            candidate_subset = [
                row for row in candidate_rows if int(row["candidate_lineage_id"]) == int(lineage_id)
            ]
            claim_row = claim_lookup.get(int(lineage_id))
            surface_exists = int(lineage_id) in actual_surface_lineages
            candidate_enumerated = bool(candidate_subset)
            restore_eligible = any(bool(row["restore_eligibility"]) for row in candidate_subset)
            claim_built = claim_row is not None
            claim_visible_final = int(lineage_id) in final_visible_lineages
            claim_pruned = int(claim_built and not claim_visible_final)
            if not surface_exists:
                claim_drop_stage = "surface_missing"
                claim_drop_reason = "no_recovery_surface"
            elif not candidate_enumerated:
                claim_drop_stage = "candidate_enumeration"
                claim_drop_reason = "surface_exists_but_candidate_not_enumerated"
            elif not restore_eligible:
                claim_drop_stage = "restore_eligibility"
                claim_drop_reason = "enumerated_but_not_restore_eligible"
            elif not claim_built:
                claim_drop_stage = "claim_builder"
                claim_drop_reason = "eligible_candidate_never_built_into_claim"
            elif not claim_visible_final:
                claim_drop_stage = "claim_pruning"
                claim_drop_reason = "claim_built_but_pruned_from_final_set"
            else:
                claim_drop_stage = "visible"
                claim_drop_reason = "claim_visible_final"

            trace_rows.append(
                {
                    "candidate_lineage_id": int(lineage_id),
                    "claim_score_total": None if claim_row is None else float(claim_row["claim_score_total"]),
                    "best_live_score": 0.0 if claim_row is None else float(claim_row["best_live_score"]),
                    "best_continuation_score": 0.0 if claim_row is None else float(claim_row["best_continuation_score"]),
                    "best_anchor_score": 0.0 if claim_row is None else float(claim_row["best_anchor_score"]),
                    "prototype_similarity_aggregate": 0.0 if claim_row is None else float(claim_row["prototype_similarity_aggregate"]),
                    "temporal_consistency_aggregate": 0.0 if claim_row is None else float(claim_row["temporal_consistency_aggregate"]),
                    "lineage_continuity_prior": 0.0 if claim_row is None else float(claim_row["lineage_continuity_prior"]),
                    "source_type_count": 0 if claim_row is None else int(claim_row["source_type_count"]),
                    "same_prototype_support": 0 if claim_row is None else int(claim_row["same_prototype_support"]),
                    "same_track_support": 0 if claim_row is None else int(claim_row["same_track_support"]),
                    "restore_eligible_count": 0 if claim_row is None else int(claim_row["restore_eligible_count"]),
                    "continuity_priority_score": 0.0 if claim_row is None else float(claim_row["continuity_priority_score"]),
                    "geometry_priority_score": 0.0 if claim_row is None else float(claim_row["geometry_priority_score"]),
                    "hint_lineage_match": int(int(lineage_id) in hint_lineage_ids),
                    "target_lineage_surface_exists": int(surface_exists),
                    "target_lineage_candidate_enumerated": int(candidate_enumerated),
                    "target_lineage_restore_eligible": int(restore_eligible),
                    "target_lineage_claim_built": int(claim_built),
                    "target_lineage_claim_pruned": int(claim_pruned),
                    "target_lineage_claim_visible_final": int(claim_visible_final),
                    "claim_drop_stage": claim_drop_stage,
                    "claim_drop_reason": claim_drop_reason,
                }
            )
        trace_rows = sorted(
            trace_rows,
            key=lambda row: (
                -int(row["target_lineage_claim_visible_final"]),
                -float(0.0 if row["claim_score_total"] is None else row["claim_score_total"]),
                int(row["candidate_lineage_id"]),
            ),
        )
        return trace_rows, final_claim_rows
    def _select_recovery_lineage(
        self,
        *,
        claim_rows: list[dict[str, object]],
    ) -> tuple[dict[str, object] | None, bool, str | None]:
        if not claim_rows:
            return None, False, None
        ordered = list(claim_rows)
        selected = ordered[0]
        applied = False
        reason = None
        if len(ordered) > 1:
            runner_up = ordered[1]
            margin = float(selected["claim_score_total"]) - float(runner_up["claim_score_total"])
            if self.enable_phase3d_identity_preference_tiebreak and margin < self.identity_preference_margin:
                selected = max(
                    ordered[:2],
                    key=lambda row: (
                        float(row["continuity_priority_score"]),
                        int(row["hint_lineage_match"]),
                        float(row["claim_score_total"]),
                        float(row["geometry_priority_score"]),
                    ),
                )
                applied = True
                reason = "identity_preference_tiebreak"
            elif margin < 0.05:
                selected = max(
                    ordered[:2],
                    key=lambda row: (
                        int(row["source_type_count"]),
                        float(row["lineage_continuity_prior"]),
                        float(row["best_continuation_score"]) + float(row["best_anchor_score"]),
                        float(row["claim_score_total"]),
                    ),
                )
                applied = True
                reason = "identity_surface_tiebreak"
        return selected, applied, reason

    def _select_old_identity_within_lineage(
        self,
        *,
        candidate_rows: list[dict[str, object]],
        selected_lineage_id: int,
        tracking_assignment: TrackAssignment,
    ) -> tuple[dict[str, object] | None, bool, str | None]:
        lineage_rows = [row for row in candidate_rows if int(row["candidate_lineage_id"]) == int(selected_lineage_id)]
        if not lineage_rows:
            return None, False, None

        by_identity: dict[tuple[int, int | None], list[dict[str, object]]] = defaultdict(list)
        for row in lineage_rows:
            track_id = int(row["candidate_track_id"])
            prototype_id = row["candidate_prototype_id"]
            by_identity[(track_id, None if prototype_id is None or int(prototype_id) < 0 else int(prototype_id))].append(row)

        identity_rows: list[dict[str, object]] = []
        for identity_key, rows in by_identity.items():
            source_types = {str(row["source_type"]) for row in rows}
            best_score = max(float(row["recovery_score_total"]) for row in rows)
            same_prototype_support = max(int(row["same_prototype_hint"]) for row in rows)
            continuation_present = int("continuation_bank" in source_types)
            anchor_present = int("recovery_anchor" in source_types)
            live_present = int(bool({"dormant", "ghost"} & source_types))
            combined_score = float(
                best_score
                + 0.12 * min(1.0, max(0, len(source_types) - 1))
                + 0.08 * same_prototype_support
                + 0.06 * continuation_present
                + 0.06 * anchor_present
                + 0.04 * live_present
            )
            identity_rows.append(
                {
                    "identity_key": identity_key,
                    "rows": rows,
                    "source_type_count": int(len(source_types)),
                    "same_prototype_support": int(same_prototype_support),
                    "continuation_present": int(continuation_present),
                    "anchor_present": int(anchor_present),
                    "live_present": int(live_present),
                    "combined_score": combined_score,
                    "best_score": float(best_score),
                }
            )

        identity_rows = sorted(
            identity_rows,
            key=lambda row: (
                -float(row["combined_score"]),
                -int(row["source_type_count"]),
                -int(row["same_prototype_support"]),
                -float(row["best_score"]),
            ),
        )
        selected_identity = identity_rows[0]
        applied = False
        reason = None
        if len(identity_rows) > 1:
            runner_up = identity_rows[1]
            margin = float(selected_identity["combined_score"]) - float(runner_up["combined_score"])
            if self.enable_phase3d_identity_preference_tiebreak and margin < self.identity_preference_margin:
                selected_identity = max(
                    identity_rows[:2],
                    key=lambda row: (
                        int(row["same_prototype_support"]),
                        int(row["continuation_present"]) + int(row["anchor_present"]),
                        int(row["live_present"]),
                        float(row["combined_score"]),
                    ),
                )
                applied = True
                reason = "identity_preference_tiebreak"
            elif margin < 0.05:
                selected_identity = max(
                    identity_rows[:2],
                    key=lambda row: (
                        int(row["source_type_count"]),
                        int(row["same_prototype_support"]),
                        int(row["continuation_present"]) + int(row["anchor_present"]),
                        float(row["combined_score"]),
                    ),
                )
                applied = True
                reason = "identity_cluster_tiebreak"

        chosen_rows = list(selected_identity["rows"])
        source_priority = {"dormant": 0, "ghost": 1, "continuation_bank": 2, "recovery_anchor": 3}
        chosen = sorted(
            chosen_rows,
            key=lambda row: (
                float(row["raw_cost"]),
                source_priority.get(str(row["source_type"]), 99),
                -float(row["recovery_score_total"]),
            ),
        )[0]
        if len(chosen_rows) > 1:
            chosen["same_track_hint"] = int(
                len({str(row["source_type"]) for row in chosen_rows}) > 1
                or (
                    tracking_assignment.prototype_hint_id is not None
                    and any(int(row["same_prototype_hint"]) == 1 for row in chosen_rows)
                )
            )
        return chosen, applied, reason

    def _restore_from_candidate_row(
        self,
        *,
        candidate_row: dict[str, object],
        proposal_track: TrackState,
        tracking_assignment: TrackAssignment,
        prototype_assignment,
        tracking_output,
        frame_index: int,
    ) -> bool:
        selected_lineage_id = int(candidate_row["candidate_lineage_id"])
        selected_prototype_id = candidate_row.get("candidate_prototype_id")
        prototype_assignment.lineage_id = int(selected_lineage_id)
        prototype_assignment.matched_lineage_id = int(selected_lineage_id)
        if selected_prototype_id is not None and int(selected_prototype_id) >= 0:
            prototype_assignment.prototype_id = int(selected_prototype_id)

        tracking_assignment.selected_recovery_lineage_id = int(selected_lineage_id)
        tracking_assignment.selected_recovery_source_type = str(candidate_row["source_type"])
        tracking_assignment.selected_recovery_track_id = int(candidate_row["candidate_track_id"])
        tracking_assignment.selected_recovery_prototype_id = (
            None
            if candidate_row.get("candidate_prototype_id") is None or int(candidate_row.get("candidate_prototype_id")) < 0
            else int(candidate_row["candidate_prototype_id"])
        )
        tracking_assignment.resurrection_attempted = True
        tracking_assignment.resurrection_cost_best = float(candidate_row["raw_cost"])

        source_type = str(candidate_row["source_type"])
        if source_type in {"dormant", "ghost"}:
            if float(candidate_row["raw_cost"]) > float(candidate_row["accept_threshold"]):
                return False
            old_track = candidate_row["_candidate_obj"]
            tracking_assignment.best_candidate_state = str(source_type)
            tracking_assignment.best_candidate_gap = int(old_track.gap_length)
            self._resurrect_old_track(
                old_track=old_track,
                temporary_track=proposal_track,
                tracking_assignment=tracking_assignment,
                prototype_assignment=prototype_assignment,
                tracking_output=tracking_output,
                assignment_source="concept_gated_resurrection",
                resurrected_from_continuation=False,
            )
            return True

        if source_type == "continuation_bank":
            if float(candidate_row["raw_cost"]) > float(self.tau_continuation):
                return False
            continuation = candidate_row["_candidate_obj"]
            tracking_assignment.continuation_attempted = True
            tracking_assignment.best_continuation_cost = float(candidate_row["raw_cost"])
            tracking_assignment.best_continuation_gap = int(frame_index - continuation.last_seen_frame)
            tracking_assignment.best_continuation_age = int(continuation.age_since_last_seen)
            continuation_track = self._get_or_create_continuation_track(continuation, proposal_track, frame_index)
            self._resurrect_old_track(
                old_track=continuation_track,
                temporary_track=proposal_track,
                tracking_assignment=tracking_assignment,
                prototype_assignment=prototype_assignment,
                tracking_output=tracking_output,
                assignment_source="continuation_resurrection",
                resurrected_from_continuation=True,
            )
            tracking_assignment.continuation_success = True
            return True

        if float(candidate_row["raw_cost"]) > float(self.tau_continuation) and not self.debug_force_anchor_consume:
            return False
        anchor = candidate_row["_candidate_obj"]
        tracking_assignment.restore_attempted_from_anchor = True
        tracking_assignment.best_anchor_uid = str(anchor.get("anchor_uid", ""))
        tracking_assignment.best_anchor_gap = int(anchor.get("age_since_last_seen", 0))
        tracking_assignment.best_candidate_state = "recovery_identity_anchor"
        anchor_track = self._get_or_create_anchor_track(anchor, proposal_track, frame_index)
        self._resurrect_old_track(
            old_track=anchor_track,
            temporary_track=proposal_track,
            tracking_assignment=tracking_assignment,
            prototype_assignment=prototype_assignment,
            tracking_output=tracking_output,
            assignment_source="anchor_resurrection",
            resurrected_from_continuation=False,
        )
        tracking_assignment.anchor_success = True
        return True

    def apply_concept_gated_resurrection(
        self,
        tracking_output,
        memory_output,
        *,
        frame_index: int,
        frame_shape: tuple[int, int],
    ) -> int:
        recovered = 0
        attempts = 0
        continuation_attempts = 0
        continuation_successes = 0
        anchor_attempts = 0
        prototype_assignments = memory_output.assignments
        continuation_lookup = getattr(memory_output, "continuation_lookup", {})
        continuation_lineage_lookup = getattr(memory_output, "continuation_lineage_lookup", {})
        temp_attach_lookup = getattr(memory_output, "temp_attach_lookup", {})
        recovery_anchor_lookup = getattr(memory_output, "recovery_anchor_lookup", {})
        assignment_by_track = {int(assignment.track_id): assignment for assignment in tracking_output.assignments}
        frame_diagonal = float(np.hypot(frame_shape[1], frame_shape[0]))
        consumed_continuation_ids: set[int] = set()
        consumed_anchor_uids: set[str] = set()

        for prototype_assignment in prototype_assignments:
            tracking_assignment = assignment_by_track.get(int(prototype_assignment.track_id))
            new_track = self._tracks.get(int(prototype_assignment.track_id))
            if tracking_assignment is None or new_track is None:
                continue

            concept_recovered = bool(
                (
                    tracking_assignment.assignment_source == "new_track"
                    or tracking_assignment.assignment_source == "rerouted_to_resurrection"
                    or tracking_assignment.final_assignment_source == "rerouted_to_resurrection"
                    or bool(tracking_assignment.was_rerouted)
                )
                and not prototype_assignment.new_prototype_created
            )
            tracking_assignment.concept_recovered = concept_recovered
            tracking_assignment.linked_prototype_id = int(prototype_assignment.prototype_id)
            tracking_assignment.linked_lineage_id = int(getattr(prototype_assignment, "lineage_id", -1))
            tracking_assignment.final_lineage_id = int(getattr(prototype_assignment, "lineage_id", -1))
            tracking_assignment.prototype_hint_id = int(prototype_assignment.prototype_id)
            tracking_assignment.prototype_hint_distance = float(prototype_assignment.distance)
            tracking_assignment.prototype_hint_similarity = float(prototype_assignment.similarity)
            attach_written = bool(
                getattr(prototype_assignment, "attach_state_written", False)
                or getattr(prototype_assignment, "recovery_attach_target", "none") != "none"
            )
            lineage_id = getattr(prototype_assignment, "lineage_id", None)
            temp_slot = (
                None
                if lineage_id is None or int(lineage_id) < 0
                else temp_attach_lookup.get(int(lineage_id))
            )
            temp_slot_available = bool(temp_slot) and not bool(temp_slot.get("expired", False))
            tracking_assignment.promotion_pending_created = bool(
                getattr(prototype_assignment, "promotion_pending_flag", False)
            )
            tracking_assignment.promotion_step_executed = bool(
                getattr(prototype_assignment, "promotion_success", False)
                or getattr(prototype_assignment, "promotion_decision", "keep_head") != "keep_head"
            )
            if (
                self.debug_force_attach_consume
                and attach_written
                and getattr(prototype_assignment, "recovery_attach_target", "none") == "temporary_attach_slot"
                and temp_slot_available
            ):
                tracking_assignment.attach_state_consumed_by_tracker = True
                tracking_assignment.restore_attempted_from_attach = True
                tracking_assignment.best_candidate_state = "temporary_attach_slot"
                tracking_assignment.best_candidate_gap = int(temp_slot.get("age_since_last_seen", 0))
            if not concept_recovered:
                continue

            consumer_lineage_id = int(getattr(prototype_assignment, "lineage_id", -1))
            stagea4_trace_enabled = bool(
                self.enable_phase3d_target_selection_trace
                or self.enable_phase3d_target_selection_repair
                or bool(tracking_assignment.was_rerouted)
            )
            selected_candidate_row = None
            if stagea4_trace_enabled:
                candidate_rows, preserve_input_rows, preserve_before_ids, preserve_after_ids = (
                    self._build_stagea4_candidate_rows(
                    proposal_track=new_track,
                    tracking_assignment=tracking_assignment,
                    prototype_assignment=prototype_assignment,
                    tracking_output=tracking_output,
                    memory_output=memory_output,
                    frame_index=frame_index,
                    frame_shape=frame_shape,
                    frame_diagonal=frame_diagonal,
                    )
                )
                claim_trace_rows, final_claim_rows = self._build_lineage_recovery_claims(
                    candidate_rows=candidate_rows,
                    tracking_assignment=tracking_assignment,
                    prototype_assignment=prototype_assignment,
                    memory_output=memory_output,
                )
                selected_claim, lineage_tiebreak_applied, lineage_tiebreak_reason = self._select_recovery_lineage(
                    claim_rows=final_claim_rows
                )
                selected_lineage_id = (
                    None if selected_claim is None else int(selected_claim["candidate_lineage_id"])
                )
                identity_tiebreak_applied = False
                identity_tiebreak_reason = None
                if selected_lineage_id is not None:
                    selected_candidate_row, identity_tiebreak_applied, identity_tiebreak_reason = (
                        self._select_old_identity_within_lineage(
                            candidate_rows=candidate_rows,
                            selected_lineage_id=selected_lineage_id,
                            tracking_assignment=tracking_assignment,
                        )
                    )
                tracking_assignment.consumer_candidate_count_seen = int(len(candidate_rows))
                tracking_assignment.selected_recovery_lineage_id = selected_lineage_id
                tracking_assignment.selected_recovery_source_type = (
                    None if selected_candidate_row is None else str(selected_candidate_row["source_type"])
                )
                tracking_assignment.selected_recovery_track_id = (
                    None if selected_candidate_row is None else int(selected_candidate_row["candidate_track_id"])
                )
                tracking_assignment.selected_recovery_prototype_id = (
                    None
                    if selected_candidate_row is None
                    or selected_candidate_row.get("candidate_prototype_id") is None
                    or int(selected_candidate_row.get("candidate_prototype_id")) < 0
                    else int(selected_candidate_row["candidate_prototype_id"])
                )
                tracking_assignment.identity_tiebreak_applied = bool(
                    lineage_tiebreak_applied or identity_tiebreak_applied
                )
                tracking_assignment.identity_tiebreak_reason = (
                    identity_tiebreak_reason or lineage_tiebreak_reason
                )

                preserve_before_text = "|".join(str(lineage_id) for lineage_id in preserve_before_ids)
                preserve_after_text = "|".join(str(lineage_id) for lineage_id in preserve_after_ids)
                runtime_hint_text = "|".join(
                    str(lineage_id)
                    for lineage_id in self._phase3d_runtime_hint_lineage_ids(
                        tracking_assignment=tracking_assignment,
                        prototype_assignment=prototype_assignment,
                    )
                )
                for row in preserve_input_rows:
                    tracking_output.preserve_input_rows.append(
                        {
                            "frame_id": int(frame_index),
                            "proposal_id": int(tracking_assignment.proposal_index),
                            "track_id": int(tracking_assignment.track_id),
                            "proposal_box": tuple(int(value) for value in tracking_assignment.box),
                            "proposal_score": float(tracking_assignment.score),
                            **row,
                        }
                    )

                for row in candidate_rows:
                    sanitized = {
                        key: value
                        for key, value in row.items()
                        if key != "_candidate_obj"
                    }
                    sanitized.update(
                        {
                            "frame_id": int(frame_index),
                            "proposal_id": int(tracking_assignment.proposal_index),
                            "track_id": int(tracking_assignment.track_id),
                            "proposal_box": tuple(int(value) for value in tracking_assignment.box),
                            "proposal_score": float(tracking_assignment.score),
                            "tentative_active_lineage": tracking_assignment.tentative_lineage_id,
                            "rerouted_flag": int(bool(tracking_assignment.was_rerouted)),
                            "candidate_count_seen_by_consumer": int(len(candidate_rows)),
                            "final_selected_lineage": selected_lineage_id,
                            "final_selected_source_type": None
                            if selected_candidate_row is None
                            else str(selected_candidate_row["source_type"]),
                            "final_selected_track_id": None
                            if selected_candidate_row is None
                            else int(selected_candidate_row["candidate_track_id"]),
                            "final_selected_prototype_id": None
                            if selected_candidate_row is None
                            or selected_candidate_row.get("candidate_prototype_id") is None
                            or int(selected_candidate_row.get("candidate_prototype_id")) < 0
                            else int(selected_candidate_row["candidate_prototype_id"]),
                        }
                    )
                    tracking_output.recovery_candidate_rows.append(sanitized)

                for row in claim_trace_rows:
                    tracking_output.lineage_claim_rows.append(
                        {
                            "frame_id": int(frame_index),
                            "proposal_id": int(tracking_assignment.proposal_index),
                            "track_id": int(tracking_assignment.track_id),
                            "proposal_box": tuple(int(value) for value in tracking_assignment.box),
                            "proposal_score": float(tracking_assignment.score),
                            **row,
                            "entered_preserve_input": int(
                                int(row["candidate_lineage_id"]) in set(preserve_before_ids)
                            ),
                            "entered_claim_builder_input": int(
                                int(row["candidate_lineage_id"]) in set(preserve_after_ids)
                            ),
                            "preserve_candidate_lineages_before_prune": preserve_before_text,
                            "preserve_candidate_lineages_after_prune": preserve_after_text,
                            "proposal_lineage_hint_topk": runtime_hint_text,
                            "selected_lineage": selected_lineage_id,
                            "claim_winner": int(
                                selected_lineage_id is not None
                                and int(row["candidate_lineage_id"]) == int(selected_lineage_id)
                            ),
                            "identity_tiebreak_applied": int(lineage_tiebreak_applied),
                            "identity_tiebreak_reason": lineage_tiebreak_reason,
                            "final_claim_visible": int(row.get("target_lineage_claim_visible_final", 0)),
                            "failure_bucket": (
                                "claim_visibility_failure"
                                if int(row.get("hint_lineage_match", 0)) == 1
                                and int(row.get("target_lineage_claim_visible_final", 0)) == 0
                                else (
                                    "visible_but_underweighted_failure"
                                    if int(row.get("hint_lineage_match", 0)) == 1
                                    and selected_lineage_id is not None
                                    and int(row["candidate_lineage_id"]) != int(selected_lineage_id)
                                    and int(row.get("target_lineage_claim_visible_final", 0)) == 1
                                    else "not_applicable"
                                )
                            ),
                        }
                    )

                tracking_output.recovery_selection_rows.append(
                    {
                        "frame_id": int(frame_index),
                        "proposal_id": int(tracking_assignment.proposal_index),
                        "track_id": int(tracking_assignment.track_id),
                        "proposal_box": tuple(int(value) for value in tracking_assignment.box),
                        "proposal_score": float(tracking_assignment.score),
                        "candidate_count_seen_by_consumer": int(len(candidate_rows)),
                        "visible_claim_count": int(len(final_claim_rows)),
                        "preserve_candidate_lineages_before_prune": preserve_before_text,
                        "preserve_candidate_lineages_after_prune": preserve_after_text,
                        "proposal_lineage_hint_topk": runtime_hint_text,
                        "selected_lineage_id": selected_lineage_id,
                        "selected_source_type": None
                        if selected_candidate_row is None
                        else str(selected_candidate_row["source_type"]),
                        "selected_track_id": None
                        if selected_candidate_row is None
                        else int(selected_candidate_row["candidate_track_id"]),
                        "selected_prototype_id": None
                        if selected_candidate_row is None
                        or selected_candidate_row.get("candidate_prototype_id") is None
                        or int(selected_candidate_row.get("candidate_prototype_id")) < 0
                        else int(selected_candidate_row["candidate_prototype_id"]),
                        "target_lineage_visible": int(
                            any(
                                int(row.get("hint_lineage_match", 0)) == 1
                                and int(row.get("target_lineage_claim_visible_final", 0)) == 1
                                for row in claim_trace_rows
                            )
                        ),
                        "target_lineage_rank": next(
                            (
                                int(index + 1)
                                for index, row in enumerate(final_claim_rows)
                                if int(row.get("hint_lineage_match", 0)) == 1
                            ),
                            None,
                        ),
                        "target_lineage_claim_score": next(
                            (
                                float(row["claim_score_total"])
                                for row in final_claim_rows
                                if int(row.get("hint_lineage_match", 0)) == 1
                            ),
                            None,
                        ),
                        "winning_lineage_claim_score": None
                        if selected_claim is None
                        else float(selected_claim["claim_score_total"]),
                        "continuity_priority_score": None
                        if selected_claim is None
                        else float(selected_claim.get("continuity_priority_score", 0.0)),
                        "geometry_priority_score": None
                        if selected_claim is None
                        else float(selected_claim.get("geometry_priority_score", 0.0)),
                        "lineage_tiebreak_applied": int(lineage_tiebreak_applied),
                        "lineage_tiebreak_reason": lineage_tiebreak_reason,
                        "identity_tiebreak_applied": int(identity_tiebreak_applied),
                        "identity_tiebreak_reason": identity_tiebreak_reason,
                        "identity_preference_tiebreak_applied": int(
                            lineage_tiebreak_reason == "identity_preference_tiebreak"
                        ),
                        "identity_preference_tiebreak_reason": None
                        if lineage_tiebreak_reason != "identity_preference_tiebreak"
                        else lineage_tiebreak_reason,
                        "identity_preference_delta": None
                        if selected_claim is None or len(final_claim_rows) < 2
                        else float(selected_claim.get("continuity_priority_score", 0.0))
                        - float(final_claim_rows[1].get("continuity_priority_score", 0.0)),
                        "two_stage_enabled": int(self.enable_phase3d_target_selection_repair),
                    }
                )

                if (
                    self.enable_phase3d_target_selection_repair
                    and bool(tracking_assignment.was_rerouted)
                    and selected_lineage_id is not None
                ):
                    consumer_lineage_id = int(selected_lineage_id)
                    prototype_assignment.lineage_id = int(selected_lineage_id)
                    prototype_assignment.matched_lineage_id = int(selected_lineage_id)
                    if selected_candidate_row is not None:
                        if str(selected_candidate_row["source_type"]) in {"dormant", "ghost"}:
                            attempts += 1
                        elif str(selected_candidate_row["source_type"]) == "continuation_bank":
                            continuation_attempts += 1
                        else:
                            anchor_attempts += 1
                        if self._restore_from_candidate_row(
                            candidate_row=selected_candidate_row,
                            proposal_track=new_track,
                            tracking_assignment=tracking_assignment,
                            prototype_assignment=prototype_assignment,
                            tracking_output=tracking_output,
                            frame_index=frame_index,
                        ):
                            if str(selected_candidate_row["source_type"]) == "continuation_bank":
                                continuation_successes += 1
                            recovered += 1
                            continue

            live_candidates = [
                track
                for track in self._tracks.values()
                if track.track_id != new_track.track_id
                and track.lineage_id == consumer_lineage_id
                and (track.dormant or track.ghost)
            ]
            prototype_continuation_candidates = [
                candidate
                for candidate in continuation_lookup.get(int(prototype_assignment.prototype_id), [])
                if int(getattr(candidate, "continuation_id", -1)) not in consumed_continuation_ids
            ]
            lineage_continuation_candidates = [
                candidate
                for candidate in continuation_lineage_lookup.get(int(consumer_lineage_id), [])
                if int(getattr(candidate, "continuation_id", -1)) not in consumed_continuation_ids
            ]
            anchor_candidates = [
                candidate
                for candidate in recovery_anchor_lookup.get(int(consumer_lineage_id), [])
                if str(candidate.get("anchor_uid", "")) not in consumed_anchor_uids
            ]
            continuation_candidates = lineage_continuation_candidates
            tracking_assignment.live_candidate_pool_size = len(live_candidates)
            tracking_assignment.continuation_bank_size = len(continuation_candidates)
            tracking_assignment.anchor_candidate_pool_size = len(anchor_candidates)
            tracking_assignment.prototype_matched_continuation_count = len(prototype_continuation_candidates)
            tracking_assignment.lineage_matched_continuation_count = len(lineage_continuation_candidates)
            tracking_assignment.continuation_bank_exists = len(continuation_candidates) > 0
            tracking_assignment.slot_candidate_pool_size = 0
            tracking_assignment.candidate_pool_size = (
                len(live_candidates) + len(continuation_candidates) + len(anchor_candidates)
            )
            tracking_assignment.candidate_pool_nonempty = tracking_assignment.candidate_pool_size > 0
            tracking_assignment.attach_state_consumed_by_tracker = bool(
                attach_written and tracking_assignment.candidate_pool_size > 0
            )
            tracking_assignment.attach_state_consumed_by_continuation = bool(
                attach_written and len(continuation_candidates) > 0
            )
            if not tracking_assignment.candidate_pool_nonempty:
                continue

            if live_candidates:
                attempts += 1
                tracking_assignment.resurrection_attempted = True
                best_candidate, best_cost = min(
                    (
                        (
                            candidate,
                            self._concept_gated_resurrection_cost(
                                proposal_track=new_track,
                                candidate_track=candidate,
                                frame_index=frame_index,
                                frame_shape=frame_shape,
                                frame_diagonal=frame_diagonal,
                            ),
                        )
                        for candidate in live_candidates
                    ),
                    key=lambda item: item[1],
                )
                tracking_assignment.resurrection_cost_best = float(best_cost)
                tracking_assignment.best_candidate_state = best_candidate.state
                tracking_assignment.best_candidate_gap = int(best_candidate.gap_length)

                gap = max(1, best_candidate.gap_length)
                gap_decay = float(np.exp(-gap / self.tau_g))
                threshold = float(self.tau_res_short + (1.0 - gap_decay) * (self.tau_res_long - self.tau_res_short))
                tracking_assignment.restore_attempted_from_attach = bool(attach_written)
                if best_cost <= threshold:
                    self._resurrect_old_track(
                        old_track=best_candidate,
                        temporary_track=new_track,
                        tracking_assignment=tracking_assignment,
                        prototype_assignment=prototype_assignment,
                        tracking_output=tracking_output,
                        assignment_source="concept_gated_resurrection",
                        resurrected_from_continuation=False,
                    )
                    recovered += 1
                    continue

            continuation_restored = False
            if continuation_candidates:
                continuation_attempts += 1
                tracking_assignment.resurrection_attempted = True
                tracking_assignment.continuation_attempted = True
                tracking_assignment.restore_attempted_from_attach = bool(attach_written)
                ordered_continuations = sorted(
                    (
                        (
                            continuation,
                            self._continuation_resurrection_cost(
                                continuation=continuation,
                                proposal_track=new_track,
                                frame_index=frame_index,
                                frame_shape=frame_shape,
                                frame_diagonal=frame_diagonal,
                            ),
                        )
                        for continuation in continuation_candidates
                    ),
                    key=lambda item: item[1],
                )
                best_continuation, best_continuation_cost = ordered_continuations[0]
                second_best_cost = (
                    ordered_continuations[1][1] if len(ordered_continuations) > 1 else float("inf")
                )
                tracking_assignment.resurrection_cost_best = float(best_continuation_cost)
                tracking_assignment.best_continuation_cost = float(best_continuation_cost)
                tracking_assignment.best_continuation_gap = int(frame_index - best_continuation.last_seen_frame)
                tracking_assignment.best_continuation_age = int(best_continuation.age_since_last_seen)

                if (
                    best_continuation_cost <= self.tau_continuation
                    and (second_best_cost - best_continuation_cost) > self.continuation_margin
                ):
                    continuation_track = self._get_or_create_continuation_track(
                        best_continuation, new_track, frame_index
                    )
                    self._resurrect_old_track(
                        old_track=continuation_track,
                        temporary_track=new_track,
                        tracking_assignment=tracking_assignment,
                        prototype_assignment=prototype_assignment,
                        tracking_output=tracking_output,
                        assignment_source="continuation_resurrection",
                        resurrected_from_continuation=True,
                    )
                    tracking_assignment.continuation_success = True
                    continuation_successes += 1
                    recovered += 1
                    consumed_continuation_ids.add(int(best_continuation.continuation_id))
                    continuation_restored = True
            if continuation_restored:
                continue

            if not anchor_candidates:
                continue

            anchor_attempts += 1
            tracking_assignment.resurrection_attempted = True
            tracking_assignment.restore_attempted_from_anchor = True
            tracking_assignment.restore_attempted_from_attach = bool(attach_written)
            ordered_anchors = sorted(
                (
                    (
                        anchor,
                        self._recovery_anchor_cost(
                            anchor=anchor,
                            proposal_track=new_track,
                            frame_index=frame_index,
                            frame_diagonal=frame_diagonal,
                        ),
                    )
                    for anchor in anchor_candidates
                ),
                key=lambda item: item[1],
            )
            best_anchor, best_anchor_cost = ordered_anchors[0]
            tracking_assignment.resurrection_cost_best = float(best_anchor_cost)
            tracking_assignment.best_candidate_state = "recovery_identity_anchor"
            tracking_assignment.best_anchor_uid = str(best_anchor.get("anchor_uid", ""))
            tracking_assignment.best_anchor_gap = int(best_anchor.get("age_since_last_seen", 0))
            threshold = float(self.tau_continuation)
            if not self.debug_force_anchor_consume and best_anchor_cost > threshold:
                continue

            anchor_track = self._get_or_create_anchor_track(best_anchor, new_track, frame_index)
            self._resurrect_old_track(
                old_track=anchor_track,
                temporary_track=new_track,
                tracking_assignment=tracking_assignment,
                prototype_assignment=prototype_assignment,
                tracking_output=tracking_output,
                assignment_source="anchor_resurrection",
                resurrected_from_continuation=False,
            )
            tracking_assignment.continuation_success = False
            tracking_assignment.anchor_success = True
            recovered += 1
            consumed_anchor_uids.add(str(best_anchor.get("anchor_uid", "")))

        tracking_output.reactivation_attempts = 0
        tracking_output.resurrection_attempts = attempts + continuation_attempts + anchor_attempts
        tracking_output.resurrection_successes = recovered
        tracking_output.slot_resurrection_attempts = 0
        tracking_output.slot_resurrection_successes = 0
        tracking_output.continuation_resurrection_attempts = continuation_attempts
        tracking_output.continuation_resurrection_successes = continuation_successes
        tracking_output.reactivated_track_ids = sorted(set(tracking_output.reactivated_track_ids))
        tracking_output.active_tracks = [_clone_track(track) for track in self._tracks_by_state("active")]
        tracking_output.dormant_tracks = [_clone_track(track) for track in self._tracks_by_state("dormant")]
        tracking_output.ghost_tracks = [_clone_track(track) for track in self._tracks_by_state("ghost")]
        tracking_output.retired_tracks = [_clone_track(track) for track in self._tracks_by_state("retired")]
        tracking_output.active_track_count = len(tracking_output.active_tracks)
        tracking_output.dormant_track_count = len(tracking_output.dormant_tracks)
        tracking_output.ghost_track_count = len(tracking_output.ghost_tracks)
        tracking_output.retired_track_count = len(tracking_output.retired_tracks)
        tracking_output.identity_slot_count = self.identity_slot_count()
        return recovered

    def _resurrect_old_track(
        self,
        *,
        old_track: TrackState,
        temporary_track: TrackState,
        tracking_assignment: TrackAssignment,
        prototype_assignment,
        tracking_output,
        assignment_source: str = "concept_gated_resurrection",
        resurrected_from_continuation: bool = False,
    ) -> None:
        previous_state = old_track.state
        old_track.box = temporary_track.box
        old_track.centroid = temporary_track.centroid
        old_track.last_bbox = temporary_track.last_bbox
        old_track.last_center = temporary_track.last_center
        old_track.last_feature = temporary_track.last_feature.copy()
        old_track.signature = _normalize_signature(
            self.signature_momentum * old_track.signature + (1.0 - self.signature_momentum) * temporary_track.signature
        )
        old_track.anchor_signature = _normalize_signature(
            self.anchor_momentum * old_track.anchor_signature
            + (1.0 - self.anchor_momentum) * temporary_track.anchor_signature
        )
        old_track.velocity = (
            0.6 * old_track.velocity.astype(np.float32) + 0.4 * temporary_track.velocity.astype(np.float32)
        ).astype(np.float32)
        old_track.score = temporary_track.score
        old_track.hits += 1
        old_track.hit_count += 1
        old_track.missed_frames = 0
        old_track.miss_count = 0
        old_track.last_seen_frame = temporary_track.last_seen_frame
        old_track.gap_length = 0
        old_track.lineage_id = int(getattr(prototype_assignment, "lineage_id", -1))
        old_track.continuity_lineage_id = (
            old_track.lineage_id
            if getattr(prototype_assignment, "continuity_lineage_id", None) is None
            else int(prototype_assignment.continuity_lineage_id)
        )
        _set_track_state(old_track, "active")

        tracking_assignment.track_id = old_track.track_id
        tracking_assignment.assignment_source = assignment_source
        final_source_lookup = {
            "concept_gated_resurrection": "resurrection_from_dormant_or_ghost",
            "continuation_resurrection": "resurrection_from_continuation_bank",
            "anchor_resurrection": "resurrection_from_recovery_anchor",
        }
        tracking_assignment.final_assignment_source = final_source_lookup.get(
            assignment_source,
            assignment_source,
        )
        tracking_assignment.reactivation_attempted = True
        tracking_assignment.reactivation_cost = float(
            tracking_assignment.resurrection_cost_best
            if tracking_assignment.resurrection_cost_best is not None
            else tracking_assignment.reactivation_cost
        )
        tracking_assignment.resurrection_success = True
        tracking_assignment.previous_state = previous_state
        tracking_assignment.linked_prototype_id = int(prototype_assignment.prototype_id)
        tracking_assignment.linked_lineage_id = int(getattr(prototype_assignment, "lineage_id", -1))
        tracking_assignment.final_lineage_id = int(getattr(prototype_assignment, "lineage_id", -1))
        tracking_assignment.prototype_hint_id = int(prototype_assignment.prototype_id)
        tracking_assignment.prototype_hint_distance = float(prototype_assignment.distance)
        tracking_assignment.prototype_hint_similarity = float(prototype_assignment.similarity)
        tracking_assignment.prototype_similarity = float(prototype_assignment.similarity)
        tracking_assignment.concept_recovered = True
        tracking_assignment.continuation_success = True
        tracking_assignment.resurrected_from_continuation = bool(resurrected_from_continuation)
        tracking_assignment.resurrected_from_slot = False
        tracking_assignment.continuity_lineage_id = (
            None
            if getattr(prototype_assignment, "continuity_lineage_id", None) is None
            else int(prototype_assignment.continuity_lineage_id)
        )
        prototype_assignment.track_id = old_track.track_id

        if temporary_track.track_id in tracking_output.new_track_ids:
            tracking_output.new_track_ids.remove(temporary_track.track_id)
        tracking_output.reactivated_track_ids.append(old_track.track_id)
        self._tracks.pop(int(temporary_track.track_id), None)

    def _refresh_identity_slots(self, frame_index: int) -> None:
        if not self.enable_identity_slots:
            self._identity_slots_by_prototype.clear()
            return
        stale_prototypes: list[int] = []
        for prototype_id, slots in self._identity_slots_by_prototype.items():
            refreshed: list[IdentitySlot] = []
            for slot in slots:
                slot.age_since_disappear = int(max(0, frame_index - slot.disappear_frame))
                # Apply a linear per-frame decay. Using total age here over-decays slots
                # quadratically and empties the pool before long-gap re-entry happens.
                slot.slot_confidence = float(max(0.0, slot.slot_confidence - self.slot_decay))
                source_track = self._tracks.get(int(slot.source_track_id))
                if source_track is not None and not source_track.retired:
                    continue
                if slot.age_since_disappear > self.slot_max_gap:
                    continue
                if slot.slot_confidence <= 0.0:
                    continue
                refreshed.append(slot)
            if refreshed:
                self._identity_slots_by_prototype[prototype_id] = deque(
                    sorted(refreshed, key=lambda item: self._identity_slot_rank(item), reverse=True)[
                        : self.slot_topk_per_proto
                    ]
                )
            else:
                stale_prototypes.append(prototype_id)
        for prototype_id in stale_prototypes:
            self._identity_slots_by_prototype.pop(prototype_id, None)

    def _archive_identity_slot(self, track: TrackState, frame_index: int) -> int:
        if not self.enable_identity_slots:
            return 0
        if track.prototype_id is None:
            return 0
        if track.age < self.min_track_age_for_slot or track.hit_count < self.min_hits_for_slot:
            return 0
        prototype_id = int(track.prototype_id)
        shape_signature = track.last_feature[10:13].copy() if track.last_feature.size >= 13 else track.last_feature.copy()
        slot = IdentitySlot(
            slot_id=self._next_slot_id,
            source_track_id=int(track.track_id),
            prototype_id=prototype_id,
            disappear_frame=int(track.last_seen_frame),
            last_center=(float(track.last_center[0]), float(track.last_center[1])),
            last_bbox=track.last_bbox,
            velocity=track.velocity.copy(),
            feature_ema=track.anchor_signature.copy(),
            shape_signature=shape_signature.astype(np.float32),
            last_objectness=float(track.score),
            hit_count=int(track.hit_count),
            track_age=int(track.age),
            age_since_disappear=int(max(0, frame_index - track.last_seen_frame)),
            slot_confidence=float(
                np.clip(
                    0.35 * min(1.0, track.hit_count / max(self.min_hits_for_slot + 2, 1))
                    + 0.35 * min(1.0, track.age / max(self.min_track_age_for_slot + 4, 1))
                    + 0.30 * np.clip(track.score, 0.0, 1.0),
                    0.0,
                    1.0,
                )
            ),
            was_occluded_before_disappear=bool(track.missed_frames <= max(2, self.keepalive_frames // 2)),
        )
        self._next_slot_id += 1
        slots = [existing for existing in self._identity_slots_by_prototype[prototype_id] if existing.source_track_id != slot.source_track_id]
        slots.append(slot)
        slots.sort(key=lambda item: self._identity_slot_rank(item), reverse=True)
        self._identity_slots_by_prototype[prototype_id] = deque(slots[: self.slot_topk_per_proto])
        return 1

    def _get_or_create_continuation_track(
        self,
        continuation,
        proposal_track: TrackState,
        frame_index: int,
    ) -> TrackState:
        existing = self._tracks.get(int(continuation.track_id))
        if existing is None:
            existing = TrackState(
                track_id=int(continuation.track_id),
                state="retired",
                box=continuation.last_bbox,
                centroid=continuation.last_center,
                last_bbox=continuation.last_bbox,
                last_center=continuation.last_center,
                signature=continuation.feature_ema.copy(),
                anchor_signature=continuation.feature_ema.copy(),
                last_feature=proposal_track.last_feature.copy(),
                prototype_id=int(continuation.prototype_id),
                lineage_id=int(getattr(continuation, "source_lineage_id", -1)),
                prototype_signature=np.zeros_like(proposal_track.signature, dtype=np.float32),
                velocity=continuation.velocity.copy(),
                score=float(continuation.last_objectness),
                hits=int(continuation.hit_count),
                hit_count=int(continuation.hit_count),
                age=int(continuation.track_age),
                missed_frames=int(continuation.age_since_last_seen),
                miss_count=int(continuation.age_since_last_seen),
                last_seen_frame=int(continuation.last_seen_frame),
                gap_length=int(continuation.age_since_last_seen),
                active=False,
                dormant=False,
                ghost=False,
                retired=True,
                continuity_lineage_id=(
                    int(getattr(continuation, "continuity_lineage_id"))
                    if getattr(continuation, "continuity_lineage_id", None) is not None
                    else int(getattr(continuation, "source_lineage_id", -1))
                ),
            )
            self._tracks[existing.track_id] = existing
            self._next_track_id = max(self._next_track_id, existing.track_id + 1)
        existing.prototype_id = int(continuation.prototype_id)
        existing.lineage_id = int(getattr(continuation, "source_lineage_id", -1))
        existing.continuity_lineage_id = (
            int(getattr(continuation, "continuity_lineage_id"))
            if getattr(continuation, "continuity_lineage_id", None) is not None
            else int(getattr(continuation, "source_lineage_id", -1))
        )
        existing.last_center = continuation.last_center
        existing.last_bbox = continuation.last_bbox
        existing.velocity = (
            0.6 * continuation.velocity.astype(np.float32) + 0.4 * proposal_track.velocity.astype(np.float32)
        ).astype(np.float32)
        existing.anchor_signature = continuation.feature_ema.copy()
        existing.signature = continuation.feature_ema.copy()
        existing.score = float(continuation.last_objectness)
        existing.last_seen_frame = int(continuation.last_seen_frame)
        existing.gap_length = int(max(0, frame_index - continuation.last_seen_frame))
        _set_track_state(existing, "retired")
        return existing

    def _continuation_resurrection_cost(
        self,
        *,
        continuation,
        proposal_track: TrackState,
        frame_index: int,
        frame_shape: tuple[int, int],
        frame_diagonal: float,
    ) -> float:
        gap = max(1, int(frame_index - continuation.last_seen_frame))
        gap_decay = float(np.exp(-gap / self.tau_g))
        weights = _continuation_resurrection_weights(gap_decay)
        predicted_center = (
            float(continuation.last_center[0] + continuation.velocity[0] * gap),
            float(continuation.last_center[1] + continuation.velocity[1] * gap),
        )
        d_geom = min(1.0, _centroid_distance(predicted_center, proposal_track.centroid) / max(frame_diagonal, 1e-6))
        d_feat = _feature_distance(continuation.feature_ema, proposal_track.signature)
        d_motion = _continuation_motion_distance(continuation, proposal_track, gap)
        d_obj = _continuation_object_consistency_distance(continuation, proposal_track)
        d_age = min(1.0, gap / max(float(gap + 16), 1.0))
        return float(
            weights["geom"] * d_geom
            + weights["feat"] * d_feat
            + weights["motion"] * d_motion
            + weights["obj"] * d_obj
            + weights["age"] * d_age
        )

    def _get_or_create_anchor_track(
        self,
        anchor: dict[str, object],
        proposal_track: TrackState,
        frame_index: int,
    ) -> TrackState:
        track_id = int(anchor.get("old_track_id", -1))
        existing = self._tracks.get(track_id)
        if existing is None:
            existing = TrackState(
                track_id=track_id,
                state="retired",
                box=tuple(int(value) for value in anchor.get("last_bbox", proposal_track.last_bbox)),
                centroid=tuple(float(value) for value in anchor.get("last_center", proposal_track.last_center)),
                last_bbox=tuple(int(value) for value in anchor.get("last_bbox", proposal_track.last_bbox)),
                last_center=tuple(float(value) for value in anchor.get("last_center", proposal_track.last_center)),
                signature=proposal_track.signature.copy(),
                anchor_signature=proposal_track.anchor_signature.copy(),
                last_feature=proposal_track.last_feature.copy(),
                prototype_id=None,
                lineage_id=int(anchor.get("old_lineage_id", -1)),
                prototype_signature=np.zeros_like(proposal_track.signature, dtype=np.float32),
                velocity=proposal_track.velocity.copy(),
                score=float(anchor.get("last_objectness", proposal_track.score)),
                hits=int(anchor.get("hit_count", proposal_track.hit_count)),
                hit_count=int(anchor.get("hit_count", proposal_track.hit_count)),
                age=int(anchor.get("track_age", proposal_track.age)),
                missed_frames=int(anchor.get("age_since_last_seen", 0)),
                miss_count=int(anchor.get("age_since_last_seen", 0)),
                last_seen_frame=int(anchor.get("last_alive_frame", max(0, frame_index - 1))),
                gap_length=int(anchor.get("age_since_last_seen", 0)),
                active=False,
                dormant=False,
                ghost=False,
                retired=True,
                continuity_lineage_id=(
                    int(anchor.get("continuity_lineage_id"))
                    if anchor.get("continuity_lineage_id") not in (None, "", "None")
                    else int(anchor.get("old_lineage_id", -1))
                ),
            )
            self._tracks[existing.track_id] = existing
            self._next_track_id = max(self._next_track_id, existing.track_id + 1)
        existing.prototype_id = (
            None if int(anchor.get("old_prototype_id", -1)) < 0 else int(anchor.get("old_prototype_id", -1))
        )
        existing.lineage_id = int(anchor.get("old_lineage_id", -1))
        existing.continuity_lineage_id = (
            int(anchor.get("continuity_lineage_id"))
            if anchor.get("continuity_lineage_id") not in (None, "", "None")
            else int(anchor.get("old_lineage_id", -1))
        )
        existing.last_center = tuple(float(value) for value in anchor.get("last_center", proposal_track.last_center))
        existing.last_bbox = tuple(int(value) for value in anchor.get("last_bbox", proposal_track.last_bbox))
        existing.centroid = existing.last_center
        existing.box = existing.last_bbox
        existing.velocity = proposal_track.velocity.copy()
        existing.anchor_signature = proposal_track.anchor_signature.copy()
        existing.signature = proposal_track.signature.copy()
        existing.score = float(anchor.get("last_objectness", proposal_track.score))
        existing.last_seen_frame = int(anchor.get("last_alive_frame", max(0, frame_index - 1)))
        existing.gap_length = int(anchor.get("age_since_last_seen", 0))
        _set_track_state(existing, "retired")
        return existing

    def _recovery_anchor_cost(
        self,
        *,
        anchor: dict[str, object],
        proposal_track: TrackState,
        frame_index: int,
        frame_diagonal: float,
    ) -> float:
        gap = max(1, int(anchor.get("age_since_last_seen", max(1, frame_index - int(anchor.get("last_alive_frame", 0))))))
        gap_decay = float(np.exp(-gap / self.tau_g))
        weights = _continuation_resurrection_weights(gap_decay)
        last_center = tuple(float(value) for value in anchor.get("last_center", proposal_track.centroid))
        velocity = np.asarray(anchor.get("velocity", np.zeros(2, dtype=np.float32)), dtype=np.float32)
        predicted_center = (
            float(last_center[0] + velocity[0] * gap),
            float(last_center[1] + velocity[1] * gap),
        )
        feature_ema = np.asarray(anchor.get("feature_ema", proposal_track.anchor_signature), dtype=np.float32)
        d_geom = min(1.0, _centroid_distance(predicted_center, proposal_track.centroid) / max(frame_diagonal, 1e-6))
        d_feat = _feature_distance(feature_ema, proposal_track.signature)
        motion_norm = max(frame_diagonal, 1e-6)
        proposal_motion = np.asarray(proposal_track.velocity, dtype=np.float32)
        d_motion = min(1.0, float(np.linalg.norm(proposal_motion - velocity)) / motion_norm)
        last_objectness = float(anchor.get("last_objectness", proposal_track.score))
        d_obj = min(1.0, abs(float(proposal_track.score) - last_objectness))
        d_age = min(1.0, gap / max(float(gap + 16), 1.0))
        return float(
            weights["geom"] * d_geom
            + weights["feat"] * d_feat
            + weights["motion"] * d_motion
            + weights["obj"] * d_obj
            + weights["age"] * d_age
        )

    def _slot_candidates_for_prototype(self, prototype_id: int, *, frame_index: int) -> list[IdentitySlot]:
        self._refresh_identity_slots(frame_index)
        return list(self._identity_slots_by_prototype.get(int(prototype_id), deque()))

    def _get_or_create_slot_track(
        self,
        slot: IdentitySlot,
        proposal_track: TrackState,
        frame_index: int,
    ) -> TrackState:
        existing = self._tracks.get(int(slot.source_track_id))
        if existing is None:
            existing = TrackState(
                track_id=int(slot.source_track_id),
                state="retired",
                box=slot.last_bbox,
                centroid=slot.last_center,
                last_bbox=slot.last_bbox,
                last_center=slot.last_center,
                signature=slot.feature_ema.copy(),
                anchor_signature=slot.feature_ema.copy(),
                last_feature=proposal_track.last_feature.copy(),
                prototype_id=int(slot.prototype_id),
                lineage_id=None,
                prototype_signature=np.zeros_like(proposal_track.signature, dtype=np.float32),
                velocity=slot.velocity.copy(),
                score=float(slot.last_objectness),
                hits=int(slot.hit_count),
                hit_count=int(slot.hit_count),
                age=int(slot.track_age),
                missed_frames=int(slot.age_since_disappear),
                miss_count=int(slot.age_since_disappear),
                last_seen_frame=int(slot.disappear_frame),
                gap_length=int(slot.age_since_disappear),
                active=False,
                dormant=False,
                ghost=False,
                retired=True,
                continuity_lineage_id=None,
            )
            self._tracks[existing.track_id] = existing
            self._next_track_id = max(self._next_track_id, existing.track_id + 1)
        existing.prototype_id = int(slot.prototype_id)
        existing.lineage_id = None
        existing.continuity_lineage_id = None
        existing.last_center = slot.last_center
        existing.last_bbox = slot.last_bbox
        existing.velocity = (
            0.6 * slot.velocity.astype(np.float32) + 0.4 * proposal_track.velocity.astype(np.float32)
        ).astype(np.float32)
        existing.anchor_signature = slot.feature_ema.copy()
        existing.signature = slot.feature_ema.copy()
        existing.score = float(slot.last_objectness)
        existing.last_seen_frame = int(slot.disappear_frame)
        existing.gap_length = int(max(0, frame_index - slot.disappear_frame))
        _set_track_state(existing, "retired")
        return existing

    def _consume_identity_slot(self, slot: IdentitySlot) -> None:
        slots = self._identity_slots_by_prototype.get(int(slot.prototype_id))
        if not slots:
            return
        kept = [existing for existing in slots if existing.slot_id != slot.slot_id]
        if kept:
            self._identity_slots_by_prototype[int(slot.prototype_id)] = deque(kept)
        else:
            self._identity_slots_by_prototype.pop(int(slot.prototype_id), None)

    def _identity_slot_rank(self, slot: IdentitySlot) -> float:
        recency = 1.0 / max(1.0, float(slot.age_since_disappear))
        stability = min(1.0, float(slot.hit_count) / max(self.min_hits_for_slot + 2, 1))
        maturity = min(1.0, float(slot.track_age) / max(self.min_track_age_for_slot + 4, 1))
        return float(0.45 * slot.slot_confidence + 0.25 * stability + 0.20 * maturity + 0.10 * recency)

    def _slot_resurrection_cost(
        self,
        *,
        slot: IdentitySlot,
        proposal_track: TrackState,
        frame_index: int,
        frame_shape: tuple[int, int],
        frame_diagonal: float,
    ) -> float:
        gap = max(1, frame_index - slot.disappear_frame)
        gap_decay = float(np.exp(-gap / self.tau_g))
        weights = _slot_resurrection_weights(gap_decay)
        predicted_center = (
            float(slot.last_center[0] + slot.velocity[0] * gap),
            float(slot.last_center[1] + slot.velocity[1] * gap),
        )
        d_geom = min(1.0, _centroid_distance(predicted_center, proposal_track.centroid) / max(frame_diagonal, 1e-6))
        d_feat = _feature_distance(slot.feature_ema, proposal_track.signature)
        d_motion = _slot_motion_distance(slot, proposal_track, gap)
        d_obj = _slot_object_consistency_distance(slot, proposal_track)
        d_age = min(1.0, gap / max(float(self.slot_max_gap), 1.0))
        return float(
            weights["geom"] * d_geom
            + weights["feat"] * d_feat
            + weights["motion"] * d_motion
            + weights["obj"] * d_obj
            + weights["age"] * d_age
        )

    def _concept_gated_resurrection_cost(
        self,
        *,
        proposal_track: TrackState,
        candidate_track: TrackState,
        frame_index: int,
        frame_shape: tuple[int, int],
        frame_diagonal: float,
    ) -> float:
        gap = max(1, frame_index - candidate_track.last_seen_frame)
        gap_decay = float(np.exp(-gap / self.tau_g))
        weights = _resurrection_weights(gap_decay)
        predicted_box, predicted_centroid = _predict_track_state(
            candidate_track,
            frame_shape=frame_shape,
            prediction_steps_cap=max(self.prediction_steps_cap, gap),
        )
        d_geom = min(1.0, _centroid_distance(predicted_centroid, proposal_track.centroid) / max(frame_diagonal, 1e-6))
        d_feat = _feature_distance(candidate_track.anchor_signature, proposal_track.signature)
        d_motion = _motion_distance(candidate_track, proposal_track, gap)
        proposal_feature = ProposalFeature(
            proposal_index=-1,
            box=proposal_track.box,
            centroid=proposal_track.centroid,
            score=proposal_track.score,
            signature=proposal_track.signature,
        )
        d_obj = _object_consistency_distance(candidate_track, proposal_feature)
        return float(
            weights["geom"] * d_geom
            + weights["feat"] * d_feat
            + weights["motion"] * d_motion
            + weights["obj"] * d_obj
        )


def _greedy_reactivation_matches(candidates: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: float(item["cost"]))
    matched_tracks: set[int] = set()
    matched_proposals: set[int] = set()
    accepted: list[dict[str, float | int]] = []
    for candidate in ordered:
        track_id = int(candidate["track_id"])
        proposal_index = int(candidate["proposal_index"])
        if track_id in matched_tracks or proposal_index in matched_proposals:
            continue
        if float(candidate["cost"]) > float(candidate["threshold"]):
            continue
        matched_tracks.add(track_id)
        matched_proposals.add(proposal_index)
        accepted.append(candidate)
    return accepted


def _build_cost_matrix(
    tracks: list[TrackState],
    projected_states: list[tuple[Box, tuple[float, float]]],
    proposals: list[ProposalFeature],
    frame_shape: tuple[int, int],
    beta_iou: float,
    beta_center: float,
    beta_feat: float,
    max_center_distance: float | None,
) -> np.ndarray:
    cost_matrix = np.zeros((len(tracks), len(proposals)), dtype=np.float32)
    frame_diagonal = float(np.hypot(frame_shape[1], frame_shape[0]))
    center_normalizer = max_center_distance if max_center_distance is not None else frame_diagonal
    center_normalizer = max(center_normalizer, 1e-6)
    for track_index, track in enumerate(tracks):
        projected_box, projected_centroid = projected_states[track_index]
        for proposal_index, proposal in enumerate(proposals):
            d_iou = 1.0 - bbox_iou(projected_box, proposal.box)
            d_center = min(1.5, _centroid_distance(projected_centroid, proposal.centroid) / center_normalizer)
            d_feat = _feature_distance(track.signature, proposal.signature)
            if track.prototype_id is not None and np.linalg.norm(track.prototype_signature) > 0.0:
                d_proto = _feature_distance(track.prototype_signature, proposal.signature)
                d_feat = 0.65 * d_feat + 0.35 * d_proto
            cost_matrix[track_index, proposal_index] = beta_iou * d_iou + beta_center * d_center + beta_feat * d_feat
    return cost_matrix


def _greedy_accept_matches(
    cost_matrix: np.ndarray,
    tracks: list[TrackState],
    proposals: list[ProposalFeature],
    max_match_cost: float,
    min_feature_similarity: float | None = None,
    missed_similarity_boost: float = 0.0,
) -> list[tuple[int, int, float]]:
    if cost_matrix.size == 0:
        return []
    candidates: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        for proposal_index, proposal in enumerate(proposals):
            candidates.append((float(cost_matrix[track_index, proposal_index]), int(track.track_id), int(proposal.proposal_index)))
    candidates.sort(key=lambda item: item[0])

    matched_tracks: set[int] = set()
    matched_proposals: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for cost, track_id, proposal_index in candidates:
        if track_id in matched_tracks or proposal_index in matched_proposals:
            continue
        track = next(track for track in tracks if track.track_id == track_id)
        proposal = next(proposal for proposal in proposals if proposal.proposal_index == proposal_index)
        dynamic_match_cost = max(
            0.30,
            float(max_match_cost) - min(0.18, 0.03 * max(track.missed_frames, 0)),
        )
        if cost > dynamic_match_cost:
            continue
        if min_feature_similarity is not None:
            similarity_floor = float(
                np.clip(min_feature_similarity + missed_similarity_boost * max(track.missed_frames, 0), 0.0, 0.98)
            )
            if _feature_similarity(track.signature, proposal.signature) < similarity_floor:
                continue
            if track.missed_frames > 0:
                if track.prototype_id is not None and np.linalg.norm(track.prototype_signature) > 0.0:
                    proto_similarity = _feature_similarity(track.prototype_signature, proposal.signature)
                else:
                    proto_similarity = _feature_similarity(track.anchor_signature, proposal.signature)
                if proto_similarity < max(0.10, similarity_floor - 0.05):
                    continue
        matched_tracks.add(track_id)
        matched_proposals.add(proposal_index)
        matches.append((track_id, proposal_index, cost))
    return matches


def _summarize_cost_matrix(cost_matrix: np.ndarray, accepted_matches: int) -> MatchCostStats:
    if cost_matrix.size == 0:
        return MatchCostStats(0.0, 0.0, 0.0, 0, accepted_matches)
    return MatchCostStats(
        min_cost=float(cost_matrix.min()),
        mean_cost=float(cost_matrix.mean()),
        max_cost=float(cost_matrix.max()),
        candidate_pairs=int(cost_matrix.size),
        accepted_matches=int(accepted_matches),
    )


def _build_proposal_feature(
    proposal_index: int,
    proposal: Proposal,
    encoding: SpikeEncoding,
    heatmap: np.ndarray,
    current_frame: np.ndarray,
) -> ProposalFeature:
    x1, y1, x2, y2 = proposal.box
    patch_on = encoding.on_spikes[y1:y2, x1:x2]
    patch_off = encoding.off_spikes[y1:y2, x1:x2]
    patch_spike = encoding.spike_response[y1:y2, x1:x2]
    patch_heatmap = heatmap[y1:y2, x1:x2]
    patch_gray = encoding.current_gray[y1:y2, x1:x2]
    patch_rgb = current_frame[y1:y2, x1:x2].astype(np.float32) / 255.0

    height, width = heatmap.shape
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    area_ratio = (box_width * box_height) / float(height * width)
    aspect_ratio = min(4.0, box_width / max(1.0, float(box_height))) / 4.0

    signature = np.array(
        [
            float(patch_on.mean()) if patch_on.size else 0.0,
            float(patch_off.mean()) if patch_off.size else 0.0,
            float(patch_spike.mean()) if patch_spike.size else 0.0,
            float(patch_heatmap.mean()) if patch_heatmap.size else 0.0,
            float(patch_heatmap.std()) if patch_heatmap.size else 0.0,
            float(patch_gray.mean()) if patch_gray.size else 0.0,
            float(patch_gray.std()) if patch_gray.size else 0.0,
            float(patch_rgb[..., 0].mean()) if patch_rgb.size else 0.0,
            float(patch_rgb[..., 1].mean()) if patch_rgb.size else 0.0,
            float(patch_rgb[..., 2].mean()) if patch_rgb.size else 0.0,
            float(area_ratio),
            float(aspect_ratio),
            float(proposal.score),
        ],
        dtype=np.float32,
    )
    return ProposalFeature(
        proposal_index=proposal_index,
        box=proposal.box,
        centroid=proposal.centroid,
        score=proposal.score,
        signature=_normalize_signature(signature),
    )


def _set_track_state(track: TrackState, state: str) -> None:
    track.state = state
    track.active = state == "active"
    track.dormant = state == "dormant"
    track.ghost = state == "ghost"
    track.retired = state == "retired"


def _clone_track(track: TrackState) -> TrackState:
    return TrackState(
        track_id=track.track_id,
        state=track.state,
        box=track.box,
        centroid=track.centroid,
        last_bbox=track.last_bbox,
        last_center=track.last_center,
        signature=track.signature.copy(),
        anchor_signature=track.anchor_signature.copy(),
        last_feature=track.last_feature.copy(),
        prototype_id=track.prototype_id,
        lineage_id=track.lineage_id,
        prototype_signature=track.prototype_signature.copy(),
        velocity=track.velocity.copy(),
        score=track.score,
        hits=track.hits,
        hit_count=track.hit_count,
        age=track.age,
        missed_frames=track.missed_frames,
        miss_count=track.miss_count,
        last_seen_frame=track.last_seen_frame,
        gap_length=track.gap_length,
        active=track.active,
        dormant=track.dormant,
        ghost=track.ghost,
        retired=track.retired,
        continuity_lineage_id=track.continuity_lineage_id,
    )


def _reactivation_weights(gap_decay: float) -> dict[str, float]:
    raw = {
        "pos": 0.50 * gap_decay + 0.10,
        "feat": 0.20 + 0.10 * gap_decay,
        "proto": 0.10 + 0.50 * (1.0 - gap_decay),
        "obj": 0.15,
    }
    total = sum(raw.values())
    return {key: float(value / total) for key, value in raw.items()}


def _track_reattach_weights(gap_decay: float) -> dict[str, float]:
    raw = {
        "pos": 0.25 * gap_decay + 0.05,
        "feat": 0.20 + 0.10 * gap_decay,
        "proto": 0.35 + 0.45 * (1.0 - gap_decay),
        "obj": 0.10,
    }
    total = sum(raw.values())
    return {key: float(value / total) for key, value in raw.items()}


def _legacy_reactivation_weights() -> dict[str, float]:
    return {"pos": 0.18, "feat": 0.32, "proto": 0.35, "obj": 0.15}


def _resurrection_weights(gap_decay: float) -> dict[str, float]:
    raw = {
        "geom": 0.45 * gap_decay + 0.05,
        "feat": 0.20 + 0.25 * (1.0 - gap_decay),
        "motion": 0.15 + 0.15 * (1.0 - gap_decay),
        "obj": 0.15,
    }
    total = sum(raw.values())
    return {key: float(value / total) for key, value in raw.items()}


def _slot_resurrection_weights(gap_decay: float) -> dict[str, float]:
    raw = {
        "geom": 0.40 * gap_decay + 0.05,
        "feat": 0.25 + 0.20 * (1.0 - gap_decay),
        "motion": 0.15 + 0.10 * (1.0 - gap_decay),
        "obj": 0.10,
        "age": 0.10,
    }
    total = sum(raw.values())
    return {key: float(value / total) for key, value in raw.items()}


def _continuation_resurrection_weights(gap_decay: float) -> dict[str, float]:
    raw = {
        "geom": 0.38 * gap_decay + 0.07,
        "feat": 0.24 + 0.18 * (1.0 - gap_decay),
        "motion": 0.16 + 0.12 * (1.0 - gap_decay),
        "obj": 0.12,
        "age": 0.10,
    }
    total = sum(raw.values())
    return {key: float(value / total) for key, value in raw.items()}


def _prototype_similarity(track: TrackState, proposal: ProposalFeature) -> float:
    if track.prototype_id is not None and np.linalg.norm(track.prototype_signature) > 0.0:
        distance = _feature_distance(track.prototype_signature, proposal.signature)
    else:
        distance = _feature_distance(track.anchor_signature, proposal.signature)
    return float(max(0.0, 1.0 - distance))


def _object_consistency_distance(track: TrackState, proposal: ProposalFeature) -> float:
    score_distance = abs(float(proposal.score) - float(track.score))
    area_distance = abs(float(proposal.signature[10]) - float(track.last_feature[10])) if track.last_feature.size > 10 else 0.0
    aspect_distance = abs(float(proposal.signature[11]) - float(track.last_feature[11])) if track.last_feature.size > 11 else 0.0
    return float(np.clip((score_distance + area_distance + aspect_distance) / 3.0, 0.0, 1.0))


def _normalize_signature(signature: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(signature))
    if norm < 1e-6:
        return np.zeros_like(signature, dtype=np.float32)
    return signature.astype(np.float32) / norm


def _feature_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 1.0
    similarity = float(np.clip(np.dot(_normalize_signature(a), _normalize_signature(b)), -1.0, 1.0))
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * similarity)) / np.sqrt(2.0))


def _feature_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(0.0, 1.0 - _feature_distance(a, b)))


def _centroid_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _motion_distance(candidate_track: TrackState, proposal_track: TrackState, gap: int) -> float:
    expected_motion = candidate_track.velocity * float(max(1, gap))
    observed_motion = np.array(
        [
            proposal_track.centroid[0] - candidate_track.last_center[0],
            proposal_track.centroid[1] - candidate_track.last_center[1],
        ],
        dtype=np.float32,
    )
    motion_error = float(np.linalg.norm(observed_motion - expected_motion))
    scale = max(1.0, float(np.linalg.norm(observed_motion) + np.linalg.norm(expected_motion)))
    magnitude_cost = min(1.0, motion_error / scale)

    expected_norm = float(np.linalg.norm(expected_motion))
    observed_norm = float(np.linalg.norm(observed_motion))
    if expected_norm < 1e-6 and observed_norm < 1e-6:
        direction_cost = 0.0
    elif expected_norm < 1e-6 or observed_norm < 1e-6:
        direction_cost = 0.5
    else:
        cosine = float(
            np.clip(
                np.dot(expected_motion, observed_motion) / max(expected_norm * observed_norm, 1e-6),
                -1.0,
                1.0,
            )
        )
        direction_cost = 0.5 * (1.0 - cosine)

    return float(np.clip(0.65 * magnitude_cost + 0.35 * direction_cost, 0.0, 1.0))


def _slot_motion_distance(slot: IdentitySlot, proposal_track: TrackState, gap: int) -> float:
    expected_motion = slot.velocity * float(max(1, gap))
    observed_motion = np.array(
        [
            proposal_track.centroid[0] - slot.last_center[0],
            proposal_track.centroid[1] - slot.last_center[1],
        ],
        dtype=np.float32,
    )
    motion_error = float(np.linalg.norm(observed_motion - expected_motion))
    scale = max(1.0, float(np.linalg.norm(observed_motion) + np.linalg.norm(expected_motion)))
    magnitude_cost = min(1.0, motion_error / scale)

    expected_norm = float(np.linalg.norm(expected_motion))
    observed_norm = float(np.linalg.norm(observed_motion))
    if expected_norm < 1e-6 and observed_norm < 1e-6:
        direction_cost = 0.0
    elif expected_norm < 1e-6 or observed_norm < 1e-6:
        direction_cost = 0.5
    else:
        cosine = float(
            np.clip(
                np.dot(expected_motion, observed_motion) / max(expected_norm * observed_norm, 1e-6),
                -1.0,
                1.0,
            )
        )
        direction_cost = 0.5 * (1.0 - cosine)

    return float(np.clip(0.65 * magnitude_cost + 0.35 * direction_cost, 0.0, 1.0))


def _continuation_motion_distance(continuation, proposal_track: TrackState, gap: int) -> float:
    expected_motion = continuation.velocity * float(max(1, gap))
    observed_motion = np.array(
        [
            proposal_track.centroid[0] - continuation.last_center[0],
            proposal_track.centroid[1] - continuation.last_center[1],
        ],
        dtype=np.float32,
    )
    motion_error = float(np.linalg.norm(observed_motion - expected_motion))
    scale = max(1.0, float(np.linalg.norm(observed_motion) + np.linalg.norm(expected_motion)))
    magnitude_cost = min(1.0, motion_error / scale)

    expected_norm = float(np.linalg.norm(expected_motion))
    observed_norm = float(np.linalg.norm(observed_motion))
    if expected_norm < 1e-6 and observed_norm < 1e-6:
        direction_cost = 0.0
    elif expected_norm < 1e-6 or observed_norm < 1e-6:
        direction_cost = 0.5
    else:
        cosine = float(
            np.clip(
                np.dot(expected_motion, observed_motion) / max(expected_norm * observed_norm, 1e-6),
                -1.0,
                1.0,
            )
        )
        direction_cost = 0.5 * (1.0 - cosine)

    return float(np.clip(0.65 * magnitude_cost + 0.35 * direction_cost, 0.0, 1.0))


def _slot_object_consistency_distance(slot: IdentitySlot, proposal_track: TrackState) -> float:
    score_distance = abs(float(proposal_track.score) - float(slot.last_objectness))
    proposal_shape = proposal_track.last_feature[10:13] if proposal_track.last_feature.size >= 13 else proposal_track.last_feature
    if slot.shape_signature.size == 0 or proposal_shape.size == 0:
        shape_distance = 0.0
    else:
        shape_distance = _feature_distance(_normalize_signature(slot.shape_signature), _normalize_signature(proposal_shape))
    return float(np.clip(0.55 * score_distance + 0.45 * shape_distance, 0.0, 1.0))


def _continuation_object_consistency_distance(continuation, proposal_track: TrackState) -> float:
    score_distance = abs(float(proposal_track.score) - float(continuation.last_objectness))
    proposal_shape = proposal_track.last_feature[10:13] if proposal_track.last_feature.size >= 13 else proposal_track.last_feature
    if continuation.shape_signature.size == 0 or proposal_shape.size == 0:
        shape_distance = 0.0
    else:
        shape_distance = _feature_distance(
            _normalize_signature(continuation.shape_signature),
            _normalize_signature(proposal_shape),
        )
    return float(np.clip(0.55 * score_distance + 0.45 * shape_distance, 0.0, 1.0))


def _project_track_state(
    track: TrackState,
    frame_shape: tuple[int, int],
    use_linear_prediction: bool,
    prediction_steps_cap: int,
) -> tuple[Box, tuple[float, float]]:
    if not use_linear_prediction:
        return track.box, track.centroid
    return _predict_track_state(track, frame_shape, prediction_steps_cap)


def _predict_track_state(
    track: TrackState,
    frame_shape: tuple[int, int],
    prediction_steps_cap: int,
) -> tuple[Box, tuple[float, float]]:
    steps = min(max(track.gap_length, 0) + 1, prediction_steps_cap)
    shift = track.velocity * float(steps)
    x1, y1, x2, y2 = track.box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    raw_box = (
        int(round(x1 + shift[0])),
        int(round(y1 + shift[1])),
        int(round(x2 + shift[0])),
        int(round(y2 + shift[1])),
    )
    predicted_box = _clamp_box(raw_box, frame_shape, width, height)
    predicted_centroid = (
        float(0.5 * (predicted_box[0] + predicted_box[2])),
        float(0.5 * (predicted_box[1] + predicted_box[3])),
    )
    return predicted_box, predicted_centroid


def _clamp_box(box: Box, frame_shape: tuple[int, int], width: int, height: int) -> Box:
    frame_height, frame_width = frame_shape
    x1 = int(np.clip(box[0], 0, max(frame_width - width, 0)))
    y1 = int(np.clip(box[1], 0, max(frame_height - height, 0)))
    x2 = min(frame_width, x1 + width)
    y2 = min(frame_height, y1 + height)
    return x1, y1, x2, y2
