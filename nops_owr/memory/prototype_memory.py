"""Prototype memory with concept recovery and prototype-centric continuation banks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nops_owr.tracking.temporal_identity import TrackAssignment, TrackState


@dataclass(slots=True)
class IdentityContinuation:
    continuation_id: int
    continuation_uid: str
    prototype_id: int
    source_prototype_id: int
    source_lineage_id: int
    track_id: int
    write_frame: int
    last_seen_frame: int
    last_center: tuple[float, float]
    last_bbox: tuple[int, int, int, int]
    velocity: np.ndarray
    feature_ema: np.ndarray
    shape_signature: np.ndarray
    last_objectness: float
    track_age: int
    hit_count: int
    age_since_last_seen: int
    continuation_confidence: float
    was_occluded_before_disappear: bool
    runtime_owner_lineage_id: int | None = None
    continuity_lineage_id: int | None = None
    origin_lineage_id: int | None = None
    old_identity_ref_track_id: int | None = None
    old_identity_ref_prototype_id: int | None = None
    continuity_key_valid: bool = False


@dataclass(slots=True)
class TemporaryAttachSlot:
    temp_attach_id: int
    lineage_id: int
    anchor_prototype_id: int | None
    start_frame: int
    last_seen_frame: int
    support_count: int
    promote_ready: bool
    expired: bool
    source_track_id: int | None
    source_prototype_id: int | None
    last_center: tuple[float, float]
    last_bbox: tuple[int, int, int, int]
    last_signature: np.ndarray
    last_objectness: float


@dataclass(slots=True)
class RecoveryIdentityAnchor:
    anchor_uid: str
    old_track_id: int
    old_prototype_id: int
    old_lineage_id: int
    source_frame_id: int
    last_alive_frame: int
    gap_length_at_creation: int
    restore_priority: float
    continuation_hint: str
    anchor_state: str
    anchor_origin: str
    age_since_last_seen: int
    last_center: tuple[float, float]
    last_bbox: tuple[int, int, int, int]
    velocity: np.ndarray
    feature_ema: np.ndarray
    last_objectness: float
    track_age: int
    hit_count: int
    runtime_owner_lineage_id: int | None = None
    continuity_lineage_id: int | None = None
    origin_lineage_id: int | None = None
    old_identity_ref_track_id: int | None = None
    old_identity_ref_prototype_id: int | None = None
    continuity_key_valid: bool = False


@dataclass(slots=True)
class PrototypeState:
    prototype_id: int
    lineage_id: int
    parent_lineage_id: int | None
    signature: np.ndarray
    hits: int
    age: int
    strength: float
    inactive_steps: int
    active: bool
    is_active: bool
    retired: bool
    birth_frame: int
    created_frame: int
    last_updated_frame: int
    last_track_id: int | None
    is_protected: bool
    protected_until_frame: int
    merged_from: tuple[int, ...] = ()
    split_from: int | None = None
    continuation_bank_ref: str = ""
    active_track_refs: list[int] = field(default_factory=list)
    continuation_bank: list[IdentityContinuation] = field(default_factory=list)
    runtime_owner_lineage_id: int | None = None
    continuity_lineage_id: int | None = None
    origin_lineage_id: int | None = None
    continuity_key_valid: bool = False


@dataclass(slots=True)
class ConceptLineage:
    lineage_id: int
    current_head_prototype_id: int | None
    active_prototype_ids: list[int] = field(default_factory=list)
    archived_prototype_ids: list[int] = field(default_factory=list)
    birth_frame: int = 0
    last_update_frame: int = 0
    status: str = "active"
    continuation_bank: list[IdentityContinuation] = field(default_factory=list)
    merged_lineage_ids: list[int] = field(default_factory=list)
    alias_lineage_ids: list[int] = field(default_factory=list)
    temp_attach_slot: TemporaryAttachSlot | None = None
    recovery_identity_anchors: list[RecoveryIdentityAnchor] = field(default_factory=list)
    promotion_candidate_id: int | None = None
    promotion_pending_flag: bool = False
    promotion_window_progress: int = 0
    promotion_support_count: int = 0
    promotion_start_frame: int | None = None
    last_promotion_decision: str = "keep_head"
    post_promotion_cooldown_until: int = -1


@dataclass(slots=True)
class PrototypeAssignment:
    track_id: int
    prototype_id: int
    lineage_id: int
    head_prototype_id: int | None
    head_prototype_id_before: int | None
    head_prototype_id_after: int | None
    similarity: float
    distance: float
    soft_similarity: float
    update_weight: float
    action: str
    action_type: str
    selected_prototype_state: str
    head_score: float | None
    best_active_sibling_score: float | None
    best_archived_sibling_score: float | None
    birth_trigger_score: float | None
    score_margin_vs_current_head: float | None
    head_switched: bool
    box: tuple[int, int, int, int]
    score: float
    previous_prototype_id: int | None
    previous_lineage_id: int | None
    concept_only_recovery: bool
    new_prototype_created: bool
    prototype_protected: bool
    prototype_signature: np.ndarray
    matched_lineage_id: int | None = None
    recovery_attach_target: str = "none"
    recovery_attach_target_id: int | None = None
    current_head_prototype_id: int | None = None
    promotion_candidate_id: int | None = None
    promotion_pending_flag: bool = False
    promotion_window_progress: int = 0
    promotion_decision: str = "keep_head"
    attach_path_source: str = ""
    attach_score_current_head: float | None = None
    attach_score_active_sibling: float | None = None
    attach_score_archived_sibling: float | None = None
    attach_score_temp_slot: float | None = None
    promote_score_candidate: float | None = None
    promote_score_current_head: float | None = None
    promotion_support_count: int = 0
    promotion_delay_frames: int | None = None
    promotion_success: bool = False
    promotion_regret_flag: bool = False
    temp_attach_used: bool = False
    temp_attach_id: int | None = None
    temp_attach_support_count: int = 0
    temp_attach_promote_ready: bool = False
    temp_attach_expired: bool = False
    attach_branch_entered: bool = False
    temp_attach_eligibility_checked: bool = False
    attach_state_written: bool = False
    temp_attach_force_mode: bool = False
    lineage_seed_id_used: int | None = None
    continuity_lineage_id: int | None = None


@dataclass(slots=True)
class FrameMemoryResult:
    frame_index: int
    assignments: list[PrototypeAssignment]
    active_prototype_ids: list[int]
    retired_prototype_ids: list[int]
    total_prototypes: int
    budget_pruned_ids: list[int]
    protected_prototype_ids: list[int]
    prototype_lineage_lookup: dict[int, int]
    lineage_prototype_lookup: dict[int, list[int]]
    lineage_head_lookup: dict[int, int | None]
    continuation_lookup: dict[int, list[IdentityContinuation]]
    continuation_lineage_lookup: dict[int, list[IdentityContinuation]]
    continuation_continuity_lookup: dict[int, list[IdentityContinuation]]
    temp_attach_lookup: dict[int, dict[str, object]]
    recovery_anchor_lookup: dict[int, list[dict[str, object]]]
    recovery_anchor_continuity_lookup: dict[int, list[dict[str, object]]]
    prototype_continuity_lookup: dict[int, int]
    continuation_bank_count: int
    recovery_anchor_count: int
    continuation_archive_events: int
    continuation_binding_mode: str
    continuation_write_rows: list[dict[str, object]]
    continuation_lifecycle_rows: list[dict[str, object]]
    recovery_anchor_rows: list[dict[str, object]]
    recovery_anchor_lifecycle_rows: list[dict[str, object]]
    prototype_lineage_rows: list[dict[str, object]]


class MinimalPrototypeMemory:
    def __init__(
        self,
        tau_birth: float = 0.40,
        tau_merge: float = 0.22,
        tau_sim: float = 0.25,
        lr_proto: float = 0.45,
        decay_rate: float = 0.02,
        decay_patience: int = 20,
        memory_budget: int = 32,
        decay_floor: float = 0.10,
        tau_proto_attach: float = 0.35,
        tau_obj_attach: float = 0.50,
        use_concept_only_recovery: bool = True,
        protect_linked_prototypes: bool = True,
        enable_explicit_lineage: bool = True,
        preserve_lineage_on_archive: bool = True,
        preserve_lineage_on_replace: bool = True,
        preserve_lineage_on_merge: bool = True,
        allow_alias_lineage: bool = True,
        enable_continuation_bank: bool = True,
        continuation_topk_per_proto: int = 4,
        continuation_topk_per_lineage: int | None = None,
        bind_continuation_to: str = "lineage",
        min_track_age_for_continuation: int = 4,
        min_hits_for_continuation: int = 3,
        continuation_max_gap: int = 96,
        tau_continuation: float | None = None,
        continuation_margin: float | None = None,
        continuation_decay: float = 0.01,
        enable_phase3p_keep_head_default: bool = False,
        enable_phase3p_grouped_gating: bool = False,
        enable_phase3p_birth_suppression: bool = False,
        enable_phase3p_full_stabilization: bool = False,
        keep_head_min: float = 0.42,
        replace_margin: float = 0.08,
        replace_consistency_window: int = 6,
        archived_reactivate_min: float = 0.58,
        birth_margin: float = 0.10,
        post_recovery_birth_suppression_window: int = 12,
        lineage_internal_cooldown_after_head_switch: int = 8,
        head_continuity_bonus: float = 0.03,
        archived_sibling_penalty: float = 0.06,
        newborn_prototype_penalty: float = 0.10,
        enable_phase3d_dual_score: bool = False,
        enable_phase3d_temp_attach: bool = False,
        enable_phase3d_deferred_promotion: bool = False,
        attach_accept_min: float = 0.28,
        promote_margin: float = 0.10,
        promote_support_min: int = 3,
        promotion_window: int = 8,
        promotion_stability_min: float = 0.60,
        post_promotion_cooldown: int = 8,
        temp_attach_ttl: int = 10,
        debug_force_temp_attach: bool = False,
        anchor_ttl: int = 192,
        anchor_expiry_policy: str = "ttl",
        anchor_prune_policy: str = "topk_priority",
        anchor_survives_retire: bool = True,
        anchor_survives_head_replace: bool = True,
        recovery_anchor_topk_per_lineage: int = 4,
        birth_threshold: float | None = None,
        merge_threshold: float | None = None,
        prototype_momentum: float | None = None,
        inactive_patience: int | None = None,
        budget_cap: int | None = None,
        **_: object,
    ) -> None:
        self.tau_birth = float(tau_birth if birth_threshold is None else birth_threshold)
        self.tau_merge = float(tau_merge if merge_threshold is None else merge_threshold)
        self.tau_sim = float(tau_sim)
        self.lr_proto = float(lr_proto if prototype_momentum is None else (1.0 - prototype_momentum))
        self.decay_rate = float(decay_rate)
        self.decay_patience = int(decay_patience if inactive_patience is None else inactive_patience)
        self.memory_budget = int(memory_budget if budget_cap is None else budget_cap)
        self.decay_floor = float(decay_floor)
        self.tau_proto_attach = float(tau_proto_attach)
        self.tau_obj_attach = float(tau_obj_attach)
        self.use_concept_only_recovery = bool(use_concept_only_recovery)
        self.protect_linked_prototypes = bool(protect_linked_prototypes)
        self.enable_explicit_lineage = bool(enable_explicit_lineage)
        self.preserve_lineage_on_archive = bool(preserve_lineage_on_archive)
        self.preserve_lineage_on_replace = bool(preserve_lineage_on_replace)
        self.preserve_lineage_on_merge = bool(preserve_lineage_on_merge)
        self.allow_alias_lineage = bool(allow_alias_lineage)
        self.enable_continuation_bank = bool(enable_continuation_bank)
        effective_topk = continuation_topk_per_proto if continuation_topk_per_lineage is None else continuation_topk_per_lineage
        self.continuation_topk_per_proto = int(max(1, continuation_topk_per_proto))
        self.continuation_topk_per_lineage = int(max(1, effective_topk))
        self.bind_continuation_to = "lineage" if str(bind_continuation_to).lower() != "prototype" else "prototype"
        self.min_track_age_for_continuation = int(max(1, min_track_age_for_continuation))
        self.min_hits_for_continuation = int(max(1, min_hits_for_continuation))
        self.continuation_max_gap = int(max(1, continuation_max_gap))
        self.tau_continuation = 0.62 if tau_continuation is None else float(tau_continuation)
        self.continuation_margin = 0.08 if continuation_margin is None else float(max(0.0, continuation_margin))
        self.continuation_decay = float(max(0.0, continuation_decay))
        self.enable_phase3p_keep_head_default = bool(enable_phase3p_keep_head_default)
        self.enable_phase3p_grouped_gating = bool(enable_phase3p_grouped_gating)
        self.enable_phase3p_birth_suppression = bool(enable_phase3p_birth_suppression)
        self.enable_phase3p_full_stabilization = bool(enable_phase3p_full_stabilization)
        self.keep_head_min = float(np.clip(keep_head_min, 0.0, 1.0))
        self.replace_margin = float(max(0.0, replace_margin))
        self.replace_consistency_window = int(max(1, replace_consistency_window))
        self.archived_reactivate_min = float(np.clip(archived_reactivate_min, 0.0, 1.0))
        self.birth_margin = float(max(0.0, birth_margin))
        self.post_recovery_birth_suppression_window = int(max(0, post_recovery_birth_suppression_window))
        self.lineage_internal_cooldown_after_head_switch = int(max(0, lineage_internal_cooldown_after_head_switch))
        self.head_continuity_bonus = float(max(0.0, head_continuity_bonus))
        self.archived_sibling_penalty = float(max(0.0, archived_sibling_penalty))
        self.newborn_prototype_penalty = float(max(0.0, newborn_prototype_penalty))
        self.enable_phase3d_dual_score = bool(enable_phase3d_dual_score)
        self.enable_phase3d_temp_attach = bool(enable_phase3d_temp_attach)
        self.enable_phase3d_deferred_promotion = bool(enable_phase3d_deferred_promotion)
        self.attach_accept_min = float(np.clip(attach_accept_min, 0.0, 1.0))
        self.promote_margin = float(max(0.0, promote_margin))
        self.promote_support_min = int(max(1, promote_support_min))
        self.promotion_window = int(max(1, promotion_window))
        self.promotion_stability_min = float(np.clip(promotion_stability_min, 0.0, 1.0))
        self.post_promotion_cooldown = int(max(0, post_promotion_cooldown))
        self.temp_attach_ttl = int(max(1, temp_attach_ttl))
        self.debug_force_temp_attach = bool(debug_force_temp_attach)
        self.anchor_ttl = int(max(1, anchor_ttl))
        self.anchor_expiry_policy = str(anchor_expiry_policy)
        self.anchor_prune_policy = str(anchor_prune_policy)
        self.anchor_survives_retire = bool(anchor_survives_retire)
        self.anchor_survives_head_replace = bool(anchor_survives_head_replace)
        self.recovery_anchor_topk_per_lineage = int(max(1, recovery_anchor_topk_per_lineage))

        self._prototypes: dict[int, PrototypeState] = {}
        self._lineages: dict[int, ConceptLineage] = {}
        self._next_prototype_id = 0
        self._next_lineage_id = 0
        self._next_continuation_id = 0
        self._next_temp_attach_id = 0
        self._lineage_recovery_lock_until: dict[int, int] = {}
        self._lineage_last_head_switch_frame: dict[int, int] = {}
        self._lineage_sibling_win_frames: dict[tuple[int, int], list[int]] = {}
        self._last_continuation_write_rows: list[dict[str, object]] = []
        self._last_continuation_lifecycle_rows: list[dict[str, object]] = []
        self._last_recovery_anchor_rows: list[dict[str, object]] = []
        self._last_recovery_anchor_lifecycle_rows: list[dict[str, object]] = []
        self._last_prototype_lineage_rows: list[dict[str, object]] = []

    def reset(self) -> None:
        self._prototypes.clear()
        self._lineages.clear()
        self._next_prototype_id = 0
        self._next_lineage_id = 0
        self._next_continuation_id = 0
        self._next_temp_attach_id = 0
        self._lineage_recovery_lock_until = {}
        self._lineage_last_head_switch_frame = {}
        self._lineage_sibling_win_frames = {}
        self._last_continuation_write_rows = []
        self._last_continuation_lifecycle_rows = []
        self._last_recovery_anchor_rows = []
        self._last_recovery_anchor_lifecycle_rows = []
        self._last_prototype_lineage_rows = []

    def update(
        self,
        assignments: list[TrackAssignment],
        frame_index: int,
        track_states: list[TrackState] | None = None,
    ) -> FrameMemoryResult:
        self._last_continuation_write_rows = []
        self._last_continuation_lifecycle_rows = []
        self._last_recovery_anchor_rows = []
        self._last_recovery_anchor_lifecycle_rows = []
        self._last_prototype_lineage_rows = []
        self._expire_temp_attach_slots(frame_index)
        protected_ids = self._refresh_protection(track_states, frame_index)
        continuation_archive_events = self._refresh_continuations(track_states, frame_index)
        self._refresh_recovery_anchors(track_states, frame_index)
        retired_ids: list[int] = []
        budget_pruned_ids: list[int] = []

        for prototype in self._prototypes.values():
            if prototype.retired:
                continue
            prototype.age += 1
            prototype.active = False
            prototype.is_active = False
            prototype.inactive_steps += 1
            decay_scale = 0.25 if prototype.is_protected else 1.0
            prototype.strength *= max(0.0, 1.0 - self.decay_rate * decay_scale)
            if not prototype.is_protected and prototype.strength < self.decay_floor and prototype.inactive_steps >= self.decay_patience:
                self._retire_prototype(prototype, frame_index=frame_index, reason="decay")
                retired_ids.append(prototype.prototype_id)

        memory_assignments = [self._assign_prototype(assignment, frame_index) for assignment in assignments]

        if self._active_count() > self.memory_budget:
            budget_pruned_ids = self._apply_budget_cap(frame_index)
            retired_ids.extend(budget_pruned_ids)

        self._capture_alive_continuation_rows(frame_index)
        self._capture_prototype_lineage_rows(frame_index)

        active_ids = sorted(
            prototype.prototype_id
            for prototype in self._prototypes.values()
            if not prototype.retired and prototype.active
        )
        protected_ids = sorted(
            prototype.prototype_id
            for prototype in self._prototypes.values()
            if not prototype.retired and prototype.is_protected
        )
        return FrameMemoryResult(
            frame_index=frame_index,
            assignments=memory_assignments,
            active_prototype_ids=active_ids,
            retired_prototype_ids=sorted(set(retired_ids)),
            total_prototypes=self._active_count(),
            budget_pruned_ids=budget_pruned_ids,
            protected_prototype_ids=protected_ids,
            prototype_lineage_lookup=self.prototype_lineage_lookup(),
            lineage_prototype_lookup=self.lineage_prototype_lookup(),
            lineage_head_lookup=self.lineage_head_lookup(),
            continuation_lookup=self.continuation_lookup(frame_index),
            continuation_lineage_lookup=self.continuation_lineage_lookup(frame_index),
            continuation_continuity_lookup=self.continuation_continuity_lookup(frame_index),
            temp_attach_lookup=self.temp_attach_lookup(frame_index),
            recovery_anchor_lookup=self.recovery_anchor_lookup(frame_index),
            recovery_anchor_continuity_lookup=self.recovery_anchor_continuity_lookup(frame_index),
            prototype_continuity_lookup=self.prototype_continuity_lookup(),
            continuation_bank_count=self.continuation_bank_count(),
            recovery_anchor_count=self.recovery_anchor_count(),
            continuation_archive_events=int(continuation_archive_events),
            continuation_binding_mode=str(self.bind_continuation_to),
            continuation_write_rows=[dict(row) for row in self._last_continuation_write_rows],
            continuation_lifecycle_rows=[dict(row) for row in self._last_continuation_lifecycle_rows],
            recovery_anchor_rows=[dict(row) for row in self._last_recovery_anchor_rows],
            recovery_anchor_lifecycle_rows=[dict(row) for row in self._last_recovery_anchor_lifecycle_rows],
            prototype_lineage_rows=[dict(row) for row in self._last_prototype_lineage_rows],
        )

    def snapshot(self) -> list[PrototypeState]:
        prototypes = [
            PrototypeState(
                prototype_id=prototype.prototype_id,
                lineage_id=prototype.lineage_id,
                parent_lineage_id=prototype.parent_lineage_id,
                signature=prototype.signature.copy(),
                hits=prototype.hits,
                age=prototype.age,
                strength=prototype.strength,
                inactive_steps=prototype.inactive_steps,
                active=prototype.active,
                is_active=prototype.is_active,
                retired=prototype.retired,
                birth_frame=prototype.birth_frame,
                created_frame=prototype.created_frame,
                last_updated_frame=prototype.last_updated_frame,
                last_track_id=prototype.last_track_id,
                is_protected=prototype.is_protected,
                protected_until_frame=prototype.protected_until_frame,
                merged_from=tuple(int(value) for value in prototype.merged_from),
                split_from=None if prototype.split_from is None else int(prototype.split_from),
                continuation_bank_ref=str(prototype.continuation_bank_ref),
                active_track_refs=list(prototype.active_track_refs),
                continuation_bank=[_clone_continuation(item) for item in prototype.continuation_bank],
                runtime_owner_lineage_id=None
                if prototype.runtime_owner_lineage_id is None
                else int(prototype.runtime_owner_lineage_id),
                continuity_lineage_id=None
                if prototype.continuity_lineage_id is None
                else int(prototype.continuity_lineage_id),
                origin_lineage_id=None
                if prototype.origin_lineage_id is None
                else int(prototype.origin_lineage_id),
                continuity_key_valid=bool(prototype.continuity_key_valid),
            )
            for prototype in self._prototypes.values()
            if not prototype.retired
        ]
        prototypes.sort(key=lambda prototype: prototype.prototype_id)
        return prototypes

    def prototype_lineage_lookup(self) -> dict[int, int]:
        return {
            int(prototype_id): int(prototype.lineage_id)
            for prototype_id, prototype in self._prototypes.items()
            if not prototype.retired
        }

    def prototype_continuity_lookup(self) -> dict[int, int]:
        return {
            int(prototype_id): int(prototype.continuity_lineage_id)
            for prototype_id, prototype in self._prototypes.items()
            if not prototype.retired and prototype.continuity_lineage_id is not None
        }

    def lineage_prototype_lookup(self) -> dict[int, list[int]]:
        lookup: dict[int, list[int]] = {}
        for lineage_id, lineage in self._lineages.items():
            prototype_ids = sorted({int(value) for value in lineage.active_prototype_ids})
            if prototype_ids:
                lookup[int(lineage_id)] = prototype_ids
        return lookup

    def lineage_head_lookup(self) -> dict[int, int | None]:
        return {
            int(lineage_id): (
                None if lineage.current_head_prototype_id is None else int(lineage.current_head_prototype_id)
            )
            for lineage_id, lineage in self._lineages.items()
        }

    def temp_attach_lookup(self, frame_index: int | None = None) -> dict[int, dict[str, object]]:
        lookup: dict[int, dict[str, object]] = {}
        for lineage_id, lineage in self._lineages.items():
            slot = lineage.temp_attach_slot
            if slot is None:
                continue
            age = 0
            if frame_index is not None:
                age = int(max(0, int(frame_index) - int(slot.last_seen_frame)))
            lookup[int(lineage_id)] = {
                "temp_attach_id": int(slot.temp_attach_id),
                "lineage_id": int(slot.lineage_id),
                "anchor_prototype_id": (
                    None if slot.anchor_prototype_id is None else int(slot.anchor_prototype_id)
                ),
                "start_frame": int(slot.start_frame),
                "last_seen_frame": int(slot.last_seen_frame),
                "support_count": int(slot.support_count),
                "promote_ready": bool(slot.promote_ready),
                "expired": bool(slot.expired),
                "source_track_id": (
                    None if slot.source_track_id is None else int(slot.source_track_id)
                ),
                "source_prototype_id": (
                    None if slot.source_prototype_id is None else int(slot.source_prototype_id)
                ),
                "age_since_last_seen": int(age),
                "last_objectness": float(slot.last_objectness),
            }
        return lookup

    def recovery_anchor_lookup(self, frame_index: int | None = None) -> dict[int, list[dict[str, object]]]:
        lookup: dict[int, list[dict[str, object]]] = {}
        for lineage_id, lineage in self._lineages.items():
            if not lineage.recovery_identity_anchors:
                continue
            anchors: list[dict[str, object]] = []
            for anchor in lineage.recovery_identity_anchors:
                anchors.append(
                    self._serialize_recovery_anchor(
                        anchor=anchor,
                        runtime_lineage_id=int(lineage_id),
                        frame_index=frame_index,
                    )
                )
            lookup[int(lineage_id)] = anchors
        return lookup

    def recovery_anchor_continuity_lookup(
        self,
        frame_index: int | None = None,
    ) -> dict[int, list[dict[str, object]]]:
        lookup: dict[int, list[dict[str, object]]] = {}
        for lineage_id, lineage in self._lineages.items():
            if not lineage.recovery_identity_anchors:
                continue
            for anchor in lineage.recovery_identity_anchors:
                continuity_lineage_id = (
                    int(anchor.continuity_lineage_id)
                    if anchor.continuity_lineage_id is not None
                    else (
                        int(anchor.origin_lineage_id)
                        if anchor.origin_lineage_id is not None
                        else int(anchor.old_lineage_id)
                    )
                )
                lookup.setdefault(int(continuity_lineage_id), []).append(
                    self._serialize_recovery_anchor(
                        anchor=anchor,
                        runtime_lineage_id=int(lineage_id),
                        frame_index=frame_index,
                    )
                )
        return lookup

    def _current_head_prototype_id(self, lineage_id: int | None) -> int | None:
        if lineage_id is None:
            return None
        lineage = self._lineages.get(int(lineage_id))
        if lineage is None:
            return None
        if lineage.current_head_prototype_id is not None:
            prototype = self._prototypes.get(int(lineage.current_head_prototype_id))
            if prototype is not None and not prototype.retired:
                return int(prototype.prototype_id)
        active_ids = [
            int(prototype_id)
            for prototype_id in lineage.active_prototype_ids
            if (prototype := self._prototypes.get(int(prototype_id))) is not None and not prototype.retired
        ]
        if not active_ids:
            return None
        head_id = max(active_ids, key=lambda prototype_id: self._prototypes[int(prototype_id)].last_updated_frame)
        lineage.current_head_prototype_id = int(head_id)
        return int(head_id)

    def _ensure_lineage(
        self,
        lineage_id: int,
        *,
        frame_index: int,
        birth_frame: int | None = None,
        parent_lineage_id: int | None = None,
    ) -> ConceptLineage:
        lineage = self._lineages.get(int(lineage_id))
        if lineage is None:
            lineage = ConceptLineage(
                lineage_id=int(lineage_id),
                current_head_prototype_id=None,
                birth_frame=int(frame_index if birth_frame is None else birth_frame),
                last_update_frame=int(frame_index),
                status="active",
            )
            if parent_lineage_id is not None and parent_lineage_id != lineage_id:
                lineage.alias_lineage_ids.append(int(parent_lineage_id))
            self._lineages[int(lineage_id)] = lineage
        return lineage

    def _mark_lineage_active(
        self,
        prototype: PrototypeState,
        *,
        frame_index: int,
    ) -> None:
        self._mark_lineage_member_active(prototype, frame_index=frame_index, as_head=True)

    def _mark_lineage_member_active(
        self,
        prototype: PrototypeState,
        *,
        frame_index: int,
        as_head: bool,
    ) -> None:
        lineage = self._ensure_lineage(
            int(prototype.lineage_id),
            frame_index=frame_index,
            birth_frame=int(prototype.birth_frame),
            parent_lineage_id=None if prototype.parent_lineage_id is None else int(prototype.parent_lineage_id),
        )
        prototype_id = int(prototype.prototype_id)
        if prototype_id not in lineage.active_prototype_ids:
            lineage.active_prototype_ids.append(prototype_id)
        if prototype_id in lineage.archived_prototype_ids:
            lineage.archived_prototype_ids.remove(prototype_id)
        if as_head or lineage.current_head_prototype_id is None:
            lineage.current_head_prototype_id = prototype_id
        lineage.last_update_frame = int(frame_index)
        lineage.status = "active"
        prototype.runtime_owner_lineage_id = int(prototype.lineage_id)
        if prototype.continuity_lineage_id is None:
            prototype.continuity_lineage_id = int(prototype.lineage_id)
        if prototype.origin_lineage_id is None:
            prototype.origin_lineage_id = int(prototype.continuity_lineage_id)
        prototype.continuity_key_valid = bool(prototype.continuity_lineage_id is not None)

    def _mark_lineage_archived(
        self,
        prototype: PrototypeState,
        *,
        frame_index: int,
    ) -> None:
        lineage = self._ensure_lineage(
            int(prototype.lineage_id),
            frame_index=frame_index,
            birth_frame=int(prototype.birth_frame),
            parent_lineage_id=None if prototype.parent_lineage_id is None else int(prototype.parent_lineage_id),
        )
        prototype_id = int(prototype.prototype_id)
        if prototype_id in lineage.active_prototype_ids:
            lineage.active_prototype_ids.remove(prototype_id)
        if prototype_id not in lineage.archived_prototype_ids:
            lineage.archived_prototype_ids.append(prototype_id)
        head_id = self._current_head_prototype_id(int(prototype.lineage_id))
        lineage.current_head_prototype_id = head_id
        lineage.last_update_frame = int(frame_index)
        lineage.status = "archived" if not lineage.active_prototype_ids else "active"
        prototype.runtime_owner_lineage_id = int(prototype.lineage_id)
        if prototype.continuity_lineage_id is None:
            prototype.continuity_lineage_id = int(prototype.lineage_id)
        if prototype.origin_lineage_id is None:
            prototype.origin_lineage_id = int(prototype.continuity_lineage_id)
        prototype.continuity_key_valid = bool(prototype.continuity_lineage_id is not None)

    def _retire_prototype(
        self,
        prototype: PrototypeState,
        *,
        frame_index: int,
        reason: str,
    ) -> None:
        if prototype.retired:
            return
        prototype.retired = True
        prototype.active = False
        prototype.is_active = False
        self._mark_lineage_archived(prototype, frame_index=frame_index)

    def _refresh_protection(self, track_states: list[TrackState] | None, frame_index: int) -> list[int]:
        for prototype in self._prototypes.values():
            prototype.is_protected = False
            prototype.active_track_refs = []
        if not self.protect_linked_prototypes or not track_states:
            return []

        protected_ids: list[int] = []
        for track in track_states:
            if track.prototype_id is None or track.retired:
                continue
            prototype = self._prototypes.get(int(track.prototype_id))
            if prototype is None or prototype.retired:
                continue
            if track.active or track.dormant or track.ghost:
                prototype.is_protected = True
                prototype.protected_until_frame = int(frame_index)
                prototype.active_track_refs.append(int(track.track_id))
        for prototype in self._prototypes.values():
            if prototype.is_protected:
                protected_ids.append(prototype.prototype_id)
        return sorted(protected_ids)

    def continuation_lookup(self, frame_index: int) -> dict[int, list[IdentityContinuation]]:
        lookup: dict[int, list[IdentityContinuation]] = {}
        if self.bind_continuation_to == "prototype":
            for prototype_id, prototype in self._prototypes.items():
                if prototype.retired or not prototype.continuation_bank:
                    continue
                lookup[int(prototype_id)] = [_clone_continuation(item) for item in prototype.continuation_bank]
            return lookup
        for prototype_id, prototype in self._prototypes.items():
            if prototype.retired:
                continue
            lineage = self._lineages.get(int(prototype.lineage_id))
            if lineage is None or not lineage.continuation_bank:
                continue
            lookup[int(prototype_id)] = [_clone_continuation(item) for item in lineage.continuation_bank]
        return lookup

    def continuation_lineage_lookup(self, frame_index: int) -> dict[int, list[IdentityContinuation]]:
        lookup: dict[int, list[IdentityContinuation]] = {}
        if self.bind_continuation_to == "prototype":
            for prototype in self._prototypes.values():
                if not prototype.continuation_bank:
                    continue
                cloned = [_clone_continuation(item) for item in prototype.continuation_bank]
                lookup.setdefault(int(prototype.lineage_id), []).extend(cloned)
        else:
            for lineage_id, lineage in self._lineages.items():
                if not lineage.continuation_bank:
                    continue
                lookup[int(lineage_id)] = [_clone_continuation(item) for item in lineage.continuation_bank]
        for lineage_id, continuations in lookup.items():
            lookup[lineage_id] = sorted(
                continuations,
                key=lambda item: (
                    int(item.age_since_last_seen),
                    -float(item.continuation_confidence),
                    -int(item.hit_count),
                ),
            )
        return lookup

    def continuation_continuity_lookup(self, frame_index: int) -> dict[int, list[IdentityContinuation]]:
        lookup: dict[int, list[IdentityContinuation]] = {}
        if self.bind_continuation_to == "prototype":
            owners = [
                (None if prototype.retired else prototype, None, prototype.continuation_bank)
                for prototype in self._prototypes.values()
                if not prototype.retired and prototype.continuation_bank
            ]
        else:
            owners = [
                (None, lineage, lineage.continuation_bank)
                for lineage in self._lineages.values()
                if lineage.continuation_bank
            ]
        for owner_prototype, owner_lineage, bank in owners:
            fallback_lineage_id = None
            if owner_prototype is not None:
                fallback_lineage_id = int(owner_prototype.continuity_lineage_id or owner_prototype.lineage_id)
            elif owner_lineage is not None:
                fallback_lineage_id = int(owner_lineage.lineage_id)
            for continuation in bank:
                continuity_lineage_id = (
                    int(continuation.continuity_lineage_id)
                    if continuation.continuity_lineage_id is not None
                    else (
                        int(continuation.origin_lineage_id)
                        if continuation.origin_lineage_id is not None
                        else fallback_lineage_id
                    )
                )
                if continuity_lineage_id is None:
                    continue
                lookup.setdefault(int(continuity_lineage_id), []).append(
                    _clone_continuation(continuation)
                )
        for lineage_id, continuations in lookup.items():
            lookup[lineage_id] = sorted(
                continuations,
                key=lambda item: (
                    int(item.age_since_last_seen),
                    -float(item.continuation_confidence),
                    -int(item.hit_count),
                ),
            )
        return lookup

    def continuation_bank_count(self) -> int:
        if self.bind_continuation_to == "prototype":
            return int(
                sum(
                    len(prototype.continuation_bank)
                    for prototype in self._prototypes.values()
                    if not prototype.retired
                )
            )
        return int(sum(len(lineage.continuation_bank) for lineage in self._lineages.values()))

    def _serialize_recovery_anchor(
        self,
        *,
        anchor: RecoveryIdentityAnchor,
        runtime_lineage_id: int,
        frame_index: int | None = None,
    ) -> dict[str, object]:
        age = int(anchor.age_since_last_seen)
        state = str(anchor.anchor_state)
        if frame_index is not None:
            age = int(max(0, int(frame_index) - int(anchor.last_alive_frame)))
            state = "expired" if age > self.anchor_ttl else ("stale" if age > max(4, int(self.anchor_ttl * 0.6)) else "alive")
        return {
            "anchor_uid": str(anchor.anchor_uid),
            "old_track_id": int(anchor.old_track_id),
            "old_prototype_id": int(anchor.old_prototype_id),
            "old_lineage_id": int(anchor.old_lineage_id),
            "source_frame_id": int(anchor.source_frame_id),
            "last_alive_frame": int(anchor.last_alive_frame),
            "gap_length_at_creation": int(anchor.gap_length_at_creation),
            "restore_priority": float(anchor.restore_priority),
            "continuation_hint": str(anchor.continuation_hint),
            "anchor_state": str(state),
            "anchor_origin": str(anchor.anchor_origin),
            "age_since_last_seen": int(age),
            "last_center": tuple(float(value) for value in anchor.last_center),
            "last_bbox": tuple(int(value) for value in anchor.last_bbox),
            "velocity": anchor.velocity.copy(),
            "feature_ema": anchor.feature_ema.copy(),
            "last_objectness": float(anchor.last_objectness),
            "track_age": int(anchor.track_age),
            "hit_count": int(anchor.hit_count),
            "runtime_owner_lineage_id": None
            if anchor.runtime_owner_lineage_id is None
            else int(anchor.runtime_owner_lineage_id),
            "continuity_lineage_id": None
            if anchor.continuity_lineage_id is None
            else int(anchor.continuity_lineage_id),
            "origin_lineage_id": None
            if anchor.origin_lineage_id is None
            else int(anchor.origin_lineage_id),
            "old_identity_ref_track_id": None
            if anchor.old_identity_ref_track_id is None
            else int(anchor.old_identity_ref_track_id),
            "old_identity_ref_prototype_id": None
            if anchor.old_identity_ref_prototype_id is None
            else int(anchor.old_identity_ref_prototype_id),
            "continuity_key_valid": int(bool(anchor.continuity_key_valid)),
            "owner_runtime_lineage_id": int(runtime_lineage_id),
        }

    def recovery_anchor_count(self) -> int:
        return int(sum(len(lineage.recovery_identity_anchors) for lineage in self._lineages.values()))

    def _resolve_lineage_seed_prototype_id(
        self,
        assignment: TrackAssignment,
        previous_prototype_id: int | None,
    ) -> int | None:
        continuity_lineage_id = getattr(assignment, "continuity_lineage_id", None)
        if continuity_lineage_id is not None:
            continuity_candidates = [
                prototype
                for prototype in self._prototypes.values()
                if (
                    not prototype.retired
                    and prototype.continuity_lineage_id is not None
                    and int(prototype.continuity_lineage_id) == int(continuity_lineage_id)
                )
            ]
            if continuity_candidates:
                continuity_candidates.sort(
                    key=lambda prototype: (
                        int(prototype.prototype_id) != int(self._current_head_prototype_id(prototype.lineage_id) or -1),
                        -int(prototype.last_updated_frame),
                        -float(prototype.strength),
                        -int(prototype.hits),
                    )
                )
                return int(continuity_candidates[0].prototype_id)
        for lineage_id in (
            assignment.linked_lineage_id,
            self._resolve_lineage_id(assignment.prototype_hint_id),
            self._resolve_lineage_id(assignment.linked_prototype_id),
            self._resolve_lineage_id(previous_prototype_id),
        ):
            head_id = self._current_head_prototype_id(lineage_id)
            if head_id is not None:
                return int(head_id)
        for candidate_id in (
            assignment.prototype_hint_id,
            assignment.linked_prototype_id,
            previous_prototype_id,
        ):
            prototype = None if candidate_id is None else self._prototypes.get(int(candidate_id))
            if prototype is None:
                continue
            return int(prototype.prototype_id)
        return None

    def _resolve_lineage_id(self, prototype_id: int | None) -> int | None:
        if prototype_id is None:
            return None
        prototype = self._prototypes.get(int(prototype_id))
        if prototype is None:
            return None
        return int(prototype.lineage_id)

    def _resolve_continuity_lineage_id(self, prototype_id: int | None) -> int | None:
        if prototype_id is None:
            return None
        prototype = self._prototypes.get(int(prototype_id))
        if prototype is None:
            return None
        if prototype.continuity_lineage_id is not None:
            return int(prototype.continuity_lineage_id)
        return int(prototype.lineage_id)

    def _prototype_candidate_score(self, distance: float) -> float:
        return float(np.exp(-float(distance) / max(self.tau_sim, 1e-6)))

    def _phase3d_enabled(self) -> bool:
        return bool(
            self.enable_phase3d_dual_score
            or self.enable_phase3d_temp_attach
            or self.enable_phase3d_deferred_promotion
        )

    def _attach_score(
        self,
        *,
        distance: float,
        assignment_score: float,
        is_head: bool,
        is_archived: bool,
    ) -> float:
        similarity = self._prototype_candidate_score(distance)
        score = 0.72 * float(similarity) + 0.28 * float(np.clip(assignment_score, 0.0, 1.0))
        if is_head:
            score += 0.02
        if is_archived:
            score -= 0.05
        return float(np.clip(score, 0.0, 1.0))

    def _promote_score(
        self,
        *,
        attach_score: float,
        is_head: bool,
        is_archived: bool,
        support_count: int,
    ) -> float:
        stability_bonus = 0.06 * float(min(1.0, float(support_count) / max(1.0, float(self.promote_support_min))))
        score = float(attach_score) + stability_bonus
        if is_head:
            score += 0.03
        if is_archived:
            score -= 0.08
        return float(np.clip(score, 0.0, 1.0))

    def _expire_temp_attach_slots(self, frame_index: int) -> None:
        for lineage in self._lineages.values():
            slot = lineage.temp_attach_slot
            if slot is None or slot.expired:
                continue
            if int(frame_index) - int(slot.last_seen_frame) > int(self.temp_attach_ttl):
                slot.expired = True

    def _start_or_refresh_temp_attach(
        self,
        *,
        lineage_id: int,
        frame_index: int,
        assignment: TrackAssignment,
        memory_signature: np.ndarray,
        anchor_prototype_id: int | None,
        source_prototype_id: int | None,
    ) -> TemporaryAttachSlot:
        lineage = self._ensure_lineage(int(lineage_id), frame_index=frame_index)
        slot = lineage.temp_attach_slot
        if slot is None or slot.expired:
            slot = TemporaryAttachSlot(
                temp_attach_id=int(self._next_temp_attach_id),
                lineage_id=int(lineage_id),
                anchor_prototype_id=None if anchor_prototype_id is None else int(anchor_prototype_id),
                start_frame=int(frame_index),
                last_seen_frame=int(frame_index),
                support_count=1,
                promote_ready=False,
                expired=False,
                source_track_id=int(assignment.track_id),
                source_prototype_id=None if source_prototype_id is None else int(source_prototype_id),
                last_center=(float(assignment.centroid[0]), float(assignment.centroid[1])),
                last_bbox=assignment.box,
                last_signature=memory_signature.copy(),
                last_objectness=float(assignment.score),
            )
            lineage.temp_attach_slot = slot
            self._next_temp_attach_id += 1
        else:
            slot.anchor_prototype_id = None if anchor_prototype_id is None else int(anchor_prototype_id)
            slot.last_seen_frame = int(frame_index)
            slot.support_count += 1
            slot.source_track_id = int(assignment.track_id)
            slot.source_prototype_id = None if source_prototype_id is None else int(source_prototype_id)
            slot.last_center = (float(assignment.centroid[0]), float(assignment.centroid[1]))
            slot.last_bbox = assignment.box
            slot.last_signature = memory_signature.copy()
            slot.last_objectness = float(assignment.score)
            slot.expired = False
        slot.promote_ready = bool(slot.support_count >= self.promote_support_min)
        return slot

    def _clear_lineage_promotion_pending(self, lineage_id: int | None) -> None:
        if lineage_id is None:
            return
        lineage = self._lineages.get(int(lineage_id))
        if lineage is None:
            return
        lineage.promotion_candidate_id = None
        lineage.promotion_pending_flag = False
        lineage.promotion_window_progress = 0
        lineage.promotion_support_count = 0
        lineage.promotion_start_frame = None
        lineage.last_promotion_decision = "drop_candidate"

    def _update_lineage_promotion_state(
        self,
        *,
        lineage_id: int | None,
        candidate_id: int | None,
        frame_index: int,
        candidate_promote_score: float | None,
        head_promote_score: float | None,
        allow_promotion: bool,
    ) -> dict[str, object]:
        if lineage_id is None or candidate_id is None:
            self._clear_lineage_promotion_pending(lineage_id)
            return {
                "promotion_candidate_id": None,
                "promotion_pending_flag": False,
                "promotion_window_progress": 0,
                "promotion_decision": "keep_head",
                "promotion_support_count": 0,
                "promotion_delay_frames": None,
                "promotion_success": False,
            }
        lineage = self._ensure_lineage(int(lineage_id), frame_index=frame_index)
        current_head_id = lineage.current_head_prototype_id
        if current_head_id is not None and int(candidate_id) == int(current_head_id):
            self._clear_lineage_promotion_pending(lineage_id)
            lineage.last_promotion_decision = "keep_head"
            return {
                "promotion_candidate_id": int(candidate_id),
                "promotion_pending_flag": False,
                "promotion_window_progress": 0,
                "promotion_decision": "keep_head",
                "promotion_support_count": 0,
                "promotion_delay_frames": None,
                "promotion_success": False,
            }

        if lineage.promotion_candidate_id is None or int(lineage.promotion_candidate_id) != int(candidate_id):
            lineage.promotion_candidate_id = int(candidate_id)
            lineage.promotion_pending_flag = True
            lineage.promotion_window_progress = 1
            lineage.promotion_support_count = 1
            lineage.promotion_start_frame = int(frame_index)
        else:
            lineage.promotion_pending_flag = True
            lineage.promotion_window_progress = min(
                int(self.promotion_window),
                int(lineage.promotion_window_progress) + 1,
            )
            lineage.promotion_support_count += 1

        cooldown_active = int(frame_index) <= int(lineage.post_promotion_cooldown_until)
        promotion_success = False
        promotion_decision = "keep_head"
        if (
            allow_promotion
            and not cooldown_active
            and candidate_promote_score is not None
            and head_promote_score is not None
            and (float(candidate_promote_score) - float(head_promote_score)) >= self.promote_margin
            and int(lineage.promotion_support_count) >= int(self.promote_support_min)
            and (float(lineage.promotion_window_progress) / float(max(1, self.promotion_window))) >= self.promotion_stability_min
        ):
            lineage.current_head_prototype_id = int(candidate_id)
            lineage.post_promotion_cooldown_until = int(frame_index + self.post_promotion_cooldown)
            promotion_success = True
            promotion_decision = "promote_candidate"
            self._lineage_last_head_switch_frame[int(lineage_id)] = int(frame_index)
            lineage.promotion_candidate_id = None
            lineage.promotion_pending_flag = False
            lineage.promotion_window_progress = 0
            lineage.promotion_support_count = 0
            lineage.promotion_start_frame = None
        else:
            promotion_decision = "keep_head"

        lineage.last_promotion_decision = str(promotion_decision)
        delay_frames = (
            None
            if lineage.promotion_start_frame is None
            else int(max(0, frame_index - int(lineage.promotion_start_frame)))
        )
        return {
            "promotion_candidate_id": None if lineage.promotion_candidate_id is None else int(lineage.promotion_candidate_id),
            "promotion_pending_flag": bool(lineage.promotion_pending_flag),
            "promotion_window_progress": int(lineage.promotion_window_progress),
            "promotion_decision": str(promotion_decision),
            "promotion_support_count": int(lineage.promotion_support_count),
            "promotion_delay_frames": delay_frames,
            "promotion_success": bool(promotion_success),
        }

    def _lineage_selector_audit(
        self,
        *,
        lineage_id: int | None,
        memory_signature: np.ndarray,
        head_prototype_id_before: int | None,
    ) -> dict[str, object]:
        if lineage_id is None:
            return {
                "head_prototype_id_before": None,
                "head_score": None,
                "best_active_sibling_id": None,
                "best_active_sibling_score": None,
                "best_archived_sibling_id": None,
                "best_archived_sibling_score": None,
            }
        lineage_prototypes = [
            prototype
            for prototype in self._prototypes.values()
            if int(prototype.lineage_id) == int(lineage_id)
        ]
        head_prototype = (
            None
            if head_prototype_id_before is None
            else self._prototypes.get(int(head_prototype_id_before))
        )
        head_score = None
        if head_prototype is not None:
            head_distance = _prototype_distance(head_prototype.signature, memory_signature)
            head_score = self._prototype_candidate_score(head_distance)

        def _best_score(
            rows: list[PrototypeState],
        ) -> tuple[int | None, float | None]:
            if not rows:
                return None, None
            best_distance, best_prototype_id = min(
                (
                    (_prototype_distance(prototype.signature, memory_signature), int(prototype.prototype_id))
                    for prototype in rows
                ),
                key=lambda item: item[0],
            )
            return int(best_prototype_id), self._prototype_candidate_score(best_distance)

        active_siblings = [
            prototype
            for prototype in lineage_prototypes
            if not prototype.retired and int(prototype.prototype_id) != int(head_prototype_id_before if head_prototype_id_before is not None else -1)
        ]
        archived_siblings = [
            prototype
            for prototype in lineage_prototypes
            if prototype.retired
        ]
        best_active_id, best_active_score = _best_score(active_siblings)
        best_archived_id, best_archived_score = _best_score(archived_siblings)
        return {
            "head_prototype_id_before": None if head_prototype_id_before is None else int(head_prototype_id_before),
            "head_score": head_score,
            "best_active_sibling_id": best_active_id,
            "best_active_sibling_score": best_active_score,
            "best_archived_sibling_id": best_archived_id,
            "best_archived_sibling_score": best_archived_score,
        }

    def _selector_action_type(
        self,
        *,
        selected_prototype_id: int,
        selected_lineage_id: int,
        selected_was_newborn: bool,
        selected_was_retired: bool,
        head_prototype_id_before: int | None,
        lineage_seed_id: int | None,
        previous_lineage_id: int | None,
    ) -> tuple[str, str]:
        if selected_was_newborn:
            if previous_lineage_id is not None and int(selected_lineage_id) == int(previous_lineage_id):
                return "birth_sibling", "newborn"
            if lineage_seed_id is not None and int(selected_lineage_id) == int(lineage_seed_id):
                return "birth_sibling", "newborn"
            return "birth_lineage", "newborn"
        if head_prototype_id_before is not None and int(selected_prototype_id) == int(head_prototype_id_before):
            return "keep_head", "head"
        if selected_was_retired:
            return "reactivate_archived", "archived_sibling"
        if lineage_seed_id is not None and int(selected_lineage_id) == int(lineage_seed_id):
            return "replace_head", "active_sibling"
        return "attach_external", "active_sibling"

    def _record_lineage_sibling_win(
        self,
        *,
        lineage_id: int,
        sibling_id: int,
        frame_index: int,
    ) -> int:
        key = (int(lineage_id), int(sibling_id))
        history = [
            int(item)
            for item in self._lineage_sibling_win_frames.get(key, [])
            if int(frame_index) - int(item) < self.replace_consistency_window
        ]
        history.append(int(frame_index))
        self._lineage_sibling_win_frames[key] = history
        return len(history)

    def _clear_lineage_sibling_wins(self, *, lineage_id: int, keep_sibling_id: int | None = None) -> None:
        stale_keys = [
            key
            for key in list(self._lineage_sibling_win_frames.keys())
            if int(key[0]) == int(lineage_id) and (keep_sibling_id is None or int(key[1]) != int(keep_sibling_id))
        ]
        for key in stale_keys:
            self._lineage_sibling_win_frames.pop(key, None)

    def _current_lineage_recovery_lock(self, lineage_id: int | None, frame_index: int) -> bool:
        if lineage_id is None:
            return False
        return int(frame_index) <= int(self._lineage_recovery_lock_until.get(int(lineage_id), -1))

    def _set_lineage_recovery_lock(self, lineage_id: int | None, frame_index: int) -> None:
        if lineage_id is None or self.post_recovery_birth_suppression_window <= 0:
            return
        self._lineage_recovery_lock_until[int(lineage_id)] = int(frame_index + self.post_recovery_birth_suppression_window)

    def _select_within_lineage_policy(
        self,
        *,
        assignment: TrackAssignment,
        memory_signature: np.ndarray,
        frame_index: int,
        lineage_seed_id: int,
        head_prototype_id_before: int | None,
        lineage_audit: dict[str, object],
    ) -> dict[str, object] | None:
        if not (
            self.enable_phase3p_keep_head_default
            or self.enable_phase3p_grouped_gating
            or self.enable_phase3p_birth_suppression
            or self.enable_phase3p_full_stabilization
        ):
            return None

        lineage_candidates = [
            prototype
            for prototype in self._prototypes.values()
            if int(prototype.lineage_id) == int(lineage_seed_id)
        ]
        if not lineage_candidates:
            return None

        head_prototype = (
            None
            if head_prototype_id_before is None
            else self._prototypes.get(int(head_prototype_id_before))
        )
        active_siblings = [
            prototype
            for prototype in lineage_candidates
            if not prototype.retired and int(prototype.prototype_id) != int(head_prototype_id_before if head_prototype_id_before is not None else -1)
        ]
        archived_siblings = [prototype for prototype in lineage_candidates if prototype.retired]

        def _score(proto: PrototypeState | None) -> tuple[float | None, float | None]:
            if proto is None:
                return None, None
            distance = _prototype_distance(proto.signature, memory_signature)
            return self._prototype_candidate_score(distance), float(distance)

        head_score, head_distance = _score(head_prototype)
        active_rows: list[tuple[PrototypeState, float, float]] = []
        for prototype in active_siblings:
            score, distance = _score(prototype)
            if score is None or distance is None:
                continue
            active_rows.append((prototype, float(score), float(distance)))
        active_rows.sort(key=lambda item: item[1], reverse=True)
        archived_rows: list[tuple[PrototypeState, float, float]] = []
        for prototype in archived_siblings:
            score, distance = _score(prototype)
            if score is None or distance is None:
                continue
            archived_rows.append((prototype, float(score), float(distance)))
        archived_rows.sort(key=lambda item: item[1], reverse=True)

        best_active = active_rows[0] if active_rows else None
        best_archived = archived_rows[0] if archived_rows else None
        use_full = bool(self.enable_phase3p_full_stabilization)
        use_grouped = bool(self.enable_phase3p_grouped_gating or self.enable_phase3p_full_stabilization)
        use_keep_head = bool(self.enable_phase3p_keep_head_default or self.enable_phase3p_full_stabilization)
        use_birth_suppression = bool(self.enable_phase3p_birth_suppression or self.enable_phase3p_full_stabilization)

        cooldown_active = False
        if use_full and self.lineage_internal_cooldown_after_head_switch > 0:
            last_switch = self._lineage_last_head_switch_frame.get(int(lineage_seed_id), -10**9)
            cooldown_active = (int(frame_index) - int(last_switch)) <= int(self.lineage_internal_cooldown_after_head_switch)
        effective_replace_margin = float(self.replace_margin + (0.05 if cooldown_active else 0.0))

        adjusted_head_score = None if head_score is None else float(head_score + (self.head_continuity_bonus if use_full else 0.0))
        adjusted_active_score = None if best_active is None else float(best_active[1])
        adjusted_archived_score = None if best_archived is None else float(best_archived[1] - (self.archived_sibling_penalty if use_full else 0.0))

        if use_grouped and head_prototype is not None and adjusted_head_score is not None and adjusted_head_score >= self.keep_head_min:
            should_replace = False
            if best_active is not None and adjusted_active_score is not None:
                margin = float(adjusted_active_score - adjusted_head_score)
                if margin >= effective_replace_margin:
                    if use_full:
                        sibling_wins = self._record_lineage_sibling_win(
                            lineage_id=int(lineage_seed_id),
                            sibling_id=int(best_active[0].prototype_id),
                            frame_index=frame_index,
                        )
                        should_replace = sibling_wins >= 2
                    else:
                        should_replace = True
                else:
                    self._clear_lineage_sibling_wins(lineage_id=int(lineage_seed_id))
            else:
                self._clear_lineage_sibling_wins(lineage_id=int(lineage_seed_id))

            if use_keep_head and not should_replace:
                return {
                    "decision": "prototype",
                    "prototype": head_prototype,
                    "distance": float(head_distance if head_distance is not None else 1.0),
                    "action": "keep_head",
                }
            if should_replace and best_active is not None:
                return {
                    "decision": "prototype",
                    "prototype": best_active[0],
                    "distance": float(best_active[2]),
                    "action": "replace_head",
                }

        if use_grouped and best_active is not None and adjusted_active_score is not None and adjusted_active_score >= self.keep_head_min:
            self._clear_lineage_sibling_wins(lineage_id=int(lineage_seed_id), keep_sibling_id=int(best_active[0].prototype_id))
            return {
                "decision": "prototype",
                "prototype": best_active[0],
                "distance": float(best_active[2]),
                "action": "replace_head",
            }

        if use_grouped and best_archived is not None and adjusted_archived_score is not None and adjusted_archived_score >= self.archived_reactivate_min:
            return {
                "decision": "prototype",
                "prototype": best_archived[0],
                "distance": float(best_archived[2]),
                "action": "reactivate_archived",
            }

        if use_birth_suppression and self._current_lineage_recovery_lock(int(lineage_seed_id), frame_index):
            if head_prototype is not None and head_distance is not None:
                return {
                    "decision": "prototype",
                    "prototype": head_prototype,
                    "distance": float(head_distance),
                    "action": "keep_head",
                }
            if best_active is not None:
                return {
                    "decision": "prototype",
                    "prototype": best_active[0],
                    "distance": float(best_active[2]),
                    "action": "replace_head",
                }
            return {"decision": "suppress_birth"}

        best_existing_score = max(
            [value for value in [adjusted_head_score, adjusted_active_score, adjusted_archived_score] if value is not None],
            default=None,
        )
        birth_score = float(max(0.0, float(assignment.score) - (self.newborn_prototype_penalty if use_full else 0.0)))
        if use_grouped and best_existing_score is not None and birth_score < float(best_existing_score + self.birth_margin):
            if head_prototype is not None and head_distance is not None:
                return {
                    "decision": "prototype",
                    "prototype": head_prototype,
                    "distance": float(head_distance),
                    "action": "keep_head",
                }
            if best_active is not None:
                return {
                    "decision": "prototype",
                    "prototype": best_active[0],
                    "distance": float(best_active[2]),
                    "action": "replace_head",
                }
            return {"decision": "suppress_birth"}
        return None

    def _select_within_lineage_attach_policy(
        self,
        *,
        assignment: TrackAssignment,
        memory_signature: np.ndarray,
        frame_index: int,
        lineage_seed_id: int,
        head_prototype_id_before: int | None,
    ) -> dict[str, object] | None:
        if not self._phase3d_enabled():
            return None

        lineage_candidates = [
            prototype
            for prototype in self._prototypes.values()
            if int(prototype.lineage_id) == int(lineage_seed_id)
        ]
        if not lineage_candidates:
            return None

        head_prototype = (
            None if head_prototype_id_before is None else self._prototypes.get(int(head_prototype_id_before))
        )
        active_siblings = [
            prototype
            for prototype in lineage_candidates
            if not prototype.retired
            and int(prototype.prototype_id) != int(head_prototype_id_before if head_prototype_id_before is not None else -1)
        ]
        archived_siblings = [prototype for prototype in lineage_candidates if prototype.retired]

        def _score_row(proto: PrototypeState | None, *, is_head: bool, is_archived: bool) -> tuple[float | None, float | None, float | None]:
            if proto is None:
                return None, None, None
            distance = _prototype_distance(proto.signature, memory_signature)
            attach_score = self._attach_score(
                distance=float(distance),
                assignment_score=float(assignment.score),
                is_head=is_head,
                is_archived=is_archived,
            )
            promote_score = self._promote_score(
                attach_score=float(attach_score),
                is_head=is_head,
                is_archived=is_archived,
                support_count=1,
            )
            return float(distance), float(attach_score), float(promote_score)

        head_distance, head_attach_score, head_promote_score = _score_row(
            head_prototype,
            is_head=True,
            is_archived=False,
        )

        active_rows: list[tuple[PrototypeState, float, float, float]] = []
        for prototype in active_siblings:
            distance, attach_score, promote_score = _score_row(prototype, is_head=False, is_archived=False)
            if distance is None or attach_score is None or promote_score is None:
                continue
            active_rows.append((prototype, distance, attach_score, promote_score))
        active_rows.sort(key=lambda item: item[2], reverse=True)

        archived_rows: list[tuple[PrototypeState, float, float, float]] = []
        for prototype in archived_siblings:
            distance, attach_score, promote_score = _score_row(prototype, is_head=False, is_archived=True)
            if distance is None or attach_score is None or promote_score is None:
                continue
            archived_rows.append((prototype, distance, attach_score, promote_score))
        archived_rows.sort(key=lambda item: item[2], reverse=True)

        best_active = active_rows[0] if active_rows else None
        best_archived = archived_rows[0] if archived_rows else None
        temp_attach_score = float(
            max(
                [score for score in [head_attach_score, None if best_active is None else best_active[2], None if best_archived is None else best_archived[2]] if score is not None],
                default=0.0,
            )
            * 0.85
            + 0.15 * float(np.clip(assignment.score, 0.0, 1.0))
        )

        selector_payload = {
            "matched_lineage_id": int(lineage_seed_id),
            "attach_score_current_head": head_attach_score,
            "attach_score_active_sibling": None if best_active is None else float(best_active[2]),
            "attach_score_archived_sibling": None if best_archived is None else float(best_archived[2]),
            "attach_score_temp_slot": temp_attach_score,
            "promote_score_current_head": head_promote_score,
            "promote_score_candidate": None,
            "attach_path_source": "lineage_similarity",
            "attach_branch_entered": True,
            "temp_attach_eligibility_checked": True,
            "temp_attach_force_mode": bool(self.debug_force_temp_attach),
            "lineage_seed_id_used": int(lineage_seed_id),
        }

        force_temp_attach = bool(self.debug_force_temp_attach and self.enable_phase3d_temp_attach)

        if (
            not force_temp_attach
            and head_prototype is not None
            and head_attach_score is not None
            and head_attach_score >= self.attach_accept_min
        ):
            promotion_state = self._update_lineage_promotion_state(
                lineage_id=int(lineage_seed_id),
                candidate_id=int(head_prototype.prototype_id),
                frame_index=frame_index,
                candidate_promote_score=head_promote_score,
                head_promote_score=head_promote_score,
                allow_promotion=bool(self.enable_phase3d_deferred_promotion),
            )
            return {
                **selector_payload,
                **promotion_state,
                "decision": "prototype",
                "prototype": head_prototype,
                "distance": float(head_distance if head_distance is not None else 1.0),
                "action": "keep_head_attach",
                "override_action_type": "keep_head_attach",
                "override_selected_state": "head",
                "recovery_attach_target": "current_head",
                "recovery_attach_target_id": int(head_prototype.prototype_id),
                "promote_to_head": True,
                "attach_state_written": True,
                "temp_attach_used": False,
                "temp_attach_id": None,
                "temp_attach_support_count": 0,
                "temp_attach_promote_ready": False,
                "temp_attach_expired": False,
            }

        if not force_temp_attach and best_active is not None and float(best_active[2]) >= self.attach_accept_min:
            promotion_state = self._update_lineage_promotion_state(
                lineage_id=int(lineage_seed_id),
                candidate_id=int(best_active[0].prototype_id),
                frame_index=frame_index,
                candidate_promote_score=float(best_active[3]),
                head_promote_score=head_promote_score,
                allow_promotion=bool(self.enable_phase3d_deferred_promotion),
            )
            return {
                **selector_payload,
                **promotion_state,
                "decision": "prototype",
                "prototype": best_active[0],
                "distance": float(best_active[1]),
                "action": "attach_active_sibling",
                "override_action_type": "attach_active_sibling",
                "override_selected_state": "active_sibling",
                "recovery_attach_target": "active_sibling",
                "recovery_attach_target_id": int(best_active[0].prototype_id),
                "promote_to_head": bool(promotion_state["promotion_success"]),
                "promote_score_candidate": float(best_active[3]),
                "attach_state_written": True,
                "temp_attach_used": False,
                "temp_attach_id": None,
                "temp_attach_support_count": 0,
                "temp_attach_promote_ready": False,
                "temp_attach_expired": False,
            }

        if (
            not force_temp_attach
            and best_archived is not None
            and float(best_archived[2]) >= float(self.attach_accept_min + 0.05)
        ):
            promotion_state = self._update_lineage_promotion_state(
                lineage_id=int(lineage_seed_id),
                candidate_id=int(best_archived[0].prototype_id),
                frame_index=frame_index,
                candidate_promote_score=float(best_archived[3]),
                head_promote_score=head_promote_score,
                allow_promotion=bool(self.enable_phase3d_deferred_promotion),
            )
            return {
                **selector_payload,
                **promotion_state,
                "decision": "prototype",
                "prototype": best_archived[0],
                "distance": float(best_archived[1]),
                "action": "attach_archived_sibling",
                "override_action_type": "attach_archived_sibling",
                "override_selected_state": "archived_sibling",
                "recovery_attach_target": "archived_sibling",
                "recovery_attach_target_id": int(best_archived[0].prototype_id),
                "promote_to_head": bool(promotion_state["promotion_success"]),
                "promote_score_candidate": float(best_archived[3]),
                "attach_state_written": True,
                "temp_attach_used": False,
                "temp_attach_id": None,
                "temp_attach_support_count": 0,
                "temp_attach_promote_ready": False,
                "temp_attach_expired": False,
            }

        if not self.enable_phase3d_temp_attach:
            return None

        anchor_prototype = head_prototype
        if anchor_prototype is None and best_active is not None:
            anchor_prototype = best_active[0]
        if anchor_prototype is None and best_archived is not None:
            anchor_prototype = best_archived[0]
        if anchor_prototype is None:
            return None

        temp_slot = self._start_or_refresh_temp_attach(
            lineage_id=int(lineage_seed_id),
            frame_index=frame_index,
            assignment=assignment,
            memory_signature=memory_signature,
            anchor_prototype_id=int(anchor_prototype.prototype_id),
            source_prototype_id=int(anchor_prototype.prototype_id),
        )
        self._clear_lineage_promotion_pending(int(lineage_seed_id))
        return {
            **selector_payload,
            "promotion_candidate_id": None,
            "promotion_pending_flag": False,
            "promotion_window_progress": 0,
            "promotion_decision": "keep_head",
            "promotion_support_count": 0,
            "promotion_delay_frames": None,
            "promotion_success": False,
            "decision": "temp_attach",
            "prototype": anchor_prototype,
            "distance": float(head_distance if head_distance is not None else (best_active[1] if best_active is not None else (best_archived[1] if best_archived is not None else 1.0))),
            "action": "temp_attach",
            "override_action_type": "temp_attach",
            "override_selected_state": "temporary_attach_slot",
            "recovery_attach_target": "temporary_attach_slot",
            "recovery_attach_target_id": int(temp_slot.temp_attach_id),
            "promote_to_head": False,
            "attach_path_source": "fallback_temp_attach",
            "attach_state_written": True,
            "temp_attach_used": True,
            "temp_attach_id": int(temp_slot.temp_attach_id),
            "temp_attach_support_count": int(temp_slot.support_count),
            "temp_attach_promote_ready": bool(temp_slot.promote_ready),
            "temp_attach_expired": bool(temp_slot.expired),
        }

    def _allocate_lineage(
        self,
        inherited_from_prototype_id: int | None,
        inherited_from_lineage_id: int | None = None,
    ) -> tuple[int, int | None]:
        if inherited_from_lineage_id is not None:
            return int(inherited_from_lineage_id), int(inherited_from_lineage_id)
        if inherited_from_prototype_id is not None:
            source = self._prototypes.get(int(inherited_from_prototype_id))
            if source is not None:
                return int(source.lineage_id), int(source.lineage_id)
        lineage_id = int(self._next_lineage_id)
        self._next_lineage_id += 1
        return lineage_id, None

    def _assign_prototype(self, assignment: TrackAssignment, frame_index: int) -> PrototypeAssignment:
        memory_signature = _to_memory_signature(assignment.signature)
        candidates = [prototype for prototype in self._prototypes.values() if not prototype.retired]
        previous_prototype_id = assignment.linked_prototype_id
        previous_lineage_id = (
            int(getattr(assignment, "continuity_lineage_id"))
            if getattr(assignment, "continuity_lineage_id", None) is not None
            and int(getattr(assignment, "continuity_lineage_id")) >= 0
            else int(assignment.linked_lineage_id)
            if assignment.linked_lineage_id is not None and int(assignment.linked_lineage_id) >= 0
            else self._resolve_lineage_id(previous_prototype_id)
        )
        lineage_seed_id = previous_lineage_id
        head_prototype_id_before = self._current_head_prototype_id(lineage_seed_id)
        lineage_audit = self._lineage_selector_audit(
            lineage_id=lineage_seed_id,
            memory_signature=memory_signature,
            head_prototype_id_before=head_prototype_id_before,
        )

        if not candidates:
            prototype_id = self._create_prototype(
                assignment,
                memory_signature,
                frame_index,
                inherited_from_prototype_id=self._resolve_lineage_seed_prototype_id(
                    assignment,
                    previous_prototype_id,
                ),
                inherited_from_lineage_id=(
                    None
                    if getattr(assignment, "continuity_lineage_id", None) is None
                    else int(assignment.continuity_lineage_id)
                ),
            )
            prototype = self._prototypes[prototype_id]
            action_type, selected_state = self._selector_action_type(
                selected_prototype_id=prototype_id,
                selected_lineage_id=int(prototype.lineage_id),
                selected_was_newborn=True,
                selected_was_retired=False,
                head_prototype_id_before=head_prototype_id_before,
                lineage_seed_id=lineage_seed_id,
                previous_lineage_id=previous_lineage_id,
            )
            return PrototypeAssignment(
                track_id=assignment.track_id,
                prototype_id=prototype_id,
                lineage_id=int(prototype.lineage_id),
                head_prototype_id=self._current_head_prototype_id(int(prototype.lineage_id)),
                head_prototype_id_before=head_prototype_id_before,
                head_prototype_id_after=self._current_head_prototype_id(int(prototype.lineage_id)),
                similarity=0.0,
                distance=1.0,
                soft_similarity=0.0,
                update_weight=1.0,
                action="birth",
                action_type=action_type,
                selected_prototype_state=selected_state,
                head_score=None if lineage_audit["head_score"] is None else float(lineage_audit["head_score"]),
                best_active_sibling_score=None
                if lineage_audit["best_active_sibling_score"] is None
                else float(lineage_audit["best_active_sibling_score"]),
                best_archived_sibling_score=None
                if lineage_audit["best_archived_sibling_score"] is None
                else float(lineage_audit["best_archived_sibling_score"]),
                birth_trigger_score=float(assignment.score),
                score_margin_vs_current_head=None,
                head_switched=False,
                box=assignment.box,
                score=assignment.score,
                previous_prototype_id=previous_prototype_id,
            previous_lineage_id=previous_lineage_id,
            concept_only_recovery=False,
            new_prototype_created=True,
            prototype_protected=prototype.is_protected,
            prototype_signature=prototype.signature.copy(),
            continuity_lineage_id=(
                int(prototype.continuity_lineage_id)
                if prototype.continuity_lineage_id is not None
                else int(prototype.lineage_id)
            ),
        )

        if assignment.linked_prototype_id is not None:
            linked = self._prototypes.get(int(assignment.linked_prototype_id))
            if linked is not None and not linked.retired:
                linked_distance = _prototype_distance(linked.signature, memory_signature)
                linked_attach_limit = max(self.tau_birth, self._concept_attach_limit(linked))
                if linked_distance <= linked_attach_limit:
                    action = "merge" if linked_distance <= self.tau_merge else "reuse"
                    return self._build_assignment(
                        assignment=assignment,
                        prototype=linked,
                        distance=linked_distance,
                        action=action,
                        frame_index=frame_index,
                        concept_only_recovery=False,
                        previous_prototype_id=previous_prototype_id,
                        lineage_seed_id=lineage_seed_id,
                        head_prototype_id_before=head_prototype_id_before,
                        lineage_audit=lineage_audit,
                    )

        hinted_score_floor = max(0.0, self.tau_obj_attach - 0.08)
        if lineage_seed_id is not None and assignment.score >= hinted_score_floor:
            selector_result = self._select_within_lineage_attach_policy(
                assignment=assignment,
                memory_signature=memory_signature,
                frame_index=frame_index,
                lineage_seed_id=int(lineage_seed_id),
                head_prototype_id_before=head_prototype_id_before,
            )
            if selector_result is not None:
                return self._build_assignment(
                    assignment=assignment,
                    prototype=selector_result["prototype"],
                    distance=float(selector_result["distance"]),
                    action=str(selector_result["action"]),
                    frame_index=frame_index,
                    concept_only_recovery=True,
                    previous_prototype_id=previous_prototype_id,
                    lineage_seed_id=lineage_seed_id,
                    head_prototype_id_before=head_prototype_id_before,
                    lineage_audit=lineage_audit,
                    promote_to_head=bool(selector_result.get("promote_to_head", True)),
                    override_action_type=selector_result.get("override_action_type"),
                    override_selected_state=selector_result.get("override_selected_state"),
                    matched_lineage_id=selector_result.get("matched_lineage_id"),
                    recovery_attach_target=selector_result.get("recovery_attach_target"),
                    recovery_attach_target_id=selector_result.get("recovery_attach_target_id"),
                    attach_path_source=selector_result.get("attach_path_source"),
                    attach_score_current_head=selector_result.get("attach_score_current_head"),
                    attach_score_active_sibling=selector_result.get("attach_score_active_sibling"),
                    attach_score_archived_sibling=selector_result.get("attach_score_archived_sibling"),
                    attach_score_temp_slot=selector_result.get("attach_score_temp_slot"),
                    promote_score_candidate=selector_result.get("promote_score_candidate"),
                    promote_score_current_head=selector_result.get("promote_score_current_head"),
                    promotion_candidate_id=selector_result.get("promotion_candidate_id"),
                    promotion_pending_flag=selector_result.get("promotion_pending_flag"),
                    promotion_window_progress=selector_result.get("promotion_window_progress"),
                    promotion_decision=selector_result.get("promotion_decision"),
                    promotion_support_count=selector_result.get("promotion_support_count"),
                    promotion_delay_frames=selector_result.get("promotion_delay_frames"),
                    promotion_success=selector_result.get("promotion_success"),
                    temp_attach_used=selector_result.get("temp_attach_used", False),
                    temp_attach_id=selector_result.get("temp_attach_id"),
                    temp_attach_support_count=selector_result.get("temp_attach_support_count", 0),
                    temp_attach_promote_ready=selector_result.get("temp_attach_promote_ready", False),
                    temp_attach_expired=selector_result.get("temp_attach_expired", False),
                    attach_branch_entered=selector_result.get("attach_branch_entered", False),
                    temp_attach_eligibility_checked=selector_result.get("temp_attach_eligibility_checked", False),
                    attach_state_written=selector_result.get("attach_state_written", False),
                    temp_attach_force_mode=selector_result.get("temp_attach_force_mode", False),
                    lineage_seed_id_used=selector_result.get("lineage_seed_id_used"),
                )
            selector_result = self._select_within_lineage_policy(
                assignment=assignment,
                memory_signature=memory_signature,
                frame_index=frame_index,
                lineage_seed_id=int(lineage_seed_id),
                head_prototype_id_before=head_prototype_id_before,
                lineage_audit=lineage_audit,
            )
            if selector_result is not None:
                if selector_result.get("decision") == "prototype":
                    return self._build_assignment(
                        assignment=assignment,
                        prototype=selector_result["prototype"],
                        distance=float(selector_result["distance"]),
                        action=str(selector_result["action"]),
                        frame_index=frame_index,
                        concept_only_recovery=True,
                        previous_prototype_id=previous_prototype_id,
                        lineage_seed_id=lineage_seed_id,
                        head_prototype_id_before=head_prototype_id_before,
                        lineage_audit=lineage_audit,
                    )
                if selector_result.get("decision") == "suppress_birth":
                    head_candidate_id = self._current_head_prototype_id(int(lineage_seed_id))
                    if head_candidate_id is not None:
                        head_candidate = self._prototypes.get(int(head_candidate_id))
                        if head_candidate is not None:
                            fallback_distance = _prototype_distance(head_candidate.signature, memory_signature)
                            return self._build_assignment(
                                assignment=assignment,
                                prototype=head_candidate,
                                distance=float(fallback_distance),
                                action="keep_head",
                                frame_index=frame_index,
                                concept_only_recovery=True,
                                previous_prototype_id=previous_prototype_id,
                                lineage_seed_id=lineage_seed_id,
                                head_prototype_id_before=head_prototype_id_before,
                                lineage_audit=lineage_audit,
                            )
            lineage_candidates = [prototype for prototype in candidates if int(prototype.lineage_id) == int(lineage_seed_id)]
            if lineage_candidates:
                best_lineage_distance, best_lineage_prototype_id = min(
                    (
                        (_prototype_distance(prototype.signature, memory_signature), prototype.prototype_id)
                        for prototype in lineage_candidates
                    ),
                    key=lambda item: item[0],
                )
                lineage_prototype = self._prototypes[int(best_lineage_prototype_id)]
                if best_lineage_distance <= self._concept_attach_limit(lineage_prototype):
                    return self._build_assignment(
                        assignment=assignment,
                        prototype=lineage_prototype,
                        distance=best_lineage_distance,
                        action="lineage_attach",
                        frame_index=frame_index,
                        concept_only_recovery=True,
                        previous_prototype_id=previous_prototype_id,
                        lineage_seed_id=lineage_seed_id,
                        head_prototype_id_before=head_prototype_id_before,
                        lineage_audit=lineage_audit,
                    )
        if self.use_concept_only_recovery and assignment.prototype_hint_id is not None and assignment.score >= hinted_score_floor:
            hinted = self._prototypes.get(int(assignment.prototype_hint_id))
            if hinted is not None and not hinted.retired:
                hinted_distance = _prototype_distance(hinted.signature, memory_signature)
                if hinted_distance <= self._concept_attach_limit(hinted):
                    return self._build_assignment(
                        assignment=assignment,
                        prototype=hinted,
                        distance=hinted_distance,
                        action="concept_reuse",
                        frame_index=frame_index,
                        concept_only_recovery=True,
                        previous_prototype_id=previous_prototype_id,
                        lineage_seed_id=lineage_seed_id,
                        head_prototype_id_before=head_prototype_id_before,
                        lineage_audit=lineage_audit,
                    )

        distances = [
            (_prototype_distance(prototype.signature, memory_signature), prototype.prototype_id)
            for prototype in candidates
        ]
        best_distance, best_prototype_id = min(distances, key=lambda item: item[0])
        prototype = self._prototypes[int(best_prototype_id)]
        concept_score_floor = max(0.0, self.tau_obj_attach - (0.08 if prototype.is_protected else 0.05))
        if (
            self.use_concept_only_recovery
            and assignment.score >= concept_score_floor
            and best_distance <= self._concept_attach_limit(prototype)
        ):
            return self._build_assignment(
                assignment=assignment,
                prototype=prototype,
                distance=best_distance,
                action="concept_attach",
                frame_index=frame_index,
                concept_only_recovery=True,
                previous_prototype_id=previous_prototype_id,
                lineage_seed_id=lineage_seed_id,
                head_prototype_id_before=head_prototype_id_before,
                lineage_audit=lineage_audit,
            )
        if best_distance <= self.tau_merge:
            return self._build_assignment(
                assignment=assignment,
                prototype=prototype,
                distance=best_distance,
                action="merge",
                frame_index=frame_index,
                concept_only_recovery=False,
                previous_prototype_id=previous_prototype_id,
                lineage_seed_id=lineage_seed_id,
                head_prototype_id_before=head_prototype_id_before,
                lineage_audit=lineage_audit,
            )
        if best_distance <= self.tau_birth:
            return self._build_assignment(
                assignment=assignment,
                prototype=prototype,
                distance=best_distance,
                action="reuse",
                frame_index=frame_index,
                concept_only_recovery=False,
                previous_prototype_id=previous_prototype_id,
                lineage_seed_id=lineage_seed_id,
                head_prototype_id_before=head_prototype_id_before,
                lineage_audit=lineage_audit,
            )

        prototype_id = self._create_prototype(
            assignment,
            memory_signature,
            frame_index,
            inherited_from_prototype_id=self._resolve_lineage_seed_prototype_id(
                assignment,
                previous_prototype_id,
            ),
            inherited_from_lineage_id=(
                None
                if getattr(assignment, "continuity_lineage_id", None) is None
                else int(assignment.continuity_lineage_id)
            ),
        )
        prototype = self._prototypes[prototype_id]
        action_type, selected_state = self._selector_action_type(
            selected_prototype_id=prototype_id,
            selected_lineage_id=int(prototype.lineage_id),
            selected_was_newborn=True,
            selected_was_retired=False,
            head_prototype_id_before=head_prototype_id_before,
            lineage_seed_id=lineage_seed_id,
            previous_lineage_id=previous_lineage_id,
        )
        return PrototypeAssignment(
            track_id=assignment.track_id,
            prototype_id=prototype_id,
            lineage_id=int(prototype.lineage_id),
            head_prototype_id=self._current_head_prototype_id(int(prototype.lineage_id)),
            head_prototype_id_before=head_prototype_id_before,
            head_prototype_id_after=self._current_head_prototype_id(int(prototype.lineage_id)),
            similarity=0.0,
            distance=float(best_distance),
            soft_similarity=float(np.exp(-best_distance / max(self.tau_sim, 1e-6))),
            update_weight=1.0,
            action="birth",
            action_type=action_type,
            selected_prototype_state=selected_state,
            head_score=None if lineage_audit["head_score"] is None else float(lineage_audit["head_score"]),
            best_active_sibling_score=None
            if lineage_audit["best_active_sibling_score"] is None
            else float(lineage_audit["best_active_sibling_score"]),
            best_archived_sibling_score=None
            if lineage_audit["best_archived_sibling_score"] is None
            else float(lineage_audit["best_archived_sibling_score"]),
            birth_trigger_score=float(assignment.score),
            score_margin_vs_current_head=None,
            head_switched=False,
            box=assignment.box,
            score=assignment.score,
            previous_prototype_id=previous_prototype_id,
            previous_lineage_id=previous_lineage_id,
            concept_only_recovery=False,
            new_prototype_created=True,
            prototype_protected=prototype.is_protected,
            prototype_signature=prototype.signature.copy(),
            continuity_lineage_id=(
                int(prototype.continuity_lineage_id)
                if prototype.continuity_lineage_id is not None
                else int(prototype.lineage_id)
            ),
        )

    def _build_assignment(
        self,
        *,
        assignment: TrackAssignment,
        prototype: PrototypeState,
        distance: float,
        action: str,
        frame_index: int,
        concept_only_recovery: bool,
        previous_prototype_id: int | None,
        lineage_seed_id: int | None,
        head_prototype_id_before: int | None,
        lineage_audit: dict[str, object],
        promote_to_head: bool = True,
        override_action_type: str | None = None,
        override_selected_state: str | None = None,
        matched_lineage_id: int | None = None,
        recovery_attach_target: str = "none",
        recovery_attach_target_id: int | None = None,
        attach_path_source: str = "",
        attach_score_current_head: float | None = None,
        attach_score_active_sibling: float | None = None,
        attach_score_archived_sibling: float | None = None,
        attach_score_temp_slot: float | None = None,
        promote_score_candidate: float | None = None,
        promote_score_current_head: float | None = None,
        promotion_candidate_id: int | None = None,
        promotion_pending_flag: bool = False,
        promotion_window_progress: int = 0,
        promotion_decision: str = "keep_head",
        promotion_support_count: int = 0,
        promotion_delay_frames: int | None = None,
        promotion_success: bool = False,
        temp_attach_used: bool = False,
        temp_attach_id: int | None = None,
        temp_attach_support_count: int = 0,
        temp_attach_promote_ready: bool = False,
        temp_attach_expired: bool = False,
        attach_branch_entered: bool = False,
        temp_attach_eligibility_checked: bool = False,
        attach_state_written: bool = False,
        temp_attach_force_mode: bool = False,
        lineage_seed_id_used: int | None = None,
    ) -> PrototypeAssignment:
        memory_signature = _to_memory_signature(assignment.signature)
        soft_similarity = float(np.exp(-distance / max(self.tau_sim, 1e-6)))
        update_weight = float(np.clip(self.lr_proto * soft_similarity, 0.0, 1.0))
        selected_was_retired = bool(prototype.retired)
        if temp_attach_used:
            update_weight = 0.0
        self._update_prototype(
            prototype,
            memory_signature,
            assignment,
            frame_index,
            soft_similarity,
            update_weight,
            promote_to_head=bool(promote_to_head),
        )
        head_prototype_id_after = self._current_head_prototype_id(int(prototype.lineage_id))
        action_type, selected_state = self._selector_action_type(
            selected_prototype_id=int(prototype.prototype_id),
            selected_lineage_id=int(prototype.lineage_id),
            selected_was_newborn=False,
            selected_was_retired=selected_was_retired,
            head_prototype_id_before=head_prototype_id_before,
            lineage_seed_id=lineage_seed_id,
            previous_lineage_id=self._resolve_lineage_id(previous_prototype_id),
        )
        if override_action_type is not None:
            action_type = str(override_action_type)
        if override_selected_state is not None:
            selected_state = str(override_selected_state)
        head_score = None if lineage_audit["head_score"] is None else float(lineage_audit["head_score"])
        score_margin = None if head_score is None else float(soft_similarity - head_score)
        if concept_only_recovery:
            self._set_lineage_recovery_lock(int(prototype.lineage_id), frame_index)
        if (
            head_prototype_id_before is not None
            and head_prototype_id_after is not None
            and int(head_prototype_id_before) != int(head_prototype_id_after)
        ):
            self._lineage_last_head_switch_frame[int(prototype.lineage_id)] = int(frame_index)
        return PrototypeAssignment(
            track_id=assignment.track_id,
            prototype_id=prototype.prototype_id,
            lineage_id=int(prototype.lineage_id),
            head_prototype_id=head_prototype_id_after,
            head_prototype_id_before=head_prototype_id_before,
            head_prototype_id_after=head_prototype_id_after,
            similarity=soft_similarity,
            distance=float(distance),
            soft_similarity=soft_similarity,
            update_weight=update_weight,
            action=action,
            action_type=action_type,
            selected_prototype_state=selected_state,
            head_score=head_score,
            best_active_sibling_score=None
            if lineage_audit["best_active_sibling_score"] is None
            else float(lineage_audit["best_active_sibling_score"]),
            best_archived_sibling_score=None
            if lineage_audit["best_archived_sibling_score"] is None
            else float(lineage_audit["best_archived_sibling_score"]),
            birth_trigger_score=float(assignment.score),
            score_margin_vs_current_head=score_margin,
            head_switched=bool(
                head_prototype_id_before is not None
                and head_prototype_id_after is not None
                and int(head_prototype_id_before) != int(head_prototype_id_after)
            ),
            box=assignment.box,
            score=assignment.score,
            previous_prototype_id=previous_prototype_id,
            previous_lineage_id=self._resolve_lineage_id(previous_prototype_id),
            concept_only_recovery=concept_only_recovery,
            new_prototype_created=False,
            prototype_protected=prototype.is_protected,
            prototype_signature=prototype.signature.copy(),
            matched_lineage_id=None if matched_lineage_id is None else int(matched_lineage_id),
            recovery_attach_target=str(recovery_attach_target),
            recovery_attach_target_id=None if recovery_attach_target_id is None else int(recovery_attach_target_id),
            current_head_prototype_id=head_prototype_id_after,
            promotion_candidate_id=None if promotion_candidate_id is None else int(promotion_candidate_id),
            promotion_pending_flag=bool(promotion_pending_flag),
            promotion_window_progress=int(promotion_window_progress),
            promotion_decision=str(promotion_decision),
            attach_path_source=str(attach_path_source),
            attach_score_current_head=None if attach_score_current_head is None else float(attach_score_current_head),
            attach_score_active_sibling=None
            if attach_score_active_sibling is None
            else float(attach_score_active_sibling),
            attach_score_archived_sibling=None
            if attach_score_archived_sibling is None
            else float(attach_score_archived_sibling),
            attach_score_temp_slot=None if attach_score_temp_slot is None else float(attach_score_temp_slot),
            promote_score_candidate=None if promote_score_candidate is None else float(promote_score_candidate),
            promote_score_current_head=None
            if promote_score_current_head is None
            else float(promote_score_current_head),
            promotion_support_count=int(promotion_support_count),
            promotion_delay_frames=None if promotion_delay_frames is None else int(promotion_delay_frames),
            promotion_success=bool(promotion_success),
            temp_attach_used=bool(temp_attach_used),
            temp_attach_id=None if temp_attach_id is None else int(temp_attach_id),
            temp_attach_support_count=int(temp_attach_support_count),
            temp_attach_promote_ready=bool(temp_attach_promote_ready),
            temp_attach_expired=bool(temp_attach_expired),
            attach_branch_entered=bool(attach_branch_entered),
            temp_attach_eligibility_checked=bool(temp_attach_eligibility_checked),
            attach_state_written=bool(attach_state_written),
            temp_attach_force_mode=bool(temp_attach_force_mode),
            lineage_seed_id_used=None if lineage_seed_id_used is None else int(lineage_seed_id_used),
            continuity_lineage_id=(
                int(prototype.continuity_lineage_id)
                if prototype.continuity_lineage_id is not None
                else int(prototype.lineage_id)
            ),
        )

    def _create_prototype(
        self,
        assignment: TrackAssignment,
        memory_signature: np.ndarray,
        frame_index: int,
        inherited_from_prototype_id: int | None = None,
        inherited_from_lineage_id: int | None = None,
    ) -> int:
        prototype_id = self._next_prototype_id
        self._next_prototype_id += 1
        lineage_id, parent_lineage_id = self._allocate_lineage(
            inherited_from_prototype_id,
            inherited_from_lineage_id=inherited_from_lineage_id,
        )
        continuity_lineage_id = (
            self._resolve_continuity_lineage_id(inherited_from_prototype_id)
            if inherited_from_prototype_id is not None
            else inherited_from_lineage_id
        )
        if continuity_lineage_id is None:
            continuity_lineage_id = int(lineage_id)
        self._prototypes[prototype_id] = PrototypeState(
            prototype_id=prototype_id,
            lineage_id=int(lineage_id),
            parent_lineage_id=parent_lineage_id,
            signature=memory_signature.copy(),
            hits=1,
            age=1,
            strength=max(assignment.score, self.decay_floor + 0.05),
            inactive_steps=0,
            active=True,
            is_active=True,
            retired=False,
            birth_frame=frame_index,
            created_frame=frame_index,
            last_updated_frame=frame_index,
            last_track_id=assignment.track_id,
            is_protected=False,
            protected_until_frame=-1,
            merged_from=(),
            split_from=None,
            continuation_bank_ref=f"lineage:{lineage_id}",
            active_track_refs=[],
            continuation_bank=[],
            runtime_owner_lineage_id=int(lineage_id),
            continuity_lineage_id=int(continuity_lineage_id),
            origin_lineage_id=int(continuity_lineage_id),
            continuity_key_valid=True,
        )
        self._mark_lineage_active(self._prototypes[prototype_id], frame_index=frame_index)
        return prototype_id

    def _update_prototype(
        self,
        prototype: PrototypeState,
        memory_signature: np.ndarray,
        assignment: TrackAssignment,
        frame_index: int,
        soft_similarity: float,
        update_weight: float,
        *,
        promote_to_head: bool = True,
    ) -> None:
        prototype.signature = (
            (1.0 - update_weight) * prototype.signature + update_weight * memory_signature
        ).astype(np.float32)
        prototype.signature = _normalize_signature(prototype.signature)
        prototype.hits += 1
        prototype.strength = min(4.0, prototype.strength + max(0.08, assignment.score * max(soft_similarity, 0.10)))
        prototype.inactive_steps = 0
        prototype.active = True
        prototype.is_active = True
        prototype.retired = False
        prototype.last_updated_frame = frame_index
        prototype.last_track_id = assignment.track_id
        prototype.runtime_owner_lineage_id = int(prototype.lineage_id)
        if prototype.continuity_lineage_id is None:
            prototype.continuity_lineage_id = int(prototype.lineage_id)
        if prototype.origin_lineage_id is None:
            prototype.origin_lineage_id = int(prototype.continuity_lineage_id)
        prototype.continuity_key_valid = bool(prototype.continuity_lineage_id is not None)
        self._mark_lineage_member_active(prototype, frame_index=frame_index, as_head=bool(promote_to_head))

    def _apply_budget_cap(self, frame_index: int) -> list[int]:
        active_prototypes = [prototype for prototype in self._prototypes.values() if not prototype.retired]
        if len(active_prototypes) <= self.memory_budget:
            return []

        active_prototypes.sort(
            key=lambda prototype: (
                prototype.is_protected,
                prototype.active,
                prototype.strength,
                prototype.hits,
                -prototype.inactive_steps,
                -prototype.last_updated_frame,
            )
        )

        retired_ids: list[int] = []
        for prototype in active_prototypes:
            if self._active_count() <= self.memory_budget:
                break
            if prototype.is_protected:
                continue
            self._retire_prototype(prototype, frame_index=frame_index, reason="budget")
            retired_ids.append(prototype.prototype_id)
        return retired_ids

    def _active_count(self) -> int:
        return sum(1 for prototype in self._prototypes.values() if not prototype.retired)

    def _concept_attach_limit(self, prototype: PrototypeState) -> float:
        return float(self.tau_proto_attach + (0.06 if prototype.is_protected else 0.0))

    def _lineage_continuation_bank(self, lineage_id: int | None) -> list[IdentityContinuation]:
        if lineage_id is None:
            return []
        lineage = self._ensure_lineage(int(lineage_id), frame_index=0)
        return lineage.continuation_bank

    def _continuation_owner_context(
        self,
        *,
        prototype: PrototypeState | None = None,
        lineage: ConceptLineage | None = None,
        lineage_id: int | None = None,
    ) -> tuple[int | None, int | None]:
        if prototype is not None:
            return int(prototype.prototype_id), int(prototype.lineage_id)
        if lineage is None and lineage_id is not None:
            lineage = self._lineages.get(int(lineage_id))
        if lineage is None:
            return None, None
        head_id = self._current_head_prototype_id(int(lineage.lineage_id))
        return (None if head_id is None else int(head_id), int(lineage.lineage_id))

    def _refresh_continuations(self, track_states: list[TrackState] | None, frame_index: int) -> int:
        archive_events = 0
        live_track_ids: set[int] = set()
        if track_states:
            live_track_ids = {
                int(track.track_id)
                for track in track_states
                if track.active
            }
        if self.bind_continuation_to == "prototype":
            owners: list[tuple[PrototypeState | None, ConceptLineage | None, list[IdentityContinuation]]] = [
                (prototype, None, prototype.continuation_bank) for prototype in self._prototypes.values()
            ]
        else:
            owners = [
                (None, lineage, lineage.continuation_bank) for lineage in self._lineages.values()
            ]

        for owner_prototype, owner_lineage, bank in owners:
            refreshed: list[IdentityContinuation] = []
            for continuation in bank:
                continuation.age_since_last_seen = int(max(0, frame_index - continuation.last_seen_frame))
                continuation.continuation_confidence = float(
                    max(0.0, continuation.continuation_confidence - self.continuation_decay)
                )
                drop_reason = "alive"
                if continuation.track_id in live_track_ids:
                    drop_reason = "source_track_active"
                    self._log_continuation_lifecycle(
                        continuation=continuation,
                        owner_prototype=owner_prototype,
                        owner_lineage=owner_lineage,
                        frame_index=frame_index,
                        is_alive=False,
                        drop_reason=drop_reason,
                    )
                    continue
                if continuation.age_since_last_seen > self.continuation_max_gap:
                    drop_reason = "gap_exceeded"
                    self._log_continuation_lifecycle(
                        continuation=continuation,
                        owner_prototype=owner_prototype,
                        owner_lineage=owner_lineage,
                        frame_index=frame_index,
                        is_alive=False,
                        drop_reason=drop_reason,
                    )
                    continue
                if continuation.continuation_confidence <= 0.0:
                    drop_reason = "confidence_expired"
                    self._log_continuation_lifecycle(
                        continuation=continuation,
                        owner_prototype=owner_prototype,
                        owner_lineage=owner_lineage,
                        frame_index=frame_index,
                        is_alive=False,
                        drop_reason=drop_reason,
                    )
                    continue
                refreshed.append(continuation)
            trimmed = self._trim_continuations(refreshed)
            if owner_prototype is not None:
                owner_prototype.continuation_bank = trimmed
            elif owner_lineage is not None:
                owner_lineage.continuation_bank = trimmed

        if not self.enable_continuation_bank or not track_states:
            return archive_events

        for track in track_states:
            if not (track.dormant or track.ghost or track.retired):
                continue
            source_lineage_id = (
                int(track.lineage_id)
                if getattr(track, "lineage_id", None) is not None and int(getattr(track, "lineage_id", -1)) >= 0
                else self._resolve_lineage_id(track.prototype_id)
            )
            if track.prototype_id is None and source_lineage_id is None:
                self._log_continuation_write(
                    track=track,
                    source_prototype_id=None,
                    source_lineage_id=None,
                    frame_index=frame_index,
                    write_success=False,
                    write_reason="missing_prototype_ref",
                    continuation_uid=None,
                )
                continue
            prototype = None if track.prototype_id is None else self._prototypes.get(int(track.prototype_id))
            if source_lineage_id is None and prototype is not None:
                source_lineage_id = int(prototype.lineage_id)
            if source_lineage_id is None:
                self._log_continuation_write(
                    track=track,
                    source_prototype_id=None if track.prototype_id is None else int(track.prototype_id),
                    source_lineage_id=None,
                    frame_index=frame_index,
                    write_success=False,
                    write_reason="owner_lineage_missing",
                    continuation_uid=None,
                )
                continue
            archive_events += self._archive_track_continuation(
                prototype,
                int(source_lineage_id),
                track,
                frame_index,
            )
        return archive_events

    def _refresh_recovery_anchors(self, track_states: list[TrackState] | None, frame_index: int) -> None:
        for lineage_id, lineage in self._lineages.items():
            refreshed: list[RecoveryIdentityAnchor] = []
            for anchor in lineage.recovery_identity_anchors:
                age_since_last_seen = int(max(0, frame_index - int(anchor.last_alive_frame)))
                anchor.age_since_last_seen = age_since_last_seen
                anchor.anchor_state = (
                    "expired"
                    if age_since_last_seen > self.anchor_ttl
                    else ("stale" if age_since_last_seen > max(4, int(self.anchor_ttl * 0.6)) else "alive")
                )
                if age_since_last_seen > self.anchor_ttl:
                    self._log_recovery_anchor_lifecycle(
                        frame_index=frame_index,
                        lineage_id=int(lineage_id),
                        anchor=anchor,
                        is_alive=False,
                        drop_reason="ttl_expired",
                    )
                    continue
                refreshed.append(anchor)
                self._log_recovery_anchor_lifecycle(
                    frame_index=frame_index,
                    lineage_id=int(lineage_id),
                    anchor=anchor,
                    is_alive=True,
                    drop_reason="alive",
                )
            lineage.recovery_identity_anchors = self._trim_recovery_anchors(refreshed)

        if not track_states:
            return

        for track in track_states:
            if not (track.dormant or track.ghost or track.retired):
                continue
            source_lineage_id = (
                int(track.lineage_id)
                if getattr(track, "lineage_id", None) is not None and int(getattr(track, "lineage_id", -1)) >= 0
                else self._resolve_lineage_id(track.prototype_id)
            )
            if source_lineage_id is None:
                continue
            self._archive_recovery_anchor(track=track, lineage_id=int(source_lineage_id), frame_index=frame_index)

    def _archive_recovery_anchor(
        self,
        *,
        track: TrackState,
        lineage_id: int,
        frame_index: int,
    ) -> int:
        if track.age < self.min_track_age_for_continuation or track.hit_count < self.min_hits_for_continuation:
            self._log_recovery_anchor_event(
                frame_index=frame_index,
                lineage_id=int(lineage_id),
                old_track_id=int(track.track_id),
                old_prototype_id=None if track.prototype_id is None else int(track.prototype_id),
                anchor_uid=None,
                event_type="skip",
                event_reason="insufficient_track_maturity",
            )
            return 0

        owner_lineage = self._ensure_lineage(int(lineage_id), frame_index=frame_index)
        prototype = None if track.prototype_id is None else self._prototypes.get(int(track.prototype_id))
        continuity_lineage_id = None
        if getattr(track, "continuity_lineage_id", None) is not None:
            continuity_lineage_id = int(getattr(track, "continuity_lineage_id"))
        elif prototype is not None and prototype.continuity_lineage_id is not None:
            continuity_lineage_id = int(prototype.continuity_lineage_id)
        else:
            continuity_lineage_id = int(lineage_id)
        age_since_last_seen = int(max(0, frame_index - int(track.last_seen_frame)))
        if age_since_last_seen > self.anchor_ttl:
            self._log_recovery_anchor_event(
                frame_index=frame_index,
                lineage_id=int(lineage_id),
                old_track_id=int(track.track_id),
                old_prototype_id=None if track.prototype_id is None else int(track.prototype_id),
                anchor_uid=None,
                event_type="skip",
                event_reason="anchor_ttl_exceeded",
            )
            return 0

        restore_priority = float(
            np.clip(
                0.40 * min(1.0, float(track.hit_count) / max(self.min_hits_for_continuation + 2, 1))
                + 0.30 * min(1.0, float(track.age) / max(self.min_track_age_for_continuation + 4, 1))
                + 0.20 * np.clip(float(track.score), 0.0, 1.0)
                + 0.10 * (1.0 if track.retired else 0.7),
                0.0,
                1.0,
            )
        )
        continuation_hint = "continuation_bank" if owner_lineage.continuation_bank else "none"
        anchor_origin = "archived_track" if track.retired else ("continuation_snapshot" if track.ghost else "explicit_recovery_seed")
        existing = next(
            (item for item in owner_lineage.recovery_identity_anchors if int(item.old_track_id) == int(track.track_id)),
            None,
        )
        if existing is not None and int(existing.last_alive_frame) >= int(track.last_seen_frame):
            self._log_recovery_anchor_event(
                frame_index=frame_index,
                lineage_id=int(lineage_id),
                old_track_id=int(track.track_id),
                old_prototype_id=None if track.prototype_id is None else int(track.prototype_id),
                anchor_uid=str(existing.anchor_uid),
                event_type="keep",
                event_reason="existing_newer",
                restore_priority=float(existing.restore_priority),
                anchor_state=str(existing.anchor_state),
            )
            return 0

        anchor = RecoveryIdentityAnchor(
            anchor_uid=f"anchor_{lineage_id}_{int(track.track_id)}",
            old_track_id=int(track.track_id),
            old_prototype_id=int(track.prototype_id if track.prototype_id is not None else -1),
            old_lineage_id=int(lineage_id),
            source_frame_id=int(frame_index),
            last_alive_frame=int(track.last_seen_frame),
            gap_length_at_creation=int(age_since_last_seen),
            restore_priority=restore_priority,
            continuation_hint=str(continuation_hint),
            anchor_state="stale" if age_since_last_seen > max(4, int(self.anchor_ttl * 0.6)) else "alive",
            anchor_origin=str(anchor_origin),
            age_since_last_seen=int(age_since_last_seen),
            last_center=(float(track.last_center[0]), float(track.last_center[1])),
            last_bbox=track.last_bbox,
            velocity=track.velocity.copy(),
            feature_ema=track.anchor_signature.copy(),
            last_objectness=float(track.score),
            track_age=int(track.age),
            hit_count=int(track.hit_count),
            runtime_owner_lineage_id=int(lineage_id),
            continuity_lineage_id=int(continuity_lineage_id),
            origin_lineage_id=int(continuity_lineage_id),
            old_identity_ref_track_id=int(track.track_id),
            old_identity_ref_prototype_id=None if track.prototype_id is None else int(track.prototype_id),
            continuity_key_valid=True,
        )
        kept = [
            item
            for item in owner_lineage.recovery_identity_anchors
            if int(item.old_track_id) != int(track.track_id)
        ]
        kept.append(anchor)
        owner_lineage.recovery_identity_anchors = self._trim_recovery_anchors(kept)
        self._log_recovery_anchor_event(
            frame_index=frame_index,
            lineage_id=int(lineage_id),
            old_track_id=int(track.track_id),
            old_prototype_id=None if track.prototype_id is None else int(track.prototype_id),
            anchor_uid=str(anchor.anchor_uid),
            event_type="write" if existing is None else "refresh",
            event_reason=str(anchor_origin),
            restore_priority=float(anchor.restore_priority),
            anchor_state=str(anchor.anchor_state),
        )
        return 1

    def _trim_recovery_anchors(self, items: list[RecoveryIdentityAnchor]) -> list[RecoveryIdentityAnchor]:
        ordered = sorted(items, key=self._recovery_anchor_rank, reverse=True)
        return ordered[: int(self.recovery_anchor_topk_per_lineage)]

    def _recovery_anchor_rank(self, anchor: RecoveryIdentityAnchor) -> float:
        recency = 1.0 / max(1.0, float(anchor.age_since_last_seen))
        stability = min(1.0, float(anchor.hit_count) / max(self.min_hits_for_continuation + 2, 1))
        maturity = min(1.0, float(anchor.track_age) / max(self.min_track_age_for_continuation + 4, 1))
        return float(
            0.45 * float(anchor.restore_priority)
            + 0.25 * stability
            + 0.15 * maturity
            + 0.15 * recency
        )

    def _archive_track_continuation(
        self,
        prototype: PrototypeState | None,
        lineage_id: int,
        track: TrackState,
        frame_index: int,
    ) -> int:
        owner_lineage = self._ensure_lineage(
            int(lineage_id),
            frame_index=frame_index,
            birth_frame=frame_index if prototype is None else int(prototype.birth_frame),
            parent_lineage_id=None if prototype is None else prototype.parent_lineage_id,
        )
        source_prototype_id = None if prototype is None else int(prototype.prototype_id)
        if track.age < self.min_track_age_for_continuation:
            self._log_continuation_write(
                track=track,
                source_prototype_id=source_prototype_id,
                source_lineage_id=int(lineage_id),
                frame_index=frame_index,
                write_success=False,
                write_reason="track_too_young",
                continuation_uid=None,
            )
            return 0

        if track.hit_count < self.min_hits_for_continuation:
            self._log_continuation_write(
                track=track,
                source_prototype_id=source_prototype_id,
                source_lineage_id=int(lineage_id),
                frame_index=frame_index,
                write_success=False,
                write_reason="insufficient_hits",
                continuation_uid=None,
            )
            return 0

        age_since_last_seen = int(max(0, frame_index - track.last_seen_frame))
        if age_since_last_seen > self.continuation_max_gap:
            self._log_continuation_write(
                track=track,
                source_prototype_id=source_prototype_id,
                source_lineage_id=int(lineage_id),
                frame_index=frame_index,
                write_success=False,
                write_reason="gap_exceeded",
                continuation_uid=None,
            )
            return 0

        bank = prototype.continuation_bank if self.bind_continuation_to == "prototype" and prototype is not None else owner_lineage.continuation_bank
        existing = next(
            (item for item in bank if int(item.track_id) == int(track.track_id)),
            None,
        )
        if existing is not None and int(existing.last_seen_frame) >= int(track.last_seen_frame):
            self._log_continuation_write(
                track=track,
                source_prototype_id=source_prototype_id,
                source_lineage_id=int(lineage_id),
                frame_index=frame_index,
                write_success=False,
                write_reason="existing_newer",
                continuation_uid=str(existing.continuation_uid),
            )
            return 0

        shape_signature = track.last_feature[10:13].copy() if track.last_feature.size >= 13 else track.last_feature.copy()
        confidence = float(
            np.clip(
                0.35 * min(1.0, track.hit_count / max(self.min_hits_for_continuation + 2, 1))
                + 0.30 * min(1.0, track.age / max(self.min_track_age_for_continuation + 4, 1))
                + 0.20 * np.clip(track.score, 0.0, 1.0)
                + 0.15 * (1.0 if track.miss_count <= max(2, self.min_track_age_for_continuation + 1) else 0.5),
                0.0,
                1.0,
            )
        )
        continuity_lineage_id = None
        if getattr(track, "continuity_lineage_id", None) is not None:
            continuity_lineage_id = int(getattr(track, "continuity_lineage_id"))
        elif prototype is not None and prototype.continuity_lineage_id is not None:
            continuity_lineage_id = int(prototype.continuity_lineage_id)
        else:
            continuity_lineage_id = int(lineage_id)
        archived = IdentityContinuation(
            continuation_id=self._next_continuation_id,
            continuation_uid=f"cont_{self._next_continuation_id}",
            prototype_id=int(source_prototype_id if source_prototype_id is not None else owner_lineage.current_head_prototype_id or -1),
            source_prototype_id=int(source_prototype_id if source_prototype_id is not None else owner_lineage.current_head_prototype_id or -1),
            source_lineage_id=int(lineage_id),
            track_id=int(track.track_id),
            write_frame=int(frame_index),
            last_seen_frame=int(track.last_seen_frame),
            last_center=(float(track.last_center[0]), float(track.last_center[1])),
            last_bbox=track.last_bbox,
            velocity=track.velocity.copy(),
            feature_ema=track.anchor_signature.copy(),
            shape_signature=shape_signature.astype(np.float32),
            last_objectness=float(track.score),
            track_age=int(track.age),
            hit_count=int(track.hit_count),
            age_since_last_seen=age_since_last_seen,
            continuation_confidence=confidence,
            was_occluded_before_disappear=bool(track.miss_count > 0),
            runtime_owner_lineage_id=int(lineage_id),
            continuity_lineage_id=int(continuity_lineage_id),
            origin_lineage_id=int(continuity_lineage_id),
            old_identity_ref_track_id=int(track.track_id),
            old_identity_ref_prototype_id=None if track.prototype_id is None else int(track.prototype_id),
            continuity_key_valid=True,
        )
        self._next_continuation_id += 1
        kept = [item for item in bank if int(item.track_id) != int(track.track_id)]
        kept.append(archived)
        trimmed = self._trim_continuations(kept)
        if self.bind_continuation_to == "prototype" and prototype is not None:
            prototype.continuation_bank = trimmed
        else:
            owner_lineage.continuation_bank = trimmed
        self._log_continuation_write(
            track=track,
            source_prototype_id=source_prototype_id,
            source_lineage_id=int(lineage_id),
            frame_index=frame_index,
            write_success=True,
            write_reason="archived",
            continuation_uid=str(archived.continuation_uid),
        )
        return 1

    def _trim_continuations(self, items: list[IdentityContinuation]) -> list[IdentityContinuation]:
        ordered = sorted(items, key=self._continuation_rank, reverse=True)
        topk = self.continuation_topk_per_proto if self.bind_continuation_to == "prototype" else self.continuation_topk_per_lineage
        return ordered[: int(topk)]

    def _continuation_rank(self, continuation: IdentityContinuation) -> float:
        recency = 1.0 / max(1.0, float(continuation.age_since_last_seen))
        maturity = min(1.0, float(continuation.track_age) / max(self.min_track_age_for_continuation + 4, 1))
        stability = min(1.0, float(continuation.hit_count) / max(self.min_hits_for_continuation + 2, 1))
        return float(
            0.40 * continuation.continuation_confidence
            + 0.25 * stability
            + 0.20 * maturity
            + 0.15 * recency
        )

    def _capture_alive_continuation_rows(self, frame_index: int) -> None:
        seen_uids = {
            str(row["continuation_uid"])
            for row in self._last_continuation_lifecycle_rows
            if int(row.get("frame_index", -1)) == int(frame_index)
        }
        if self.bind_continuation_to == "prototype":
            owners = [(prototype, None, prototype.continuation_bank) for prototype in self._prototypes.values()]
        else:
            owners = [(None, lineage, lineage.continuation_bank) for lineage in self._lineages.values()]
        for owner_prototype, owner_lineage, bank in owners:
            for continuation in bank:
                if str(continuation.continuation_uid) in seen_uids:
                    continue
                self._log_continuation_lifecycle(
                    continuation=continuation,
                    owner_prototype=owner_prototype,
                    owner_lineage=owner_lineage,
                    frame_index=frame_index,
                    is_alive=True,
                    drop_reason="alive",
                )

    def _capture_prototype_lineage_rows(self, frame_index: int) -> None:
        for prototype in sorted(self._prototypes.values(), key=lambda item: item.prototype_id):
            lineage = self._lineages.get(int(prototype.lineage_id))
            temp_slot = None if lineage is None else lineage.temp_attach_slot
            self._last_prototype_lineage_rows.append(
                {
                    "frame_index": int(frame_index),
                    "prototype_id": int(prototype.prototype_id),
                    "lineage_id": int(prototype.lineage_id),
                    "runtime_owner_lineage_id": None
                    if prototype.runtime_owner_lineage_id is None
                    else int(prototype.runtime_owner_lineage_id),
                    "continuity_lineage_id": None
                    if prototype.continuity_lineage_id is None
                    else int(prototype.continuity_lineage_id),
                    "origin_lineage_id": None
                    if prototype.origin_lineage_id is None
                    else int(prototype.origin_lineage_id),
                    "continuity_key_valid": int(bool(prototype.continuity_key_valid)),
                    "parent_lineage_id": None if prototype.parent_lineage_id is None else int(prototype.parent_lineage_id),
                    "birth_frame": int(prototype.birth_frame),
                    "last_update_frame": int(prototype.last_updated_frame),
                    "is_active": int(bool(prototype.is_active and not prototype.retired)),
                    "is_retired": int(bool(prototype.retired)),
                    "strength": float(prototype.strength),
                    "head_prototype_id": self._current_head_prototype_id(int(prototype.lineage_id)),
                    "continuation_bank_size": int(
                        len(prototype.continuation_bank)
                        if self.bind_continuation_to == "prototype"
                        else len(self._lineages.get(int(prototype.lineage_id), ConceptLineage(int(prototype.lineage_id), None)).continuation_bank)
                    ),
                    "continuation_bank_ref": str(prototype.continuation_bank_ref),
                    "merged_from": ",".join(str(int(value)) for value in prototype.merged_from),
                    "split_from": None if prototype.split_from is None else int(prototype.split_from),
                    "last_track_id": None if prototype.last_track_id is None else int(prototype.last_track_id),
                    "temp_attach_id": None if temp_slot is None else int(temp_slot.temp_attach_id),
                    "temp_attach_anchor_prototype_id": None
                    if temp_slot is None or temp_slot.anchor_prototype_id is None
                    else int(temp_slot.anchor_prototype_id),
                    "temp_attach_support_count": 0 if temp_slot is None else int(temp_slot.support_count),
                    "temp_attach_expired": 0 if temp_slot is None else int(bool(temp_slot.expired)),
                    "recovery_identity_anchor_count": 0
                    if lineage is None
                    else int(len(lineage.recovery_identity_anchors)),
                    "promotion_pending_flag": 0 if lineage is None else int(bool(lineage.promotion_pending_flag)),
                    "promotion_candidate_id": None
                    if lineage is None or lineage.promotion_candidate_id is None
                    else int(lineage.promotion_candidate_id),
                }
            )

    def _log_continuation_write(
        self,
        *,
        track: TrackState,
        source_prototype_id: int | None,
        source_lineage_id: int | None,
        frame_index: int,
        write_success: bool,
        write_reason: str,
        continuation_uid: str | None,
    ) -> None:
        self._last_continuation_write_rows.append(
            {
                "frame_index": int(frame_index),
                "track_id": int(track.track_id),
                "source_prototype_id": source_prototype_id,
                "source_lineage_id": source_lineage_id,
                "write_frame": int(frame_index),
                "write_success": int(bool(write_success)),
                "write_reason": str(write_reason),
                "continuation_uid": continuation_uid or "",
                "track_runtime_owner_lineage_id": None
                if getattr(track, "lineage_id", None) in (None, "", "None")
                else int(getattr(track, "lineage_id")),
                "track_continuity_lineage_id": None
                if getattr(track, "continuity_lineage_id", None) in (None, "", "None")
                else int(getattr(track, "continuity_lineage_id")),
                "track_old_identity_ref_track_id": int(track.track_id),
                "track_old_identity_ref_prototype_id": None
                if track.prototype_id is None
                else int(track.prototype_id),
            }
        )

    def _log_continuation_lifecycle(
        self,
        *,
        continuation: IdentityContinuation,
        owner_prototype: PrototypeState | None,
        owner_lineage: ConceptLineage | None = None,
        frame_index: int,
        is_alive: bool,
        drop_reason: str,
    ) -> None:
        owner_prototype_id, owner_lineage_id = self._continuation_owner_context(
            prototype=owner_prototype,
            lineage=owner_lineage,
            lineage_id=continuation.source_lineage_id,
        )
        self._last_continuation_lifecycle_rows.append(
            {
                "frame_index": int(frame_index),
                "continuation_uid": str(continuation.continuation_uid),
                "continuation_id": int(continuation.continuation_id),
                "track_id": int(continuation.track_id),
                "source_prototype_id": int(continuation.source_prototype_id),
                "source_lineage_id": int(continuation.source_lineage_id),
                "current_owner_prototype_id": owner_prototype_id,
                "current_owner_lineage_id": owner_lineage_id,
                "runtime_owner_lineage_id": None
                if continuation.runtime_owner_lineage_id is None
                else int(continuation.runtime_owner_lineage_id),
                "continuity_lineage_id": None
                if continuation.continuity_lineage_id is None
                else int(continuation.continuity_lineage_id),
                "origin_lineage_id": None
                if continuation.origin_lineage_id is None
                else int(continuation.origin_lineage_id),
                "old_identity_ref_track_id": None
                if continuation.old_identity_ref_track_id is None
                else int(continuation.old_identity_ref_track_id),
                "old_identity_ref_prototype_id": None
                if continuation.old_identity_ref_prototype_id is None
                else int(continuation.old_identity_ref_prototype_id),
                "continuity_key_valid": int(bool(continuation.continuity_key_valid)),
                "is_alive": int(bool(is_alive)),
                "write_frame": int(continuation.write_frame),
                "age_since_write": int(max(0, frame_index - continuation.write_frame)),
                "age_since_last_seen": int(continuation.age_since_last_seen),
                "decay_score": float(continuation.continuation_confidence),
                "drop_reason": str(drop_reason),
            }
        )

    def _log_recovery_anchor_event(
        self,
        *,
        frame_index: int,
        lineage_id: int,
        old_track_id: int,
        old_prototype_id: int | None,
        anchor_uid: str | None,
        event_type: str,
        event_reason: str,
        restore_priority: float | None = None,
        anchor_state: str | None = None,
    ) -> None:
        self._last_recovery_anchor_rows.append(
            {
                "frame_index": int(frame_index),
                "lineage_id": int(lineage_id),
                "old_track_id": int(old_track_id),
                "old_prototype_id": None if old_prototype_id is None else int(old_prototype_id),
                "anchor_uid": "" if anchor_uid is None else str(anchor_uid),
                "event_type": str(event_type),
                "event_reason": str(event_reason),
                "restore_priority": None if restore_priority is None else float(restore_priority),
                "anchor_state": "" if anchor_state is None else str(anchor_state),
                "runtime_owner_lineage_id": int(lineage_id),
            }
        )

    def _log_recovery_anchor_lifecycle(
        self,
        *,
        frame_index: int,
        lineage_id: int,
        anchor: RecoveryIdentityAnchor,
        is_alive: bool,
        drop_reason: str,
    ) -> None:
        self._last_recovery_anchor_lifecycle_rows.append(
            {
                "frame_index": int(frame_index),
                "lineage_id": int(lineage_id),
                "anchor_uid": str(anchor.anchor_uid),
                "old_track_id": int(anchor.old_track_id),
                "old_prototype_id": int(anchor.old_prototype_id),
                "runtime_owner_lineage_id": None
                if anchor.runtime_owner_lineage_id is None
                else int(anchor.runtime_owner_lineage_id),
                "continuity_lineage_id": None
                if anchor.continuity_lineage_id is None
                else int(anchor.continuity_lineage_id),
                "origin_lineage_id": None
                if anchor.origin_lineage_id is None
                else int(anchor.origin_lineage_id),
                "old_identity_ref_track_id": None
                if anchor.old_identity_ref_track_id is None
                else int(anchor.old_identity_ref_track_id),
                "old_identity_ref_prototype_id": None
                if anchor.old_identity_ref_prototype_id is None
                else int(anchor.old_identity_ref_prototype_id),
                "continuity_key_valid": int(bool(anchor.continuity_key_valid)),
                "last_alive_frame": int(anchor.last_alive_frame),
                "age_since_last_seen": int(anchor.age_since_last_seen),
                "restore_priority": float(anchor.restore_priority),
                "anchor_state": str(anchor.anchor_state),
                "is_alive": int(bool(is_alive)),
                "drop_reason": str(drop_reason),
            }
        )


def _normalize_signature(signature: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(signature))
    if norm < 1e-6:
        return np.zeros_like(signature, dtype=np.float32)
    return signature.astype(np.float32) / norm


def _prototype_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 1.0
    a = _normalize_signature(a)
    b = _normalize_signature(b)
    return float(np.linalg.norm(a - b) / np.sqrt(2.0))


def _to_memory_signature(signature: np.ndarray) -> np.ndarray:
    signature = signature.astype(np.float32)
    weights = np.ones_like(signature, dtype=np.float32)
    if signature.size >= 13:
        weights[:5] = np.array([0.15, 0.15, 0.20, 0.20, 0.15], dtype=np.float32)
        weights[5:7] = np.array([1.00, 0.85], dtype=np.float32)
        weights[7:10] = np.array([1.25, 1.25, 1.25], dtype=np.float32)
        weights[10:13] = np.array([0.80, 0.80, 0.60], dtype=np.float32)
    return _normalize_signature(signature * weights)


def _clone_continuation(continuation: IdentityContinuation) -> IdentityContinuation:
    return IdentityContinuation(
        continuation_id=int(continuation.continuation_id),
        continuation_uid=str(continuation.continuation_uid),
        prototype_id=int(continuation.prototype_id),
        source_prototype_id=int(continuation.source_prototype_id),
        source_lineage_id=int(continuation.source_lineage_id),
        track_id=int(continuation.track_id),
        write_frame=int(continuation.write_frame),
        last_seen_frame=int(continuation.last_seen_frame),
        last_center=(float(continuation.last_center[0]), float(continuation.last_center[1])),
        last_bbox=tuple(int(value) for value in continuation.last_bbox),
        velocity=continuation.velocity.copy(),
        feature_ema=continuation.feature_ema.copy(),
        shape_signature=continuation.shape_signature.copy(),
        last_objectness=float(continuation.last_objectness),
        track_age=int(continuation.track_age),
        hit_count=int(continuation.hit_count),
        age_since_last_seen=int(continuation.age_since_last_seen),
        continuation_confidence=float(continuation.continuation_confidence),
        was_occluded_before_disappear=bool(continuation.was_occluded_before_disappear),
        runtime_owner_lineage_id=None
        if continuation.runtime_owner_lineage_id is None
        else int(continuation.runtime_owner_lineage_id),
        continuity_lineage_id=None
        if continuation.continuity_lineage_id is None
        else int(continuation.continuity_lineage_id),
        origin_lineage_id=None
        if continuation.origin_lineage_id is None
        else int(continuation.origin_lineage_id),
        old_identity_ref_track_id=None
        if continuation.old_identity_ref_track_id is None
        else int(continuation.old_identity_ref_track_id),
        old_identity_ref_prototype_id=None
        if continuation.old_identity_ref_prototype_id is None
        else int(continuation.old_identity_ref_prototype_id),
        continuity_key_valid=bool(continuation.continuity_key_valid),
    )


def _clone_recovery_anchor(anchor: RecoveryIdentityAnchor) -> RecoveryIdentityAnchor:
    return RecoveryIdentityAnchor(
        anchor_uid=str(anchor.anchor_uid),
        old_track_id=int(anchor.old_track_id),
        old_prototype_id=int(anchor.old_prototype_id),
        old_lineage_id=int(anchor.old_lineage_id),
        source_frame_id=int(anchor.source_frame_id),
        last_alive_frame=int(anchor.last_alive_frame),
        gap_length_at_creation=int(anchor.gap_length_at_creation),
        restore_priority=float(anchor.restore_priority),
        continuation_hint=str(anchor.continuation_hint),
        anchor_state=str(anchor.anchor_state),
        anchor_origin=str(anchor.anchor_origin),
        age_since_last_seen=int(anchor.age_since_last_seen),
        last_center=(float(anchor.last_center[0]), float(anchor.last_center[1])),
        last_bbox=tuple(int(value) for value in anchor.last_bbox),
        velocity=anchor.velocity.copy(),
        feature_ema=anchor.feature_ema.copy(),
        last_objectness=float(anchor.last_objectness),
        track_age=int(anchor.track_age),
        hit_count=int(anchor.hit_count),
        runtime_owner_lineage_id=None
        if anchor.runtime_owner_lineage_id is None
        else int(anchor.runtime_owner_lineage_id),
        continuity_lineage_id=None
        if anchor.continuity_lineage_id is None
        else int(anchor.continuity_lineage_id),
        origin_lineage_id=None
        if anchor.origin_lineage_id is None
        else int(anchor.origin_lineage_id),
        old_identity_ref_track_id=None
        if anchor.old_identity_ref_track_id is None
        else int(anchor.old_identity_ref_track_id),
        old_identity_ref_prototype_id=None
        if anchor.old_identity_ref_prototype_id is None
        else int(anchor.old_identity_ref_prototype_id),
        continuity_key_valid=bool(anchor.continuity_key_valid),
    )
