"""Shared helpers for Phase 3R re-entry repair experiments."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from datasets import SyntheticStreamGenerator, load_synth_dataset_config
from experiments.scenario_presets import build_phase3_track_scenarios
from metrics.metrics_core import greedy_match_boxes
from nops_owr.evaluation import StreamingEpisodeEvaluator

REENTRY_WINDOW = 16
IOU_THRESHOLD = 0.5
GAP_BUCKETS = (
    ("short_gap", 1, 5),
    ("medium_gap", 6, 15),
    ("long_gap", 16, 30),
    ("very_long_gap", 31, 10**9),
)


def load_config_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_phase3_scenario_map(config_path: str | Path) -> dict[str, Any]:
    base_config = load_synth_dataset_config(config_path)
    scenarios = build_phase3_track_scenarios(base_config)
    return {scenario["name"]: scenario["config"] for scenario in scenarios}


def evaluate_phase3_scenarios(
    config_path: str | Path,
    *,
    tracking_override: dict[str, Any] | None = None,
    memory_override: dict[str, Any] | None = None,
    scenario_names: list[str] | None = None,
    collect_frames: bool = False,
    frame_record_mode: str = "full",
    seed: int = 42,
) -> list[dict[str, Any]]:
    payload = load_config_payload(config_path)
    scenario_map = build_phase3_scenario_map(config_path)
    selected_names = scenario_names or list(scenario_map.keys())
    evaluator = StreamingEpisodeEvaluator(
        payload,
        tracking_override=tracking_override,
        memory_override=memory_override,
    )

    outputs: list[dict[str, Any]] = []
    for scenario_name in selected_names:
        scenario_config = scenario_map[scenario_name]
        sequence = SyntheticStreamGenerator(
            scenario_config,
            seed=seed,
        ).generate_sequence(0)
        result = evaluator.evaluate(sequence, collect_frames=collect_frames, frame_record_mode=frame_record_mode)
        outputs.append(
            {
                "scenario_name": scenario_name,
                "config": scenario_config,
                "sequence": sequence,
                "result": result,
            }
        )
    return outputs


def baseline_phase3_overrides() -> tuple[dict[str, Any], dict[str, Any]]:
    tracking_override = {
        "use_gap_aware_matching": False,
        "keepalive_frames": 12,
        "dormant_frames": 192,
        "min_match_similarity": 0.18,
        "missed_match_similarity_boost": 0.05,
        "tau_proto_attach": 0.35,
        "tau_obj_attach": 0.50,
    }
    memory_override = {
        "decay_patience": 16,
        "use_concept_only_recovery": False,
        "protect_linked_prototypes": False,
        "tau_proto_attach": 0.35,
        "tau_obj_attach": 0.50,
    }
    return tracking_override, memory_override


def extract_reentry_events(
    scenario_name: str,
    sequence,
    result,
    *,
    recovery_window: int = REENTRY_WINDOW,
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame_maps = [_build_frame_instance_map(frame_record, iou_threshold=iou_threshold) for frame_record in result.frame_records]
    history_by_instance: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame_map in frame_maps:
        visible_ids = set(frame_map["instances"].keys())
        known_ids = set(visible_ids) | set(history_by_instance.keys())
        for instance_id in known_ids:
            history_by_instance[instance_id].append(
                frame_map["instances"].get(
                    instance_id,
                    {
                        "frame_index": frame_map["frame_index"],
                        "visible": False,
                        "proposal_detected": False,
                        "objectness_score": 0.0,
                        "track_id": None,
                        "prototype_id": None,
                        "prototype_lineage_id": None,
                        "prototype_continuity_lineage_id": None,
                        "pre_memory_linked_lineage_id": None,
                        "prototype_hint_lineage_id": None,
                        "head_prototype_id_before": None,
                        "head_prototype_id_after": None,
                        "selected_prototype_state": None,
                        "action_type": None,
                        "head_score": None,
                        "best_active_sibling_score": None,
                        "best_archived_sibling_score": None,
                        "birth_trigger_score": None,
                        "score_margin_vs_current_head": None,
                        "head_switched": False,
                        "matched_lineage_id": None,
                        "recovery_attach_target": "none",
                        "recovery_attach_target_id": None,
                        "current_head_prototype_id": None,
                        "attach_path_source": "",
                        "attach_score_current_head": None,
                        "attach_score_active_sibling": None,
                        "attach_score_archived_sibling": None,
                        "attach_score_temp_slot": None,
                        "promotion_candidate_id": None,
                        "promotion_pending_flag": False,
                        "promotion_window_progress": 0,
                        "promotion_decision": "keep_head",
                        "promote_score_candidate": None,
                        "promote_score_current_head": None,
                        "promotion_support_count": 0,
                        "promotion_delay_frames": None,
                        "promotion_success": False,
                        "promotion_regret_flag": False,
                        "temp_attach_used": False,
                        "temp_attach_id": None,
                        "temp_attach_support_count": 0,
                        "temp_attach_promote_ready": False,
                        "temp_attach_expired": False,
                        "attach_branch_entered": False,
                        "temp_attach_eligibility_checked": False,
                        "attach_state_written": False,
                        "temp_attach_force_mode": False,
                        "lineage_seed_id_used": None,
                        "new_track_created": False,
                        "new_prototype_created": False,
                        "reactivation_attempted": False,
                        "reactivation_cost": None,
                        "prototype_similarity": None,
                        "position_error": None,
                        "concept_only_recovery": False,
                        "concept_recovered": False,
                        "candidate_pool_size": 0,
                        "live_candidate_pool_size": 0,
                        "continuation_bank_size": 0,
                        "prototype_matched_continuation_count": 0,
                        "lineage_matched_continuation_count": 0,
                        "continuation_bank_exists": False,
                        "slot_candidate_pool_size": 0,
                        "candidate_pool_nonempty": False,
                        "resurrection_attempted": False,
                        "resurrection_success": False,
                        "resurrection_cost_best": None,
                        "best_candidate_state": None,
                        "best_candidate_gap": None,
                        "continuation_attempted": False,
                        "continuation_success": False,
                        "best_continuation_cost": None,
                        "best_continuation_gap": None,
                        "best_continuation_age": None,
                        "resurrected_from_continuation": False,
                        "slot_attempted": False,
                        "slot_success": False,
                        "best_slot_cost": None,
                        "best_slot_gap": None,
                        "best_slot_age": None,
                        "resurrected_from_slot": False,
                        "attach_state_consumed_by_tracker": False,
                        "attach_state_consumed_by_continuation": False,
                        "restore_attempted_from_attach": False,
                        "promotion_pending_created": False,
                        "promotion_step_executed": False,
                    },
                )
            )

    event_rows: list[dict[str, Any]] = []
    frame_event_counter: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    event_id = 0

    for instance_id, history in history_by_instance.items():
        last_visible_index: int | None = None
        last_track_id: int | None = None
        last_prototype_id: int | None = None
        last_lineage_id: int | None = None
        last_continuity_lineage_id: int | None = None
        for index, record in enumerate(history):
            if record["visible"]:
                if last_visible_index is not None and index - last_visible_index > 1:
                    window = history[index : min(len(history), index + max(1, recovery_window))]
                    visible_window = [item for item in window if item["visible"]]
                    first_visible = visible_window[0] if visible_window else record
                    proposal_window = [item for item in visible_window if item["proposal_detected"]]
                    anchor_record = proposal_window[0] if proposal_window else first_visible
                    same_track_rows = [
                        item for item in visible_window if item["track_id"] == last_track_id and item["track_id"] is not None
                    ]
                    same_prototype_rows = [
                        item
                        for item in visible_window
                        if item["prototype_id"] == last_prototype_id and item["prototype_id"] is not None
                    ]
                    same_lineage_rows = [
                        item
                        for item in visible_window
                        if item.get("prototype_lineage_id") == last_lineage_id and item.get("prototype_lineage_id") is not None
                    ]
                    same_continuity_lineage_rows = [
                        item
                        for item in visible_window
                        if item.get("prototype_continuity_lineage_id") == last_continuity_lineage_id
                        and item.get("prototype_continuity_lineage_id") is not None
                    ]
                    same_track = bool(same_track_rows)
                    same_prototype = bool(same_prototype_rows)
                    same_lineage = bool(same_lineage_rows)
                    same_continuity_lineage = bool(same_continuity_lineage_rows)
                    concept_recovered = bool(same_prototype)
                    concept_window = same_prototype_rows if same_prototype_rows else visible_window
                    lineage_window = same_lineage_rows if same_lineage_rows else visible_window
                    continuity_lineage_window = (
                        same_continuity_lineage_rows if same_continuity_lineage_rows else visible_window
                    )
                    candidate_pool_size = max(int(item.get("candidate_pool_size", 0)) for item in concept_window) if concept_window else 0
                    live_candidate_pool_size = (
                        max(int(item.get("live_candidate_pool_size", 0)) for item in concept_window) if concept_window else 0
                    )
                    continuation_bank_size = (
                        max(int(item.get("continuation_bank_size", 0)) for item in concept_window) if concept_window else 0
                    )
                    prototype_matched_continuation_count = (
                        max(int(item.get("prototype_matched_continuation_count", 0)) for item in lineage_window)
                        if lineage_window
                        else 0
                    )
                    lineage_matched_continuation_count = (
                        max(int(item.get("lineage_matched_continuation_count", 0)) for item in lineage_window)
                        if lineage_window
                        else 0
                    )
                    slot_candidate_pool_size = (
                        max(int(item.get("slot_candidate_pool_size", 0)) for item in concept_window) if concept_window else 0
                    )
                    candidate_pool_nonempty = bool(
                        any(bool(item.get("candidate_pool_nonempty")) for item in concept_window)
                        or candidate_pool_size > 0
                    )
                    resurrection_attempted = any(bool(item.get("resurrection_attempted")) for item in concept_window)
                    resurrection_success = any(bool(item.get("resurrection_success")) for item in concept_window)
                    cost_rows = [item for item in concept_window if _safe_float(item.get("resurrection_cost_best")) is not None]
                    best_attempt = (
                        min(cost_rows, key=lambda item: float(item["resurrection_cost_best"]))
                        if cost_rows
                        else None
                    )
                    continuation_attempted = any(bool(item.get("continuation_attempted")) for item in concept_window)
                    continuation_success = any(bool(item.get("continuation_success")) for item in concept_window)
                    continuation_cost_rows = [
                        item for item in concept_window if _safe_float(item.get("best_continuation_cost")) is not None
                    ]
                    best_continuation_attempt = (
                        min(continuation_cost_rows, key=lambda item: float(item["best_continuation_cost"]))
                        if continuation_cost_rows
                        else None
                    )
                    slot_attempted = any(bool(item.get("slot_attempted")) for item in concept_window)
                    slot_success = any(bool(item.get("slot_success")) for item in concept_window)
                    slot_cost_rows = [item for item in concept_window if _safe_float(item.get("best_slot_cost")) is not None]
                    best_slot_attempt = (
                        min(slot_cost_rows, key=lambda item: float(item["best_slot_cost"]))
                        if slot_cost_rows
                        else None
                    )
                    resurrected_from_slot = any(bool(item.get("resurrected_from_slot")) for item in concept_window)
                    resurrected_from_continuation = any(
                        bool(item.get("resurrected_from_continuation")) for item in concept_window
                    )
                    new_track_created = any(bool(item["new_track_created"]) for item in visible_window)
                    new_prototype_created = any(bool(item["new_prototype_created"]) for item in visible_window)
                    event_rows.append(
                        {
                            "event_id": event_id,
                            "scenario_name": scenario_name,
                            "instance_id": instance_id,
                            "old_track_id": last_track_id,
                            "old_prototype_id": last_prototype_id,
                            "old_lineage_id": last_lineage_id,
                            "old_continuity_lineage_id": last_continuity_lineage_id,
                            "disappear_frame": history[last_visible_index]["frame_index"],
                            "reappear_frame": first_visible["frame_index"],
                            "gap_length": first_visible["frame_index"] - history[last_visible_index]["frame_index"] - 1,
                            "gap_bucket": bucket_gap_length(
                                first_visible["frame_index"] - history[last_visible_index]["frame_index"] - 1
                            ),
                            "proposal_detected": int(any(bool(item["proposal_detected"]) for item in visible_window)),
                            "matched_prototype_id": None if anchor_record.get("prototype_id") is None else int(anchor_record["prototype_id"]),
                            "matched_lineage_id": None
                            if anchor_record.get("prototype_lineage_id") is None
                            else int(anchor_record["prototype_lineage_id"]),
                            "matched_continuity_lineage_id": None
                            if anchor_record.get("prototype_continuity_lineage_id") is None
                            else int(anchor_record["prototype_continuity_lineage_id"]),
                            "selected_prototype_id": None
                            if anchor_record.get("prototype_id") is None
                            else int(anchor_record["prototype_id"]),
                            "selected_prototype_state": anchor_record.get("selected_prototype_state"),
                            "action_type": anchor_record.get("action_type"),
                            "frame_id": int(first_visible["frame_index"]),
                            "head_prototype_id_before": None
                            if anchor_record.get("head_prototype_id_before") is None
                            else int(anchor_record["head_prototype_id_before"]),
                            "head_prototype_id_after": None
                            if anchor_record.get("head_prototype_id_after") is None
                            else int(anchor_record["head_prototype_id_after"]),
                            "head_score": _safe_float(anchor_record.get("head_score")),
                            "best_active_sibling_score": _safe_float(anchor_record.get("best_active_sibling_score")),
                            "best_archived_sibling_score": _safe_float(anchor_record.get("best_archived_sibling_score")),
                            "birth_trigger_score": _safe_float(anchor_record.get("birth_trigger_score")),
                            "score_margin_vs_current_head": _safe_float(anchor_record.get("score_margin_vs_current_head")),
                            "head_switched": int(bool(anchor_record.get("head_switched", False))),
                            "matched_lineage_id": None
                            if anchor_record.get("matched_lineage_id") is None
                            else int(anchor_record["matched_lineage_id"]),
                            "recovery_attach_target": str(anchor_record.get("recovery_attach_target", "none")),
                            "recovery_attach_target_id": None
                            if anchor_record.get("recovery_attach_target_id") is None
                            else int(anchor_record["recovery_attach_target_id"]),
                            "current_head_prototype_id": None
                            if anchor_record.get("current_head_prototype_id") is None
                            else int(anchor_record["current_head_prototype_id"]),
                            "attach_path_source": str(anchor_record.get("attach_path_source", "")),
                            "attach_score_current_head": _safe_float(anchor_record.get("attach_score_current_head")),
                            "attach_score_active_sibling": _safe_float(anchor_record.get("attach_score_active_sibling")),
                            "attach_score_archived_sibling": _safe_float(anchor_record.get("attach_score_archived_sibling")),
                            "attach_score_temp_slot": _safe_float(anchor_record.get("attach_score_temp_slot")),
                            "promotion_candidate_id": None
                            if anchor_record.get("promotion_candidate_id") is None
                            else int(anchor_record["promotion_candidate_id"]),
                            "promotion_pending_flag": int(bool(anchor_record.get("promotion_pending_flag", False))),
                            "promotion_window_progress": int(anchor_record.get("promotion_window_progress", 0) or 0),
                            "promotion_decision": str(anchor_record.get("promotion_decision", "keep_head")),
                            "promote_score_candidate": _safe_float(anchor_record.get("promote_score_candidate")),
                            "promote_score_current_head": _safe_float(anchor_record.get("promote_score_current_head")),
                            "promotion_support_count": int(anchor_record.get("promotion_support_count", 0) or 0),
                            "promotion_delay_frames": None
                            if anchor_record.get("promotion_delay_frames") is None
                            else int(anchor_record["promotion_delay_frames"]),
                            "promotion_success": int(bool(anchor_record.get("promotion_success", False))),
                            "promotion_regret_flag": int(bool(anchor_record.get("promotion_regret_flag", False))),
                            "temp_attach_used": int(bool(anchor_record.get("temp_attach_used", False))),
                            "temp_attach_id": None
                            if anchor_record.get("temp_attach_id") is None
                            else int(anchor_record["temp_attach_id"]),
                            "temp_attach_support_count": int(anchor_record.get("temp_attach_support_count", 0) or 0),
                            "temp_attach_promote_ready": int(bool(anchor_record.get("temp_attach_promote_ready", False))),
                            "temp_attach_expired": int(bool(anchor_record.get("temp_attach_expired", False))),
                            "pre_memory_linked_lineage_id": None
                            if anchor_record.get("pre_memory_linked_lineage_id") is None
                            else int(anchor_record["pre_memory_linked_lineage_id"]),
                            "prototype_hint_lineage_id": None
                            if anchor_record.get("prototype_hint_lineage_id") is None
                            else int(anchor_record["prototype_hint_lineage_id"]),
                            "attach_branch_entered": int(bool(anchor_record.get("attach_branch_entered", False))),
                            "temp_attach_eligibility_checked": int(
                                bool(anchor_record.get("temp_attach_eligibility_checked", False))
                            ),
                            "attach_state_written": int(bool(anchor_record.get("attach_state_written", False))),
                            "temp_attach_force_mode": int(bool(anchor_record.get("temp_attach_force_mode", False))),
                            "lineage_seed_id_used": None
                            if anchor_record.get("lineage_seed_id_used") is None
                            else int(anchor_record["lineage_seed_id_used"]),
                            "attach_state_consumed_by_tracker": int(
                                bool(anchor_record.get("attach_state_consumed_by_tracker", False))
                            ),
                            "attach_state_consumed_by_continuation": int(
                                bool(anchor_record.get("attach_state_consumed_by_continuation", False))
                            ),
                            "restore_attempted_from_attach": int(
                                bool(anchor_record.get("restore_attempted_from_attach", False))
                            ),
                            "promotion_pending_created": int(
                                bool(anchor_record.get("promotion_pending_created", False))
                            ),
                            "promotion_step_executed": int(bool(anchor_record.get("promotion_step_executed", False))),
                            "same_prototype_id": int(
                                anchor_record.get("prototype_id") is not None and anchor_record.get("prototype_id") == last_prototype_id
                            ),
                            "same_lineage_id": int(
                                anchor_record.get("prototype_lineage_id") is not None
                                and anchor_record.get("prototype_lineage_id") == last_lineage_id
                            ),
                            "same_continuity_lineage_id": int(
                                anchor_record.get("prototype_continuity_lineage_id") is not None
                                and anchor_record.get("prototype_continuity_lineage_id") == last_continuity_lineage_id
                            ),
                            "matched_same_track": int(bool(same_track)),
                            "matched_same_prototype": int(bool(same_prototype)),
                            "matched_same_lineage_prototype": int(bool(same_lineage)),
                            "matched_same_continuity_lineage_prototype": int(bool(same_continuity_lineage)),
                            "new_track_created": int(bool(new_track_created)),
                            "new_prototype_created": int(bool(new_prototype_created)),
                            "reactivation_attempted": int(bool(anchor_record["reactivation_attempted"] or resurrection_attempted)),
                            "reactivation_cost": (
                                _safe_float(best_attempt["resurrection_cost_best"])
                                if best_attempt is not None
                                else _safe_float(anchor_record["reactivation_cost"])
                            ),
                            "prototype_similarity": _safe_float(
                                next(
                                    (
                                        item["prototype_similarity"]
                                        for item in concept_window
                                        if _safe_float(item.get("prototype_similarity")) is not None
                                    ),
                                    anchor_record["prototype_similarity"],
                                )
                            ),
                            "position_error": _safe_float(
                                next(
                                    (
                                        item["position_error"]
                                        for item in concept_window
                                        if _safe_float(item.get("position_error")) is not None
                                    ),
                                    anchor_record["position_error"],
                                )
                            ),
                            "objectness_at_reentry": float(anchor_record["objectness_score"]),
                            "concept_only_recovery": int(any(bool(item["concept_only_recovery"]) for item in window if item["visible"])),
                            "concept_recovered": int(bool(concept_recovered)),
                            "candidate_pool_size": int(candidate_pool_size),
                            "live_candidate_pool_size": int(live_candidate_pool_size),
                            "continuation_bank_size": int(continuation_bank_size),
                            "prototype_matched_continuation_count": int(prototype_matched_continuation_count),
                            "lineage_matched_continuation_count": int(lineage_matched_continuation_count),
                            "continuation_bank_exists": int(int(continuation_bank_size) > 0),
                            "slot_candidate_pool_size": int(slot_candidate_pool_size),
                            "candidate_pool_nonempty": int(bool(candidate_pool_nonempty)),
                            "resurrection_attempted": int(bool(resurrection_attempted)),
                            "resurrection_success": int(bool(resurrection_success)),
                            "resurrection_cost_best": (
                                None if best_attempt is None else _safe_float(best_attempt["resurrection_cost_best"])
                            ),
                            "best_candidate_state": None if best_attempt is None else best_attempt["best_candidate_state"],
                            "best_candidate_gap": None if best_attempt is None else _safe_int(best_attempt.get("best_candidate_gap")),
                            "continuation_attempted": int(bool(continuation_attempted)),
                            "continuation_success": int(bool(continuation_success)),
                            "best_continuation_cost": (
                                None
                                if best_continuation_attempt is None
                                else _safe_float(best_continuation_attempt["best_continuation_cost"])
                            ),
                            "best_continuation_gap": (
                                None
                                if best_continuation_attempt is None
                                else _safe_int(best_continuation_attempt.get("best_continuation_gap"))
                            ),
                            "best_continuation_age": (
                                None
                                if best_continuation_attempt is None
                                else _safe_int(best_continuation_attempt.get("best_continuation_age"))
                            ),
                            "resurrected_from_continuation": int(bool(resurrected_from_continuation)),
                            "slot_attempted": int(bool(slot_attempted)),
                            "slot_success": int(bool(slot_success)),
                            "best_slot_cost": None if best_slot_attempt is None else _safe_float(best_slot_attempt["best_slot_cost"]),
                            "best_slot_gap": None if best_slot_attempt is None else _safe_int(best_slot_attempt.get("best_slot_gap")),
                            "best_slot_age": None if best_slot_attempt is None else _safe_int(best_slot_attempt.get("best_slot_age")),
                            "resurrected_from_slot": int(bool(resurrected_from_slot)),
                            "same_track_after_concept_recovery": int(bool(concept_recovered and same_track)),
                            "same_track_after_lineage_recovery": int(bool(same_lineage and same_track)),
                            "same_track_after_attach": int(
                                bool(anchor_record.get("recovery_attach_target", "none") != "none" and same_track)
                            ),
                            "same_prototype_after_attach": int(
                                bool(anchor_record.get("recovery_attach_target", "none") != "none" and same_prototype)
                            ),
                            "continuation_access_used": int(
                                bool(
                                    continuation_bank_size > 0
                                    or prototype_matched_continuation_count > 0
                                    or lineage_matched_continuation_count > 0
                                )
                            ),
                            "continuation_access_success": int(bool(continuation_success or resurrected_from_continuation)),
                            "recovery_path_preserved": int(
                                bool(
                                    anchor_record.get("recovery_attach_target", "none") != "none"
                                    and (
                                        continuation_bank_size > 0
                                        or candidate_pool_nonempty
                                        or same_track
                                    )
                                )
                            ),
                            "sequence_id": 0,
                            "gt_object_id": int(instance_id),
                            "recovered_lineage_id": None
                            if anchor_record.get("prototype_lineage_id") is None
                            else int(anchor_record["prototype_lineage_id"]),
                            "recovered_continuity_lineage_id": None
                            if anchor_record.get("prototype_continuity_lineage_id") is None
                            else int(anchor_record["prototype_continuity_lineage_id"]),
                            "pfr_delta_if_any": int(bool(new_prototype_created and not same_prototype)),
                            "idsw_delta_if_any": _count_identity_switches_in_window(window, key="track_id"),
                            "idsw_after_reentry_window": _count_identity_switches_in_window(window, key="track_id"),
                        }
                    )
                    frame_event_counter[anchor_record["frame_index"]]["reactivation_success_same_track"] += int(bool(same_track))
                    frame_event_counter[anchor_record["frame_index"]]["reactivation_success_same_prototype"] += int(bool(same_prototype))
                    frame_event_counter[anchor_record["frame_index"]]["prototype_gated_resurrection_attempts"] += int(bool(resurrection_attempted))
                    frame_event_counter[anchor_record["frame_index"]]["prototype_gated_resurrection_successes"] += int(bool(resurrection_success))
                    frame_event_counter[anchor_record["frame_index"]]["new_tracks_from_reentry"] += int(bool(new_track_created))
                    frame_event_counter[anchor_record["frame_index"]]["new_prototypes_from_reentry"] += int(bool(new_prototype_created))
                    event_id += 1
                last_visible_index = index
                if record["track_id"] is not None:
                    last_track_id = int(record["track_id"])
                if record["prototype_id"] is not None:
                    last_prototype_id = int(record["prototype_id"])
                if record.get("prototype_lineage_id") is not None:
                    last_lineage_id = int(record["prototype_lineage_id"])
                if record.get("prototype_continuity_lineage_id") is not None:
                    last_continuity_lineage_id = int(record["prototype_continuity_lineage_id"])

    frame_log_rows: list[dict[str, Any]] = []
    for frame_record in result.frame_records:
        tracking_output = frame_record.tracking_output
        frame_events = frame_event_counter.get(frame_record.frame_index, {})
        frame_log_rows.append(
            {
                "frame_id": int(frame_record.frame_index),
                "scenario_name": scenario_name,
                "num_active_tracks": int(tracking_output.active_track_count),
                "num_dormant_tracks": int(tracking_output.dormant_track_count),
                "num_ghost_tracks": int(tracking_output.ghost_track_count),
                "num_retired_tracks": int(tracking_output.retired_track_count),
                "num_continuations": int(getattr(frame_record.memory_output, "continuation_bank_count", 0)),
                "num_identity_slots": int(getattr(tracking_output, "identity_slot_count", 0)),
                "reactivation_attempts": int(tracking_output.reactivation_attempts),
                "resurrection_attempts": int(tracking_output.resurrection_attempts),
                "resurrection_successes": int(tracking_output.resurrection_successes),
                "continuation_archive_events": int(getattr(frame_record.memory_output, "continuation_archive_events", 0)),
                "continuation_resurrection_attempts": int(
                    getattr(tracking_output, "continuation_resurrection_attempts", 0)
                ),
                "continuation_resurrection_successes": int(
                    getattr(tracking_output, "continuation_resurrection_successes", 0)
                ),
                "slot_archive_events": int(getattr(tracking_output, "slot_archive_events", 0)),
                "slot_resurrection_attempts": int(getattr(tracking_output, "slot_resurrection_attempts", 0)),
                "slot_resurrection_successes": int(getattr(tracking_output, "slot_resurrection_successes", 0)),
                "reactivation_success_same_track": int(frame_events.get("reactivation_success_same_track", 0)),
                "reactivation_success_same_prototype": int(frame_events.get("reactivation_success_same_prototype", 0)),
                "prototype_gated_resurrection_attempts": int(frame_events.get("prototype_gated_resurrection_attempts", 0)),
                "prototype_gated_resurrection_successes": int(frame_events.get("prototype_gated_resurrection_successes", 0)),
                "new_tracks_from_reentry": int(frame_events.get("new_tracks_from_reentry", 0)),
                "new_prototypes_from_reentry": int(frame_events.get("new_prototypes_from_reentry", 0)),
            }
        )

    return event_rows, frame_log_rows


def summarize_gap_buckets(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[(str(row["scenario_name"]), str(row["gap_bucket"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (scenario_name, gap_bucket), rows in sorted(grouped.items()):
        costs = [value for row in rows if (value := _safe_float(row.get("reactivation_cost"))) is not None]
        proto_sims = [value for row in rows if (value := _safe_float(row.get("prototype_similarity"))) is not None]
        candidate_sizes = [int(row.get("candidate_pool_size", 0)) for row in rows]
        same_prototype_recovery = _mean_int(rows, "matched_same_prototype")
        same_lineage_recovery = _mean_int(rows, "matched_same_lineage_prototype")
        concept_recovered_events = sum(int(row["concept_recovered"]) for row in rows)
        lineage_concept_recovered_events = sum(int(row["matched_same_lineage_prototype"]) for row in rows)
        candidate_exists_events = sum(
            int(bool(row["concept_recovered"]) and int(row.get("candidate_pool_size", 0)) > 0) for row in rows
        )
        summary_rows.append(
            {
                "scenario_name": scenario_name,
                "gap_bucket": gap_bucket,
                "num_events": len(rows),
                "proposal_detect_rate": _mean_int(rows, "proposal_detected"),
                "same_track_recovery_rate": _mean_int(rows, "matched_same_track"),
                "same_prototype_recovery_rate": same_prototype_recovery,
                "same_lineage_prototype_recovery_rate": same_lineage_recovery,
                "same_track_after_concept_recovery_rate": (
                    sum(int(row["same_track_after_concept_recovery"]) for row in rows) / concept_recovered_events
                    if concept_recovered_events
                    else 0.0
                ),
                "same_track_after_lineage_recovery_rate": (
                    sum(int(row["same_track_after_lineage_recovery"]) for row in rows) / lineage_concept_recovered_events
                    if lineage_concept_recovered_events
                    else 0.0
                ),
                "new_track_rate": _mean_int(rows, "new_track_created"),
                "new_prototype_rate": _mean_int(rows, "new_prototype_created"),
                "reentry_fragmentation_rate": float(max(0.0, 1.0 - same_prototype_recovery)),
                "mean_reactivation_cost": _mean_values(costs),
                "mean_prototype_similarity": _mean_values(proto_sims),
                "mean_candidate_pool_size": _mean_values(candidate_sizes),
                "candidate_pool_nonempty_rate": (
                    sum(int(row["candidate_pool_nonempty"]) for row in rows) / concept_recovered_events
                    if concept_recovered_events
                    else 0.0
                ),
                "continuation_bank_nonempty_rate": (
                    sum(int(int(row.get("continuation_bank_size", 0)) > 0) for row in rows) / concept_recovered_events
                    if concept_recovered_events
                    else 0.0
                ),
                "slot_pool_nonempty_rate": (
                    sum(int(int(row.get("slot_candidate_pool_size", 0)) > 0) for row in rows) / concept_recovered_events
                    if concept_recovered_events
                    else 0.0
                ),
                "prototype_gated_resurrection_attempt_rate": (
                    sum(int(row["resurrection_attempted"]) for row in rows) / concept_recovered_events
                    if concept_recovered_events
                    else 0.0
                ),
                "slot_resurrection_attempt_rate": (
                    sum(int(row["slot_attempted"]) for row in rows) / concept_recovered_events
                    if concept_recovered_events
                    else 0.0
                ),
                "continuation_attempt_rate": (
                    sum(int(row["continuation_attempted"]) for row in rows) / concept_recovered_events
                    if concept_recovered_events
                    else 0.0
                ),
                "slot_resurrection_success_rate": (
                    sum(int(row["slot_success"]) for row in rows) / max(1, sum(int(row["slot_attempted"]) for row in rows))
                    if sum(int(row["slot_attempted"]) for row in rows)
                    else 0.0
                ),
                "continuation_success_rate": (
                    sum(int(row["continuation_success"]) for row in rows)
                    / max(1, sum(int(row["continuation_attempted"]) for row in rows))
                    if sum(int(row["continuation_attempted"]) for row in rows)
                    else 0.0
                ),
                "new_track_with_old_prototype_rate": (
                    sum(
                        int(bool(row["concept_recovered"]) and bool(row["new_track_created"]) and not bool(row["matched_same_track"]))
                        for row in rows
                    )
                    / concept_recovered_events
                    if concept_recovered_events
                    else 0.0
                ),
                "resurrection_success_given_candidate_exists": (
                    sum(int(row["resurrection_success"]) for row in rows if int(row.get("candidate_pool_size", 0)) > 0)
                    / candidate_exists_events
                    if candidate_exists_events
                    else 0.0
                ),
                "idsw_after_reentry_window": _mean_int(rows, "idsw_after_reentry_window"),
            }
        )
    return summary_rows


def summarize_reentry_events(event_rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not event_rows:
        return {
            "num_events": 0,
            "same_track_reentry_recovery": 0.0,
            "same_prototype_reentry_recovery": 0.0,
            "proposal_detect_rate": 0.0,
            "new_track_rate": 0.0,
            "new_prototype_rate": 0.0,
            "concept_only_recovery_rate": 0.0,
            "reactivation_attempt_rate": 0.0,
            "concept_recovered_events": 0,
            "lineage_aware_concept_recovered_events": 0,
            "same_track_after_concept_recovery": 0.0,
            "same_lineage_prototype_reentry_recovery": 0.0,
            "same_track_after_lineage_recovery": 0.0,
            "prototype_gated_resurrection_attempt_rate": 0.0,
            "resurrection_success_given_candidate_exists": 0.0,
            "candidate_exists_events": 0,
            "mean_candidate_pool_size": 0.0,
            "candidate_pool_nonempty_rate": 0.0,
            "continuation_bank_nonempty_rate": 0.0,
            "slot_pool_nonempty_rate": 0.0,
            "continuation_attempt_rate": 0.0,
            "continuation_success_rate": 0.0,
            "slot_resurrection_attempt_rate": 0.0,
            "slot_resurrection_success_rate": 0.0,
            "new_track_with_old_prototype_rate": 0.0,
        }
    concept_recovered_events = sum(int(row["concept_recovered"]) for row in event_rows)
    lineage_concept_recovered_events = sum(int(row["matched_same_lineage_prototype"]) for row in event_rows)
    candidate_exists_events = sum(
        int(bool(row["concept_recovered"]) and int(row.get("candidate_pool_size", 0)) > 0) for row in event_rows
    )
    return {
        "num_events": len(event_rows),
        "same_track_reentry_recovery": _mean_int(event_rows, "matched_same_track"),
        "same_prototype_reentry_recovery": _mean_int(event_rows, "matched_same_prototype"),
        "same_lineage_prototype_reentry_recovery": _mean_int(event_rows, "matched_same_lineage_prototype"),
        "proposal_detect_rate": _mean_int(event_rows, "proposal_detected"),
        "new_track_rate": _mean_int(event_rows, "new_track_created"),
        "new_prototype_rate": _mean_int(event_rows, "new_prototype_created"),
        "concept_only_recovery_rate": _mean_int(event_rows, "concept_only_recovery"),
        "reactivation_attempt_rate": _mean_int(event_rows, "reactivation_attempted"),
        "concept_recovered_events": concept_recovered_events,
        "lineage_aware_concept_recovered_events": lineage_concept_recovered_events,
        "same_track_after_concept_recovery": (
            sum(int(row["same_track_after_concept_recovery"]) for row in event_rows) / concept_recovered_events
            if concept_recovered_events
            else 0.0
        ),
        "same_track_after_lineage_recovery": (
            sum(int(row["same_track_after_lineage_recovery"]) for row in event_rows) / lineage_concept_recovered_events
            if lineage_concept_recovered_events
            else 0.0
        ),
        "prototype_gated_resurrection_attempt_rate": (
            sum(int(row["resurrection_attempted"]) for row in event_rows) / concept_recovered_events
            if concept_recovered_events
            else 0.0
        ),
        "resurrection_success_given_candidate_exists": (
            sum(
                int(row["resurrection_success"])
                for row in event_rows
                if int(row.get("candidate_pool_size", 0)) > 0 and int(row["concept_recovered"])
            )
            / candidate_exists_events
            if candidate_exists_events
            else 0.0
        ),
        "candidate_exists_events": candidate_exists_events,
        "mean_candidate_pool_size": _mean_values([int(row.get("candidate_pool_size", 0)) for row in event_rows]),
        "candidate_pool_nonempty_rate": (
            sum(int(row["candidate_pool_nonempty"]) for row in event_rows) / concept_recovered_events
            if concept_recovered_events
            else 0.0
        ),
        "continuation_bank_nonempty_rate": (
            sum(int(int(row.get("continuation_bank_size", 0)) > 0) for row in event_rows) / concept_recovered_events
            if concept_recovered_events
            else 0.0
        ),
        "slot_pool_nonempty_rate": (
            sum(int(int(row.get("slot_candidate_pool_size", 0)) > 0) for row in event_rows) / concept_recovered_events
            if concept_recovered_events
            else 0.0
        ),
        "continuation_attempt_rate": (
            sum(int(row["continuation_attempted"]) for row in event_rows) / concept_recovered_events
            if concept_recovered_events
            else 0.0
        ),
        "continuation_success_rate": (
            sum(int(row["continuation_success"]) for row in event_rows)
            / max(1, sum(int(row["continuation_attempted"]) for row in event_rows))
            if sum(int(row["continuation_attempted"]) for row in event_rows)
            else 0.0
        ),
        "slot_resurrection_attempt_rate": (
            sum(int(row["slot_attempted"]) for row in event_rows) / concept_recovered_events
            if concept_recovered_events
            else 0.0
        ),
        "slot_resurrection_success_rate": (
            sum(int(row["slot_success"]) for row in event_rows) / max(1, sum(int(row["slot_attempted"]) for row in event_rows))
            if sum(int(row["slot_attempted"]) for row in event_rows)
            else 0.0
        ),
        "new_track_with_old_prototype_rate": (
            sum(
                int(bool(row["concept_recovered"]) and bool(row["new_track_created"]) and not bool(row["matched_same_track"]))
                for row in event_rows
            )
            / concept_recovered_events
            if concept_recovered_events
            else 0.0
        ),
    }


def bucket_gap_length(gap_length: int) -> str:
    for name, low, high in GAP_BUCKETS:
        if low <= gap_length <= high:
            return name
    return "very_long_gap"


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_frame_instance_map(frame_record, *, iou_threshold: float) -> dict[str, Any]:
    objectness_matches = greedy_match_boxes(
        frame_record.gt_boxes,
        [proposal.box for proposal in frame_record.objectness_output.proposals],
        iou_threshold=iou_threshold,
    )
    tracking_matches = greedy_match_boxes(
        frame_record.gt_boxes,
        [assignment.box for assignment in frame_record.tracking_output.assignments],
        iou_threshold=iou_threshold,
    )
    memory_matches = greedy_match_boxes(
        frame_record.gt_boxes,
        [assignment.box for assignment in frame_record.memory_output.assignments],
        iou_threshold=iou_threshold,
    )

    objectness_by_gt = {gt_index: proposal_index for gt_index, proposal_index, _ in objectness_matches}
    tracking_by_gt = {gt_index: proposal_index for gt_index, proposal_index, _ in tracking_matches}
    memory_by_gt = {gt_index: proposal_index for gt_index, proposal_index, _ in memory_matches}

    instances: dict[int, dict[str, Any]] = {}
    for gt_index, instance_id in enumerate(frame_record.instance_ids):
        objectness_index = objectness_by_gt.get(gt_index)
        tracking_index = tracking_by_gt.get(gt_index)
        memory_index = memory_by_gt.get(gt_index)
        proposal = None if objectness_index is None else frame_record.objectness_output.proposals[objectness_index]
        track_assignment = None if tracking_index is None else frame_record.tracking_output.assignments[tracking_index]
        memory_assignment = None if memory_index is None else frame_record.memory_output.assignments[memory_index]
        instances[int(instance_id)] = {
            "frame_index": int(frame_record.frame_index),
            "visible": True,
            "proposal_detected": proposal is not None,
            "objectness_score": 0.0 if proposal is None else float(proposal.score),
            "track_id": None if track_assignment is None else int(track_assignment.track_id),
            "prototype_id": None if memory_assignment is None else int(memory_assignment.prototype_id),
            "prototype_lineage_id": None if memory_assignment is None else int(memory_assignment.lineage_id),
            "prototype_continuity_lineage_id": None
            if memory_assignment is None or getattr(memory_assignment, "continuity_lineage_id", None) is None
            else int(memory_assignment.continuity_lineage_id),
            "pre_memory_linked_lineage_id": None
            if track_assignment is None or track_assignment.pre_memory_linked_lineage_id is None
            else int(track_assignment.pre_memory_linked_lineage_id),
            "prototype_hint_lineage_id": None
            if track_assignment is None or track_assignment.prototype_hint_lineage_id is None
            else int(track_assignment.prototype_hint_lineage_id),
            "head_prototype_id_before": None
            if memory_assignment is None or memory_assignment.head_prototype_id_before is None
            else int(memory_assignment.head_prototype_id_before),
            "head_prototype_id_after": None
            if memory_assignment is None or memory_assignment.head_prototype_id_after is None
            else int(memory_assignment.head_prototype_id_after),
            "selected_prototype_state": None if memory_assignment is None else str(memory_assignment.selected_prototype_state),
            "action_type": None if memory_assignment is None else str(memory_assignment.action_type),
            "head_score": None
            if memory_assignment is None or memory_assignment.head_score is None
            else float(memory_assignment.head_score),
            "best_active_sibling_score": None
            if memory_assignment is None or memory_assignment.best_active_sibling_score is None
            else float(memory_assignment.best_active_sibling_score),
            "best_archived_sibling_score": None
            if memory_assignment is None or memory_assignment.best_archived_sibling_score is None
            else float(memory_assignment.best_archived_sibling_score),
            "birth_trigger_score": None
            if memory_assignment is None or memory_assignment.birth_trigger_score is None
            else float(memory_assignment.birth_trigger_score),
            "score_margin_vs_current_head": None
            if memory_assignment is None or memory_assignment.score_margin_vs_current_head is None
            else float(memory_assignment.score_margin_vs_current_head),
            "head_switched": False if memory_assignment is None else bool(memory_assignment.head_switched),
            "matched_lineage_id": None
            if memory_assignment is None or memory_assignment.matched_lineage_id is None
            else int(memory_assignment.matched_lineage_id),
            "recovery_attach_target": "none"
            if memory_assignment is None
            else str(memory_assignment.recovery_attach_target),
            "recovery_attach_target_id": None
            if memory_assignment is None or memory_assignment.recovery_attach_target_id is None
            else int(memory_assignment.recovery_attach_target_id),
            "current_head_prototype_id": None
            if memory_assignment is None or memory_assignment.current_head_prototype_id is None
            else int(memory_assignment.current_head_prototype_id),
            "attach_path_source": "" if memory_assignment is None else str(memory_assignment.attach_path_source),
            "attach_score_current_head": None
            if memory_assignment is None or memory_assignment.attach_score_current_head is None
            else float(memory_assignment.attach_score_current_head),
            "attach_score_active_sibling": None
            if memory_assignment is None or memory_assignment.attach_score_active_sibling is None
            else float(memory_assignment.attach_score_active_sibling),
            "attach_score_archived_sibling": None
            if memory_assignment is None or memory_assignment.attach_score_archived_sibling is None
            else float(memory_assignment.attach_score_archived_sibling),
            "attach_score_temp_slot": None
            if memory_assignment is None or memory_assignment.attach_score_temp_slot is None
            else float(memory_assignment.attach_score_temp_slot),
            "promotion_candidate_id": None
            if memory_assignment is None or memory_assignment.promotion_candidate_id is None
            else int(memory_assignment.promotion_candidate_id),
            "promotion_pending_flag": False
            if memory_assignment is None
            else bool(memory_assignment.promotion_pending_flag),
            "promotion_window_progress": 0
            if memory_assignment is None
            else int(memory_assignment.promotion_window_progress),
            "promotion_decision": "keep_head"
            if memory_assignment is None
            else str(memory_assignment.promotion_decision),
            "promote_score_candidate": None
            if memory_assignment is None or memory_assignment.promote_score_candidate is None
            else float(memory_assignment.promote_score_candidate),
            "promote_score_current_head": None
            if memory_assignment is None or memory_assignment.promote_score_current_head is None
            else float(memory_assignment.promote_score_current_head),
            "promotion_support_count": 0
            if memory_assignment is None
            else int(memory_assignment.promotion_support_count),
            "promotion_delay_frames": None
            if memory_assignment is None or memory_assignment.promotion_delay_frames is None
            else int(memory_assignment.promotion_delay_frames),
            "promotion_success": False if memory_assignment is None else bool(memory_assignment.promotion_success),
            "promotion_regret_flag": False if memory_assignment is None else bool(memory_assignment.promotion_regret_flag),
            "temp_attach_used": False if memory_assignment is None else bool(memory_assignment.temp_attach_used),
            "temp_attach_id": None
            if memory_assignment is None or memory_assignment.temp_attach_id is None
            else int(memory_assignment.temp_attach_id),
            "temp_attach_support_count": 0
            if memory_assignment is None
            else int(memory_assignment.temp_attach_support_count),
            "temp_attach_promote_ready": False
            if memory_assignment is None
            else bool(memory_assignment.temp_attach_promote_ready),
            "temp_attach_expired": False if memory_assignment is None else bool(memory_assignment.temp_attach_expired),
            "attach_branch_entered": False if memory_assignment is None else bool(memory_assignment.attach_branch_entered),
            "temp_attach_eligibility_checked": False
            if memory_assignment is None
            else bool(memory_assignment.temp_attach_eligibility_checked),
            "attach_state_written": False if memory_assignment is None else bool(memory_assignment.attach_state_written),
            "temp_attach_force_mode": False if memory_assignment is None else bool(memory_assignment.temp_attach_force_mode),
            "lineage_seed_id_used": None
            if memory_assignment is None or memory_assignment.lineage_seed_id_used is None
            else int(memory_assignment.lineage_seed_id_used),
            "new_track_created": False if track_assignment is None else track_assignment.assignment_source == "new_track",
            "new_prototype_created": False if memory_assignment is None else bool(memory_assignment.new_prototype_created),
            "reactivation_attempted": False if track_assignment is None else bool(track_assignment.reactivation_attempted),
            "reactivation_cost": None if track_assignment is None else float(track_assignment.reactivation_cost),
            "prototype_similarity": None if track_assignment is None else float(track_assignment.prototype_similarity),
            "position_error": None if track_assignment is None else float(track_assignment.position_error),
            "concept_only_recovery": False if memory_assignment is None else bool(memory_assignment.concept_only_recovery),
            "concept_recovered": False if track_assignment is None else bool(track_assignment.concept_recovered),
            "candidate_pool_size": 0 if track_assignment is None else int(track_assignment.candidate_pool_size),
            "live_candidate_pool_size": 0 if track_assignment is None else int(track_assignment.live_candidate_pool_size),
            "continuation_bank_size": 0 if track_assignment is None else int(track_assignment.continuation_bank_size),
            "prototype_matched_continuation_count": 0
            if track_assignment is None
            else int(track_assignment.prototype_matched_continuation_count),
            "lineage_matched_continuation_count": 0
            if track_assignment is None
            else int(track_assignment.lineage_matched_continuation_count),
            "continuation_bank_exists": False
            if track_assignment is None
            else bool(track_assignment.continuation_bank_exists),
            "slot_candidate_pool_size": 0 if track_assignment is None else int(track_assignment.slot_candidate_pool_size),
            "candidate_pool_nonempty": False if track_assignment is None else bool(track_assignment.candidate_pool_nonempty),
            "resurrection_attempted": False if track_assignment is None else bool(track_assignment.resurrection_attempted),
            "resurrection_success": False if track_assignment is None else bool(track_assignment.resurrection_success),
            "resurrection_cost_best": None
            if track_assignment is None or track_assignment.resurrection_cost_best is None
            else float(track_assignment.resurrection_cost_best),
            "best_candidate_state": None if track_assignment is None else track_assignment.best_candidate_state,
            "best_candidate_gap": None
            if track_assignment is None or track_assignment.best_candidate_gap is None
            else int(track_assignment.best_candidate_gap),
            "continuation_attempted": False if track_assignment is None else bool(track_assignment.continuation_attempted),
            "continuation_success": False if track_assignment is None else bool(track_assignment.continuation_success),
            "best_continuation_cost": None
            if track_assignment is None or track_assignment.best_continuation_cost is None
            else float(track_assignment.best_continuation_cost),
            "best_continuation_gap": None
            if track_assignment is None or track_assignment.best_continuation_gap is None
            else int(track_assignment.best_continuation_gap),
            "best_continuation_age": None
            if track_assignment is None or track_assignment.best_continuation_age is None
            else int(track_assignment.best_continuation_age),
            "resurrected_from_continuation": False
            if track_assignment is None
            else bool(track_assignment.resurrected_from_continuation),
            "slot_attempted": False if track_assignment is None else bool(track_assignment.slot_attempted),
            "slot_success": False if track_assignment is None else bool(track_assignment.slot_success),
            "best_slot_cost": None
            if track_assignment is None or track_assignment.best_slot_cost is None
            else float(track_assignment.best_slot_cost),
            "best_slot_gap": None
            if track_assignment is None or track_assignment.best_slot_gap is None
            else int(track_assignment.best_slot_gap),
            "best_slot_age": None
            if track_assignment is None or track_assignment.best_slot_age is None
            else int(track_assignment.best_slot_age),
            "resurrected_from_slot": False if track_assignment is None else bool(track_assignment.resurrected_from_slot),
            "attach_state_consumed_by_tracker": False
            if track_assignment is None
            else bool(track_assignment.attach_state_consumed_by_tracker),
            "attach_state_consumed_by_continuation": False
            if track_assignment is None
            else bool(track_assignment.attach_state_consumed_by_continuation),
            "restore_attempted_from_attach": False
            if track_assignment is None
            else bool(track_assignment.restore_attempted_from_attach),
            "promotion_pending_created": False
            if track_assignment is None
            else bool(track_assignment.promotion_pending_created),
            "promotion_step_executed": False
            if track_assignment is None
            else bool(track_assignment.promotion_step_executed),
        }
    return {"frame_index": int(frame_record.frame_index), "instances": instances}


def _count_identity_switches_in_window(window: list[dict[str, Any]], *, key: str) -> int:
    last_value = None
    switches = 0
    for item in window:
        value = item.get(key)
        if value is None:
            continue
        if last_value is not None and last_value != value:
            switches += 1
        last_value = value
    return switches


def _mean_int(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(int(row[key]) for row in rows) / len(rows))


def _mean_values(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        value = stripped
    return float(value)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        value = stripped
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
