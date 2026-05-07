from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator
from experiments.phase3r_utils import build_phase3_scenario_map, load_config_payload, write_csv
from experiments.run_v3_stage_e2r_continuity_source_repair import _tracking_override, _snapshot_state, _safe_int
from experiments.v3_utils import TRACK_C_NAME
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.memory import MinimalPrototypeMemory
from nops_owr.objectness import MinimalObjectnessField
from nops_owr.tracking import MinimalTemporalIdentityTracker

FOCUS_EVENT_IDS = {"M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"}
TARGET_PROTOTYPE_ID = 6
TARGET_TRACK_ID = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 E2R-M owner remap and canonical target mapping audit.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--event-audit", default="results/v3_e1/stage_E1_event_audit_v1.csv")
    parser.add_argument("--e2r-focus-summary", default="results/v3_e2r/stage_E2R_focus_event_summary_v3.csv")
    parser.add_argument("--e2r-source-trace", default="results/v3_e2r/stage_E2R_source_builder_trace_v3.csv")
    parser.add_argument("--output-dir", default="results/v3_e2rm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-version", default="v1")
    return parser.parse_args()


def _load_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _target_anchor_uid(row: dict[str, Any]) -> str:
    return "::".join([
        str(row.get("scenario_name", "track_c_long_horizon")),
        f"inst_{row.get('instance_id', '')}",
        f"old_track_{row.get('old_track_id', '')}",
        f"old_proto_{row.get('old_prototype_id', '')}",
        f"e1_lineage_{row.get('old_continuity_lineage_id') or row.get('old_lineage_id')}",
    ])


def _parse_candidate_ids(text: str) -> tuple[int | None, int | None]:
    track_id = None
    proto_id = None
    for part in str(text).split("|"):
        if part.startswith("track:"):
            track_id = _safe_int(part.split(":", 1)[1])
        elif part.startswith("proto:"):
            proto_id = _safe_int(part.split(":", 1)[1])
    return track_id, proto_id


def _capture_relevant_rows(frame_idx: int, tracker, memory) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    proto = memory._prototypes.get(TARGET_PROTOTYPE_ID)
    if proto is not None:
        rows.append({
            "frame_idx": frame_idx,
            "object_kind": "prototype",
            "object_id": int(proto.prototype_id),
            "state": "retired" if proto.retired else "active" if proto.is_active else "inactive",
            "runtime_owner_lineage_id": None if proto.runtime_owner_lineage_id is None else int(proto.runtime_owner_lineage_id),
            "continuity_lineage_id": None if proto.continuity_lineage_id is None else int(proto.continuity_lineage_id),
            "bound_prototype_id": int(proto.prototype_id),
            "bound_track_id": None if proto.last_track_id is None else int(proto.last_track_id),
            "source_track_id": None if proto.last_track_id is None else int(proto.last_track_id),
            "source_prototype_id": int(proto.prototype_id),
            "source_anchor_id": "",
            "source_continuation_id": "",
        })
    track = tracker._tracks.get(TARGET_TRACK_ID)
    if track is not None:
        rows.append({
            "frame_idx": frame_idx,
            "object_kind": f"track_{track.state}",
            "object_id": int(track.track_id),
            "state": str(track.state),
            "runtime_owner_lineage_id": None if track.lineage_id is None else int(track.lineage_id),
            "continuity_lineage_id": None if getattr(track, 'continuity_lineage_id', None) is None else int(track.continuity_lineage_id),
            "bound_prototype_id": None if track.prototype_id is None else int(track.prototype_id),
            "bound_track_id": int(track.track_id),
            "source_track_id": int(track.track_id),
            "source_prototype_id": None if track.prototype_id is None else int(track.prototype_id),
            "source_anchor_id": "",
            "source_continuation_id": "",
        })
    seen_cont: set[str] = set()
    seen_anchor: set[str] = set()
    for lineage in memory._lineages.values():
        for cont in lineage.continuation_bank:
            uid = str(cont.continuation_uid)
            if uid in seen_cont:
                continue
            if not (
                int(getattr(cont, 'prototype_id', -1)) == TARGET_PROTOTYPE_ID
                or int(getattr(cont, 'source_prototype_id', -1)) == TARGET_PROTOTYPE_ID
                or int(getattr(cont, 'track_id', -1)) == TARGET_TRACK_ID
                or int(getattr(cont, 'old_identity_ref_track_id', -1) or -1) == TARGET_TRACK_ID
                or int(getattr(cont, 'old_identity_ref_prototype_id', -1) or -1) == TARGET_PROTOTYPE_ID
            ):
                continue
            seen_cont.add(uid)
            rows.append({
                "frame_idx": frame_idx,
                "object_kind": "continuation",
                "object_id": int(cont.continuation_id),
                "state": "alive",
                "runtime_owner_lineage_id": None if cont.runtime_owner_lineage_id is None else int(cont.runtime_owner_lineage_id),
                "continuity_lineage_id": None if cont.continuity_lineage_id is None else int(cont.continuity_lineage_id),
                "bound_prototype_id": int(cont.prototype_id),
                "bound_track_id": int(cont.track_id),
                "source_track_id": int(cont.track_id),
                "source_prototype_id": int(cont.source_prototype_id),
                "source_anchor_id": "",
                "source_continuation_id": uid,
            })
        for anchor in lineage.recovery_identity_anchors:
            uid = str(anchor.anchor_uid)
            if uid in seen_anchor:
                continue
            if not (
                int(getattr(anchor, 'old_prototype_id', -1)) == TARGET_PROTOTYPE_ID
                or int(getattr(anchor, 'old_track_id', -1)) == TARGET_TRACK_ID
                or int(getattr(anchor, 'old_identity_ref_track_id', -1) or -1) == TARGET_TRACK_ID
                or int(getattr(anchor, 'old_identity_ref_prototype_id', -1) or -1) == TARGET_PROTOTYPE_ID
            ):
                continue
            seen_anchor.add(uid)
            rows.append({
                "frame_idx": frame_idx,
                "object_kind": "recovery_anchor",
                "object_id": uid,
                "state": str(anchor.anchor_state),
                "runtime_owner_lineage_id": None if anchor.runtime_owner_lineage_id is None else int(anchor.runtime_owner_lineage_id),
                "continuity_lineage_id": None if anchor.continuity_lineage_id is None else int(anchor.continuity_lineage_id),
                "bound_prototype_id": int(anchor.old_prototype_id),
                "bound_track_id": int(anchor.old_track_id),
                "source_track_id": int(anchor.old_track_id),
                "source_prototype_id": int(anchor.old_prototype_id),
                "source_anchor_id": uid,
                "source_continuation_id": "",
            })
    return rows


def _lifecycle_rows(rows: list[dict[str, Any]], *, filter_fn) -> list[dict[str, Any]]:
    rows = [row for row in rows if filter_fn(row)]
    rows.sort(key=lambda row: (str(row["object_kind"]), str(row["object_id"]), int(row["frame_idx"])))
    out: list[dict[str, Any]] = []
    prev_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["object_kind"]), str(row["object_id"]))
        prev = prev_by_key.get(key)
        runtime_now = row.get("runtime_owner_lineage_id")
        continuity_now = row.get("continuity_lineage_id")
        if prev is None:
            out.append({
                "frame_idx": int(row["frame_idx"]),
                "object_kind": row["object_kind"],
                "object_id": row["object_id"],
                "before_runtime_owner_lineage_id": None,
                "after_runtime_owner_lineage_id": runtime_now,
                "before_continuity_lineage_id": None,
                "after_continuity_lineage_id": continuity_now,
                "trigger_function": "snapshot_diff",
                "trigger_reason": "first_seen",
                "caller_stage": "e2rm_replay_diff",
                "source_track_id": row.get("source_track_id"),
                "source_prototype_id": row.get("source_prototype_id"),
                "source_anchor_id": row.get("source_anchor_id"),
                "source_continuation_id": row.get("source_continuation_id"),
                "is_owner_changed": 1,
                "is_runtime_owner_changed": 1,
                "is_continuity_owner_changed": 1,
                "is_illegal_continuity_rewrite_candidate": 0,
            })
        elif prev.get("runtime_owner_lineage_id") != runtime_now or prev.get("continuity_lineage_id") != continuity_now or prev.get("state") != row.get("state"):
            out.append({
                "frame_idx": int(row["frame_idx"]),
                "object_kind": row["object_kind"],
                "object_id": row["object_id"],
                "before_runtime_owner_lineage_id": prev.get("runtime_owner_lineage_id"),
                "after_runtime_owner_lineage_id": runtime_now,
                "before_continuity_lineage_id": prev.get("continuity_lineage_id"),
                "after_continuity_lineage_id": continuity_now,
                "trigger_function": "snapshot_diff",
                "trigger_reason": "owner_or_state_changed",
                "caller_stage": "e2rm_replay_diff",
                "source_track_id": row.get("source_track_id"),
                "source_prototype_id": row.get("source_prototype_id"),
                "source_anchor_id": row.get("source_anchor_id"),
                "source_continuation_id": row.get("source_continuation_id"),
                "is_owner_changed": int(prev.get("runtime_owner_lineage_id") != runtime_now or prev.get("continuity_lineage_id") != continuity_now),
                "is_runtime_owner_changed": int(prev.get("runtime_owner_lineage_id") != runtime_now),
                "is_continuity_owner_changed": int(prev.get("continuity_lineage_id") != continuity_now),
                "is_illegal_continuity_rewrite_candidate": int(prev.get("continuity_lineage_id") not in (None, "") and continuity_now not in (None, "") and prev.get("continuity_lineage_id") != continuity_now),
            })
        prev_by_key[key] = row
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    e1_rows = _load_csv(args.event_audit)
    focus_e1 = [row for row in e1_rows if row.get("ledger_event_id") in FOCUS_EVENT_IDS]
    focus_map = {str(row["ledger_event_id"]): row for row in focus_e1}
    focus_summary_rows = _load_csv(args.e2r_focus_summary)
    focus_summary_map = {str(row["event_id"]): row for row in focus_summary_rows if str(row.get("event_id")) in FOCUS_EVENT_IDS}
    source_trace_rows = _load_csv(args.e2r_source_trace)

    payload = load_config_payload(args.config)
    scenario_map = build_phase3_scenario_map(args.config)
    sequence = SyntheticStreamGenerator(scenario_map[TRACK_C_NAME], seed=args.seed).generate_sequence(0)
    encoder = MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    field = MinimalObjectnessField(**payload["field"])
    tracking_config = dict(payload["tracking"])
    tracking_config.update(_tracking_override())
    tracker = MinimalTemporalIdentityTracker(**tracking_config)
    memory = MinimalPrototypeMemory(**payload["memory"])

    lifecycle_capture: list[dict[str, Any]] = []
    prev_memory_output = None
    for frame_offset in range(1, len(sequence.frames)):
        prev_frame = sequence.frames[frame_offset - 1]
        current_frame = sequence.frames[frame_offset]
        frame_idx = int(current_frame.frame_index)
        encoding = encoder.encode(prev_frame.frame, current_frame.frame)
        objectness_output = field.compute(encoding)
        tracking_output = tracker.update(
            proposals=objectness_output.proposals,
            encoding=encoding,
            heatmap=objectness_output.heatmap,
            current_frame=current_frame.frame,
            frame_index=current_frame.frame_index,
            memory_context=prev_memory_output,
        )
        memory_output = memory.update(
            tracking_output.assignments,
            frame_index=current_frame.frame_index,
            track_states=(tracking_output.active_tracks + tracking_output.dormant_tracks + tracking_output.ghost_tracks + tracking_output.retired_tracks),
        )
        tracker.apply_concept_gated_resurrection(
            tracking_output,
            memory_output,
            frame_index=current_frame.frame_index,
            frame_shape=objectness_output.heatmap.shape,
        )
        tracker.bind_prototypes(memory_output.assignments)
        lifecycle_capture.extend(_capture_relevant_rows(frame_idx, tracker, memory))
        prev_memory_output = memory_output

    proto_lifecycle = _lifecycle_rows(lifecycle_capture, filter_fn=lambda row: row.get("source_prototype_id") == TARGET_PROTOTYPE_ID or (row.get("object_kind") == "prototype" and int(row.get("object_id")) == TARGET_PROTOTYPE_ID))
    track_lifecycle = _lifecycle_rows(lifecycle_capture, filter_fn=lambda row: row.get("source_track_id") == TARGET_TRACK_ID or (str(row.get("object_kind", "")).startswith("track_") and int(row.get("object_id")) == TARGET_TRACK_ID))

    final_lineages = memory._lineages
    alias_to: dict[int, int] = {}
    merged_into: dict[int, int] = {}
    for lineage in final_lineages.values():
        for alias in lineage.alias_lineage_ids:
            alias_to[int(alias)] = int(lineage.lineage_id)
        for merged in lineage.merged_lineage_ids:
            merged_into[int(merged)] = int(lineage.lineage_id)
    lineage_rows: list[dict[str, Any]] = []
    lineage_ids = sorted(set(final_lineages.keys()) | {1, 3})
    current_tracks = list(tracker._tracks.values())
    for lineage_id in lineage_ids:
        lineage = final_lineages.get(int(lineage_id))
        prototype_ids = [] if lineage is None else [int(v) for v in lineage.active_prototype_ids + lineage.archived_prototype_ids]
        track_ids = [int(t.track_id) for t in current_tracks if t.lineage_id is not None and int(t.lineage_id) == int(lineage_id)]
        anchor_ids = [] if lineage is None else [str(a.anchor_uid) for a in lineage.recovery_identity_anchors]
        cont_ids = [] if lineage is None else [str(c.continuation_uid) for c in lineage.continuation_bank]
        lineage_rows.append({
            "lineage_id": int(lineage_id),
            "canonical_lineage_id": int(alias_to.get(int(lineage_id), merged_into.get(int(lineage_id), int(lineage_id)))),
            "root_lineage_id": int(alias_to.get(int(lineage_id), merged_into.get(int(lineage_id), int(lineage_id)))),
            "parent_lineage_id": "" if lineage is None else "|".join(str(v) for v in lineage.alias_lineage_ids),
            "alias_to_lineage_id": alias_to.get(int(lineage_id), ""),
            "merged_into_lineage_id": merged_into.get(int(lineage_id), ""),
            "created_frame": "" if lineage is None else int(lineage.birth_frame),
            "created_by_function": "runtime_allocate_lineage" if lineage is not None else "absent_in_e2r_runtime",
            "created_from_track_id": "",
            "created_from_prototype_id": "",
            "prototype_ids": "|".join(str(v) for v in sorted(set(prototype_ids))),
            "track_ids": "|".join(str(v) for v in sorted(set(track_ids))),
            "anchor_ids": "|".join(anchor_ids),
            "continuation_ids": "|".join(cont_ids),
            "contains_old_prototype_6": int(TARGET_PROTOTYPE_ID in prototype_ids),
            "contains_old_track_2": int(TARGET_TRACK_ID in track_ids),
            "is_e1_target_lineage": int(int(lineage_id) == 3),
            "is_e2_runtime_lineage_for_old_target": int(TARGET_PROTOTYPE_ID in prototype_ids or TARGET_TRACK_ID in track_ids),
        })

    focus_source_rows: list[dict[str, Any]] = []
    classifications: dict[str, str] = {}
    summary_rows: list[dict[str, Any]] = []

    proto_states = [row for row in lifecycle_capture if row.get("object_kind") == "prototype" and int(row.get("object_id")) == TARGET_PROTOTYPE_ID]
    track_states = [row for row in lifecycle_capture if str(row.get("object_kind", "")).startswith("track_") and int(row.get("object_id")) == TARGET_TRACK_ID]
    proto_ever_target3 = any(int(row.get("continuity_lineage_id")) == 3 or int(row.get("runtime_owner_lineage_id")) == 3 for row in proto_states if row.get("continuity_lineage_id") not in (None, "") or row.get("runtime_owner_lineage_id") not in (None, ""))
    track_ever_target3 = any(int(row.get("continuity_lineage_id")) == 3 or int(row.get("runtime_owner_lineage_id")) == 3 for row in track_states if row.get("continuity_lineage_id") not in (None, "") or row.get("runtime_owner_lineage_id") not in (None, ""))
    proto_first_runtime = None if not proto_states else proto_states[0].get("runtime_owner_lineage_id")
    proto_first_cont = None if not proto_states else proto_states[0].get("continuity_lineage_id")
    track_first_runtime = None if not track_states else track_states[0].get("runtime_owner_lineage_id")
    track_first_cont = None if not track_states else track_states[0].get("continuity_lineage_id")

    focus_source_base = [
        row for row in source_trace_rows
        if row.get("event_id") in FOCUS_EVENT_IDS and row.get("audit_mode") == "actual_dual_owner"
    ]
    grouped_focus: dict[str, list[dict[str, Any]]] = {eid: [] for eid in FOCUS_EVENT_IDS}
    for row in focus_source_base:
        grouped_focus[str(row["event_id"])].append(row)

    for event_id in sorted(FOCUS_EVENT_IDS):
        e1 = focus_map[event_id]
        focus_summary = focus_summary_map[event_id]
        target_lineage_id = _safe_int(e1.get("old_continuity_lineage_id") or e1.get("old_lineage_id"), -1)
        old_track_id = _safe_int(e1.get("old_track_id"), -1)
        old_prototype_id = _safe_int(e1.get("old_prototype_id"), -1)
        anchor_uid = _target_anchor_uid(e1)
        event_rows = grouped_focus.get(event_id, [])
        matching_rows = []
        for row in event_rows:
            cand_track, cand_proto = _parse_candidate_ids(str(row.get("candidate_object_id", "")))
            if cand_track == old_track_id or cand_proto == old_prototype_id:
                matching_rows.append(row)
        raw_lineage_visible = int(any(_safe_int(row.get("source_lineage_id"), -999) == target_lineage_id for row in matching_rows))
        target_anchor_visible = int(len(matching_rows) > 0)
        runtime_lineages = sorted({_safe_int(row.get("source_runtime_owner_lineage_id"), -1) for row in matching_rows if _safe_int(row.get("source_runtime_owner_lineage_id"), -1) >= 0})
        continuity_lineages = sorted({_safe_int(row.get("source_continuity_owner_lineage_id"), -1) for row in matching_rows if _safe_int(row.get("source_continuity_owner_lineage_id"), -1) >= 0})
        legal_alias = int(alias_to.get(target_lineage_id, -999) in runtime_lineages or merged_into.get(target_lineage_id, -999) in runtime_lineages)
        proto_changed_3_to_other = int(any(int(r.get("before_continuity_lineage_id")) == 3 and int(r.get("after_continuity_lineage_id")) != 3 for r in proto_lifecycle if r.get("before_continuity_lineage_id") not in (None, "") and r.get("after_continuity_lineage_id") not in (None, "")))
        track_changed_3_to_other = int(any(int(r.get("before_continuity_lineage_id")) == 3 and int(r.get("after_continuity_lineage_id")) != 3 for r in track_lifecycle if r.get("before_continuity_lineage_id") not in (None, "") and r.get("after_continuity_lineage_id") not in (None, "")))
        if raw_lineage_visible:
            classification = "raw_lineage_match"
        elif legal_alias:
            classification = "legal_alias_or_merge"
        elif proto_changed_3_to_other or track_changed_3_to_other:
            classification = "illegal_owner_overwrite"
        elif track_ever_target3 and not proto_ever_target3 and proto_first_cont not in (None, "") and int(proto_first_cont) != target_lineage_id:
            classification = "prototype_inheritance_error"
        elif target_anchor_visible and not proto_ever_target3 and not track_ever_target3:
            classification = "runtime_namespace_shift"
        elif target_anchor_visible and proto_ever_target3:
            classification = "continuation_anchor_write_error"
        elif not target_anchor_visible:
            classification = "target_anchor_missing"
        else:
            classification = "metric_comparison_error"
        canonical_lineage_visible = int(classification in {"raw_lineage_match", "legal_alias_or_merge", "runtime_namespace_shift"} and target_anchor_visible)
        classifications[event_id] = classification
        summary_rows.append({
            "event_id": event_id,
            "target_anchor_uid": anchor_uid,
            "raw_lineage_visible": raw_lineage_visible,
            "canonical_lineage_visible": canonical_lineage_visible,
            "target_anchor_visible": target_anchor_visible,
            "classification": classification,
            "e1_target_lineage_id": target_lineage_id,
            "e2_old_prototype_runtime_lineage_id": focus_summary.get("old_prototype_runtime_lineages", ""),
            "e2_old_prototype_continuity_lineage_id": focus_summary.get("old_prototype_continuity_lineages", ""),
        })
        for row in event_rows:
            cand_track, cand_proto = _parse_candidate_ids(str(row.get("candidate_object_id", "")))
            matches_anchor = int(cand_track == old_track_id or cand_proto == old_prototype_id)
            focus_source_rows.append({
                "event_id": event_id,
                "proposal_id": row.get("proposal_id"),
                "candidate_object_kind": row.get("candidate_object_kind"),
                "candidate_object_id": row.get("candidate_object_id"),
                "linked_old_prototype_id": old_prototype_id if cand_proto == old_prototype_id else "",
                "linked_old_track_id": old_track_id if cand_track == old_track_id else "",
                "source_kind": row.get("source_kind"),
                "source_lineage_id": row.get("source_lineage_id"),
                "source_runtime_owner_lineage_id": row.get("source_runtime_owner_lineage_id"),
                "source_continuity_owner_lineage_id": row.get("source_continuity_owner_lineage_id"),
                "source_target_anchor_uid": anchor_uid,
                "matches_e1_target_anchor": matches_anchor,
                "matches_e1_raw_lineage": int(_safe_int(row.get("source_lineage_id"), -999) == target_lineage_id),
                "matches_canonical_lineage": int(matches_anchor and classification in {"raw_lineage_match", "legal_alias_or_merge", "runtime_namespace_shift"}),
                "emitted": row.get("emitted_source"),
                "filtered": row.get("filtered"),
                "filter_reason": row.get("filter_reason"),
                "in_final_source_pool": row.get("in_final_source_pool"),
            })

    cross_run_rows: list[dict[str, Any]] = []
    for event_id in sorted(FOCUS_EVENT_IDS):
        e1 = focus_map[event_id]
        focus_summary = focus_summary_map[event_id]
        summary = next(row for row in summary_rows if row["event_id"] == event_id)
        cross_run_rows.append({
            "event_id": event_id,
            "sequence_id": e1.get("sequence_id"),
            "e1_target_lineage_id": e1.get("old_continuity_lineage_id") or e1.get("old_lineage_id"),
            "e1_old_prototype_id": e1.get("old_prototype_id"),
            "e1_old_track_id": e1.get("old_track_id"),
            "target_anchor_uid": summary["target_anchor_uid"],
            "e2_old_prototype_present": focus_summary.get("old_prototype_present_in_memory"),
            "e2_old_prototype_runtime_lineage_id": focus_summary.get("old_prototype_runtime_lineages"),
            "e2_old_prototype_continuity_lineage_id": focus_summary.get("old_prototype_continuity_lineages"),
            "e2_old_track_runtime_lineage_id": track_first_runtime,
            "e2_old_track_continuity_lineage_id": track_first_cont,
            "e2_lineage_id_holding_old_prototype": focus_summary.get("old_prototype_runtime_lineages"),
            "e2_lineage_id_holding_old_track": track_first_runtime,
            "raw_lineage_match": summary["raw_lineage_visible"],
            "object_anchor_match": summary["target_anchor_visible"],
            "canonical_anchor_match": summary["canonical_lineage_visible"],
            "classification": summary["classification"],
        })

    explained = sum(1 for row in summary_rows if row["classification"] != "")
    report = {
        "focus_events": len(summary_rows),
        "raw_lineage_visible_count": sum(int(row["raw_lineage_visible"]) for row in summary_rows),
        "canonical_lineage_visible_count": sum(int(row["canonical_lineage_visible"]) for row in summary_rows),
        "target_anchor_visible_count": sum(int(row["target_anchor_visible"]) for row in summary_rows),
        "owner_remap_explained_rate": float(explained / max(len(summary_rows), 1)),
        "classifications": {row["event_id"]: row["classification"] for row in summary_rows},
        "proto6_first_runtime": proto_first_runtime,
        "proto6_first_continuity": proto_first_cont,
        "proto6_ever_target3": proto_ever_target3,
        "track2_first_runtime": track_first_runtime,
        "track2_first_continuity": track_first_cont,
        "track2_ever_target3": track_ever_target3,
    }

    write_csv(output_dir / f"stage_E2R_prototype6_owner_lifecycle_{args.artifact_version}.csv", proto_lifecycle)
    write_csv(output_dir / f"stage_E2R_old_track2_owner_lifecycle_{args.artifact_version}.csv", track_lifecycle)
    write_csv(output_dir / f"stage_E2R_lineage_relation_graph_{args.artifact_version}.csv", lineage_rows)
    write_csv(output_dir / f"stage_E2R_cross_run_target_alignment_{args.artifact_version}.csv", cross_run_rows)
    write_csv(output_dir / f"stage_E2R_focus_source_object_trace_{args.artifact_version}.csv", focus_source_rows)
    (output_dir / f"stage_E2RM_summary_{args.artifact_version}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Stage E2R-M Summary", "", "## Global", "",
        f"- `raw_lineage_visible_count = {report['raw_lineage_visible_count']}/{report['focus_events']}`",
        f"- `canonical_lineage_visible_count = {report['canonical_lineage_visible_count']}/{report['focus_events']}`",
        f"- `target_anchor_visible_count = {report['target_anchor_visible_count']}/{report['focus_events']}`",
        f"- `owner_remap_explained_rate = {report['owner_remap_explained_rate']:.4f}`",
        f"- `proto6_first_runtime = {report['proto6_first_runtime']}`",
        f"- `proto6_first_continuity = {report['proto6_first_continuity']}`",
        f"- `proto6_ever_target3 = {report['proto6_ever_target3']}`",
        f"- `track2_first_runtime = {report['track2_first_runtime']}`",
        f"- `track2_first_continuity = {report['track2_first_continuity']}`",
        f"- `track2_ever_target3 = {report['track2_ever_target3']}`",
        "", "## Focus Event Classification", "",
    ]
    for row in summary_rows:
        lines.extend([
            f"### {row['event_id']}", "",
            f"- `classification = {row['classification']}`",
            f"- `raw_lineage_visible = {row['raw_lineage_visible']}`",
            f"- `canonical_lineage_visible = {row['canonical_lineage_visible']}`",
            f"- `target_anchor_visible = {row['target_anchor_visible']}`",
            f"- `target_anchor_uid = {row['target_anchor_uid']}`",
            "",
        ])
    (output_dir / f"stage_E2RM_summary_{args.artifact_version}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
