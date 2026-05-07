from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator
from experiments.phase3r_utils import build_phase3_scenario_map, load_config_payload, write_csv
from experiments.v3_utils import TRACK_C_NAME
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.memory import MinimalPrototypeMemory
from nops_owr.objectness import MinimalObjectnessField
from nops_owr.tracking import MinimalTemporalIdentityTracker

FOCUS_EVENT_IDS = {"M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 E2R continuity source lifecycle repair audit.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--event-audit", default="results/v3_e1/stage_E1_event_audit_v1.csv")
    parser.add_argument("--output-dir", default="results/v3_e2r")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-version", default="v1")
    return parser.parse_args()


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _iou(box_a, box_b) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def _box_text(box) -> str:
    if box is None:
        return ""
    return "(" + ", ".join(str(int(v)) for v in box) + ")"


def _target_lineage_id(event: dict[str, Any]) -> int | None:
    value = event.get("old_continuity_lineage_id") or event.get("old_lineage_id")
    return None if value in (None, "") else _safe_int(value)


def _pick_target_proposal(proposals, gt_box):
    if gt_box is None:
        return None
    best = None
    best_iou = -1.0
    for proposal_id, proposal in enumerate(proposals):
        box = tuple(int(v) for v in proposal.box)
        iou = _iou(box, gt_box)
        if iou > best_iou:
            best_iou = iou
            best = {
                "proposal_id": int(proposal_id),
                "proposal_box": box,
                "proposal_score": float(proposal.score),
                "proposal_iou_to_gt": float(iou),
            }
    return best


def _load_events(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("scenario_name", "")) == TRACK_C_NAME:
                rows.append(dict(row))
    rows.sort(key=lambda row: (_safe_int(row.get("reappear_frame"), -1), _safe_int(row.get("instance_id"), -1)))
    return rows


def _tracking_override() -> dict[str, Any]:
    return {
        "enable_phase3d_routing_repair": True,
        "enable_phase3d_target_selection_trace": False,
        "enable_phase3d_target_selection_repair": False,
        "enable_phase3d_claim_preservation_repair": False,
        "enable_phase3d_identity_preference_tiebreak": False,
        "enable_phase3d_preserve_input_trace": False,
        "enable_phase3d_continuity_lineage_repair": False,
        "enable_phase3d_three_source_preserve_input": False,
        "enable_phasea_dual_owner_source_enumeration": True,
    }

def _snapshot_state(frame_idx: int, tracker, memory, memory_output) -> dict[tuple[str, int], dict[str, Any]]:
    snap: dict[tuple[str, int], dict[str, Any]] = {}
    lp = {} if memory_output is None else getattr(memory_output, "lineage_prototype_lookup", {})
    pc = {} if memory_output is None else getattr(memory_output, "prototype_continuity_lookup", {})
    cr = {} if memory_output is None else getattr(memory_output, "continuation_lineage_lookup", {})
    cc = {} if memory_output is None else getattr(memory_output, "continuation_continuity_lookup", {})
    ar = {} if memory_output is None else getattr(memory_output, "recovery_anchor_lookup", {})
    ac = {} if memory_output is None else getattr(memory_output, "recovery_anchor_continuity_lookup", {})
    cont_runtime = {f"{lid}:{getattr(c,'continuation_uid','')}" for lid, items in cr.items() for c in items}
    cont_cont = {f"{lid}:{getattr(c,'continuation_uid','')}" for lid, items in cc.items() for c in items}
    anc_runtime = {f"{lid}:{a.get('anchor_uid','')}" for lid, items in ar.items() for a in items}
    anc_cont = {f"{lid}:{a.get('anchor_uid','')}" for lid, items in ac.items() for a in items}

    for track in tracker._tracks.values():
        snap[("track", int(track.track_id))] = {
            "frame_idx": frame_idx,
            "object_kind": f"track_{track.state}",
            "object_id": int(track.track_id),
            "state": str(track.state),
            "runtime_owner_lineage_id": None if track.lineage_id is None else int(track.lineage_id),
            "continuity_lineage_id": None if getattr(track, 'continuity_lineage_id', None) is None else int(track.continuity_lineage_id),
            "lineage_id": None if track.lineage_id is None else int(track.lineage_id),
            "prototype_id": None if track.prototype_id is None else int(track.prototype_id),
            "registered_in_runtime_index": int(track.lineage_id is not None),
            "registered_in_continuity_index": int(getattr(track, 'continuity_lineage_id', None) is not None),
            "registered_in_anchor_bank": 0,
            "registered_in_continuation_bank": 0,
            "registered_in_archive": 0,
            "source_family": "track",
            "last_seen_frame": int(track.last_seen_frame),
            "last_updated_frame": int(track.last_seen_frame),
        }

    for lineage_id, lineage in memory._lineages.items():
        snap[("lineage", int(lineage_id))] = {
            "frame_idx": frame_idx,
            "object_kind": "lineage_registry",
            "object_id": int(lineage_id),
            "state": str(lineage.status),
            "runtime_owner_lineage_id": int(lineage_id),
            "continuity_lineage_id": int(lineage_id),
            "lineage_id": int(lineage_id),
            "prototype_id": None,
            "registered_in_runtime_index": 1,
            "registered_in_continuity_index": 1,
            "registered_in_anchor_bank": int(len(lineage.recovery_identity_anchors) > 0),
            "registered_in_continuation_bank": int(len(lineage.continuation_bank) > 0),
            "registered_in_archive": int(len(lineage.archived_prototype_ids) > 0),
            "source_family": "lineage_registry",
            "last_seen_frame": int(lineage.last_update_frame),
            "last_updated_frame": int(lineage.last_update_frame),
        }

    for proto in memory._prototypes.values():
        lineage = memory._lineages.get(int(proto.lineage_id))
        archived = int(bool(proto.retired or (lineage is not None and int(proto.prototype_id) in set(int(v) for v in lineage.archived_prototype_ids))))
        runtime_owner = None if proto.runtime_owner_lineage_id is None else int(proto.runtime_owner_lineage_id)
        continuity_owner = None if proto.continuity_lineage_id is None else int(proto.continuity_lineage_id)
        snap[("prototype", int(proto.prototype_id))] = {
            "frame_idx": frame_idx,
            "object_kind": "prototype",
            "object_id": int(proto.prototype_id),
            "state": "retired" if proto.retired else "active" if proto.is_active else "inactive",
            "runtime_owner_lineage_id": runtime_owner,
            "continuity_lineage_id": continuity_owner,
            "lineage_id": int(proto.lineage_id),
            "prototype_id": int(proto.prototype_id),
            "registered_in_runtime_index": int(runtime_owner is not None and int(proto.prototype_id) in set(int(v) for v in lp.get(runtime_owner, []))),
            "registered_in_continuity_index": int(continuity_owner is not None and pc.get(int(proto.prototype_id)) == continuity_owner),
            "registered_in_anchor_bank": 0,
            "registered_in_continuation_bank": int(len(proto.continuation_bank) > 0),
            "registered_in_archive": archived,
            "source_family": "prototype",
            "last_seen_frame": int(proto.last_updated_frame),
            "last_updated_frame": int(proto.last_updated_frame),
        }

    if memory_output is not None:
        for row in getattr(memory_output, 'continuation_lifecycle_rows', []):
            uid = str(row.get('continuation_uid', ''))
            rid = _safe_int(row.get('runtime_owner_lineage_id'))
            cid = _safe_int(row.get('continuity_lineage_id'))
            snap[("continuation", _safe_int(row.get('continuation_id'), -1))] = {
                "frame_idx": frame_idx, "object_kind": "continuation", "object_id": _safe_int(row.get('continuation_id'), -1),
                "state": 'alive' if int(_safe_int(row.get('is_alive'), 0) or 0) == 1 else str(row.get('drop_reason', 'dead')),
                "runtime_owner_lineage_id": rid, "continuity_lineage_id": cid, "lineage_id": _safe_int(row.get('current_owner_lineage_id')),
                "prototype_id": _safe_int(row.get('source_prototype_id')), "registered_in_runtime_index": int(rid is not None and f"{rid}:{uid}" in cont_runtime),
                "registered_in_continuity_index": int(cid is not None and f"{cid}:{uid}" in cont_cont), "registered_in_anchor_bank": 0,
                "registered_in_continuation_bank": 1, "registered_in_archive": 0, "source_family": "continuation",
                "last_seen_frame": _safe_int(row.get('age_since_last_seen'), 0), "last_updated_frame": _safe_int(row.get('write_frame'), 0),
            }
        for row in getattr(memory_output, 'recovery_anchor_lifecycle_rows', []):
            uid = str(row.get('anchor_uid', ''))
            rid = _safe_int(row.get('runtime_owner_lineage_id'))
            cid = _safe_int(row.get('continuity_lineage_id'))
            snap[("anchor", abs(hash(uid)) % 1000000007)] = {
                "frame_idx": frame_idx, "object_kind": "recovery_anchor", "object_id": abs(hash(uid)) % 1000000007,
                "state": 'alive' if int(_safe_int(row.get('is_alive'), 0) or 0) == 1 else str(row.get('drop_reason', 'dead')),
                "runtime_owner_lineage_id": rid, "continuity_lineage_id": cid, "lineage_id": _safe_int(row.get('lineage_id')),
                "prototype_id": _safe_int(row.get('old_prototype_id')), "registered_in_runtime_index": int(rid is not None and f"{rid}:{uid}" in anc_runtime),
                "registered_in_continuity_index": int(cid is not None and f"{cid}:{uid}" in anc_cont), "registered_in_anchor_bank": 1,
                "registered_in_continuation_bank": 0, "registered_in_archive": 0, "source_family": "anchor",
                "last_seen_frame": _safe_int(row.get('age_since_last_seen'), 0), "last_updated_frame": _safe_int(row.get('last_alive_frame'), 0),
            }
    return snap


def _inventory_rows(event: dict[str, Any], snap: dict[tuple[str, int], dict[str, Any]], target_lineage_id: int | None) -> list[dict[str, Any]]:
    rows = []
    for item in snap.values():
        row = dict(item)
        row.update({
            "event_id": str(event.get("ledger_event_id", "")),
            "sequence_id": _safe_int(event.get("sequence_id"), 0),
            "event_frame_idx": _safe_int(event.get("reappear_frame"), -1),
            "target_lineage_id": target_lineage_id,
            "proposal_detected": _safe_int(event.get("proposal_detected"), 0),
            "is_target_runtime_owner": int(target_lineage_id is not None and row.get("runtime_owner_lineage_id") == target_lineage_id),
            "is_target_continuity_owner": int(target_lineage_id is not None and row.get("continuity_lineage_id") == target_lineage_id),
            "is_target_lineage_object": int(target_lineage_id is not None and (row.get("runtime_owner_lineage_id") == target_lineage_id or row.get("continuity_lineage_id") == target_lineage_id or row.get("lineage_id") == target_lineage_id)),
        })
        rows.append(row)
    return rows

def _no_filter_rows(event: dict[str, Any], snap: dict[tuple[str, int], dict[str, Any]], target_lineage_id: int | None, target_proposal) -> list[dict[str, Any]]:
    if target_proposal is None:
        return []
    rows = []
    for item in snap.values():
        runtime_owner = item.get("runtime_owner_lineage_id")
        continuity_owner = item.get("continuity_lineage_id")
        if runtime_owner is not None and item.get("source_family") in {"track", "prototype", "continuation", "anchor", "lineage_registry"}:
            rows.append({
                "audit_mode": "dual_owner_no_filter", "event_id": str(event.get("ledger_event_id", "")), "frame_idx": _safe_int(event.get("reappear_frame"), -1),
                "proposal_id": int(target_proposal["proposal_id"]), "proposal_box": _box_text(target_proposal["proposal_box"]),
                "candidate_object_kind": item.get("object_kind"), "candidate_object_id": item.get("object_id"),
                "candidate_runtime_owner_lineage_id": runtime_owner, "candidate_continuity_lineage_id": continuity_owner,
                "emitted_source": 1, "source_kind": "runtime_owner", "source_lineage_id": runtime_owner,
                "source_runtime_owner_lineage_id": runtime_owner, "source_continuity_owner_lineage_id": continuity_owner,
                "filtered": 0, "filter_stage": "", "filter_reason": "", "in_final_source_pool": 1,
                "is_target_lineage_source": int(target_lineage_id is not None and runtime_owner == target_lineage_id),
            })
        if continuity_owner is not None and item.get("source_family") in {"track", "prototype", "continuation", "anchor", "lineage_registry"}:
            rows.append({
                "audit_mode": "dual_owner_no_filter", "event_id": str(event.get("ledger_event_id", "")), "frame_idx": _safe_int(event.get("reappear_frame"), -1),
                "proposal_id": int(target_proposal["proposal_id"]), "proposal_box": _box_text(target_proposal["proposal_box"]),
                "candidate_object_kind": item.get("object_kind"), "candidate_object_id": item.get("object_id"),
                "candidate_runtime_owner_lineage_id": runtime_owner, "candidate_continuity_lineage_id": continuity_owner,
                "emitted_source": 1, "source_kind": "continuity_owner", "source_lineage_id": continuity_owner,
                "source_runtime_owner_lineage_id": runtime_owner, "source_continuity_owner_lineage_id": continuity_owner,
                "filtered": 0, "filter_stage": "", "filter_reason": "", "in_final_source_pool": 1,
                "is_target_lineage_source": int(target_lineage_id is not None and continuity_owner == target_lineage_id),
            })
    return rows


def _actual_rows(event: dict[str, Any], tracking_output, target_lineage_id: int | None, target_proposal) -> list[dict[str, Any]]:
    if target_proposal is None:
        return []
    proposal_id = int(target_proposal["proposal_id"])
    rows = []
    for row in getattr(tracking_output, 'recovery_candidate_rows', []):
        if _safe_int(row.get('proposal_id'), -1) != proposal_id:
            continue
        source_kind = str(row.get('source_kind', row.get('source_type', '')))
        candidate_lineage_id = _safe_int(row.get('candidate_lineage_id'), -1)
        rows.append({
            "audit_mode": "actual_dual_owner", "event_id": str(event.get("ledger_event_id", "")), "frame_idx": _safe_int(event.get("reappear_frame"), -1),
            "proposal_id": proposal_id, "proposal_box": _box_text(target_proposal["proposal_box"]),
            "candidate_object_kind": source_kind,
            "candidate_object_id": f"track:{row.get('candidate_track_id')}|proto:{row.get('candidate_prototype_id')}",
            "candidate_runtime_owner_lineage_id": _safe_int(row.get('source_runtime_owner_id')),
            "candidate_continuity_lineage_id": _safe_int(row.get('source_continuity_owner_id')),
            "emitted_source": 1, "source_kind": str(row.get('source_owner_mode', 'runtime_owner')), "source_lineage_id": candidate_lineage_id,
            "source_runtime_owner_lineage_id": _safe_int(row.get('source_runtime_owner_id')),
            "source_continuity_owner_lineage_id": _safe_int(row.get('source_continuity_owner_id')),
            "filtered": int(row.get('filtered_flag', 0) or 0), "filter_stage": str(row.get('filter_stage', '')), "filter_reason": str(row.get('filter_reason', '')),
            "in_final_source_pool": 1, "is_target_lineage_source": int(target_lineage_id is not None and candidate_lineage_id == target_lineage_id),
            "restore_eligibility": int(bool(row.get('restore_eligibility', False))), "recovery_score_total": float(row.get('recovery_score_total', 0.0)),
        })
    return rows


def _owner_diff(previous: dict[tuple[str, int], dict[str, Any]], current: dict[tuple[str, int], dict[str, Any]], frame_idx: int, event_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(previous.keys()) | set(current.keys())):
        before = previous.get(key)
        after = current.get(key)
        if before is not None and after is not None and before.get("runtime_owner_lineage_id") == after.get("runtime_owner_lineage_id") and before.get("continuity_lineage_id") == after.get("continuity_lineage_id"):
            continue
        if before is None and after is None:
            continue
        runtime_before = None if before is None else before.get("runtime_owner_lineage_id")
        runtime_after = None if after is None else after.get("runtime_owner_lineage_id")
        continuity_before = None if before is None else before.get("continuity_lineage_id")
        continuity_after = None if after is None else after.get("continuity_lineage_id")
        obj = after or before
        rows.append({
            "frame_idx": frame_idx, "event_id_if_known": "|".join(v for v in event_ids if v), "object_kind": obj.get("object_kind"), "object_id": obj.get("object_id"),
            "field_name": "owner_fields", "before_value": f"runtime={runtime_before}|continuity={continuity_before}", "after_value": f"runtime={runtime_after}|continuity={continuity_after}",
            "trigger_function": "snapshot_diff", "trigger_reason": "first_seen" if before is None else "disappeared" if after is None else "owner_field_changed", "caller_stage": "memory_runtime_snapshot_diff",
            "runtime_owner_lineage_id_before": runtime_before, "runtime_owner_lineage_id_after": runtime_after,
            "continuity_lineage_id_before": continuity_before, "continuity_lineage_id_after": continuity_after,
            "is_continuity_overwrite": int(continuity_before not in (None, "") and continuity_after not in (None, "") and continuity_before != continuity_after),
            "is_target_lineage_lost": int(continuity_before == 3 and continuity_after != 3),
        })
    return rows


def _classify(
    proposal_detected: int,
    pre_runtime_target: int,
    pre_cont_target: int,
    registered_cont_target: int,
    target_in_no_filter_any_pool: int,
    target_in_any_pool: int,
    oracle_visible: int,
    old_prototype_present: int,
    old_prototype_target_owner_match: int,
) -> str:
    if int(proposal_detected) == 0:
        return "proposal_missing_upstream"
    if int(pre_runtime_target) == 0 and int(pre_cont_target) == 0:
        if int(old_prototype_present) == 1 and int(old_prototype_target_owner_match) == 0:
            return "target_owner_remapped_or_namespace_shift"
        return "target_absent_from_memory"
    if int(pre_cont_target) > 0 and int(registered_cont_target) == 0:
        return "target_present_but_not_indexed"
    if int(target_in_no_filter_any_pool) == 0:
        return "target_indexed_but_not_enumerated"
    if int(target_in_no_filter_any_pool) == 1 and int(target_in_any_pool) == 0:
        return "target_enumerated_but_filtered"
    if int(target_in_any_pool) == 0 and int(oracle_visible) == 0:
        return "target_in_pool_but_metric_mismatch"
    return "target_visible"


def _focus_summary(event: dict[str, Any], snap: dict[tuple[str, int], dict[str, Any]], target_lineage_id: int | None, target_proposal, actual_rows: list[dict[str, Any]]) -> dict[str, Any]:
    inventory = list(snap.values())
    old_prototype_id = _safe_int(event.get("old_prototype_id"), -1)
    pre_runtime_target = sum(int(row.get("runtime_owner_lineage_id") == target_lineage_id) for row in inventory if target_lineage_id is not None)
    pre_cont_target = sum(int(row.get("continuity_lineage_id") == target_lineage_id) for row in inventory if target_lineage_id is not None)
    pre_anchor_target = sum(int(row.get("object_kind") == "recovery_anchor" and row.get("continuity_lineage_id") == target_lineage_id) for row in inventory if target_lineage_id is not None)
    pre_continuation_target = sum(int(row.get("object_kind") == "continuation" and row.get("continuity_lineage_id") == target_lineage_id) for row in inventory if target_lineage_id is not None)
    pre_archive_target = sum(int(row.get("registered_in_archive", 0) == 1 and (row.get("continuity_lineage_id") == target_lineage_id or row.get("runtime_owner_lineage_id") == target_lineage_id)) for row in inventory if target_lineage_id is not None)
    registered_cont_target = sum(int(row.get("registered_in_continuity_index", 0) == 1 and row.get("continuity_lineage_id") == target_lineage_id) for row in inventory if target_lineage_id is not None)
    old_proto_rows = [
        row for row in inventory
        if (old_prototype_id is not None and old_prototype_id >= 0)
        and (
            _safe_int(row.get("prototype_id"), -1) == old_prototype_id
            or (str(row.get("object_kind", "")) == "prototype" and _safe_int(row.get("object_id"), -1) == old_prototype_id)
        )
    ]
    old_proto_runtime_lineages = sorted({int(row["runtime_owner_lineage_id"]) for row in old_proto_rows if row.get("runtime_owner_lineage_id") not in (None, "", "None")})
    old_proto_continuity_lineages = sorted({int(row["continuity_lineage_id"]) for row in old_proto_rows if row.get("continuity_lineage_id") not in (None, "", "None")})
    old_prototype_present = int(len(old_proto_rows) > 0)
    old_prototype_target_owner_match = int(
        target_lineage_id is not None and (
            int(target_lineage_id) in old_proto_runtime_lineages or int(target_lineage_id) in old_proto_continuity_lineages
        )
    )
    nf_rows = _no_filter_rows(event, snap, target_lineage_id, target_proposal)
    actual_runtime = sorted({int(row["source_lineage_id"]) for row in actual_rows if row.get("source_kind") == "runtime_owner" and row.get("source_lineage_id") not in (None, "")})
    actual_cont = sorted({int(row["source_lineage_id"]) for row in actual_rows if row.get("source_kind") == "continuity_owner" and row.get("source_lineage_id") not in (None, "")})
    all_actual = sorted({int(row["source_lineage_id"]) for row in actual_rows if row.get("source_lineage_id") not in (None, "")})
    nf_runtime = sorted({int(row["source_lineage_id"]) for row in nf_rows if row.get("source_kind") == "runtime_owner" and row.get("source_lineage_id") not in (None, "")})
    nf_cont = sorted({int(row["source_lineage_id"]) for row in nf_rows if row.get("source_kind") == "continuity_owner" and row.get("source_lineage_id") not in (None, "")})
    target_in_any = int(target_lineage_id is not None and int(target_lineage_id) in all_actual)
    target_in_no_filter_any = int(target_lineage_id is not None and (int(target_lineage_id) in nf_runtime or int(target_lineage_id) in nf_cont))
    oracle_visible = int(target_lineage_id is not None and target_proposal is not None)
    return {
        "event_id": str(event.get("ledger_event_id", "")), "runtime_event_index": _safe_int(event.get("runtime_event_index"), -1), "frame_idx": _safe_int(event.get("reappear_frame"), -1),
        "instance_id": _safe_int(event.get("instance_id"), -1), "target_lineage_id": target_lineage_id, "old_prototype_id": old_prototype_id, "proposal_detected": _safe_int(event.get("proposal_detected"), 0),
        "focus_event": int(str(event.get("ledger_event_id", "")) in FOCUS_EVENT_IDS),
        "pre_event_count_runtime_owner_target": pre_runtime_target, "pre_event_count_continuity_owner_target": pre_cont_target,
        "pre_event_count_anchor_target": pre_anchor_target, "pre_event_count_continuation_target": pre_continuation_target, "pre_event_count_archive_target": pre_archive_target,
        "registered_continuity_index_target_count": registered_cont_target,
        "old_prototype_present_in_memory": old_prototype_present,
        "old_prototype_runtime_lineages": "|".join(str(v) for v in old_proto_runtime_lineages),
        "old_prototype_continuity_lineages": "|".join(str(v) for v in old_proto_continuity_lineages),
        "old_prototype_target_owner_match": old_prototype_target_owner_match,
        "runtime_source_pool_lineages": "|".join(str(v) for v in actual_runtime), "continuity_source_pool_lineages": "|".join(str(v) for v in actual_cont), "all_source_pool_lineages": "|".join(str(v) for v in all_actual),
        "no_filter_runtime_pool_lineages": "|".join(str(v) for v in nf_runtime), "no_filter_continuity_pool_lineages": "|".join(str(v) for v in nf_cont),
        "target_in_runtime_pool": int(target_lineage_id is not None and int(target_lineage_id) in actual_runtime), "target_in_continuity_pool": int(target_lineage_id is not None and int(target_lineage_id) in actual_cont),
        "target_in_any_pool": target_in_any, "target_in_no_filter_any_pool": target_in_no_filter_any, "oracle_visible_if_injected": oracle_visible,
        "if_not_visible_reason": _classify(_safe_int(event.get("proposal_detected"), 0), pre_runtime_target, pre_cont_target, registered_cont_target, target_in_no_filter_any, target_in_any, oracle_visible, old_prototype_present, old_prototype_target_owner_match),
    }

def _render(summary: dict[str, Any], focus_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage E2R Summary", "", "## Global", "",
        f"- `actual_runtime_owner_rows_total = {int(summary['actual_runtime_owner_rows_total'])}`",
        f"- `actual_continuity_owner_rows_total = {int(summary['actual_continuity_owner_rows_total'])}`",
        f"- `continuity_owner_branch_materialized = {bool(summary['continuity_owner_branch_materialized'])}`",
        f"- `actual_unique_source_lineages = {summary['actual_unique_source_lineages']}`",
        f"- `actual_unique_continuity_source_lineages = {summary['actual_unique_continuity_source_lineages']}`",
        f"- `continuity_overwrite_rows = {int(summary['continuity_overwrite_rows'])}`",
        f"- `target_lineage_lost_rows = {int(summary['target_lineage_lost_rows'])}`", "", "## Focus Events", "",
    ]
    for row in focus_rows:
        if int(row.get("focus_event", 0) or 0) != 1:
            continue
        lines.extend([
            f"### {row['event_id']}", "",
            f"- `target_lineage_id = {row['target_lineage_id']}`",
            f"- `proposal_detected = {row['proposal_detected']}`",
            f"- `pre_event_count_runtime_owner_target = {row['pre_event_count_runtime_owner_target']}`",
            f"- `pre_event_count_continuity_owner_target = {row['pre_event_count_continuity_owner_target']}`",
            f"- `registered_continuity_index_target_count = {row['registered_continuity_index_target_count']}`",
            f"- `runtime_source_pool_lineages = {row['runtime_source_pool_lineages']}`",
            f"- `continuity_source_pool_lineages = {row['continuity_source_pool_lineages']}`",
            f"- `no_filter_continuity_pool_lineages = {row['no_filter_continuity_pool_lineages']}`",
            f"- `target_in_runtime_pool = {row['target_in_runtime_pool']}`",
            f"- `target_in_continuity_pool = {row['target_in_continuity_pool']}`",
            f"- `target_in_no_filter_any_pool = {row['target_in_no_filter_any_pool']}`",
            f"- `oracle_visible_if_injected = {row['oracle_visible_if_injected']}`",
            f"- `if_not_visible_reason = {row['if_not_visible_reason']}`", "",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = _load_events(args.event_audit)
    payload = load_config_payload(args.config)
    scenario_map = build_phase3_scenario_map(args.config)
    sequence = SyntheticStreamGenerator(scenario_map[TRACK_C_NAME], seed=args.seed).generate_sequence(0)

    encoder = MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    field = MinimalObjectnessField(**payload["field"])
    tracking_config = dict(payload["tracking"])
    tracking_config.update(_tracking_override())
    tracker = MinimalTemporalIdentityTracker(**tracking_config)
    memory = MinimalPrototypeMemory(**payload["memory"])

    events_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        events_by_frame[_safe_int(event.get("reappear_frame"), -1)].append(event)

    prev_memory_output = None
    previous_snapshot: dict[tuple[str, int], dict[str, Any]] = {}
    inventory_rows: list[dict[str, Any]] = []
    owner_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    focus_rows: list[dict[str, Any]] = []
    counter: Counter[str] = Counter()

    for frame_offset in range(1, len(sequence.frames)):
        prev_frame = sequence.frames[frame_offset - 1]
        current_frame = sequence.frames[frame_offset]
        frame_idx = int(current_frame.frame_index)
        current_events = list(events_by_frame.get(frame_idx, []))
        gt_boxes = {int(i): tuple(int(v) for v in b) for i, b in zip(current_frame.instance_ids, current_frame.boxes)}
        encoding = encoder.encode(prev_frame.frame, current_frame.frame)
        objectness_output = field.compute(encoding)

        if current_events:
            pre_snap = _snapshot_state(frame_idx - 1, tracker, memory, prev_memory_output)
            for event in current_events:
                target_lineage_id = _target_lineage_id(event)
                inventory_rows.extend(_inventory_rows(event, pre_snap, target_lineage_id))
                target_prop = _pick_target_proposal(objectness_output.proposals, gt_boxes.get(_safe_int(event.get("instance_id"), -1)))
                source_rows.extend(_no_filter_rows(event, pre_snap, target_lineage_id, target_prop))
        else:
            pre_snap = None

        tracking_output = tracker.update(proposals=objectness_output.proposals, encoding=encoding, heatmap=objectness_output.heatmap, current_frame=current_frame.frame, frame_index=current_frame.frame_index, memory_context=prev_memory_output)
        memory_output = memory.update(tracking_output.assignments, frame_index=current_frame.frame_index, track_states=(tracking_output.active_tracks + tracking_output.dormant_tracks + tracking_output.ghost_tracks + tracking_output.retired_tracks))
        tracker.apply_concept_gated_resurrection(tracking_output, memory_output, frame_index=current_frame.frame_index, frame_shape=objectness_output.heatmap.shape)
        tracker.bind_prototypes(memory_output.assignments)
        post_snap = _snapshot_state(frame_idx, tracker, memory, memory_output)
        owner_rows.extend(_owner_diff(previous_snapshot, post_snap, frame_idx, [str(e.get("ledger_event_id", "")) for e in current_events]))
        previous_snapshot = post_snap

        actual_cont = [row for row in tracking_output.recovery_candidate_rows if str(row.get("source_owner_mode", "")) == "continuity_owner"]
        counter["actual_continuity_owner_rows_total"] += len(actual_cont)
        counter["actual_runtime_owner_rows_total"] += len([row for row in tracking_output.recovery_candidate_rows if str(row.get("source_owner_mode", "")) == "runtime_owner"])

        if current_events:
            for event in current_events:
                target_lineage_id = _target_lineage_id(event)
                target_prop = _pick_target_proposal(objectness_output.proposals, gt_boxes.get(_safe_int(event.get("instance_id"), -1)))
                actual = _actual_rows(event, tracking_output, target_lineage_id, target_prop)
                source_rows.extend(actual)
                focus_rows.append(_focus_summary(event, pre_snap or {}, target_lineage_id, target_prop, actual))

        prev_memory_output = memory_output

    actual_rows = [row for row in source_rows if str(row.get("audit_mode")) == "actual_dual_owner"]
    cont_rows = [row for row in actual_rows if str(row.get("source_kind")) == "continuity_owner"]
    summary = {
        "track_c_only": True,
        "actual_runtime_owner_rows_total": int(counter.get("actual_runtime_owner_rows_total", 0)),
        "actual_continuity_owner_rows_total": int(counter.get("actual_continuity_owner_rows_total", 0)),
        "actual_unique_source_lineages": sorted({int(row["source_lineage_id"]) for row in actual_rows if row.get("source_lineage_id") not in (None, "")}),
        "actual_unique_continuity_source_lineages": sorted({int(row["source_lineage_id"]) for row in cont_rows if row.get("source_lineage_id") not in (None, "")}),
        "continuity_owner_branch_materialized": bool(len(cont_rows) > 0),
        "continuity_overwrite_rows": int(sum(int(row.get("is_continuity_overwrite", 0) or 0) for row in owner_rows)),
        "target_lineage_lost_rows": int(sum(int(row.get("is_target_lineage_lost", 0) or 0) for row in owner_rows)),
    }

    write_csv(output_dir / f"stage_E2R_pre_event_inventory_{args.artifact_version}.csv", inventory_rows)
    write_csv(output_dir / f"stage_E2R_owner_lifecycle_{args.artifact_version}.csv", owner_rows)
    write_csv(output_dir / f"stage_E2R_source_builder_trace_{args.artifact_version}.csv", source_rows)
    write_csv(output_dir / f"stage_E2R_focus_event_summary_{args.artifact_version}.csv", focus_rows)
    (output_dir / f"stage_E2R_summary_{args.artifact_version}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / f"stage_E2R_summary_{args.artifact_version}.md").write_text(_render(summary, focus_rows), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
