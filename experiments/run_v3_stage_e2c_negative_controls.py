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

from experiments.phase3r_utils import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 Stage E2C negative controls.")
    parser.add_argument("--event-audit", default="results/v3_e1/stage_E1_event_audit_v1.csv")
    parser.add_argument("--e2c-events", default="results/v3_e2c/canonical_visibility_events_dual_owner_v1.csv")
    parser.add_argument("--source-trace", default="results/v3_e2r/stage_E2R_source_builder_trace_v3.csv")
    parser.add_argument("--output-dir", default="results/v3_e2c")
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


def _load_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _target_anchor_uid(row: dict[str, Any]) -> str:
    return "::".join([
        str(row.get("scenario_name", "")),
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


def _anchor_visible(rows: list[dict[str, Any]], *, track_id: int | None, proto_id: int | None) -> bool:
    for row in rows:
        cand_track, cand_proto = _parse_candidate_ids(str(row.get("candidate_object_id", "")))
        if track_id is not None and cand_track == track_id:
            return True
        if proto_id is not None and cand_proto == proto_id:
            return True
    return False


def _prototype_visible(rows: list[dict[str, Any]], *, proto_id: int | None) -> bool:
    if proto_id is None:
        return False
    for row in rows:
        _, cand_proto = _parse_candidate_ids(str(row.get("candidate_object_id", "")))
        if cand_proto == proto_id:
            return True
    return False


def _pick_wrong_proto(event_id: str) -> int:
    # 选择同序列非目标 old prototype，且在对应 focus frame 的 source rows 中不出现。
    mapping = {
        "M-RE-TC-012": 0,
        "M-RE-TC-013": 7,
        "M-RE-TC-014": 7,
    }
    return mapping[event_id]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    event_rows = [r for r in _load_csv(args.event_audit) if r.get("scenario_name") == "track_c_long_horizon"]
    e2c_rows = [r for r in _load_csv(args.e2c_events) if r.get("scenario_name") == "track_c_long_horizon"]
    source_rows = [r for r in _load_csv(args.source_trace) if r.get("audit_mode") == "actual_dual_owner"]

    e1_by_event = {str(r["ledger_event_id"]): r for r in event_rows}
    e2c_by_event = {str(r["ledger_event_id"]): r for r in e2c_rows}

    source_by_event: dict[str, list[dict[str, Any]]] = {}
    for row in source_rows:
        source_by_event.setdefault(str(row.get("event_id", "")), []).append(row)

    proposal_events = [r for r in event_rows if int(_safe_int(r.get("proposal_detected"), 0) or 0) == 1]
    proposal_ids = [str(r["ledger_event_id"]) for r in proposal_events]
    # rotate shuffle，保证 deterministic 且无 event 与自身对齐
    shuffled_lookup = {proposal_ids[i]: proposal_ids[(i + 1) % len(proposal_ids)] for i in range(len(proposal_ids))}

    focus_ids = ["M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"]
    normal_ids = []
    for row in e2c_rows:
        eid = str(row["ledger_event_id"])
        if eid in focus_ids:
            continue
        if row.get("proposal_detected") == "1" and row.get("raw_lineage_visible") == "1" and row.get("canonical_lineage_visible") == "1" and row.get("target_anchor_visible") == "1":
            normal_ids.append(eid)
    normal_ids = normal_ids[:3]

    out_rows: list[dict[str, Any]] = []

    # real baseline over all proposal-detected track_c events
    real_visible = 0
    shuffled_visible = 0
    for eid in proposal_ids:
        event = e1_by_event[eid]
        rows = source_by_event.get(eid, [])
        real_track = _safe_int(event.get("old_track_id"), None)
        real_proto = _safe_int(event.get("old_prototype_id"), None)
        vis_real = _anchor_visible(rows, track_id=real_track, proto_id=real_proto)
        real_visible += int(vis_real)

        shuf_event = e1_by_event[shuffled_lookup[eid]]
        shuf_track = _safe_int(shuf_event.get("old_track_id"), None)
        shuf_proto = _safe_int(shuf_event.get("old_prototype_id"), None)
        vis_shuf = _anchor_visible(rows, track_id=shuf_track, proto_id=shuf_proto)
        shuffled_visible += int(vis_shuf)

        out_rows.append({
            "control_name": "shuffled_anchor",
            "event_id": eid,
            "proposal_detected": 1,
            "target_anchor_uid": _target_anchor_uid(event),
            "test_anchor_uid": _target_anchor_uid(shuf_event),
            "real_old_track_id": real_track,
            "real_old_prototype_id": real_proto,
            "test_old_track_id": shuf_track,
            "test_old_prototype_id": shuf_proto,
            "anchor_visible": int(vis_shuf),
            "reference_anchor_visible": int(vis_real),
            "reference_raw_lineage_visible": int(e2c_by_event[eid]["raw_lineage_visible"]),
            "reference_canonical_lineage_visible": int(e2c_by_event[eid]["canonical_lineage_visible"]),
            "reference_target_anchor_visible": int(e2c_by_event[eid]["target_anchor_visible"]),
        })

    # focus wrong-prototype control
    wrong_proto_visible = 0
    focus_real_visible = 0
    focus_shuffled_visible = 0
    for eid in focus_ids:
        event = e1_by_event[eid]
        rows = source_by_event.get(eid, [])
        real_track = _safe_int(event.get("old_track_id"), None)
        real_proto = _safe_int(event.get("old_prototype_id"), None)
        vis_real = _anchor_visible(rows, track_id=real_track, proto_id=real_proto)
        focus_real_visible += int(vis_real)

        shuf_event = e1_by_event[shuffled_lookup[eid]]
        vis_shuf = _anchor_visible(rows, track_id=_safe_int(shuf_event.get("old_track_id"), None), proto_id=_safe_int(shuf_event.get("old_prototype_id"), None))
        focus_shuffled_visible += int(vis_shuf)

        wrong_proto = _pick_wrong_proto(eid)
        vis_wrong = _prototype_visible(rows, proto_id=wrong_proto)
        wrong_proto_visible += int(vis_wrong)

        out_rows.append({
            "control_name": "wrong_old_prototype",
            "event_id": eid,
            "proposal_detected": 1,
            "target_anchor_uid": _target_anchor_uid(event),
            "test_anchor_uid": f"wrong_proto_{wrong_proto}",
            "real_old_track_id": real_track,
            "real_old_prototype_id": real_proto,
            "test_old_track_id": "",
            "test_old_prototype_id": wrong_proto,
            "anchor_visible": int(vis_wrong),
            "reference_anchor_visible": int(vis_real),
            "reference_raw_lineage_visible": int(e2c_by_event[eid]["raw_lineage_visible"]),
            "reference_canonical_lineage_visible": int(e2c_by_event[eid]["canonical_lineage_visible"]),
            "reference_target_anchor_visible": int(e2c_by_event[eid]["target_anchor_visible"]),
        })

    # normal non-remap reference
    normal_triple_match = 0
    for eid in normal_ids:
        row = e2c_by_event[eid]
        ok = int(row["raw_lineage_visible"]) == 1 and int(row["canonical_lineage_visible"]) == 1 and int(row["target_anchor_visible"]) == 1
        normal_triple_match += int(ok)
        out_rows.append({
            "control_name": "normal_non_remap_reference",
            "event_id": eid,
            "proposal_detected": int(row["proposal_detected"]),
            "target_anchor_uid": row["target_anchor_uid"],
            "test_anchor_uid": row["target_anchor_uid"],
            "real_old_track_id": row["old_track_id"],
            "real_old_prototype_id": row["old_prototype_id"],
            "test_old_track_id": row["old_track_id"],
            "test_old_prototype_id": row["old_prototype_id"],
            "anchor_visible": int(row["target_anchor_visible"]),
            "reference_anchor_visible": int(row["target_anchor_visible"]),
            "reference_raw_lineage_visible": int(row["raw_lineage_visible"]),
            "reference_canonical_lineage_visible": int(row["canonical_lineage_visible"]),
            "reference_target_anchor_visible": int(row["target_anchor_visible"]),
        })

    summary = {
        "scope": "track_c_long_horizon_dual_owner_only",
        "proposal_detected_events": len(proposal_ids),
        "real_anchor_visible_count": real_visible,
        "real_anchor_svr": float(real_visible / max(len(proposal_ids), 1)),
        "shuffled_anchor_visible_count": shuffled_visible,
        "shuffled_anchor_svr": float(shuffled_visible / max(len(proposal_ids), 1)),
        "focus_real_anchor_visible_count": focus_real_visible,
        "focus_shuffled_anchor_visible_count": focus_shuffled_visible,
        "focus_wrong_old_prototype_visible_count": wrong_proto_visible,
        "normal_reference_event_count": len(normal_ids),
        "normal_triple_match_count": normal_triple_match,
        "pass": {
            "shuffled_anchor_lower_than_real": shuffled_visible < real_visible,
            "focus_wrong_old_prototype_zero": wrong_proto_visible == 0,
            "normal_reference_triple_match": normal_triple_match == len(normal_ids),
        },
    }

    write_csv(output_dir / f"stage_E2C_negative_control_events_{args.artifact_version}.csv", out_rows)
    (output_dir / f"stage_E2C_negative_control_summary_{args.artifact_version}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
