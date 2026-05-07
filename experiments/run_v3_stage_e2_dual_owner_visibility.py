from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator
from experiments.phase3r_utils import build_phase3_scenario_map, load_config_payload, write_csv
from experiments.v3_utils import TRACK_A_NAME, TRACK_C_NAME
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.memory import MinimalPrototypeMemory
from nops_owr.objectness import MinimalObjectnessField
from nops_owr.tracking import MinimalTemporalIdentityTracker


SCENARIO_NAMES = (TRACK_A_NAME, TRACK_C_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 Stage E2 dual-owner visibility comparison.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--event-audit", default="results/v3_e1/stage_E1_event_audit_v1.csv")
    parser.add_argument("--output-dir", default="results/v3_e2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-version", default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    event_rows = _load_event_rows(args.event_audit)

    mode_rows: dict[str, list[dict[str, Any]]] = {}
    for mode_name, dual_owner in (("runtime_only", False), ("dual_owner", True)):
        rows = _run_mode(
            config_path=args.config,
            seed=args.seed,
            dual_owner=dual_owner,
            event_rows=event_rows,
        )
        mode_rows[mode_name] = rows
        write_csv(output_dir / f"source_visibility_events_{mode_name}_{args.artifact_version}.csv", rows)

    compare_rows = _build_compare_rows(mode_rows["runtime_only"], mode_rows["dual_owner"])
    write_csv(output_dir / f"source_visibility_compare_{args.artifact_version}.csv", compare_rows)

    summary = {
        "runtime_only": _summarize_mode(mode_rows["runtime_only"]),
        "dual_owner": _summarize_mode(mode_rows["dual_owner"]),
        "paired": _summarize_pairs(compare_rows),
    }
    summary_path = output_dir / f"stage_E2_summary_{args.artifact_version}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = output_dir / f"stage_E2_report_{args.artifact_version}.md"
    report_path.write_text(_render_report(summary), encoding="utf-8")

    print(f"saved_runtime_only={output_dir / f'source_visibility_events_runtime_only_{args.artifact_version}.csv'}")
    print(f"saved_dual_owner={output_dir / f'source_visibility_events_dual_owner_{args.artifact_version}.csv'}")
    print(f"saved_compare={output_dir / f'source_visibility_compare_{args.artifact_version}.csv'}")
    print(f"saved_summary={summary_path}")
    print(f"saved_report={report_path}")
    print(json.dumps(summary["paired"], ensure_ascii=False))


def _load_event_rows(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scenario_name = str(row.get("scenario_name", ""))
            if scenario_name not in SCENARIO_NAMES:
                continue
            by_scenario[scenario_name].append(dict(row))
    for scenario_name in by_scenario:
        by_scenario[scenario_name].sort(
            key=lambda row: (
                _safe_int(row.get("runtime_event_index"), -1),
                _safe_int(row.get("reappear_frame"), -1),
                _safe_int(row.get("instance_id"), -1),
            )
        )
    return by_scenario


def _tracking_override(dual_owner: bool) -> dict[str, Any]:
    return {
        "enable_phase3d_routing_repair": True,
        "enable_phase3d_target_selection_trace": False,
        "enable_phase3d_target_selection_repair": False,
        "enable_phase3d_claim_preservation_repair": False,
        "enable_phase3d_identity_preference_tiebreak": False,
        "enable_phase3d_preserve_input_trace": False,
        "enable_phase3d_continuity_lineage_repair": False,
        "enable_phase3d_three_source_preserve_input": False,
        "enable_phasea_dual_owner_source_enumeration": bool(dual_owner),
    }


def _run_mode(*, config_path: str | Path, seed: int, dual_owner: bool, event_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    payload = load_config_payload(config_path)
    scenario_map = build_phase3_scenario_map(config_path)
    rows: list[dict[str, Any]] = []
    for scenario_name in SCENARIO_NAMES:
        sequence = SyntheticStreamGenerator(scenario_map[scenario_name], seed=seed).generate_sequence(0)
        result = _replay_sequence(payload, sequence, tracking_override=_tracking_override(dual_owner))
        frame_lookup = {int(fr.frame_index): fr for fr in result.frame_records}
        for event in event_rows.get(scenario_name, []):
            rows.append(
                _build_visibility_row(
                    scenario_name=scenario_name,
                    event=event,
                    frame_lookup=frame_lookup,
                    dual_owner=dual_owner,
                )
            )
    return rows


def _replay_sequence(payload: dict[str, Any], sequence, *, tracking_override: dict[str, Any]):
    encoder = MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    objectness = MinimalObjectnessField(**payload["field"])
    tracking_config = dict(payload["tracking"])
    tracking_config.update(tracking_override)
    tracker = MinimalTemporalIdentityTracker(**tracking_config)
    memory = MinimalPrototypeMemory(**payload["memory"])
    prev_memory_output = None
    frame_records: list[Any] = []
    for frame_offset in range(1, len(sequence.frames)):
        prev_frame = sequence.frames[frame_offset - 1]
        current_frame = sequence.frames[frame_offset]
        encoding = encoder.encode(prev_frame.frame, current_frame.frame)
        objectness_output = objectness.compute(encoding)
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
            track_states=(
                tracking_output.active_tracks
                + tracking_output.dormant_tracks
                + tracking_output.ghost_tracks
                + tracking_output.retired_tracks
            ),
        )
        tracker.apply_concept_gated_resurrection(
            tracking_output,
            memory_output,
            frame_index=current_frame.frame_index,
            frame_shape=objectness_output.heatmap.shape,
        )
        tracker.bind_prototypes(memory_output.assignments)
        prev_memory_output = memory_output
        frame_records.append(
            SimpleNamespace(
                frame_index=int(current_frame.frame_index),
                frame=None,
                gt_boxes=list(current_frame.boxes),
                masks=[],
                instance_ids=list(current_frame.instance_ids),
                concept_ids=list(current_frame.concept_ids),
                objectness_output=objectness_output,
                tracking_output=tracking_output,
                memory_output=memory_output,
                predicted_boxes=[assignment.box for assignment in memory_output.assignments],
                predicted_ids=[assignment.prototype_id for assignment in memory_output.assignments],
                matches=[],
                recall_hit=0.0,
                frame_purity=0.0,
                objectness_recall=0.0,
                false_hot_area=0.0,
            )
        )
    return SimpleNamespace(frame_records=frame_records)


def _build_visibility_row(
    *,
    scenario_name: str,
    event: dict[str, Any],
    frame_lookup: dict[int, Any],
    dual_owner: bool,
) -> dict[str, Any]:
    frame_id = _safe_int(event.get("reappear_frame"), 0)
    instance_id = _safe_int(event.get("instance_id"), -1)
    proposal_detected = int(_safe_int(event.get("proposal_detected"), 0) or 0)
    target_lineage_id = event.get("old_continuity_lineage_id") or event.get("old_lineage_id")
    target_lineage_id = None if target_lineage_id in (None, "") else _safe_int(target_lineage_id)
    frame_record = frame_lookup.get(frame_id)
    gt_box = _gt_box(frame_record, instance_id) if frame_record is not None else None

    source_rows = [] if frame_record is None else list(getattr(frame_record.tracking_output, "recovery_candidate_rows", []))
    claim_rows = [] if frame_record is None else list(getattr(frame_record.tracking_output, "lineage_claim_rows", []))
    frame_source_rows = [row for row in source_rows if _safe_int(row.get("frame_id"), -1) == frame_id]
    frame_claim_rows = [row for row in claim_rows if _safe_int(row.get("frame_id"), -1) == frame_id]

    best_iou = None
    target_proposal_id = None
    target_source_rows: list[dict[str, Any]] = []
    target_claim_rows: list[dict[str, Any]] = []

    if proposal_detected and gt_box is not None:
        target_proposal_id, best_iou = _pick_target_proposal_id(frame_source_rows, gt_box)
        if target_proposal_id is not None:
            target_source_rows = [
                row for row in frame_source_rows if _safe_int(row.get("proposal_id"), -1) == int(target_proposal_id)
            ]
            target_claim_rows = [
                row for row in frame_claim_rows if _safe_int(row.get("proposal_id"), -1) == int(target_proposal_id)
            ]

    source_visible = False
    runtime_owner_visible = False
    continuity_owner_visible = False
    claim_visible = False
    claim_rank = None
    if target_lineage_id is not None and target_proposal_id is not None:
        for row in target_source_rows:
            if _safe_int(row.get("candidate_lineage_id"), -1) == int(target_lineage_id):
                source_visible = True
                owner_mode = str(row.get("source_owner_mode", ""))
                runtime_owner_visible = runtime_owner_visible or owner_mode == "runtime_owner"
                continuity_owner_visible = continuity_owner_visible or owner_mode == "continuity_owner"
        ranked_claim_rows = [row for row in target_claim_rows if row.get("candidate_lineage_id") not in (None, "")]
        ranked_claim_rows.sort(key=lambda row: float(row.get("claim_score_total", 0.0)), reverse=True)
        for idx, row in enumerate(ranked_claim_rows, start=1):
            if _safe_int(row.get("candidate_lineage_id"), -1) == int(target_lineage_id):
                claim_visible = True
                claim_rank = idx
                break

    if target_lineage_id is None:
        drop_reason = "missing_target_lineage"
    elif frame_record is None:
        drop_reason = "missing_frame_record"
    elif proposal_detected == 0:
        drop_reason = "proposal_missing_upstream"
    elif target_proposal_id is None:
        drop_reason = "missing_target_proposal_trace"
    elif not source_visible:
        drop_reason = "target_lineage_not_in_source_pool"
    elif not claim_visible:
        drop_reason = "target_lineage_not_in_claim_rows"
    else:
        drop_reason = "visible"

    return {
        "mode": "dual_owner" if dual_owner else "runtime_only",
        "scenario_name": scenario_name,
        "ledger_event_id": event.get("ledger_event_id"),
        "runtime_event_index": _safe_int(event.get("runtime_event_index"), -1),
        "instance_id": instance_id,
        "disappear_frame": _safe_int(event.get("disappear_frame"), 0),
        "reappear_frame": frame_id,
        "gap_length": _safe_int(event.get("gap_length"), 0),
        "gap_bucket": str(event.get("gap_bucket", "")),
        "event_type": str(event.get("event_type", "")),
        "target_lineage_id": target_lineage_id,
        "proposal_detected": proposal_detected,
        "candidate_pool_nonempty_e1": _safe_int(event.get("candidate_pool_nonempty"), 0),
        "continuation_bank_exists_e1": _safe_int(event.get("continuation_bank_exists"), 0),
        "failure_layer_e1": str(event.get("failure_layer", "")),
        "failure_reason_e1": str(event.get("failure_reason", "")),
        "target_proposal_id": target_proposal_id,
        "best_iou_to_gt": best_iou,
        "frame_source_row_count": len(frame_source_rows),
        "frame_claim_row_count": len(frame_claim_rows),
        "source_row_count": len(target_source_rows),
        "claim_row_count": len(target_claim_rows),
        "source_visible": int(source_visible),
        "runtime_owner_visible": int(runtime_owner_visible),
        "continuity_owner_visible": int(continuity_owner_visible),
        "claim_visible": int(claim_visible),
        "claim_rank": claim_rank,
        "drop_reason": drop_reason,
    }


def _event_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("scenario_name", "")),
        str(row.get("ledger_event_id", "")),
        int(_safe_int(row.get("instance_id"), -1)),
        int(_safe_int(row.get("reappear_frame"), -1)),
    )


def _build_compare_rows(runtime_rows: list[dict[str, Any]], dual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runtime_lookup = {_event_key(row): row for row in runtime_rows}
    dual_lookup = {_event_key(row): row for row in dual_rows}
    keys = sorted(set(runtime_lookup.keys()) | set(dual_lookup.keys()))
    rows: list[dict[str, Any]] = []
    for key in keys:
        before = runtime_lookup.get(key, {})
        after = dual_lookup.get(key, {})
        rows.append(
            {
                "scenario_name": key[0],
                "ledger_event_id": key[1],
                "instance_id": key[2],
                "reappear_frame": key[3],
                "target_lineage_id": after.get("target_lineage_id", before.get("target_lineage_id")),
                "proposal_detected": after.get("proposal_detected", before.get("proposal_detected")),
                "drop_reason_before": before.get("drop_reason"),
                "drop_reason_after": after.get("drop_reason"),
                "source_visible_before": int(before.get("source_visible", 0) or 0),
                "source_visible_after": int(after.get("source_visible", 0) or 0),
                "claim_visible_before": int(before.get("claim_visible", 0) or 0),
                "claim_visible_after": int(after.get("claim_visible", 0) or 0),
                "runtime_owner_visible_before": int(before.get("runtime_owner_visible", 0) or 0),
                "runtime_owner_visible_after": int(after.get("runtime_owner_visible", 0) or 0),
                "continuity_owner_visible_before": int(before.get("continuity_owner_visible", 0) or 0),
                "continuity_owner_visible_after": int(after.get("continuity_owner_visible", 0) or 0),
                "visibility_delta": int(after.get("source_visible", 0) or 0) - int(before.get("source_visible", 0) or 0),
                "claim_delta": int(after.get("claim_visible", 0) or 0) - int(before.get("claim_visible", 0) or 0),
            }
        )
    return rows


def _summarize_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "overall": _summarize_rows(rows),
        "proposal_detected_only": _summarize_rows([row for row in rows if int(row.get("proposal_detected", 0) or 0) == 1]),
        "scenarios": {},
    }
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[str(row["scenario_name"])].append(row)
    for scenario_name, scenario_rows in by_scenario.items():
        summary["scenarios"][scenario_name] = {
            "overall": _summarize_rows(scenario_rows),
            "proposal_detected_only": _summarize_rows([
                row for row in scenario_rows if int(row.get("proposal_detected", 0) or 0) == 1
            ]),
        }
    return summary


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "events": 0,
            "svr": 0.0,
            "claim_visibility": 0.0,
            "continuity_owner_visible": 0.0,
            "runtime_owner_visible": 0.0,
        }
    return {
        "events": len(rows),
        "svr": sum(int(row.get("source_visible", 0) or 0) for row in rows) / len(rows),
        "claim_visibility": sum(int(row.get("claim_visible", 0) or 0) for row in rows) / len(rows),
        "continuity_owner_visible": sum(int(row.get("continuity_owner_visible", 0) or 0) for row in rows) / len(rows),
        "runtime_owner_visible": sum(int(row.get("runtime_owner_visible", 0) or 0) for row in rows) / len(rows),
    }


def _summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"events": 0}
    proposal_rows = [row for row in rows if int(row.get("proposal_detected", 0) or 0) == 1]
    return {
        "events": len(rows),
        "proposal_detected_events": len(proposal_rows),
        "source_visible_before": sum(int(row.get("source_visible_before", 0) or 0) for row in rows),
        "source_visible_after": sum(int(row.get("source_visible_after", 0) or 0) for row in rows),
        "claim_visible_before": sum(int(row.get("claim_visible_before", 0) or 0) for row in rows),
        "claim_visible_after": sum(int(row.get("claim_visible_after", 0) or 0) for row in rows),
        "improved_visibility_events": sum(int(int(row.get("visibility_delta", 0) or 0) > 0) for row in rows),
        "improved_claim_events": sum(int(int(row.get("claim_delta", 0) or 0) > 0) for row in rows),
        "new_continuity_owner_visible_events": sum(
            int(int(row.get("continuity_owner_visible_before", 0) or 0) == 0 and int(row.get("continuity_owner_visible_after", 0) or 0) == 1)
            for row in rows
        ),
        "proposal_detected_source_visible_before": sum(int(row.get("source_visible_before", 0) or 0) for row in proposal_rows),
        "proposal_detected_source_visible_after": sum(int(row.get("source_visible_after", 0) or 0) for row in proposal_rows),
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage E2 Report",
        "",
        "## 目标",
        "",
        "只比较 source enumeration：runtime-owner only vs runtime+continuity dual-owner。",
        "事件锚点使用 E1 产出的 event audit，不再用 E2 replay 自行重建 target metadata。",
        "",
        "## Paired Summary",
        "",
        f"- `events = {int(summary['paired']['events'])}`",
        f"- `proposal_detected_events = {int(summary['paired']['proposal_detected_events'])}`",
        f"- `source_visible_before = {int(summary['paired']['source_visible_before'])}`",
        f"- `source_visible_after = {int(summary['paired']['source_visible_after'])}`",
        f"- `claim_visible_before = {int(summary['paired']['claim_visible_before'])}`",
        f"- `claim_visible_after = {int(summary['paired']['claim_visible_after'])}`",
        f"- `improved_visibility_events = {int(summary['paired']['improved_visibility_events'])}`",
        f"- `improved_claim_events = {int(summary['paired']['improved_claim_events'])}`",
        f"- `new_continuity_owner_visible_events = {int(summary['paired']['new_continuity_owner_visible_events'])}`",
        f"- `proposal_detected_source_visible_before = {int(summary['paired']['proposal_detected_source_visible_before'])}`",
        f"- `proposal_detected_source_visible_after = {int(summary['paired']['proposal_detected_source_visible_after'])}`",
        "",
        "## Mode Summary",
        "",
    ]
    for mode_name in ("runtime_only", "dual_owner"):
        mode_summary = summary[mode_name]
        lines.append(f"### {mode_name}")
        lines.append("")
        lines.append(f"- `overall = {mode_summary['overall']}`")
        lines.append(f"- `proposal_detected_only = {mode_summary['proposal_detected_only']}`")
        for scenario_name, scenario_summary in mode_summary["scenarios"].items():
            lines.append(f"- `{scenario_name} overall = {scenario_summary['overall']}`")
            lines.append(f"- `{scenario_name} proposal_detected_only = {scenario_summary['proposal_detected_only']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _pick_target_proposal_id(source_rows: list[dict[str, Any]], gt_box) -> tuple[int | None, float | None]:
    if gt_box is None or not source_rows:
        return None, None
    best_iou = -1.0
    best_proposal_id = None
    seen: set[int] = set()
    for row in source_rows:
        proposal_id = _safe_int(row.get("proposal_id"), -1)
        if proposal_id in seen:
            continue
        seen.add(proposal_id)
        proposal_box = _parse_box(row.get("proposal_box"))
        if proposal_box is None:
            continue
        iou = _iou(proposal_box, gt_box)
        if iou > best_iou:
            best_iou = iou
            best_proposal_id = proposal_id
    if best_proposal_id is None:
        return None, None
    return best_proposal_id, best_iou


def _gt_box(frame_record, instance_id: int):
    if frame_record is None:
        return None
    try:
        index = list(frame_record.instance_ids).index(int(instance_id))
    except ValueError:
        return None
    return tuple(int(v) for v in frame_record.gt_boxes[index])


def _parse_box(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(int(v) for v in value)
    text = str(value).strip()
    if not text:
        return None
    text = text.strip("()[]")
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        return None
    try:
        return tuple(int(float(p)) for p in parts)
    except ValueError:
        return None


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
    if union <= 0.0:
        return 0.0
    return inter / union


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


if __name__ == "__main__":
    main()
