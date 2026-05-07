from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3r_utils import write_csv
from experiments.run_v3_stage_e2_dual_owner_visibility import (
    SCENARIO_NAMES,
    _event_key,
    _gt_box,
    _load_event_rows,
    _pick_target_proposal_id,
    _replay_sequence,
    _safe_int,
    _tracking_override,
)
from experiments.v3_utils import TRACK_A_NAME, TRACK_C_NAME
from datasets import SyntheticStreamGenerator
from experiments.phase3r_utils import build_phase3_scenario_map, load_config_payload

CANONICAL_CLASSES = {"raw_lineage_match", "legal_alias_or_merge", "runtime_namespace_shift"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 Stage E2C raw/canonical/anchor visibility comparison.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--event-audit", default="results/v3_e1/stage_E1_event_audit_v1.csv")
    parser.add_argument("--cross-run-alignment", default="results/v3_e2rm/stage_E2R_cross_run_target_alignment_v1.csv")
    parser.add_argument("--output-dir", default="results/v3_e2c")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-version", default="v1")
    parser.add_argument("--mode", choices=("all", "runtime_only", "dual_owner"), default="all")
    return parser.parse_args()


def _target_anchor_uid(row: dict[str, Any]) -> str:
    return "::".join([
        str(row.get("scenario_name", "")),
        f"inst_{row.get('instance_id', '')}",
        f"old_track_{row.get('old_track_id', '')}",
        f"old_proto_{row.get('old_prototype_id', '')}",
        f"e1_lineage_{row.get('old_continuity_lineage_id') or row.get('old_lineage_id')}",
    ])


def _load_alignment_map(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[str(row.get("event_id", ""))] = dict(row)
    return rows


def _matches_target_anchor(row: dict[str, Any], event: dict[str, Any]) -> bool:
    old_track_id = _safe_int(event.get("old_track_id"), -1)
    old_prototype_id = _safe_int(event.get("old_prototype_id"), -1)
    candidate_track_id = _safe_int(row.get("candidate_track_id"), -1)
    candidate_prototype_id = _safe_int(row.get("candidate_prototype_id"), -1)
    if old_track_id is not None and old_track_id >= 0 and candidate_track_id == old_track_id:
        return True
    if old_prototype_id is not None and old_prototype_id >= 0 and candidate_prototype_id == old_prototype_id:
        return True
    return False


def _run_mode(*, config_path: str | Path, seed: int, dual_owner: bool, event_rows: dict[str, list[dict[str, Any]]], alignment_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
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
                    alignment_map=alignment_map,
                )
            )
    return rows


def _build_visibility_row(*, scenario_name: str, event: dict[str, Any], frame_lookup: dict[int, Any], dual_owner: bool, alignment_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    frame_id = _safe_int(event.get("reappear_frame"), 0)
    instance_id = _safe_int(event.get("instance_id"), -1)
    proposal_detected = int(_safe_int(event.get("proposal_detected"), 0) or 0)
    ledger_event_id = str(event.get("ledger_event_id", ""))
    target_lineage_id = event.get("old_continuity_lineage_id") or event.get("old_lineage_id")
    target_lineage_id = None if target_lineage_id in (None, "") else _safe_int(target_lineage_id)
    frame_record = frame_lookup.get(frame_id)
    gt_box = _gt_box(frame_record, instance_id) if frame_record is not None else None
    target_anchor_uid = _target_anchor_uid(event)
    alignment_row = alignment_map.get(ledger_event_id, {})
    alignment_classification = str(alignment_row.get("classification", ""))

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
            target_source_rows = [row for row in frame_source_rows if _safe_int(row.get("proposal_id"), -1) == int(target_proposal_id)]
            target_claim_rows = [row for row in frame_claim_rows if _safe_int(row.get("proposal_id"), -1) == int(target_proposal_id)]

    raw_lineage_visible = False
    raw_runtime_owner_visible = False
    raw_continuity_owner_visible = False
    target_anchor_visible = False
    claim_visible = False
    claim_rank = None

    if target_proposal_id is not None:
        for row in target_source_rows:
            if _matches_target_anchor(row, event):
                target_anchor_visible = True
            if target_lineage_id is not None and _safe_int(row.get("candidate_lineage_id"), -1) == int(target_lineage_id):
                raw_lineage_visible = True
                owner_mode = str(row.get("source_owner_mode", ""))
                raw_runtime_owner_visible = raw_runtime_owner_visible or owner_mode == "runtime_owner"
                raw_continuity_owner_visible = raw_continuity_owner_visible or owner_mode == "continuity_owner"
        ranked_claim_rows = [row for row in target_claim_rows if row.get("candidate_lineage_id") not in (None, "")]
        ranked_claim_rows.sort(key=lambda row: float(row.get("claim_score_total", 0.0)), reverse=True)
        for idx, row in enumerate(ranked_claim_rows, start=1):
            if target_lineage_id is not None and _safe_int(row.get("candidate_lineage_id"), -1) == int(target_lineage_id):
                claim_visible = True
                claim_rank = idx
                break

    canonical_lineage_visible = False
    canonical_visibility_reason = ""
    if raw_lineage_visible:
        canonical_lineage_visible = True
        canonical_visibility_reason = "raw_lineage_match"
    elif target_anchor_visible and alignment_classification in CANONICAL_CLASSES:
        canonical_lineage_visible = True
        canonical_visibility_reason = alignment_classification or "target_anchor_match"
    elif target_anchor_visible:
        canonical_visibility_reason = "target_anchor_visible_but_canonical_unresolved"
    else:
        canonical_visibility_reason = "target_anchor_not_visible"

    if target_lineage_id is None:
        drop_reason = "missing_target_lineage"
    elif frame_record is None:
        drop_reason = "missing_frame_record"
    elif proposal_detected == 0:
        drop_reason = "proposal_missing_upstream"
    elif target_proposal_id is None:
        drop_reason = "missing_target_proposal_trace"
    elif raw_lineage_visible:
        drop_reason = "raw_lineage_visible"
    elif canonical_lineage_visible:
        drop_reason = f"canonical_visible_via_{canonical_visibility_reason}"
    elif target_anchor_visible:
        drop_reason = "target_anchor_visible_but_raw_and_canonical_miss"
    else:
        drop_reason = "target_not_visible_even_by_anchor"

    return {
        "mode": "dual_owner" if dual_owner else "runtime_only",
        "scenario_name": scenario_name,
        "ledger_event_id": ledger_event_id,
        "runtime_event_index": _safe_int(event.get("runtime_event_index"), -1),
        "instance_id": instance_id,
        "disappear_frame": _safe_int(event.get("disappear_frame"), 0),
        "reappear_frame": frame_id,
        "gap_length": _safe_int(event.get("gap_length"), 0),
        "gap_bucket": str(event.get("gap_bucket", "")),
        "event_type": str(event.get("event_type", "")),
        "target_lineage_id": target_lineage_id,
        "target_anchor_uid": target_anchor_uid,
        "alignment_classification": alignment_classification,
        "proposal_detected": proposal_detected,
        "candidate_pool_nonempty_e1": _safe_int(event.get("candidate_pool_nonempty"), 0),
        "continuation_bank_exists_e1": _safe_int(event.get("continuation_bank_exists"), 0),
        "failure_layer_e1": str(event.get("failure_layer", "")),
        "failure_reason_e1": str(event.get("failure_reason", "")),
        "old_track_id": _safe_int(event.get("old_track_id"), -1),
        "old_prototype_id": _safe_int(event.get("old_prototype_id"), -1),
        "target_proposal_id": target_proposal_id,
        "best_iou_to_gt": best_iou,
        "frame_source_row_count": len(frame_source_rows),
        "frame_claim_row_count": len(frame_claim_rows),
        "source_row_count": len(target_source_rows),
        "claim_row_count": len(target_claim_rows),
        "raw_lineage_visible": int(raw_lineage_visible),
        "canonical_lineage_visible": int(canonical_lineage_visible),
        "target_anchor_visible": int(target_anchor_visible),
        "raw_runtime_owner_visible": int(raw_runtime_owner_visible),
        "raw_continuity_owner_visible": int(raw_continuity_owner_visible),
        "claim_visible": int(claim_visible),
        "claim_rank": claim_rank,
        "canonical_visibility_reason": canonical_visibility_reason,
        "drop_reason": drop_reason,
    }


def _build_compare_rows(runtime_rows: list[dict[str, Any]], dual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runtime_lookup = {_event_key(row): row for row in runtime_rows}
    dual_lookup = {_event_key(row): row for row in dual_rows}
    keys = sorted(set(runtime_lookup.keys()) | set(dual_lookup.keys()))
    rows: list[dict[str, Any]] = []
    for key in keys:
        before = runtime_lookup.get(key, {})
        after = dual_lookup.get(key, {})
        rows.append({
            "scenario_name": key[0],
            "ledger_event_id": key[1],
            "instance_id": key[2],
            "reappear_frame": key[3],
            "target_lineage_id": after.get("target_lineage_id", before.get("target_lineage_id")),
            "target_anchor_uid": after.get("target_anchor_uid", before.get("target_anchor_uid")),
            "proposal_detected": after.get("proposal_detected", before.get("proposal_detected")),
            "alignment_classification_before": before.get("alignment_classification", ""),
            "alignment_classification_after": after.get("alignment_classification", ""),
            "drop_reason_before": before.get("drop_reason"),
            "drop_reason_after": after.get("drop_reason"),
            "raw_lineage_visible_before": int(before.get("raw_lineage_visible", 0) or 0),
            "raw_lineage_visible_after": int(after.get("raw_lineage_visible", 0) or 0),
            "canonical_lineage_visible_before": int(before.get("canonical_lineage_visible", 0) or 0),
            "canonical_lineage_visible_after": int(after.get("canonical_lineage_visible", 0) or 0),
            "target_anchor_visible_before": int(before.get("target_anchor_visible", 0) or 0),
            "target_anchor_visible_after": int(after.get("target_anchor_visible", 0) or 0),
            "raw_continuity_owner_visible_before": int(before.get("raw_continuity_owner_visible", 0) or 0),
            "raw_continuity_owner_visible_after": int(after.get("raw_continuity_owner_visible", 0) or 0),
            "raw_runtime_owner_visible_before": int(before.get("raw_runtime_owner_visible", 0) or 0),
            "raw_runtime_owner_visible_after": int(after.get("raw_runtime_owner_visible", 0) or 0),
            "raw_delta": int(after.get("raw_lineage_visible", 0) or 0) - int(before.get("raw_lineage_visible", 0) or 0),
            "canonical_delta": int(after.get("canonical_lineage_visible", 0) or 0) - int(before.get("canonical_lineage_visible", 0) or 0),
            "anchor_delta": int(after.get("target_anchor_visible", 0) or 0) - int(before.get("target_anchor_visible", 0) or 0),
        })
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "events": 0,
            "raw_svr": 0.0,
            "canonical_svr": 0.0,
            "anchor_svr": 0.0,
            "claim_visibility": 0.0,
            "raw_continuity_owner_visible": 0.0,
            "raw_runtime_owner_visible": 0.0,
        }
    return {
        "events": len(rows),
        "raw_svr": sum(int(row.get("raw_lineage_visible", 0) or 0) for row in rows) / len(rows),
        "canonical_svr": sum(int(row.get("canonical_lineage_visible", 0) or 0) for row in rows) / len(rows),
        "anchor_svr": sum(int(row.get("target_anchor_visible", 0) or 0) for row in rows) / len(rows),
        "claim_visibility": sum(int(row.get("claim_visible", 0) or 0) for row in rows) / len(rows),
        "raw_continuity_owner_visible": sum(int(row.get("raw_continuity_owner_visible", 0) or 0) for row in rows) / len(rows),
        "raw_runtime_owner_visible": sum(int(row.get("raw_runtime_owner_visible", 0) or 0) for row in rows) / len(rows),
    }


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
            "proposal_detected_only": _summarize_rows([row for row in scenario_rows if int(row.get("proposal_detected", 0) or 0) == 1]),
        }
    return summary


def _summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"events": 0}
    proposal_rows = [row for row in rows if int(row.get("proposal_detected", 0) or 0) == 1]
    return {
        "events": len(rows),
        "proposal_detected_events": len(proposal_rows),
        "raw_visible_before": sum(int(row.get("raw_lineage_visible_before", 0) or 0) for row in rows),
        "raw_visible_after": sum(int(row.get("raw_lineage_visible_after", 0) or 0) for row in rows),
        "canonical_visible_before": sum(int(row.get("canonical_lineage_visible_before", 0) or 0) for row in rows),
        "canonical_visible_after": sum(int(row.get("canonical_lineage_visible_after", 0) or 0) for row in rows),
        "anchor_visible_before": sum(int(row.get("target_anchor_visible_before", 0) or 0) for row in rows),
        "anchor_visible_after": sum(int(row.get("target_anchor_visible_after", 0) or 0) for row in rows),
        "improved_raw_visibility_events": sum(int(int(row.get("raw_delta", 0) or 0) > 0) for row in rows),
        "improved_canonical_visibility_events": sum(int(int(row.get("canonical_delta", 0) or 0) > 0) for row in rows),
        "improved_anchor_visibility_events": sum(int(int(row.get("anchor_delta", 0) or 0) > 0) for row in rows),
        "new_canonical_visible_events": sum(int(int(row.get("canonical_lineage_visible_before", 0) or 0) == 0 and int(row.get("canonical_lineage_visible_after", 0) or 0) == 1) for row in rows),
        "new_anchor_visible_events": sum(int(int(row.get("target_anchor_visible_before", 0) or 0) == 0 and int(row.get("target_anchor_visible_after", 0) or 0) == 1) for row in rows),
        "proposal_detected_raw_visible_before": sum(int(row.get("raw_lineage_visible_before", 0) or 0) for row in proposal_rows),
        "proposal_detected_raw_visible_after": sum(int(row.get("raw_lineage_visible_after", 0) or 0) for row in proposal_rows),
        "proposal_detected_canonical_visible_before": sum(int(row.get("canonical_lineage_visible_before", 0) or 0) for row in proposal_rows),
        "proposal_detected_canonical_visible_after": sum(int(row.get("canonical_lineage_visible_after", 0) or 0) for row in proposal_rows),
        "proposal_detected_anchor_visible_before": sum(int(row.get("target_anchor_visible_before", 0) or 0) for row in proposal_rows),
        "proposal_detected_anchor_visible_after": sum(int(row.get("target_anchor_visible_after", 0) or 0) for row in proposal_rows),
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage E2C Report",
        "",
        "## 目标",
        "",
        "E2C 不再只看 raw runtime lineage id，可见性拆成三口径：raw lineage / canonical lineage / target anchor。",
        "这一步用于区分：真正没看见旧 source，还是 runtime namespace shift 下旧记忆锚点仍然可见。",
        "",
        "## Paired Summary",
        "",
        f"- `events = {int(summary['paired']['events'])}`",
        f"- `proposal_detected_events = {int(summary['paired']['proposal_detected_events'])}`",
        f"- `raw_visible_before = {int(summary['paired']['raw_visible_before'])}`",
        f"- `raw_visible_after = {int(summary['paired']['raw_visible_after'])}`",
        f"- `canonical_visible_before = {int(summary['paired']['canonical_visible_before'])}`",
        f"- `canonical_visible_after = {int(summary['paired']['canonical_visible_after'])}`",
        f"- `anchor_visible_before = {int(summary['paired']['anchor_visible_before'])}`",
        f"- `anchor_visible_after = {int(summary['paired']['anchor_visible_after'])}`",
        f"- `improved_raw_visibility_events = {int(summary['paired']['improved_raw_visibility_events'])}`",
        f"- `improved_canonical_visibility_events = {int(summary['paired']['improved_canonical_visibility_events'])}`",
        f"- `improved_anchor_visibility_events = {int(summary['paired']['improved_anchor_visibility_events'])}`",
        f"- `new_canonical_visible_events = {int(summary['paired']['new_canonical_visible_events'])}`",
        f"- `new_anchor_visible_events = {int(summary['paired']['new_anchor_visible_events'])}`",
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


def _load_mode_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    event_rows = _load_event_rows(args.event_audit)
    alignment_map = _load_alignment_map(args.cross_run_alignment)

    requested_modes = [("runtime_only", False), ("dual_owner", True)] if args.mode == "all" else [(args.mode, args.mode == "dual_owner")]

    for mode_name, dual_owner in requested_modes:
        rows = _run_mode(
            config_path=args.config,
            seed=args.seed,
            dual_owner=dual_owner,
            event_rows=event_rows,
            alignment_map=alignment_map,
        )
        write_csv(output_dir / f"canonical_visibility_events_{mode_name}_{args.artifact_version}.csv", rows)

    runtime_path = output_dir / f"canonical_visibility_events_runtime_only_{args.artifact_version}.csv"
    dual_path = output_dir / f"canonical_visibility_events_dual_owner_{args.artifact_version}.csv"
    runtime_rows = _load_mode_csv(runtime_path)
    dual_rows = _load_mode_csv(dual_path)

    print(f"saved_runtime_only={runtime_path if runtime_path.exists() else ''}")
    print(f"saved_dual_owner={dual_path if dual_path.exists() else ''}")

    if not runtime_rows or not dual_rows:
        print(json.dumps({
            "status": "partial",
            "have_runtime_only": bool(runtime_rows),
            "have_dual_owner": bool(dual_rows),
        }, ensure_ascii=False))
        return

    compare_rows = _build_compare_rows(runtime_rows, dual_rows)
    write_csv(output_dir / f"canonical_visibility_compare_{args.artifact_version}.csv", compare_rows)

    summary = {
        "runtime_only": _summarize_mode(runtime_rows),
        "dual_owner": _summarize_mode(dual_rows),
        "paired": _summarize_pairs(compare_rows),
    }
    summary_path = output_dir / f"stage_E2C_summary_{args.artifact_version}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = output_dir / f"stage_E2C_report_{args.artifact_version}.md"
    report_path.write_text(_render_report(summary), encoding="utf-8")

    print(f"saved_compare={output_dir / f'canonical_visibility_compare_{args.artifact_version}.csv'}")
    print(f"saved_summary={summary_path}")
    print(f"saved_report={report_path}")
    print(json.dumps(summary["paired"], ensure_ascii=False))


if __name__ == "__main__":
    main()
