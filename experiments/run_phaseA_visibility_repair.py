"""Phase A: continuity source visibility repair runner."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator, load_synth_dataset_config  # noqa: E402
from experiments.phase3d_utils import (  # noqa: E402
    default_phase3d_stagea_memory_override,
    default_phase3d_stagea_tracking_override,
)
from experiments.phase3s_utils import TRACK_A_NAME, TRACK_C_NAME  # noqa: E402
from experiments.phase3r_utils import evaluate_phase3_scenarios  # noqa: E402
from experiments.phase3r_utils import _build_frame_instance_map  # noqa: E402
from experiments.run_phase3d_stage_a5_claim_preservation_trace import (  # noqa: E402
    TARGET_EVENT_ID,
    TARGET_FRAME,
    TARGET_GT_OBJECT_ID,
    _gt_box,
    _iou,
    _load_target_metadata,
    _parse_box,
)
from experiments.scenario_presets import build_phase3_track_scenarios  # noqa: E402
from nops_owr.encoder import MinimalSpikeEncoder  # noqa: E402
from nops_owr.memory import MinimalPrototypeMemory  # noqa: E402
from nops_owr.objectness import MinimalObjectnessField  # noqa: E402
from nops_owr.tracking import MinimalTemporalIdentityTracker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase A continuity visibility repair.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--output-dir", default="results/phaseA")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _stagea_tracking_override(*, force_target_reroute: bool = False) -> dict[str, Any]:
    override = default_phase3d_stagea_tracking_override()
    override.update(
        {
            "enable_phase3d_routing_repair": True,
            "enable_phase3d_target_selection_trace": True,
            "enable_phase3d_target_selection_repair": False,
            "enable_phase3d_claim_preservation_repair": True,
            "enable_phase3d_identity_preference_tiebreak": False,
            "enable_phase3d_preserve_input_trace": True,
            "enable_phase3d_continuity_lineage_repair": True,
            "enable_phase3d_three_source_preserve_input": False,
            "enable_phasea_dual_owner_source_enumeration": True,
            "routing_recovery_max_distance": 0.70,
            "routing_recovery_min_confidence": 0.30,
            "routing_active_claim_override_margin": 0.20,
            "routing_topk": 4,
            "claim_preserve_min_score": 0.25,
            "continuity_hint_min_score": 0.15,
        }
    )
    if force_target_reroute:
        override["debug_force_reroute_frame"] = TARGET_FRAME
    return override


def _stagea_memory_override() -> dict[str, Any]:
    return default_phase3d_stagea_memory_override()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_track_sequence(config_path: Path, scenario_name: str, *, seed: int):
    base_config = load_synth_dataset_config(config_path)
    scenario_map = {item["name"]: item["config"] for item in build_phase3_track_scenarios(base_config)}
    return SyntheticStreamGenerator(scenario_map[scenario_name], seed=seed).generate_sequence(0)


def _target_selection_row(selection_rows: list[dict[str, Any]], frame_id: int, gt_box) -> dict[str, Any] | None:
    rows = [row for row in selection_rows if int(row.get("frame_id", -1)) == int(frame_id)]
    if not rows:
        return None
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        proposal_box = _parse_box(row.get("proposal_box"))
        scored.append((_iou(proposal_box, gt_box), row))
    scored.sort(key=lambda item: -item[0])
    return scored[0][1]


def _derive_target_metadata_from_run(
    *,
    result,
    gt_object_id: int,
    target_frame: int,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    last_visible: dict[str, Any] | None = None
    for frame_record in result.frame_records:
        if int(frame_record.frame_index) >= int(target_frame):
            break
        frame_map = _build_frame_instance_map(frame_record, iou_threshold=iou_threshold)
        instance_row = frame_map["instances"].get(int(gt_object_id))
        if instance_row is not None and bool(instance_row.get("visible", False)):
            last_visible = dict(instance_row)
    if last_visible is None:
        return {
            "old_track_id": None,
            "old_prototype_id": None,
            "old_lineage_id": None,
            "old_continuity_lineage_id": None,
        }
    return {
        "old_track_id": last_visible.get("track_id"),
        "old_prototype_id": last_visible.get("prototype_id"),
        "old_lineage_id": last_visible.get("prototype_lineage_id"),
        "old_continuity_lineage_id": last_visible.get("prototype_continuity_lineage_id"),
    }


def _manual_track_c_trace(
    *,
    config_path: Path,
    seed: int,
) -> dict[str, Any]:
    sequence = _load_track_sequence(config_path, TRACK_C_NAME, seed=seed)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    tracking_config = dict(payload["tracking"])
    tracking_config.update(_stagea_tracking_override(force_target_reroute=True))
    memory_config = dict(payload["memory"])
    memory_config.update(_stagea_memory_override())

    encoder = MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    objectness = MinimalObjectnessField(**payload["field"])
    tracker = MinimalTemporalIdentityTracker(**tracking_config)
    memory = MinimalPrototypeMemory(**memory_config)

    offline_target_metadata = _load_target_metadata()
    offline_expected_lineage_id = int(
        offline_target_metadata.get("old_continuity_lineage_id")
        or offline_target_metadata.get("old_lineage_id")
        or 2
    )
    source_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    attach_rows: list[dict[str, Any]] = []
    prototype_owner_rows: list[dict[str, Any]] = []
    budget_suppression_rows: list[dict[str, Any]] = []
    recovery_selection_rows: list[dict[str, Any]] = []
    trace_frame_records: list[Any] = []
    prev_memory_output = None
    trace_stop_frame = TARGET_FRAME + 24

    for frame_offset in range(1, len(sequence.frames)):
        prev_frame = sequence.frames[frame_offset - 1]
        current_frame = sequence.frames[frame_offset]
        if int(current_frame.frame_index) > int(trace_stop_frame):
            break

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
        trace_frame_records.append(
            SimpleNamespace(
                frame_index=int(current_frame.frame_index),
                instance_ids=list(current_frame.instance_ids),
                gt_boxes=list(current_frame.boxes),
                objectness_output=objectness_output,
                tracking_output=tracking_output,
                memory_output=memory_output,
            )
        )

        gt_box = _gt_box(current_frame, TARGET_GT_OBJECT_ID)
        target_selection = _target_selection_row(
            tracking_output.recovery_selection_rows,
            current_frame.frame_index,
            gt_box,
        )
        target_proposal_id = None if target_selection is None else int(target_selection["proposal_id"])

        for row in tracking_output.recovery_candidate_rows:
            if int(row.get("frame_id", -1)) != int(current_frame.frame_index):
                continue
            proposal_box = _parse_box(row.get("proposal_box"))
            proposal_iou_to_gt = _iou(proposal_box, gt_box)
            annotated = dict(row)
            annotated.update(
                {
                    "scenario_name": TRACK_C_NAME,
                    "target_event_id": TARGET_EVENT_ID,
                    "target_gt_object_id": TARGET_GT_OBJECT_ID,
                    "target_frame": TARGET_FRAME,
                    "target_lineage_id": offline_expected_lineage_id,
                    "offline_expected_lineage_id": offline_expected_lineage_id,
                    "proposal_iou_to_gt": proposal_iou_to_gt,
                    "is_target_proposal": int(
                        target_proposal_id is not None
                        and int(row.get("proposal_id", -1)) == int(target_proposal_id)
                    ),
                    "is_target_frame": int(int(current_frame.frame_index) == int(TARGET_FRAME)),
                }
            )
            source_rows.append(annotated)

        for row in tracking_output.lineage_claim_rows:
            if int(row.get("frame_id", -1)) != int(current_frame.frame_index):
                continue
            proposal_box = _parse_box(row.get("proposal_box"))
            proposal_iou_to_gt = _iou(proposal_box, gt_box)
            annotated = dict(row)
            annotated.update(
                {
                    "scenario_name": TRACK_C_NAME,
                    "target_event_id": TARGET_EVENT_ID,
                    "target_gt_object_id": TARGET_GT_OBJECT_ID,
                    "target_frame": TARGET_FRAME,
                    "target_lineage_id": offline_expected_lineage_id,
                    "offline_expected_lineage_id": offline_expected_lineage_id,
                    "proposal_iou_to_gt": proposal_iou_to_gt,
                    "is_target_proposal": int(
                        target_proposal_id is not None
                        and int(row.get("proposal_id", -1)) == int(target_proposal_id)
                    ),
                    "is_target_frame": int(int(current_frame.frame_index) == int(TARGET_FRAME)),
                }
            )
            claim_rows.append(annotated)

        for row in tracking_output.recovery_selection_rows:
            if int(row.get("frame_id", -1)) != int(current_frame.frame_index):
                continue
            proposal_box = _parse_box(row.get("proposal_box"))
            proposal_iou_to_gt = _iou(proposal_box, gt_box)
            annotated = dict(row)
            annotated.update(
                {
                    "scenario_name": TRACK_C_NAME,
                    "target_event_id": TARGET_EVENT_ID,
                    "target_gt_object_id": TARGET_GT_OBJECT_ID,
                    "target_frame": TARGET_FRAME,
                    "target_lineage_id": offline_expected_lineage_id,
                    "offline_expected_lineage_id": offline_expected_lineage_id,
                    "proposal_iou_to_gt": proposal_iou_to_gt,
                    "is_target_proposal": int(
                        target_proposal_id is not None
                        and int(row.get("proposal_id", -1)) == int(target_proposal_id)
                    ),
                    "is_target_frame": int(int(current_frame.frame_index) == int(TARGET_FRAME)),
                }
            )
            recovery_selection_rows.append(annotated)

        for assignment in memory_output.assignments:
            attach_rows.append(
                {
                    "scenario_name": TRACK_C_NAME,
                    "frame_id": int(current_frame.frame_index),
                    "track_id": int(assignment.track_id),
                    "prototype_id": int(assignment.prototype_id),
                    "lineage_id": int(assignment.lineage_id),
                    "continuity_lineage_id": None
                    if getattr(assignment, "continuity_lineage_id", None) is None
                    else int(assignment.continuity_lineage_id),
                    "matched_lineage_id": None
                    if assignment.matched_lineage_id is None
                    else int(assignment.matched_lineage_id),
                    "recovery_attach_target": str(assignment.recovery_attach_target),
                    "recovery_attach_target_id": assignment.recovery_attach_target_id,
                    "attach_path_source": str(assignment.attach_path_source),
                    "attach_score_current_head": assignment.attach_score_current_head,
                    "attach_score_active_sibling": assignment.attach_score_active_sibling,
                    "attach_score_archived_sibling": assignment.attach_score_archived_sibling,
                    "attach_score_temp_slot": assignment.attach_score_temp_slot,
                    "promotion_pending_flag": int(bool(assignment.promotion_pending_flag)),
                    "promotion_candidate_id": assignment.promotion_candidate_id,
                    "promotion_decision": str(assignment.promotion_decision),
                    "temp_attach_used": int(bool(assignment.temp_attach_used)),
                    "temp_attach_id": assignment.temp_attach_id,
                    "temp_attach_expired": int(bool(assignment.temp_attach_expired)),
                    "attach_branch_entered": int(bool(assignment.attach_branch_entered)),
                    "attach_state_written": int(bool(assignment.attach_state_written)),
                    "lineage_seed_id_used": assignment.lineage_seed_id_used,
                }
            )

        prototype_owner_rows.extend(
            {
                **row,
                "scenario_name": TRACK_C_NAME,
            }
            for row in memory_output.prototype_lineage_rows
        )

    runtime_target_metadata = _derive_target_metadata_from_run(
        result=SimpleNamespace(frame_records=trace_frame_records),
        gt_object_id=TARGET_GT_OBJECT_ID,
        target_frame=TARGET_FRAME,
    )
    effective_target_lineage_id = int(
        runtime_target_metadata.get("old_continuity_lineage_id")
        or runtime_target_metadata.get("old_lineage_id")
        or offline_expected_lineage_id
    )
    for row_set in (source_rows, claim_rows, recovery_selection_rows, attach_rows):
        for row in row_set:
            row["runtime_target_lineage_id"] = None if runtime_target_metadata.get("old_lineage_id") is None else int(runtime_target_metadata["old_lineage_id"])
            row["runtime_target_continuity_lineage_id"] = (
                None
                if runtime_target_metadata.get("old_continuity_lineage_id") is None
                else int(runtime_target_metadata["old_continuity_lineage_id"])
            )
            row["effective_target_lineage_id"] = int(effective_target_lineage_id)
    target_source_rows = [
        row
        for row in source_rows
        if int(row.get("frame_id", -1)) == int(TARGET_FRAME)
        and int(row.get("is_target_proposal", 0)) == 1
    ]
    target_claim_rows = [
        row
        for row in claim_rows
        if int(row.get("frame_id", -1)) == int(TARGET_FRAME)
        and int(row.get("is_target_proposal", 0)) == 1
    ]
    target_selection_rows = [
        row
        for row in recovery_selection_rows
        if int(row.get("frame_id", -1)) == int(TARGET_FRAME)
        and int(row.get("is_target_proposal", 0)) == 1
    ]

    target_continuity_source_visible = any(
        int(row.get("candidate_lineage_id", -1)) == int(effective_target_lineage_id)
        for row in target_source_rows
    )
    target_claim_visible = any(
        int(row.get("candidate_lineage_id", -1)) == int(effective_target_lineage_id)
        and int(row.get("target_lineage_claim_visible_final", 0)) == 1
        for row in target_claim_rows
    )
    target_final_selection = None
    if target_selection_rows:
        target_final_selection = target_selection_rows[0].get("selected_lineage_id")
        if target_final_selection not in (None, "", "None"):
            target_final_selection = int(target_final_selection)
        else:
            target_final_selection = None

    if not target_continuity_source_visible:
        target_failure_class = "visibility_failure"
    elif not target_claim_visible:
        target_failure_class = "claim_builder_failure"
    elif target_final_selection != int(effective_target_lineage_id):
        target_failure_class = "attach_or_ranking_failure"
    else:
        target_failure_class = "visibility_restored"

    return {
        "source_rows": source_rows,
        "claim_rows": claim_rows,
        "attach_rows": attach_rows,
        "prototype_owner_rows": prototype_owner_rows,
        "budget_suppression_rows": budget_suppression_rows,
        "target_summary": {
            "target_event_id": TARGET_EVENT_ID,
            "target_gt_object_id": TARGET_GT_OBJECT_ID,
            "target_frame": TARGET_FRAME,
            "target_lineage_id": int(effective_target_lineage_id),
            "offline_expected_lineage_id": int(offline_expected_lineage_id),
            "runtime_old_lineage_id": runtime_target_metadata.get("old_lineage_id"),
            "runtime_old_continuity_lineage_id": runtime_target_metadata.get("old_continuity_lineage_id"),
            "runtime_old_track_id": runtime_target_metadata.get("old_track_id"),
            "runtime_old_prototype_id": runtime_target_metadata.get("old_prototype_id"),
            "target_continuity_source_visible": int(target_continuity_source_visible),
            "target_claim_visible": int(target_claim_visible),
            "target_final_selected_lineage": target_final_selection,
            "target_failure_class": target_failure_class,
            "target_source_row_count": len(target_source_rows),
            "target_claim_row_count": len(target_claim_rows),
        },
    }


def _load_overwrite_rows() -> list[dict[str, Any]]:
    path = Path("results/phase3d/phase3d_stagea7_remap_trace.csv")
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            overwritten = row.get("was_continuity_key_overwritten", "0")
            overwrite_bug = row.get("overwrite_bug_flag", "0")
            if int(overwritten or 0) == 1 or int(overwrite_bug or 0) == 1:
                rows.append(dict(row))
    return rows


def _write_summary(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    trace_payload: dict[str, Any],
) -> None:
    summary_lookup = {str(row["scenario_name"]): row for row in summary_rows}
    track_a = summary_lookup.get(TRACK_A_NAME, {})
    track_c = summary_lookup.get(TRACK_C_NAME, {})
    target = trace_payload["target_summary"]
    lines = [
        "# Phase A Result Summary",
        "",
        "## 目标",
        "",
        "阶段A只修一件事：让正确 continuity owner 真正变成可枚举、可追踪、可进入 preserve-input / claim-builder 的 recovery source。",
        "",
        "## 最小回归",
        "",
        f"- `track_a_bridge / U-Recall = {float(track_a.get('u_recall', 0.0)):.4f}`",
        f"- `track_a_bridge / PFR = {float(track_a.get('pfr', 0.0)):.4f}`",
        f"- `track_a_bridge / IDSW = {int(track_a.get('track_idsw', 0))}`",
        f"- `track_c_long_horizon / U-Recall = {float(track_c.get('u_recall', 0.0)):.4f}`",
        f"- `track_c_long_horizon / PFR = {float(track_c.get('pfr', 0.0)):.4f}`",
        f"- `track_c_long_horizon / IDSW = {int(track_c.get('track_idsw', 0))}`",
        "",
        "## Target Event",
        "",
        f"- `event_id = {int(target['target_event_id'])}`",
        f"- `frame = {int(target['target_frame'])}`",
        f"- `target_lineage_id = {int(target['target_lineage_id'])}`",
        f"- `offline_expected_lineage_id = {int(target['offline_expected_lineage_id'])}`",
        f"- `runtime_old_lineage_id = {target['runtime_old_lineage_id']}`",
        f"- `runtime_old_continuity_lineage_id = {target['runtime_old_continuity_lineage_id']}`",
        f"- `continuity_source_visible = {int(target['target_continuity_source_visible'])}`",
        f"- `claim_visible = {int(target['target_claim_visible'])}`",
        f"- `final_selected_lineage = {target['target_final_selected_lineage']}`",
        f"- `failure_class = {target['target_failure_class']}`",
        "",
        "## 解释",
        "",
        "阶段A只看 source visibility。若 target continuity lineage 重新进入 source pool / claim-builder，则说明问题已从 visibility failure 推进到 attach/ranking failure；若它仍不可见，则说明 dual-owner 枚举还没有真正接通。",
        "",
        "## A.7 overwrite 审计复用",
        "",
        "- `results/phase3d/phase3d_stagea7_remap_trace.csv` 继续作为 continuity key overwrite 审计基准。",
        "- 本轮统一 runner 只把 overwrite bug 抽取成 `continuity_key_overwrite_events.csv`，不重复跑更深的 remap trace。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_debug_commands(path: Path) -> None:
    lines = [
        "# Phase A Debug Commands",
        "",
        "```powershell",
        "python experiments/run_phaseA_visibility_repair.py --config configs/bridge_synth_generic_v1.yaml --output-dir results/phaseA",
        "python experiments/run_phase3d_stage_a7_remap_trace.py --config configs/bridge_synth_generic_v1.yaml --output-dir results/phase3d",
        "```",
        "",
        "重点文件：",
        "",
        "- `results/phaseA/source_visibility_trace.jsonl`",
        "- `results/phaseA/claim_builder_source_breakdown.csv`",
        "- `results/phaseA/attach_decisions.jsonl`",
        "- `results/phaseA/continuity_key_overwrite_events.csv`",
        "- `results/phaseA/phasea_result_summary.md`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_runs = evaluate_phase3_scenarios(
        args.config,
        tracking_override=_stagea_tracking_override(force_target_reroute=False),
        memory_override=_stagea_memory_override(),
        seed=args.seed,
        scenario_names=[TRACK_A_NAME, TRACK_C_NAME],
    )
    trace_payload = _manual_track_c_trace(
        config_path=Path(args.config),
        seed=args.seed,
    )

    regression_rows = [
        {
            "scenario_name": str(run["scenario_name"]),
            "u_recall": float(run["result"].summary.u_recall),
            "pfr": float(run["result"].summary.pfr),
            "track_idsw": int(run["result"].primary_monitoring["track_idsw"]),
            "memory_growth": float(run["result"].summary.memory_growth),
            "purity": float(run["result"].summary.purity),
        }
        for run in scenario_runs
    ]
    overwrite_rows = _load_overwrite_rows()

    _write_jsonl(output_dir / "source_visibility_trace.jsonl", trace_payload["source_rows"])
    _write_csv(output_dir / "claim_builder_source_breakdown.csv", trace_payload["source_rows"])
    _write_jsonl(output_dir / "attach_decisions.jsonl", trace_payload["attach_rows"])
    _write_jsonl(output_dir / "prototype_owner_transitions.jsonl", trace_payload["prototype_owner_rows"])
    _write_csv(output_dir / "continuity_key_overwrite_events.csv", overwrite_rows)
    _write_csv(output_dir / "budget_suppression_trace.csv", trace_payload["budget_suppression_rows"])
    _write_csv(output_dir / "minimal_regression_summary.csv", regression_rows)
    (output_dir / "phasea_target_event_summary.json").write_text(
        json.dumps(trace_payload["target_summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_summary(
        output_dir / "phasea_result_summary.md",
        summary_rows=regression_rows,
        trace_payload=trace_payload,
    )
    _write_debug_commands(output_dir / "phasea_debug_commands.md")

    print(f"saved={output_dir / 'phasea_result_summary.md'}")
    print(f"saved={output_dir / 'source_visibility_trace.jsonl'}")
    print(f"saved={output_dir / 'claim_builder_source_breakdown.csv'}")
    print(f"saved={output_dir / 'attach_decisions.jsonl'}")
    print(f"saved={output_dir / 'prototype_owner_transitions.jsonl'}")
    print(f"saved={output_dir / 'continuity_key_overwrite_events.csv'}")
    print(f"saved={output_dir / 'minimal_regression_summary.csv'}")


if __name__ == "__main__":
    main()
