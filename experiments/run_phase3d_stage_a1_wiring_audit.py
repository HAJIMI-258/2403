"""Run Phase 3D Stage A.1: attach path wiring audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3d_utils import (  # noqa: E402
    _build_run_frame_maps,
    _figure_to_array,
    _to_rgb,
    _window_frames,
    default_phase3d_stagea_memory_override,
    default_phase3d_stagea_tracking_override,
)
from experiments.phase3l_utils import evaluate_phase3l_bundle  # noqa: E402
from experiments.phase3r_utils import extract_reentry_events, write_csv  # noqa: E402
from experiments.phase3s_utils import TRACK_C_NAME  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3D Stage A.1 attach wiring audit.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--output-dir", default="results/phase3d")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _evaluate_track_c(
    config_path: str | Path,
    *,
    seed: int,
    run_label: str,
    tracking_patch: dict[str, Any] | None = None,
    memory_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tracking_override = default_phase3d_stagea_tracking_override()
    if tracking_patch:
        tracking_override.update(tracking_patch)
    memory_override = default_phase3d_stagea_memory_override()
    if memory_patch:
        memory_override.update(memory_patch)

    bundle = evaluate_phase3l_bundle(
        config_path,
        tracking_override=tracking_override,
        memory_override=memory_override,
        seed=seed,
        scenario_names=[TRACK_C_NAME],
        frame_record_mode="full",
    )
    run = bundle["runs"][0]
    summary_row = dict(bundle["rows"][0])
    event_rows, frame_logs = extract_reentry_events(TRACK_C_NAME, run["sequence"], run["result"])
    return {
        "label": run_label,
        "tracking_override": tracking_override,
        "memory_override": memory_override,
        "bundle": bundle,
        "run": run,
        "summary_row": summary_row,
        "event_rows": event_rows,
        "frame_logs": frame_logs,
    }


def _event_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("gt_object_id", row.get("instance_id", -1))),
        int(row.get("reappear_frame", -1)),
        int(row.get("gap_length", -1)),
    )


def _build_branch_coverage_rows(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in run_payload["event_rows"]:
        matched_lineage_established = int(row.get("matched_lineage_id") is not None)
        entered_temp_attach_branch = int(
            bool(row.get("temp_attach_used", 0))
            or str(row.get("recovery_attach_target", "none")) == "temporary_attach_slot"
        )
        continuation_access_attempted = int(
            bool(row.get("attach_state_consumed_by_continuation", 0))
            or bool(row.get("continuation_attempted", 0))
        )
        restore_attempted = int(bool(row.get("restore_attempted_from_attach", 0)))

        if matched_lineage_established == 0:
            failure_stage = "matched_lineage_missing"
        elif int(row.get("attach_branch_entered", 0)) == 0:
            failure_stage = "attach_branch_not_entered"
        elif int(row.get("attach_state_written", 0)) == 0:
            failure_stage = "attach_state_not_written"
        elif continuation_access_attempted == 0 and restore_attempted == 0:
            failure_stage = "attach_written_but_not_consumed"
        elif int(row.get("candidate_pool_nonempty", 0)) == 0:
            failure_stage = "no_recovery_candidates"
        elif restore_attempted == 1 and int(row.get("matched_same_track", 0)) == 0:
            failure_stage = "restore_failed_after_attach"
        else:
            failure_stage = "recovery_path_alive"

        rows.append(
            {
                "run_label": str(run_payload["label"]),
                "scenario_name": str(row.get("scenario_name", TRACK_C_NAME)),
                "sequence_id": int(row.get("sequence_id", 0)),
                "event_id": int(row.get("event_id", 0)),
                "gt_object_id": int(row.get("gt_object_id", row.get("instance_id", -1))),
                "old_track_id": None if row.get("old_track_id") is None else int(row.get("old_track_id")),
                "old_lineage_id": None if row.get("old_lineage_id") is None else int(row.get("old_lineage_id")),
                "old_prototype_id": None if row.get("old_prototype_id") is None else int(row.get("old_prototype_id")),
                "gap_length": int(row.get("gap_length", 0)),
                "reappear_frame": int(row.get("reappear_frame", 0)),
                "matched_lineage_established": matched_lineage_established,
                "matched_lineage_id": None
                if row.get("matched_lineage_id") is None
                else int(row.get("matched_lineage_id")),
                "prototype_hint_id": None
                if row.get("matched_prototype_id") is None
                else int(row.get("matched_prototype_id")),
                "prototype_hint_lineage_id": None
                if row.get("prototype_hint_lineage_id") is None
                else int(row.get("prototype_hint_lineage_id")),
                "pre_memory_linked_lineage_id": None
                if row.get("pre_memory_linked_lineage_id") is None
                else int(row.get("pre_memory_linked_lineage_id")),
                "entered_attach_branch": int(bool(row.get("attach_branch_entered", 0))),
                "temp_attach_eligibility_checked": int(bool(row.get("temp_attach_eligibility_checked", 0))),
                "entered_temp_attach_branch": entered_temp_attach_branch,
                "attach_state_written": int(bool(row.get("attach_state_written", 0))),
                "attach_state_consumed_by_tracker": int(bool(row.get("attach_state_consumed_by_tracker", 0))),
                "attach_state_consumed_by_continuation": int(
                    bool(row.get("attach_state_consumed_by_continuation", 0))
                ),
                "restore_attempted_from_attach": restore_attempted,
                "promotion_pending_created": int(bool(row.get("promotion_pending_created", 0))),
                "promotion_step_executed": int(bool(row.get("promotion_step_executed", 0))),
                "recovery_attach_target": str(row.get("recovery_attach_target", "none")),
                "attach_path_source": str(row.get("attach_path_source", "")),
                "temp_attach_used": int(bool(row.get("temp_attach_used", 0))),
                "temp_attach_force_mode": int(bool(row.get("temp_attach_force_mode", 0))),
                "lineage_seed_id_used": None
                if row.get("lineage_seed_id_used") is None
                else int(row.get("lineage_seed_id_used")),
                "candidate_pool_size": int(row.get("candidate_pool_size", 0)),
                "live_candidate_pool_size": int(row.get("live_candidate_pool_size", 0)),
                "continuation_bank_size": int(row.get("continuation_bank_size", 0)),
                "prototype_matched_continuation_count": int(row.get("prototype_matched_continuation_count", 0)),
                "lineage_matched_continuation_count": int(row.get("lineage_matched_continuation_count", 0)),
                "continuation_access_used": int(bool(row.get("continuation_access_used", 0))),
                "continuation_access_success": int(bool(row.get("continuation_access_success", 0))),
                "continuation_attempted": int(bool(row.get("continuation_attempted", 0))),
                "same_track_after_attach": int(bool(row.get("same_track_after_attach", 0))),
                "same_prototype_after_attach": int(bool(row.get("same_prototype_after_attach", 0))),
                "same_track_final": int(bool(row.get("matched_same_track", 0))),
                "same_prototype_final": int(bool(row.get("matched_same_prototype", 0))),
                "failure_stage_stagea1": failure_stage,
            }
        )
    return rows


def _mean_flag(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(int(bool(row.get(key, 0))) for row in rows) / len(rows))


def _summarize_wiring_run(run_payload: dict[str, Any], branch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    event_rows = run_payload["event_rows"]
    matched_lineage_rows = [row for row in branch_rows if int(row["matched_lineage_established"]) == 1]
    concept_rows = [row for row in event_rows if int(row.get("concept_recovered", 0)) == 1]
    hinted_rows = [row for row in event_rows if row.get("prototype_hint_lineage_id") is not None]
    return {
        "num_events": len(event_rows),
        "concept_recovered_events": len(concept_rows),
        "matched_lineage_events": len(matched_lineage_rows),
        "hinted_lineage_events": len(hinted_rows),
        "attach_success_rate_given_matched_lineage": (
            float(
                sum(int(row["recovery_attach_target"] != "none") for row in matched_lineage_rows)
                / len(matched_lineage_rows)
            )
            if matched_lineage_rows
            else 0.0
        ),
        "attach_branch_enter_rate_given_matched_lineage": _mean_flag(matched_lineage_rows, "entered_attach_branch"),
        "temp_attach_usage_rate": _mean_flag(branch_rows, "temp_attach_used"),
        "continuation_access_rate": float(
            run_payload["summary_row"].get("continuation_bank_access_rate_given_concept_recovery", 0.0)
        ),
        "continuation_access_used_rate": _mean_flag(branch_rows, "continuation_access_used"),
        "same_track_after_attach": _mean_flag(branch_rows, "same_track_after_attach"),
        "same_track_reentry_recovery": float(run_payload["summary_row"].get("same_track_reentry_recovery", 0.0)),
        "same_track_after_concept_recovery": float(
            run_payload["summary_row"].get("same_track_after_concept_recovery", 0.0)
        ),
        "same_prototype_reentry_recovery": float(
            run_payload["summary_row"].get("same_prototype_reentry_recovery", 0.0)
        ),
        "pfr": float(run_payload["summary_row"].get("pfr", 0.0)),
        "track_idsw": int(run_payload["summary_row"].get("track_idsw", 0)),
        "drop_lineage_seed_before_memory_rate": (
            float(
                sum(
                    int(
                        row.get("prototype_hint_lineage_id") is not None
                        and row.get("pre_memory_linked_lineage_id") is None
                    )
                    for row in event_rows
                )
                / len(hinted_rows)
            )
            if hinted_rows
            else 0.0
        ),
        "attach_not_consumed_rate": _mean_flag(
            [row for row in branch_rows if int(row.get("attach_state_written", 0)) == 1],
            "attach_state_consumed_by_tracker",
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _bool_text(value: Any) -> str:
    return "yes" if _as_bool(value) else "no"


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"", "0", "false", "no", "none"}:
            return False
        if stripped in {"1", "true", "yes"}:
            return True
    return bool(value)


def _diagnose_event(row: dict[str, Any]) -> str:
    matched_lineage = row.get("matched_lineage_established")
    if matched_lineage is None:
        matched_lineage = int(row.get("matched_lineage_id") is not None)
    if int(matched_lineage) == 0:
        return "matched lineage missing before attach stage"
    if row.get("prototype_hint_lineage_id") is not None and row.get("pre_memory_linked_lineage_id") is None:
        return "prototype hint carried lineage, but lineage seed was dropped before memory assignment"
    if int(row.get("entered_attach_branch", 0)) == 0:
        return "attach selector did not enter despite matched lineage"
    if int(row.get("attach_state_written", 0)) == 0:
        return "attach selector entered, but no runtime attach state was written"
    if int(row.get("attach_state_consumed_by_tracker", 0)) == 0 and int(row.get("attach_state_consumed_by_continuation", 0)) == 0:
        return "attach state was written, but neither tracker nor continuation path consumed it"
    if int(row.get("restore_attempted_from_attach", 0)) == 0 and int(row.get("continuation_access_used", 0)) == 0:
        return "attach state reached runtime, but no continuation lookup or restore attempt followed"
    if int(row.get("same_track_final", 0)) == 0:
        return "restore path executed but did not recover the old track"
    return "attach path reached recovery and restored the old track"


def _write_attach_state_trace(
    path: Path,
    baseline_rows: list[dict[str, Any]],
    forced_rows: list[dict[str, Any]],
) -> None:
    baseline_by_key = {_event_key(row): row for row in baseline_rows}
    forced_by_key = {_event_key(row): row for row in forced_rows}
    ordered_keys = sorted(set(baseline_by_key) | set(forced_by_key))

    lines = [
        "# Attach State Trace",
        "",
        "Per-event comparison between baseline Stage A wiring and forced temp-attach wiring.",
        "",
    ]
    for event_index, key in enumerate(ordered_keys, start=1):
        base = baseline_by_key.get(key)
        forced = forced_by_key.get(key)
        ref = forced or base
        if ref is None:
            continue
        lines.extend(
            [
                f"## Event {event_index:02d}",
                "",
                f"- object: `{ref.get('gt_object_id')}`",
                f"- reappear_frame: `{ref.get('reappear_frame')}`",
                f"- gap_length: `{ref.get('gap_length')}`",
                "",
                "### Baseline",
                "",
            ]
        )
        lines.extend(_trace_block(base))
        lines.extend(
            [
                "",
                "### Forced Temp Attach",
                "",
            ]
        )
        lines.extend(_trace_block(forced))
        diagnosis = _diagnose_event(forced or base)
        lines.extend(
            [
                "",
                "### Diagnosis",
                "",
                f"- {diagnosis}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _trace_block(row: dict[str, Any] | None) -> list[str]:
    if row is None:
        return ["- event missing in this run"]
    same_track_final = _as_bool(row.get("matched_same_track", row.get("same_track_final", 0)))
    same_prototype_final = _as_bool(row.get("matched_same_prototype", row.get("same_prototype_final", 0)))
    attach_branch = row.get("attach_branch_entered", row.get("entered_attach_branch", 0))
    temp_attach_used = row.get("temp_attach_used", row.get("entered_temp_attach_branch", 0))
    return [
        f"- decision made: attach_target=`{row.get('recovery_attach_target', 'none')}` action=`{row.get('action_type', '')}`",
        f"- lineage seed: hint=`{row.get('prototype_hint_lineage_id')}` pre-memory=`{row.get('pre_memory_linked_lineage_id')}` used=`{row.get('lineage_seed_id_used')}`",
        f"- state written where: attach_branch=`{_bool_text(attach_branch)}`, attach_written=`{_bool_text(row.get('attach_state_written'))}`, temp_attach=`{_bool_text(temp_attach_used)}`, promotion_pending=`{_bool_text(row.get('promotion_pending_created'))}`",
        f"- state consumed where: tracker=`{_bool_text(row.get('attach_state_consumed_by_tracker'))}`, continuation=`{_bool_text(row.get('attach_state_consumed_by_continuation'))}`, restore_attempt=`{_bool_text(row.get('restore_attempted_from_attach'))}`, promotion_step=`{_bool_text(row.get('promotion_step_executed'))}`",
        f"- downstream effect: continuation_used=`{int(_as_bool(row.get('continuation_access_used', 0)))}`, continuation_success=`{int(_as_bool(row.get('continuation_access_success', 0)))}`, same_track_after_attach=`{int(_as_bool(row.get('same_track_after_attach', 0)))}`, same_track_final=`{int(same_track_final)}`, same_prototype_final=`{int(same_prototype_final)}`",
    ]


def _write_stagea1_md(
    path: Path,
    baseline_summary: dict[str, Any],
    forced_summary: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    forced_rows: list[dict[str, Any]],
) -> None:
    baseline_not_entered = sum(int(row["failure_stage_stagea1"] == "attach_branch_not_entered") for row in baseline_rows)
    baseline_unconsumed = sum(int(row["failure_stage_stagea1"] == "attach_written_but_not_consumed") for row in baseline_rows)
    forced_unconsumed = sum(int(row["failure_stage_stagea1"] == "attach_written_but_not_consumed") for row in forced_rows)
    forced_temp = sum(int(row["temp_attach_used"]) for row in forced_rows)
    lines = [
        "# Phase 3D Stage A.1 Wiring Audit",
        "",
        "## Baseline",
        "",
        f"- matched_lineage_events = `{baseline_summary['matched_lineage_events']}`",
        f"- attach_branch_enter_rate_given_matched_lineage = `{baseline_summary['attach_branch_enter_rate_given_matched_lineage']:.4f}`",
        f"- temp_attach_usage_rate = `{baseline_summary['temp_attach_usage_rate']:.4f}`",
        f"- continuation_access_rate = `{baseline_summary['continuation_access_rate']:.4f}`",
        f"- same_track_after_attach = `{baseline_summary['same_track_after_attach']:.4f}`",
        f"- dropped_lineage_seed_before_memory_rate = `{baseline_summary['drop_lineage_seed_before_memory_rate']:.4f}`",
        f"- attach_branch_not_entered_events = `{baseline_not_entered}`",
        f"- attach_written_but_not_consumed_events = `{baseline_unconsumed}`",
        "",
        "## Forced Temp Attach",
        "",
        f"- matched_lineage_events = `{forced_summary['matched_lineage_events']}`",
        f"- attach_branch_enter_rate_given_matched_lineage = `{forced_summary['attach_branch_enter_rate_given_matched_lineage']:.4f}`",
        f"- temp_attach_usage_rate = `{forced_summary['temp_attach_usage_rate']:.4f}`",
        f"- continuation_access_rate = `{forced_summary['continuation_access_rate']:.4f}`",
        f"- same_track_after_attach = `{forced_summary['same_track_after_attach']:.4f}`",
        f"- forced_temp_attach_events = `{forced_temp}`",
        f"- attach_written_but_not_consumed_events = `{forced_unconsumed}`",
        "",
        "## Wiring Conclusion",
        "",
        "- `attach_success_rate_given_matched_lineage = 1.0` is a small-denominator signal here, because Track C re-entry events only produce one matched-lineage event per run under the current Stage A path.",
        "- In both baseline and forced runs, the matched-lineage event writes attach state, but `candidate_pool_size = 0`, `continuation_bank_size = 0`, and no restore attempt follows. That is a wiring/availability failure, not a successful recovery attach.",
        "- Forced temp attach proves the temp slot can be instantiated, but it still does not drive continuation lookup or old-track restore. The remaining break is downstream of attach-state write, at candidate-pool / continuation access consumption.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _select_forced_cases(
    baseline_rows: list[dict[str, Any]],
    forced_rows: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    baseline_by_key = {_event_key(row): row for row in baseline_rows}
    selected: list[tuple[str, dict[str, Any]]] = []

    for row in forced_rows:
        key = _event_key(row)
        baseline = baseline_by_key.get(key)
        if baseline is not None and int(baseline.get("attach_branch_entered", 0)) == 0 and int(row.get("attach_branch_entered", 0)) == 1:
            selected.append(("forced", row))
            break
    for row in forced_rows:
        if int(row.get("temp_attach_used", 0)) == 1:
            if _event_key(row) not in {_event_key(item[1]) for item in selected}:
                selected.append(("forced", row))
                break
    for row in forced_rows:
        if int(row.get("temp_attach_used", 0)) == 1 and int(row.get("matched_same_track", 0)) == 0:
            if _event_key(row) not in {_event_key(item[1]) for item in selected}:
                selected.append(("forced", row))
                break
    for row in forced_rows:
        if _event_key(row) not in {_event_key(item[1]) for item in selected}:
            selected.append(("forced", row))
        if len(selected) >= 3:
            break
    return selected[:3]


def _draw_wiring_axis(axis, frame_record, instance_row: dict[str, Any] | None, event_row: dict[str, Any], run_label: str) -> None:
    frame = frame_record.frame
    if frame is None:
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
    axis.imshow(_to_rgb(frame))
    axis.axis("off")
    for gt_box in frame_record.gt_boxes:
        axis.add_patch(Rectangle((gt_box[0], gt_box[1]), gt_box[2], gt_box[3], fill=False, lw=1.6, ec="#22c55e"))
    for pred_box in frame_record.predicted_boxes:
        axis.add_patch(Rectangle((pred_box[0], pred_box[1]), pred_box[2], pred_box[3], fill=False, lw=1.2, ec="#f59e0b"))
    for proposal in getattr(frame_record.objectness_output, "proposals", [])[:8]:
        axis.add_patch(
            Rectangle((proposal.box[0], proposal.box[1]), proposal.box[2], proposal.box[3], fill=False, lw=0.8, ec="#38bdf8", alpha=0.45)
        )

    frame_id = int(frame_record.frame_index)
    instance = instance_row or {}
    text_lines = [
        f"{run_label} f={frame_id} obj={event_row['gt_object_id']}",
        f"matchedL={instance.get('matched_lineage_id')} attachBranch={_bool_text(instance.get('attach_branch_entered', False))}",
        f"tempAttach={_bool_text(instance.get('temp_attach_used', False))} contLookup={_bool_text(instance.get('attach_state_consumed_by_continuation', False) or instance.get('continuation_attempted', False))}",
        f"restore={_bool_text(instance.get('restore_attempted_from_attach', False))} promoPending={_bool_text(instance.get('promotion_pending_created', False) or instance.get('promotion_pending_flag', False))}",
        f"track={instance.get('track_id')} proto={instance.get('prototype_id')} sameT={int(instance.get('track_id') == event_row.get('old_track_id')) if instance.get('track_id') is not None else 0} sameP={int(instance.get('prototype_id') == event_row.get('old_prototype_id')) if instance.get('prototype_id') is not None else 0}",
    ]
    axis.text(
        4,
        6,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.72, "pad": 3},
    )


def _save_preview_gif(run_payload: dict[str, Any], cases: list[tuple[str, dict[str, Any]]], path: Path) -> None:
    frame_lookup, instance_lookup = _build_run_frame_maps(run_payload["run"])
    images: list[Image.Image] = []
    for run_label, case in cases:
        frame_ids = _window_frames(frame_lookup, center_frame=int(case["reappear_frame"]), left=4, right=7)
        for frame_id in frame_ids:
            fig, axis = plt.subplots(1, 1, figsize=(5.6, 5.2), constrained_layout=True)
            _draw_wiring_axis(
                axis,
                frame_lookup[frame_id],
                instance_lookup.get(frame_id, {}).get(int(case["gt_object_id"])),
                case,
                run_label,
            )
            images.append(Image.fromarray(_figure_to_array(fig)))
    if images:
        images[0].save(path, save_all=True, append_images=images[1:], duration=180, loop=0)


def _save_failure_gallery(run_payload: dict[str, Any], path: Path) -> None:
    frame_lookup, instance_lookup = _build_run_frame_maps(run_payload["run"])
    failures = [
        row
        for row in run_payload["event_rows"]
        if int(row.get("matched_same_track", 0)) == 0 or int(row.get("continuation_access_used", 0)) == 0
    ]
    failures.sort(
        key=lambda row: (
            int(row.get("attach_branch_entered", 0)),
            int(row.get("temp_attach_used", 0)),
            int(row.get("continuation_access_used", 0)),
            int(row.get("matched_same_track", 0)),
            int(row.get("matched_same_prototype", 0)),
        )
    )
    selected = failures[:6] if failures else run_payload["event_rows"][:1]
    ncols = min(3, len(selected))
    nrows = int(np.ceil(len(selected) / max(1, ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.6 * nrows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    for axis in axes_array.ravel():
        axis.axis("off")
    for axis, case in zip(axes_array.ravel(), selected):
        frame_id = int(case["reappear_frame"])
        _draw_wiring_axis(
            axis,
            frame_lookup[frame_id],
            instance_lookup.get(frame_id, {}).get(int(case["gt_object_id"])),
            case,
            str(run_payload["label"]),
        )
    Image.fromarray(_figure_to_array(fig)).save(path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _evaluate_track_c(args.config, seed=args.seed, run_label="baseline")
    forced = _evaluate_track_c(
        args.config,
        seed=args.seed,
        run_label="forced_temp_attach",
        tracking_patch={"debug_inherit_lineage_from_hint": True},
        memory_patch={"debug_force_temp_attach": True},
    )

    baseline_branch_rows = _build_branch_coverage_rows(baseline)
    forced_branch_rows = _build_branch_coverage_rows(forced)
    branch_rows = baseline_branch_rows + forced_branch_rows

    baseline_summary = _summarize_wiring_run(baseline, baseline_branch_rows)
    forced_summary = _summarize_wiring_run(forced, forced_branch_rows)
    combined_summary = {
        "baseline": baseline_summary,
        "forced_temp_attach": forced_summary,
        "delta": {
            "attach_branch_enter_rate_given_matched_lineage": float(
                forced_summary["attach_branch_enter_rate_given_matched_lineage"]
                - baseline_summary["attach_branch_enter_rate_given_matched_lineage"]
            ),
            "temp_attach_usage_rate": float(
                forced_summary["temp_attach_usage_rate"] - baseline_summary["temp_attach_usage_rate"]
            ),
            "continuation_access_rate": float(
                forced_summary["continuation_access_rate"] - baseline_summary["continuation_access_rate"]
            ),
            "same_track_after_attach": float(
                forced_summary["same_track_after_attach"] - baseline_summary["same_track_after_attach"]
            ),
        },
        "inference": {
            "lineage_seed_drop_blocks_attach_branch": bool(
                baseline_summary["drop_lineage_seed_before_memory_rate"] > 0.5
                and baseline_summary["attach_branch_enter_rate_given_matched_lineage"] < 0.5
                and forced_summary["attach_branch_enter_rate_given_matched_lineage"]
                > baseline_summary["attach_branch_enter_rate_given_matched_lineage"]
            ),
            "forced_temp_attach_moves_execution_off_zero": bool(
                forced_summary["temp_attach_usage_rate"] > 0.0
                or forced_summary["continuation_access_used_rate"] > 0.0
                or forced_summary["same_track_after_attach"] > 0.0
            ),
            "downstream_consumption_still_missing": bool(
                forced_summary["attach_branch_enter_rate_given_matched_lineage"] > 0.0
                and forced_summary["continuation_access_used_rate"] == 0.0
                and forced_summary["same_track_after_attach"] == 0.0
            ),
        },
    }

    write_csv(output_dir / "phase3d_stagea1_branch_coverage.csv", branch_rows)
    _write_json(output_dir / "phase3d_stagea1_forced_temp_attach_summary.json", combined_summary)
    _write_attach_state_trace(output_dir / "attach_state_trace.md", baseline["event_rows"], forced["event_rows"])
    _write_stagea1_md(
        output_dir / "phase3d_stagea1_wiring_audit.md",
        baseline_summary,
        forced_summary,
        baseline_branch_rows,
        forced_branch_rows,
    )

    forced_cases = _select_forced_cases(baseline["event_rows"], forced["event_rows"])
    _save_preview_gif(forced, forced_cases, output_dir / "forced_temp_attach_preview.gif")
    _save_failure_gallery(forced, output_dir / "forced_temp_attach_failure_gallery.png")

    print(f"saved_branch_coverage={output_dir / 'phase3d_stagea1_branch_coverage.csv'}")
    print(f"saved_forced_summary={output_dir / 'phase3d_stagea1_forced_temp_attach_summary.json'}")
    print(f"saved_attach_trace={output_dir / 'attach_state_trace.md'}")
    print(f"saved_wiring_audit={output_dir / 'phase3d_stagea1_wiring_audit.md'}")
    print(f"saved_preview={output_dir / 'forced_temp_attach_preview.gif'}")
    print(f"saved_gallery={output_dir / 'forced_temp_attach_failure_gallery.png'}")


if __name__ == "__main__":
    main()
