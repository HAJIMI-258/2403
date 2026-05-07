from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

from experiments.phase3l_utils import (
    default_phase3l_memory_override,
    default_phase3l_tracking_override,
    evaluate_phase3l_bundle,
)
from experiments.phase3r_utils import _build_frame_instance_map, extract_reentry_events, write_csv
from experiments.phase3s_utils import TRACK_A_NAME, TRACK_C_NAME


def default_phase3d_stagea_tracking_override() -> dict[str, Any]:
    return default_phase3l_tracking_override()


def default_phase3d_stagea_memory_override() -> dict[str, Any]:
    override = default_phase3l_memory_override()
    override.update(
        {
            "enable_phase3d_dual_score": True,
            "enable_phase3d_temp_attach": True,
            "enable_phase3d_deferred_promotion": False,
            "attach_accept_min": 0.28,
            "promote_margin": 0.10,
            "promote_support_min": 3,
            "promotion_window": 8,
            "promotion_stability_min": 0.60,
            "post_promotion_cooldown": 8,
            "temp_attach_ttl": 10,
            "enable_phase3p_keep_head_default": False,
            "enable_phase3p_grouped_gating": False,
            "enable_phase3p_birth_suppression": False,
            "enable_phase3p_full_stabilization": False,
        }
    )
    return override


def evaluate_phase3d_stagea_bundle(
    config_path: str | Path,
    *,
    seed: int = 42,
    scenario_names: list[str] | None = None,
) -> dict[str, Any]:
    return evaluate_phase3l_bundle(
        config_path,
        tracking_override=default_phase3d_stagea_tracking_override(),
        memory_override=default_phase3d_stagea_memory_override(),
        seed=seed,
        scenario_names=scenario_names or [TRACK_A_NAME, TRACK_C_NAME],
        frame_record_mode="full",
    )


def build_phase3d_event_audit_rows(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in event_rows:
        attach_target_type = str(row.get("recovery_attach_target", "none"))
        head_kept = int(attach_target_type == "current_head" and int(row.get("head_switched", 0)) == 0)
        temp_attach_used = int(row.get("temp_attach_used", 0))
        recovery_path_preserved = int(row.get("recovery_path_preserved", 0))
        if int(row.get("proposal_detected", 0)) == 0:
            failure_type = "proposal_missing"
        elif attach_target_type == "none":
            failure_type = "attach_failed"
        elif temp_attach_used and int(row.get("same_track_after_attach", 0)) == 0:
            failure_type = "temp_attach_only"
        elif head_kept and recovery_path_preserved == 0:
            failure_type = "kept_head_recovery_starved"
        elif int(row.get("head_switched", 0)) == 1 and int(row.get("same_prototype_id", 0)) == 0:
            failure_type = "switched_head_fragmentation"
        else:
            failure_type = "recovered_or_neutral"

        rows.append(
            {
                "sequence_id": int(row.get("sequence_id", 0)),
                "frame_id": int(row.get("frame_id", row.get("reappear_frame", 0))),
                "event_id": int(row.get("event_id", 0)),
                "gt_object_id": int(row.get("gt_object_id", row.get("instance_id", -1))),
                "old_track_id": int(row.get("old_track_id", -1)) if row.get("old_track_id") is not None else None,
                "old_lineage_id": int(row.get("old_lineage_id", -1)) if row.get("old_lineage_id") is not None else None,
                "old_prototype_id": int(row.get("old_prototype_id", -1)) if row.get("old_prototype_id") is not None else None,
                "matched_lineage_id": int(row.get("matched_lineage_id", -1))
                if row.get("matched_lineage_id") is not None
                else None,
                "attach_decision": str(row.get("action_type", "")),
                "attach_target_type": attach_target_type,
                "attach_target_id": int(row.get("recovery_attach_target_id", -1))
                if row.get("recovery_attach_target_id") is not None
                else None,
                "attach_score_current_head": _safe_float(row.get("attach_score_current_head")),
                "attach_score_active_sibling": _safe_float(row.get("attach_score_active_sibling")),
                "attach_score_archived_sibling": _safe_float(row.get("attach_score_archived_sibling")),
                "attach_score_temp_slot": _safe_float(row.get("attach_score_temp_slot")),
                "continuation_access_used": int(row.get("continuation_access_used", 0)),
                "continuation_access_success": int(row.get("continuation_access_success", 0)),
                "same_track_after_attach": int(row.get("same_track_after_attach", 0)),
                "same_prototype_after_attach": int(row.get("same_prototype_after_attach", 0)),
                "promotion_pending": int(row.get("promotion_pending_flag", 0)),
                "promotion_candidate_id": int(row.get("promotion_candidate_id", -1))
                if row.get("promotion_candidate_id") is not None
                else None,
                "promote_score_candidate": _safe_float(row.get("promote_score_candidate")),
                "promote_score_current_head": _safe_float(row.get("promote_score_current_head")),
                "support_count": int(row.get("promotion_support_count", 0)),
                "support_window_progress": int(row.get("promotion_window_progress", 0)),
                "promotion_decision": str(row.get("promotion_decision", "keep_head")),
                "promotion_delay_frames": int(row.get("promotion_delay_frames", -1))
                if row.get("promotion_delay_frames") is not None
                else None,
                "promotion_success": int(row.get("promotion_success", 0)),
                "promotion_regret_flag": int(row.get("promotion_regret_flag", 0)),
                "recovery_path_preserved": recovery_path_preserved,
                "head_kept": head_kept,
                "head_switched": int(row.get("head_switched", 0)),
                "temp_attach_used": temp_attach_used,
                "temp_attach_expired_without_promotion": int(
                    bool(row.get("temp_attach_expired", 0)) and not bool(row.get("promotion_success", 0))
                ),
                "same_track_final": int(row.get("matched_same_track", 0)),
                "same_prototype_final": int(row.get("matched_same_prototype", 0)),
                "pfr_delta": int(row.get("pfr_delta_if_any", 0)),
                "idsw_delta": int(row.get("idsw_delta_if_any", 0)),
                "failure_type": failure_type,
                "scenario_name": str(row.get("scenario_name", "")),
                "gap_length": int(row.get("gap_length", 0)),
                "reappear_frame": int(row.get("reappear_frame", 0)),
            }
        )
    return rows


def summarize_phase3d_stagea(audit_rows: list[dict[str, Any]], bundle: dict[str, Any]) -> dict[str, Any]:
    track_c_rows = [row for row in audit_rows if row["scenario_name"] == TRACK_C_NAME]
    matched_lineage_rows = [row for row in track_c_rows if row.get("matched_lineage_id") is not None]
    summary_lookup = {str(row["scenario_name"]): row for row in bundle["rows"]}
    track_c = summary_lookup.get(TRACK_C_NAME, {})
    track_a = summary_lookup.get(TRACK_A_NAME, {})
    return {
        "track_c": track_c,
        "track_a": track_a,
        "attach_success_rate_given_matched_lineage": _mean_bool(
            matched_lineage_rows,
            lambda item: item["attach_target_type"] != "none",
        ),
        "temp_attach_usage_rate": _mean_bool(track_c_rows, lambda item: bool(item["temp_attach_used"])),
        "temp_attach_to_promotion_rate": _ratio(
            sum(int(item["promotion_success"]) for item in track_c_rows if int(item["temp_attach_used"]) == 1),
            sum(int(item["temp_attach_used"]) for item in track_c_rows),
        ),
        "temp_attach_expiry_rate": _ratio(
            sum(int(item["temp_attach_expired_without_promotion"]) for item in track_c_rows),
            sum(int(item["temp_attach_used"]) for item in track_c_rows),
        ),
        "promotion_success_rate": _mean_bool(track_c_rows, lambda item: bool(item["promotion_success"])),
        "promotion_delay_mean": _mean_values(
            [
                float(item["promotion_delay_frames"])
                for item in track_c_rows
                if item["promotion_delay_frames"] is not None
            ]
        ),
        "promotion_regret_rate": _mean_bool(track_c_rows, lambda item: bool(item["promotion_regret_flag"])),
        "same_track_after_attach": _mean_bool(track_c_rows, lambda item: bool(item["same_track_after_attach"])),
        "same_prototype_after_attach": _mean_bool(track_c_rows, lambda item: bool(item["same_prototype_after_attach"])),
    }


def write_phase3d_design_notes(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Phase 3D Design Notes",
                "",
                "## Core Change",
                "",
                "Phase 3D Stage A separates lineage match, temporary recovery attach, and head promotion state.",
                "",
                "## Stage A Structures",
                "",
                "- dual score channels: `attach_score` vs `promote_score`",
                "- lineage-local temporary attach slot",
                "- deferred promotion state carried on the lineage backbone",
                "",
                "## Stage A Rules",
                "",
                "- matched lineage may attach to head, active sibling, archived sibling, or temp attach slot",
                "- attach acceptance is looser than promotion eligibility",
                "- temp attach does not count as sibling birth and does not change current head ownership",
                "- promotion state is logged, but Stage A keeps promotion conservative and mostly deferred",
                "",
                "## Intended Fix",
                "",
                "Stage A is only meant to prove that recovery attach can remain alive without immediately forcing head replacement.",
            ]
        ),
        encoding="utf-8",
    )


def write_phase3d_audit_summary(path: Path, summary: dict[str, Any]) -> None:
    track_c = summary["track_c"]
    track_a = summary["track_a"]
    lines = [
        "# Phase 3D Audit Summary",
        "",
        "## Track C",
        "",
        f"- `same_track = {float(track_c.get('same_track_reentry_recovery', 0.0)):.4f}`",
        f"- `same_track_after_concept = {float(track_c.get('same_track_after_concept_recovery', 0.0)):.4f}`",
        f"- `same_prototype = {float(track_c.get('same_prototype_reentry_recovery', 0.0)):.4f}`",
        f"- `same_lineage_prototype = {float(track_c.get('same_lineage_prototype_reentry_recovery', 0.0)):.4f}`",
        f"- `continuation_access = {float(track_c.get('continuation_bank_access_rate_given_concept_recovery', 0.0)):.4f}`",
        f"- `lineage_mismatch = {float(track_c.get('concept_recovered_but_lineage_mismatch_rate', 0.0)):.4f}`",
        f"- `PFR = {float(track_c.get('pfr', 0.0)):.4f}`",
        f"- `IDSW = {int(track_c.get('track_idsw', 0))}`",
        "",
        "## Attach / Promotion Diagnostics",
        "",
        f"- `attach_success_rate_given_matched_lineage = {float(summary['attach_success_rate_given_matched_lineage']):.4f}`",
        f"- `temp_attach_usage_rate = {float(summary['temp_attach_usage_rate']):.4f}`",
        f"- `temp_attach_to_promotion_rate = {float(summary['temp_attach_to_promotion_rate']):.4f}`",
        f"- `temp_attach_expiry_rate = {float(summary['temp_attach_expiry_rate']):.4f}`",
        f"- `promotion_success_rate = {float(summary['promotion_success_rate']):.4f}`",
        f"- `promotion_delay_mean = {float(summary['promotion_delay_mean']):.4f}`",
        f"- `promotion_regret_rate = {float(summary['promotion_regret_rate']):.4f}`",
        f"- `same_track_after_attach = {float(summary['same_track_after_attach']):.4f}`",
        f"- `same_prototype_after_attach = {float(summary['same_prototype_after_attach']):.4f}`",
        "",
        "## Track A Safety Check",
        "",
        f"- `U-Recall = {float(track_a.get('u_recall', 0.0)):.4f}`",
        f"- `same_prototype = {float(track_a.get('same_prototype_reentry_recovery', 0.0)):.4f}`",
        f"- `same_lineage_prototype = {float(track_a.get('same_lineage_prototype_reentry_recovery', 0.0)):.4f}`",
        f"- `PFR = {float(track_a.get('pfr', 0.0)):.4f}`",
        f"- `IDSW = {int(track_a.get('track_idsw', 0))}`",
        "",
        "## Interpretation",
        "",
        "Stage A is a structural validation stage. The decision criterion is whether temporary attach keeps the recovery path alive without forcing immediate head promotion.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_run_artifacts(
    *,
    run: dict[str, Any],
    summary_row: dict[str, Any],
    audit_rows: list[dict[str, Any]],
    output_dir: Path,
    config_path: str | Path,
    tracking_override: dict[str, Any],
    memory_override: dict[str, Any],
) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "key_cases").mkdir(parents=True, exist_ok=True)

    metrics_summary = dict(summary_row)
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "config_path": str(config_path),
                "tracking_override": tracking_override,
                "memory_override": memory_override,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_csv(output_dir / "event_audit.csv", audit_rows)

    visuals: list[dict[str, str]] = []
    case_rows = [row for row in audit_rows if row["scenario_name"] == run["scenario_name"]]
    if not case_rows:
        return visuals

    success_case = next((row for row in case_rows if int(row["same_track_final"]) == 1), case_rows[0])
    failure_case = next(
        (
            row
            for row in case_rows
            if int(row["same_track_final"]) == 0 and int(row["recovery_path_preserved"]) == 0
        ),
        case_rows[-1],
    )
    temp_case = next((row for row in case_rows if int(row["temp_attach_used"]) == 1), None)

    preview_case = temp_case or success_case or failure_case
    strip_path = output_dir / "run_preview_strip.png"
    strip_frames = _render_case_strip(run, preview_case, num_frames=10)
    Image.fromarray(strip_frames).save(strip_path)
    visuals.append({"file": str(strip_path), "kind": "preview_strip", "case": _case_label(preview_case)})

    gif_cases = [case for case in [success_case, failure_case, temp_case] if case is not None]
    gif_path = output_dir / "run_preview.gif"
    _save_case_gif(run, gif_cases, gif_path)
    visuals.append({"file": str(gif_path), "kind": "preview_gif", "case": ", ".join(_case_label(case) for case in gif_cases)})

    failure_gallery_path = output_dir / "failure_gallery.png"
    _render_failure_gallery(run, case_rows, failure_gallery_path)
    visuals.append({"file": str(failure_gallery_path), "kind": "failure_gallery", "case": "top_failures"})

    for index, case in enumerate(gif_cases[:3], start=1):
        case_path = output_dir / "key_cases" / f"case_{index:03d}.gif"
        _save_case_gif(run, [case], case_path)
        visuals.append({"file": str(case_path), "kind": "case_gif", "case": _case_label(case)})

    return visuals


def write_visual_manifest(path: Path, visuals: list[dict[str, str]]) -> None:
    lines = ["# Visual Manifest", ""]
    for item in visuals:
        lines.append(f"- `{item['file']}`: {item['kind']} | {item['case']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _case_label(row: dict[str, Any]) -> str:
    return (
        f"{row['scenario_name']} event={row['event_id']} "
        f"attach={row['attach_target_type']} sameT={row['same_track_final']} sameP={row['same_prototype_final']}"
    )


def _render_case_strip(run: dict[str, Any], event_row: dict[str, Any], *, num_frames: int) -> np.ndarray:
    frame_lookup, instance_lookup = _build_run_frame_maps(run)
    frame_ids = _window_frames(
        frame_lookup,
        center_frame=int(event_row["reappear_frame"]),
        left=max(2, num_frames // 4),
        right=max(5, num_frames - max(2, num_frames // 4) - 1),
    )
    ncols = len(frame_ids)
    fig, axes = plt.subplots(1, ncols, figsize=(2.4 * ncols, 3.4), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    for axis, frame_id in zip(axes, frame_ids):
        _draw_case_axis(
            axis,
            frame_lookup[frame_id],
            instance_lookup.get(frame_id, {}).get(int(event_row["gt_object_id"])),
            event_row,
        )
    return _figure_to_array(fig)


def _save_case_gif(run: dict[str, Any], cases: list[dict[str, Any]], path: Path) -> None:
    frame_lookup, instance_lookup = _build_run_frame_maps(run)
    images: list[Image.Image] = []
    for case in cases:
        frame_ids = _window_frames(
            frame_lookup,
            center_frame=int(case["reappear_frame"]),
            left=4,
            right=7,
        )
        for frame_id in frame_ids:
            fig, axis = plt.subplots(1, 1, figsize=(5.6, 5.2), constrained_layout=True)
            _draw_case_axis(
                axis,
                frame_lookup[frame_id],
                instance_lookup.get(frame_id, {}).get(int(case["gt_object_id"])),
                case,
            )
            image = Image.fromarray(_figure_to_array(fig))
            images.append(image)
    if not images:
        return
    images[0].save(path, save_all=True, append_images=images[1:], duration=180, loop=0)


def _render_failure_gallery(run: dict[str, Any], case_rows: list[dict[str, Any]], path: Path) -> None:
    frame_lookup, instance_lookup = _build_run_frame_maps(run)
    failures = [
        row
        for row in case_rows
        if int(row["same_track_final"]) == 0 or int(row["same_prototype_final"]) == 0
    ]
    failures.sort(key=lambda row: (int(row["same_prototype_final"]), -int(row["idsw_delta"]), -int(row["pfr_delta"])))
    selected = failures[:6] if failures else case_rows[:1]
    ncols = min(3, len(selected))
    nrows = int(np.ceil(len(selected) / max(1, ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    for axis in axes_array.ravel():
        axis.axis("off")
    for axis, case in zip(axes_array.ravel(), selected):
        frame_id = int(case["reappear_frame"])
        _draw_case_axis(
            axis,
            frame_lookup[frame_id],
            instance_lookup.get(frame_id, {}).get(int(case["gt_object_id"])),
            case,
        )
    Image.fromarray(_figure_to_array(fig)).save(path)


def _draw_case_axis(axis, frame_record, instance_row: dict[str, Any] | None, event_row: dict[str, Any]) -> None:
    frame = frame_record.frame
    if frame is None:
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame = _to_rgb(frame)
    axis.imshow(frame)
    axis.axis("off")
    for gt_box in frame_record.gt_boxes:
        axis.add_patch(Rectangle((gt_box[0], gt_box[1]), gt_box[2], gt_box[3], fill=False, lw=1.4, ec="#22c55e"))
    for pred_box in frame_record.predicted_boxes:
        axis.add_patch(Rectangle((pred_box[0], pred_box[1]), pred_box[2], pred_box[3], fill=False, lw=1.1, ec="#f59e0b"))
    for proposal in getattr(frame_record.objectness_output, "proposals", [])[:8]:
        axis.add_patch(Rectangle((proposal.box[0], proposal.box[1]), proposal.box[2], proposal.box[3], fill=False, lw=0.8, ec="#38bdf8", alpha=0.5))

    frame_id = int(frame_record.frame_index)
    current_track = None if instance_row is None else instance_row.get("track_id")
    current_proto = None if instance_row is None else instance_row.get("prototype_id")
    current_head = None if instance_row is None else instance_row.get("current_head_prototype_id")
    attach_target = "none" if instance_row is None else str(instance_row.get("recovery_attach_target", "none"))
    promotion_pending = 0 if instance_row is None else int(bool(instance_row.get("promotion_pending_flag", False)))
    same_track_now = int(current_track == event_row.get("old_track_id")) if current_track is not None else 0
    same_proto_now = int(current_proto == event_row.get("old_prototype_id")) if current_proto is not None else 0
    text_lines = [
        f"f={frame_id} obj={event_row['gt_object_id']}",
        f"L={event_row.get('matched_lineage_id')} head={current_head}",
        f"attach={attach_target} promo_pending={promotion_pending}",
        f"track={current_track} proto={current_proto}",
        f"sameT={same_track_now} sameP={same_proto_now}",
    ]
    axis.text(
        4,
        6,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 3},
    )


def _build_run_frame_maps(run: dict[str, Any]) -> tuple[dict[int, Any], dict[int, dict[int, dict[str, Any]]]]:
    frame_lookup = {int(record.frame_index): record for record in run["result"].frame_records}
    instance_lookup = {
        int(record.frame_index): _build_frame_instance_map(record, iou_threshold=0.5)["instances"]
        for record in run["result"].frame_records
    }
    return frame_lookup, instance_lookup


def _window_frames(frame_lookup: dict[int, Any], *, center_frame: int, left: int, right: int) -> list[int]:
    available = sorted(frame_lookup.keys())
    selected = [frame_id for frame_id in available if (center_frame - left) <= frame_id <= (center_frame + right)]
    if selected:
        return selected
    return available[: min(8, len(available))]


def _figure_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    array = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return array


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=2)
    if array.ndim == 3 and array.shape[2] == 1:
        return np.repeat(array, 3, axis=2)
    return array


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        value = stripped
    return float(value)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _mean_bool(rows: list[dict[str, Any]], fn) -> float:
    if not rows:
        return 0.0
    return float(sum(int(bool(fn(row))) for row in rows) / len(rows))


def _mean_values(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
