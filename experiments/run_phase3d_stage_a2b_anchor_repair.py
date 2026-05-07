"""Phase 3D Stage A.2b: target-lineage aligned audit + recovery anchor probe."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator, load_synth_dataset_config
from experiments.phase3d_utils import (
    default_phase3d_stagea_memory_override,
    default_phase3d_stagea_tracking_override,
)
from experiments.scenario_presets import build_phase3_track_scenarios
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.memory import MinimalPrototypeMemory
from nops_owr.objectness import MinimalObjectnessField
from nops_owr.tracking import MinimalTemporalIdentityTracker


TRACK_C_NAME = "track_c_long_horizon"
WINDOW_LEFT = 20
WINDOW_RIGHT = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3D Stage A.2b anchor repair audit.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--stagea1-coverage", default="results/phase3d/phase3d_stagea1_branch_coverage.csv")
    parser.add_argument("--output-dir", default="results/phase3d")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_track_c_sequence(config_path: Path, *, seed: int):
    base_config = load_synth_dataset_config(config_path)
    scenario_map = {s["name"]: s["config"] for s in build_phase3_track_scenarios(base_config)}
    return SyntheticStreamGenerator(scenario_map[TRACK_C_NAME], seed=seed).generate_sequence(0)


def _select_target_event(path: Path) -> dict[str, Any]:
    rows = _read_csv_rows(path)
    targets = [
        row
        for row in rows
        if row.get("scenario_name") == TRACK_C_NAME
        and row.get("run_label") == "baseline"
        and str(row.get("matched_lineage_established", "0")) == "1"
    ]
    if not targets:
        raise RuntimeError("No matched-lineage baseline row found in Stage A.1 coverage.")
    targets.sort(key=lambda row: int(row.get("reappear_frame", "0")))
    return targets[0]


def _gt_box_for_instance(frame_sample, target_gt_object_id: int) -> tuple[int, int, int, int] | None:
    for instance_id, box in zip(frame_sample.instance_ids, frame_sample.boxes):
        if int(instance_id) == int(target_gt_object_id):
            return tuple(int(value) for value in box)
    return None


def _observed_lineage_id(tracking_assignment, prototype_assignment) -> int | None:
    for value in (
        getattr(prototype_assignment, "lineage_id", None),
        getattr(prototype_assignment, "matched_lineage_id", None),
        getattr(tracking_assignment, "linked_lineage_id", None),
        getattr(tracking_assignment, "pre_memory_linked_lineage_id", None),
        getattr(tracking_assignment, "prototype_hint_lineage_id", None),
    ):
        if value is None:
            continue
        if int(value) < 0:
            continue
        return int(value)
    return None


def _figure_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    array = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return array


def _draw_box(axis, box: tuple[int, int, int, int] | None, *, color: str, label: str | None = None, lw: float = 1.6) -> None:
    if box is None:
        return
    x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    axis.add_patch(Rectangle((x1, y1), max(1, x2 - x1), max(1, y2 - y1), fill=False, lw=lw, ec=color))
    if label:
        axis.text(x1, max(0, y1 - 4), label, color=color, fontsize=7, bbox={"facecolor": "black", "alpha": 0.55, "pad": 1})


def _lineage_surface_summary(
    memory_output,
    tracking_output,
    *,
    lineage_id: int,
    frame_index: int,
) -> dict[str, Any]:
    state_counts: dict[str, int] = {"active": 0, "dormant": 0, "ghost": 0, "retired": 0}
    state_track_ids: dict[str, list[int]] = {"active": [], "dormant": [], "ghost": [], "retired": []}
    for state_name, tracks in (
        ("active", tracking_output.active_tracks),
        ("dormant", tracking_output.dormant_tracks),
        ("ghost", tracking_output.ghost_tracks),
        ("retired", tracking_output.retired_tracks),
    ):
        for track in tracks:
            track_lineage_id = getattr(track, "lineage_id", None)
            if track_lineage_id is None or int(track_lineage_id) != int(lineage_id):
                continue
            state_counts[state_name] += 1
            state_track_ids[state_name].append(int(track.track_id))

    temp_slot = getattr(memory_output, "temp_attach_lookup", {}).get(int(lineage_id))
    anchors = getattr(memory_output, "recovery_anchor_lookup", {}).get(int(lineage_id), [])
    prototype_rows = [
        row for row in getattr(memory_output, "prototype_lineage_rows", []) if int(row.get("lineage_id", -1)) == int(lineage_id)
    ]
    continuation_bank_size = 0
    if prototype_rows:
        continuation_bank_size = max(int(row.get("continuation_bank_size", 0)) for row in prototype_rows)
    else:
        continuation_bank_size = len(getattr(memory_output, "continuation_lineage_lookup", {}).get(int(lineage_id), []))
    recovery_surface_evaporated = int(
        state_counts["active"] == 0
        and state_counts["dormant"] == 0
        and state_counts["ghost"] == 0
        and continuation_bank_size == 0
        and len(anchors) == 0
        and state_counts["retired"] > 0
    )
    return {
        "frame_id": int(frame_index),
        "lineage_id": int(lineage_id),
        "active_count": int(state_counts["active"]),
        "dormant_count": int(state_counts["dormant"]),
        "ghost_count": int(state_counts["ghost"]),
        "retired_count": int(state_counts["retired"]),
        "active_track_ids": "|".join(map(str, state_track_ids["active"])),
        "dormant_track_ids": "|".join(map(str, state_track_ids["dormant"])),
        "ghost_track_ids": "|".join(map(str, state_track_ids["ghost"])),
        "retired_track_ids": "|".join(map(str, state_track_ids["retired"])),
        "continuation_bank_size": int(continuation_bank_size),
        "recovery_identity_anchor_count": int(len(anchors)),
        "temp_attach_id": None if temp_slot is None else int(temp_slot.get("temp_attach_id", -1)),
        "temp_attach_alive": int(bool(temp_slot) and not bool(temp_slot.get("expired", False))),
        "temp_attach_expired": int(False if temp_slot is None else bool(temp_slot.get("expired", False))),
        "temp_attach_source_track_id": None if temp_slot is None else temp_slot.get("source_track_id"),
        "recovery_surface_evaporated": int(recovery_surface_evaporated),
        "anchor_uids": "|".join(str(anchor.get("anchor_uid", "")) for anchor in anchors),
        "anchor_track_ids": "|".join(str(anchor.get("old_track_id", "")) for anchor in anchors),
    }


def _collect_window_rows(
    *,
    run_label: str,
    frame_index: int,
    target_event: dict[str, Any],
    surface: dict[str, Any],
    tracking_output,
    memory_output,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_lineage_id = int(target_event["matched_lineage_id"])
    target_gt_object_id = int(target_event["gt_object_id"])
    local_offset = int(frame_index - int(target_event["reappear_frame"]))
    assignments = list(zip(tracking_output.assignments, memory_output.assignments))

    for tracking_assignment, prototype_assignment in assignments:
        observed_lineage_id = _observed_lineage_id(tracking_assignment, prototype_assignment)
        attach_written = int(bool(getattr(prototype_assignment, "attach_state_written", False)))
        is_target_lineage_row = int(observed_lineage_id == target_lineage_id)
        is_cross_lineage_pollution = int(
            attach_written == 1 and observed_lineage_id is not None and int(observed_lineage_id) != int(target_lineage_id)
        )
        if not (is_target_lineage_row or is_cross_lineage_pollution or int(frame_index) == int(target_event["reappear_frame"])):
            continue
        rows.append(
            {
                "run_label": str(run_label),
                "target_event_id": int(target_event["event_id"]),
                "target_gt_object_id": int(target_gt_object_id),
                "target_lineage_id": int(target_lineage_id),
                "observed_lineage_id": observed_lineage_id,
                "is_target_lineage_row": int(is_target_lineage_row),
                "is_cross_lineage_pollution": int(is_cross_lineage_pollution),
                "frame_id": int(frame_index),
                "frame_local_offset": int(local_offset),
                "is_same_target_window": 1,
                "assignment_track_id": int(tracking_assignment.track_id),
                "assignment_source": str(tracking_assignment.assignment_source),
                "prototype_id": int(prototype_assignment.prototype_id),
                "recovery_attach_target": str(getattr(prototype_assignment, "recovery_attach_target", "none")),
                "attach_state_written": int(bool(getattr(prototype_assignment, "attach_state_written", False))),
                "attach_state_consumed_by_tracker": int(bool(tracking_assignment.attach_state_consumed_by_tracker)),
                "attach_state_consumed_by_continuation": int(bool(tracking_assignment.attach_state_consumed_by_continuation)),
                "restore_attempted_from_attach": int(bool(tracking_assignment.restore_attempted_from_attach)),
                "restore_attempted_from_anchor": int(bool(getattr(tracking_assignment, "restore_attempted_from_anchor", False))),
                "anchor_candidate_pool_size": int(getattr(tracking_assignment, "anchor_candidate_pool_size", 0)),
                "candidate_pool_size": int(tracking_assignment.candidate_pool_size),
                "live_candidate_pool_size": int(tracking_assignment.live_candidate_pool_size),
                "continuation_bank_size_runtime": int(tracking_assignment.continuation_bank_size),
                "best_candidate_state": tracking_assignment.best_candidate_state,
                "best_candidate_gap": tracking_assignment.best_candidate_gap,
                "best_anchor_uid": getattr(tracking_assignment, "best_anchor_uid", None),
                "best_anchor_gap": getattr(tracking_assignment, "best_anchor_gap", None),
                "anchor_success": int(bool(getattr(tracking_assignment, "anchor_success", False))),
                "concept_recovered": int(bool(tracking_assignment.concept_recovered)),
                **surface,
            }
        )

    if not any(int(row["is_target_lineage_row"]) == 1 for row in rows):
        rows.append(
            {
                "run_label": str(run_label),
                "target_event_id": int(target_event["event_id"]),
                "target_gt_object_id": int(target_gt_object_id),
                "target_lineage_id": int(target_lineage_id),
                "observed_lineage_id": None,
                "is_target_lineage_row": 0,
                "is_cross_lineage_pollution": 0,
                "frame_id": int(frame_index),
                "frame_local_offset": int(local_offset),
                "is_same_target_window": 1,
                "assignment_track_id": None,
                "assignment_source": "none",
                "prototype_id": None,
                "recovery_attach_target": "none",
                "attach_state_written": 0,
                "attach_state_consumed_by_tracker": 0,
                "attach_state_consumed_by_continuation": 0,
                "restore_attempted_from_attach": 0,
                "restore_attempted_from_anchor": 0,
                "anchor_candidate_pool_size": 0,
                "candidate_pool_size": 0,
                "live_candidate_pool_size": 0,
                "continuation_bank_size_runtime": 0,
                "best_candidate_state": None,
                "best_candidate_gap": None,
                "best_anchor_uid": None,
                "best_anchor_gap": None,
                "anchor_success": 0,
                "concept_recovered": 0,
                **surface,
            }
        )
    return rows


def _run_trace(
    *,
    config_path: Path,
    seed: int,
    target_event: dict[str, Any],
    run_label: str,
    tracking_patch: dict[str, Any] | None = None,
    memory_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sequence = _load_track_c_sequence(config_path, seed=seed)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target_frame = int(target_event["reappear_frame"])
    target_lineage_id = int(target_event["matched_lineage_id"])
    window_start = max(1, target_frame - WINDOW_LEFT)
    window_end = target_frame + WINDOW_RIGHT

    tracking_config = dict(payload["tracking"])
    tracking_config.update(default_phase3d_stagea_tracking_override())
    if tracking_patch:
        tracking_config.update(tracking_patch)

    memory_config = dict(payload["memory"])
    memory_config.update(default_phase3d_stagea_memory_override())
    if memory_patch:
        memory_config.update(memory_patch)

    encoder = MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    objectness = MinimalObjectnessField(**payload["field"])
    tracker = MinimalTemporalIdentityTracker(**tracking_config)
    memory = MinimalPrototypeMemory(**memory_config)

    audit_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    anchor_lifecycle_rows: list[dict[str, Any]] = []
    frame_debug: dict[int, dict[str, Any]] = {}

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

        surface = _lineage_surface_summary(
            memory_output,
            tracking_output,
            lineage_id=target_lineage_id,
            frame_index=current_frame.frame_index,
        )
        timeline_rows.append({"run_label": str(run_label), **surface})
        for row in memory_output.recovery_anchor_rows:
            if int(row.get("lineage_id", -1)) == target_lineage_id:
                anchor_rows.append({"run_label": str(run_label), **dict(row)})
        for row in memory_output.recovery_anchor_lifecycle_rows:
            if int(row.get("lineage_id", -1)) == target_lineage_id:
                anchor_lifecycle_rows.append({"run_label": str(run_label), **dict(row)})

        if window_start <= int(current_frame.frame_index) <= int(window_end):
            audit_rows.extend(
                _collect_window_rows(
                    run_label=run_label,
                    frame_index=current_frame.frame_index,
                    target_event=target_event,
                    surface=surface,
                    tracking_output=tracking_output,
                    memory_output=memory_output,
                )
            )
            frame_debug[int(current_frame.frame_index)] = {
                "frame": current_frame.frame.copy(),
                "gt_box": _gt_box_for_instance(current_frame, int(target_event["gt_object_id"])),
                "surface": dict(surface),
            }
        if int(current_frame.frame_index) >= int(window_end):
            break

    return {
        "audit_rows": audit_rows,
        "timeline_rows": timeline_rows,
        "anchor_rows": anchor_rows,
        "anchor_lifecycle_rows": anchor_lifecycle_rows,
        "frame_debug": frame_debug,
    }


def _render_surface_timeline(path: Path, timeline_rows: list[dict[str, Any]]) -> None:
    baseline = [row for row in timeline_rows if row["run_label"] == "baseline"]
    frames = [int(row["frame_id"]) for row in baseline]
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 6.4), sharex=True)
    axes[0].plot(frames, [int(row["active_count"]) for row in baseline], label="active", color="#22c55e")
    axes[0].plot(frames, [int(row["dormant_count"]) for row in baseline], label="dormant", color="#3b82f6")
    axes[0].plot(frames, [int(row["ghost_count"]) for row in baseline], label="ghost", color="#a855f7")
    axes[0].plot(frames, [int(row["retired_count"]) for row in baseline], label="retired", color="#ef4444")
    axes[0].set_ylabel("track count")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Target lineage recovery surface")

    axes[1].plot(frames, [int(row["continuation_bank_size"]) for row in baseline], label="continuation bank", color="#0ea5e9")
    axes[1].plot(frames, [int(row["recovery_identity_anchor_count"]) for row in baseline], label="anchor count", color="#f59e0b")
    axes[1].plot(frames, [int(row["temp_attach_alive"]) for row in baseline], label="temp attach alive", color="#8b5cf6")
    axes[1].plot(frames, [int(row["recovery_surface_evaporated"]) for row in baseline], label="surface evaporated", color="#111827")
    axes[1].set_ylabel("surface state")
    axes[1].set_xlabel("frame")
    axes[1].legend(loc="upper right")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _render_core_event_strip(path: Path, frame_debug: dict[int, dict[str, Any]], target_event: dict[str, Any]) -> None:
    frame_ids = sorted(frame_debug.keys())
    if not frame_ids:
        return
    ncols = int(np.ceil(len(frame_ids) / 2))
    fig, axes = plt.subplots(2, ncols, figsize=(2.2 * ncols, 6.8), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(2, ncols)
    for axis in axes_array.ravel():
        axis.axis("off")
    for axis, frame_id in zip(axes_array.ravel(), frame_ids):
        debug = frame_debug[frame_id]
        axis.imshow(debug["frame"])
        axis.axis("off")
        _draw_box(axis, debug["gt_box"], color="#22c55e", label=f"GT obj={target_event['gt_object_id']}")
        surface = debug["surface"]
        lines = [
            f"f={frame_id} L={target_event['matched_lineage_id']}",
            f"A/D/G/R={surface['active_count']}/{surface['dormant_count']}/{surface['ghost_count']}/{surface['retired_count']}",
            f"bank={surface['continuation_bank_size']} anchor={surface['recovery_identity_anchor_count']}",
            f"temp_alive={surface['temp_attach_alive']} expired={surface['temp_attach_expired']}",
        ]
        axis.text(
            4,
            6,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=7,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
        )
    Image.fromarray(_figure_to_array(fig)).save(path)


def _render_pollution_gallery(path: Path, frame_debug: dict[int, dict[str, Any]], polluted_rows: list[dict[str, Any]], target_event: dict[str, Any]) -> None:
    if not polluted_rows:
        return
    selected = polluted_rows[:4]
    frame_id = int(selected[0]["frame_id"])
    debug = frame_debug.get(frame_id)
    if debug is None:
        return
    ncols = len(selected)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.8), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    for axis, row in zip(axes, selected):
        axis.imshow(debug["frame"])
        axis.axis("off")
        _draw_box(axis, debug["gt_box"], color="#22c55e", label="target GT")
        lines = [
            f"targetL={target_event['matched_lineage_id']}",
            f"observedL={row['observed_lineage_id']}",
            f"track={row['assignment_track_id']}",
            f"src={row['assignment_source']}",
            "cross-lineage pollution",
        ]
        axis.text(
            4,
            6,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 3},
        )
    Image.fromarray(_figure_to_array(fig)).save(path)


def _build_forced_summary(*, baseline_rows: list[dict[str, Any]], forced_rows: list[dict[str, Any]], target_event: dict[str, Any]) -> dict[str, Any]:
    def _max_int(rows: list[dict[str, Any]], key: str) -> int:
        values = [int(row.get(key, 0) or 0) for row in rows]
        return max(values) if values else 0

    summary = {
        "target_event_id": int(target_event["event_id"]),
        "target_gt_object_id": int(target_event["gt_object_id"]),
        "target_lineage_id": int(target_event["matched_lineage_id"]),
        "baseline_candidate_pool_size_max": _max_int(baseline_rows, "candidate_pool_size"),
        "baseline_anchor_candidate_pool_size_max": _max_int(baseline_rows, "anchor_candidate_pool_size"),
        "baseline_restore_attempted_from_anchor_max": _max_int(baseline_rows, "restore_attempted_from_anchor"),
        "baseline_attach_state_consumed_by_tracker_max": _max_int(baseline_rows, "attach_state_consumed_by_tracker"),
        "forced_candidate_pool_size_max": _max_int(forced_rows, "candidate_pool_size"),
        "forced_anchor_candidate_pool_size_max": _max_int(forced_rows, "anchor_candidate_pool_size"),
        "forced_restore_attempted_from_anchor_max": _max_int(forced_rows, "restore_attempted_from_anchor"),
        "forced_attach_state_consumed_by_tracker_max": _max_int(forced_rows, "attach_state_consumed_by_tracker"),
        "forced_anchor_success_max": _max_int(forced_rows, "anchor_success"),
    }
    pulled_off_zero: list[str] = []
    for base_key, forced_key, label in (
        ("baseline_candidate_pool_size_max", "forced_candidate_pool_size_max", "candidate_pool_size"),
        ("baseline_anchor_candidate_pool_size_max", "forced_anchor_candidate_pool_size_max", "anchor_candidate_pool_size"),
        ("baseline_restore_attempted_from_anchor_max", "forced_restore_attempted_from_anchor_max", "restore_attempted_from_anchor"),
        ("baseline_attach_state_consumed_by_tracker_max", "forced_attach_state_consumed_by_tracker_max", "attach_state_consumed_by_tracker"),
    ):
        if int(summary[base_key]) == 0 and int(summary[forced_key]) > 0:
            pulled_off_zero.append(label)
    summary["pulled_off_zero_metrics"] = pulled_off_zero
    return summary


def _write_summary_md(path: Path, *, target_event: dict[str, Any], baseline_rows: list[dict[str, Any]], forced_rows: list[dict[str, Any]], timeline_rows: list[dict[str, Any]]) -> None:
    target_frame = int(target_event["reappear_frame"])
    baseline_target_frame_rows = [row for row in baseline_rows if int(row["frame_id"]) == target_frame]
    target_lineage_rows = [row for row in baseline_target_frame_rows if int(row["is_target_lineage_row"]) == 1]
    pollution_rows = [row for row in baseline_target_frame_rows if int(row["is_cross_lineage_pollution"]) == 1]
    surface = next((row for row in timeline_rows if row["run_label"] == "baseline" and int(row["frame_id"]) == target_frame), None)
    lines = [
        "# Phase 3D Stage A.2b Target-Aligned Summary",
        "",
        "## Target Event",
        "",
        f"- `event_id = {target_event['event_id']}`",
        f"- `target_gt_object_id = {target_event['gt_object_id']}`",
        f"- `target_lineage_id = {target_event['matched_lineage_id']}`",
        f"- `target_frame = {target_frame}`",
        "",
        "## Target-Lineage Alignment",
        "",
        f"- baseline target-lineage rows at target frame: `{len(target_lineage_rows)}`",
        f"- baseline cross-lineage pollution rows at target frame: `{len(pollution_rows)}`",
        f"- observed pollution lineages: `{', '.join(str(row['observed_lineage_id']) for row in pollution_rows) if pollution_rows else 'none'}`",
        "",
    ]
    if surface is not None:
        lines.extend(
            [
                "## Recovery Surface At Target Frame",
                "",
                f"- `active/dormant/ghost/retired = {surface['active_count']}/{surface['dormant_count']}/{surface['ghost_count']}/{surface['retired_count']}`",
                f"- `continuation_bank_size = {surface['continuation_bank_size']}`",
                f"- `recovery_identity_anchor_count = {surface['recovery_identity_anchor_count']}`",
                f"- `temp_attach_alive = {surface['temp_attach_alive']}`",
                f"- `temp_attach_expired = {surface['temp_attach_expired']}`",
                f"- `recovery_surface_evaporated = {surface['recovery_surface_evaporated']}`",
                "",
            ]
        )
    forced_anchor_attempt_rows = [row for row in forced_rows if int(row["restore_attempted_from_anchor"]) == 1]
    lines.extend(
        [
            "## Forced Anchor Consume Probe",
            "",
            f"- forced rows with `restore_attempted_from_anchor = 1`: `{len(forced_anchor_attempt_rows)}`",
            f"- forced rows with `anchor_success = 1`: `{sum(int(row['anchor_success']) for row in forced_rows)}`",
            "",
            "## Main Reading",
            "",
            "1. Target-aligned audit no longer treats other lineages' attach writes as target-lineage recovery evidence.",
            "2. The target lineage state is read frame-locally from its own active/dormant/ghost/retired, continuation-bank, temp-attach, and recovery-anchor counts.",
            "3. Temporary attach remains an observation state only. The explicit recovery source added in Stage A.2b is the lineage-level `RecoveryIdentityAnchor`.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_design_notes(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Phase 3D Stage A.2b Design Notes",
                "",
                "## Split",
                "",
                "- `TemporaryObservationAttach`: current observation state near a lineage/prototype; not a legal old-track restore source.",
                "- `RecoveryIdentityAnchor`: lineage-level minimal old-identity source that survives retire/archive and can be consumed by resurrection.",
                "",
                "## Consumer Order",
                "",
                "1. same-lineage dormant/ghost tracks",
                "2. same-lineage continuation bank",
                "3. same-lineage recovery identity anchor",
            ]
        ),
        encoding="utf-8",
    )


def _write_anchor_trace(path: Path, *, anchor_rows: list[dict[str, Any]], anchor_lifecycle_rows: list[dict[str, Any]], timeline_rows: list[dict[str, Any]], target_event: dict[str, Any]) -> None:
    baseline_anchor_rows = [row for row in anchor_rows if row["run_label"] == "baseline"]
    baseline_lifecycle_rows = [row for row in anchor_lifecycle_rows if row["run_label"] == "baseline"]
    evaporation_row = next((row for row in timeline_rows if row["run_label"] == "baseline" and int(row["recovery_surface_evaporated"]) == 1), None)
    first_write = next((row for row in baseline_anchor_rows if row.get("event_type") in {"write", "refresh"}), None)
    lines = [
        "# Stage A.2b Anchor Creation Trace",
        "",
        f"- target lineage: `{target_event['matched_lineage_id']}`",
        f"- target frame: `{target_event['reappear_frame']}`",
        f"- first anchor write/refresh row: `{first_write}`",
        f"- first recovery-surface evaporated frame: `{None if evaporation_row is None else evaporation_row['frame_id']}`",
        "",
    ]
    for row in baseline_anchor_rows[-12:]:
        lines.append(
            f"- f={row['frame_index']} track={row['old_track_id']} type={row['event_type']} reason={row['event_reason']} state={row.get('anchor_state', '')} priority={row.get('restore_priority', '')}"
        )
    lines.extend(["", "## Recent Anchor Lifecycle", ""])
    for row in baseline_lifecycle_rows[-12:]:
        lines.append(
            f"- f={row['frame_index']} uid={row['anchor_uid']} alive={row['is_alive']} state={row['anchor_state']} age={row['age_since_last_seen']} reason={row['drop_reason']}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_consume_trace_md(path: Path, *, target_event: dict[str, Any], forced_summary: dict[str, Any], surface_row: dict[str, Any] | None) -> None:
    lines = [
        "# Phase 3D Stage A.2b Consume Trace",
        "",
        f"- target event: `{target_event['event_id']}`",
        f"- target lineage: `{target_event['matched_lineage_id']}`",
        f"- target frame: `{target_event['reappear_frame']}`",
        "",
    ]
    if surface_row is not None:
        lines.extend(
            [
                "## Baseline Surface",
                "",
                f"- `active/dormant/ghost/retired = {surface_row['active_count']}/{surface_row['dormant_count']}/{surface_row['ghost_count']}/{surface_row['retired_count']}`",
                f"- `continuation_bank_size = {surface_row['continuation_bank_size']}`",
                f"- `recovery_identity_anchor_count = {surface_row['recovery_identity_anchor_count']}`",
                f"- `temp_attach_alive = {surface_row['temp_attach_alive']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Forced Anchor Consume Probe",
            "",
            f"- `forced_candidate_pool_size_max = {forced_summary['forced_candidate_pool_size_max']}`",
            f"- `forced_anchor_candidate_pool_size_max = {forced_summary['forced_anchor_candidate_pool_size_max']}`",
            f"- `forced_restore_attempted_from_anchor_max = {forced_summary['forced_restore_attempted_from_anchor_max']}`",
            f"- `forced_attach_state_consumed_by_tracker_max = {forced_summary['forced_attach_state_consumed_by_tracker_max']}`",
            f"- `forced_anchor_success_max = {forced_summary['forced_anchor_success_max']}`",
            f"- `pulled_off_zero_metrics = {forced_summary['pulled_off_zero_metrics']}`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_recommendation(path: Path, *, forced_summary: dict[str, Any]) -> None:
    if forced_summary["forced_restore_attempted_from_anchor_max"] > 0:
        verdict = "Forced anchor consume reaches the consumer. Next step is baseline recovery-surface preservation, not Stage B."
    else:
        verdict = "Forced anchor consume still does not move the target-lineage consume path. The remaining break is consumer binding, not parameterization."
    path.write_text(
        "\n".join(
            [
                "# Phase 3D Stage A.2b Recommendation",
                "",
                verdict,
                "",
                "Do not enter Stage B until the target-lineage anchor is present on the baseline recovery surface and the consumer attempts restore from it without debug forcing.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_event = _select_target_event(Path(args.stagea1_coverage))
    baseline = _run_trace(
        config_path=config_path,
        seed=args.seed,
        target_event=target_event,
        run_label="baseline",
    )
    forced = _run_trace(
        config_path=config_path,
        seed=args.seed,
        target_event=target_event,
        run_label="forced_anchor_consume",
        tracking_patch={"debug_force_anchor_consume": True},
    )

    audit_rows = baseline["audit_rows"] + forced["audit_rows"]
    timeline_rows = baseline["timeline_rows"] + forced["timeline_rows"]
    anchor_rows = baseline["anchor_rows"] + forced["anchor_rows"]
    anchor_lifecycle_rows = baseline["anchor_lifecycle_rows"] + forced["anchor_lifecycle_rows"]

    _write_csv(output_dir / "phase3d_stagea2b_target_aligned_audit.csv", audit_rows)
    forced_summary = _build_forced_summary(
        baseline_rows=[row for row in audit_rows if row["run_label"] == "baseline"],
        forced_rows=[row for row in audit_rows if row["run_label"] == "forced_anchor_consume"],
        target_event=target_event,
    )
    (output_dir / "phase3d_stagea2b_forced_anchor_consume_summary.json").write_text(
        json.dumps(forced_summary, indent=2),
        encoding="utf-8",
    )

    target_surface_row = next(
        (
            row
            for row in timeline_rows
            if row["run_label"] == "baseline" and int(row["frame_id"]) == int(target_event["reappear_frame"])
        ),
        None,
    )
    _write_summary_md(
        output_dir / "phase3d_stagea2b_target_aligned_summary.md",
        target_event=target_event,
        baseline_rows=[row for row in audit_rows if row["run_label"] == "baseline"],
        forced_rows=[row for row in audit_rows if row["run_label"] == "forced_anchor_consume"],
        timeline_rows=timeline_rows,
    )
    _write_design_notes(output_dir / "phase3d_stagea2b_design_notes.md")
    _write_anchor_trace(
        output_dir / "stagea2b_anchor_creation_trace.md",
        anchor_rows=anchor_rows,
        anchor_lifecycle_rows=anchor_lifecycle_rows,
        timeline_rows=timeline_rows,
        target_event=target_event,
    )
    _write_consume_trace_md(
        output_dir / "phase3d_stagea2b_consume_trace.md",
        target_event=target_event,
        forced_summary=forced_summary,
        surface_row=target_surface_row,
    )
    _write_recommendation(output_dir / "phase3d_stagea2b_recommendation.md", forced_summary=forced_summary)

    _render_surface_timeline(output_dir / "target_lineage_recovery_surface_timeline.png", timeline_rows)
    _render_core_event_strip(output_dir / "target_lineage_core_event_strip.png", baseline["frame_debug"], target_event)
    pollution_rows = [row for row in audit_rows if row["run_label"] == "baseline" and int(row["is_cross_lineage_pollution"]) == 1]
    _render_pollution_gallery(
        output_dir / "cross_lineage_pollution_gallery.png",
        baseline["frame_debug"],
        pollution_rows,
        target_event,
    )


if __name__ == "__main__":
    main()
