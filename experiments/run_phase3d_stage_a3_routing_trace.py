"""Phase 3D Stage A.3: frame-local assignment routing repair trace."""

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

from datasets import SyntheticStreamGenerator, load_synth_dataset_config  # noqa: E402
from experiments.phase3d_utils import (  # noqa: E402
    default_phase3d_stagea_memory_override,
    default_phase3d_stagea_tracking_override,
)
from experiments.scenario_presets import build_phase3_track_scenarios  # noqa: E402
from nops_owr.encoder import MinimalSpikeEncoder  # noqa: E402
from nops_owr.memory import MinimalPrototypeMemory  # noqa: E402
from nops_owr.objectness import MinimalObjectnessField  # noqa: E402
from nops_owr.tracking import MinimalTemporalIdentityTracker  # noqa: E402


TRACK_C_NAME = "track_c_long_horizon"
TARGET_EVENT_ID = 6
TARGET_GT_OBJECT_ID = 2
TARGET_FRAME = 990
WINDOW_LEFT = 12
WINDOW_RIGHT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3D Stage A.3 routing trace.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--output-dir", default="results/phase3d")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _figure_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    array = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return array


def _draw_box(axis, box: tuple[int, int, int, int] | None, *, color: str, label: str | None = None, lw: float = 1.6) -> None:
    if box is None:
        return
    x1, y1, x2, y2 = [int(v) for v in box]
    axis.add_patch(Rectangle((x1, y1), max(1, x2 - x1), max(1, y2 - y1), fill=False, ec=color, lw=lw))
    if label:
        axis.text(
            x1,
            max(0, y1 - 4),
            label,
            color=color,
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 1},
        )


def _iou(box_a: tuple[int, int, int, int] | None, box_b: tuple[int, int, int, int] | None) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _load_track_c_sequence(config_path: Path, *, seed: int):
    base_config = load_synth_dataset_config(config_path)
    scenario_map = {s["name"]: s["config"] for s in build_phase3_track_scenarios(base_config)}
    return SyntheticStreamGenerator(scenario_map[TRACK_C_NAME], seed=seed).generate_sequence(0)


def _load_target_metadata() -> dict[str, Any]:
    path = Path("results/phase3d/phase3d_event_audit.csv")
    metadata = {
        "event_id": TARGET_EVENT_ID,
        "gt_object_id": TARGET_GT_OBJECT_ID,
        "target_frame": TARGET_FRAME,
        "old_track_id": None,
        "old_lineage_id": None,
        "old_prototype_id": None,
    }
    if not path.exists():
        return metadata
    for row in _read_csv(path):
        if int(row.get("event_id", -1)) != TARGET_EVENT_ID:
            continue
        metadata["old_track_id"] = None if row.get("old_track_id") in (None, "", "None") else int(row["old_track_id"])
        metadata["old_lineage_id"] = None if row.get("old_lineage_id") in (None, "", "None") else int(row["old_lineage_id"])
        metadata["old_prototype_id"] = None if row.get("old_prototype_id") in (None, "", "None") else int(row["old_prototype_id"])
        break
    return metadata


def _gt_box(frame_sample, gt_object_id: int) -> tuple[int, int, int, int] | None:
    for instance_id, box in zip(frame_sample.instance_ids, frame_sample.boxes):
        if int(instance_id) == int(gt_object_id):
            return tuple(int(v) for v in box)
    return None


def _build_frame_surface(tracking_output, memory_output) -> dict[str, int]:
    return {
        "active_count": int(len(tracking_output.active_tracks)),
        "dormant_count": int(len(tracking_output.dormant_tracks)),
        "ghost_count": int(len(tracking_output.ghost_tracks)),
        "retired_count": int(len(tracking_output.retired_tracks)),
        "continuation_bank_count": int(getattr(memory_output, "continuation_bank_count", 0)),
        "recovery_anchor_count": int(getattr(memory_output, "recovery_anchor_count", 0)),
    }


def _run_trace(
    *,
    config_path: Path,
    seed: int,
    run_label: str,
    tracking_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sequence = _load_track_c_sequence(config_path, seed=seed)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    tracking_config = dict(payload["tracking"])
    tracking_config.update(default_phase3d_stagea_tracking_override())
    tracking_config.update(
        {
            "routing_recovery_max_distance": 0.70,
            "routing_recovery_min_confidence": 0.30,
            "routing_active_claim_override_margin": 0.20,
            "routing_topk": 3,
        }
    )
    if tracking_patch:
        tracking_config.update(tracking_patch)

    memory_config = dict(payload["memory"])
    memory_config.update(default_phase3d_stagea_memory_override())

    encoder = MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    objectness = MinimalObjectnessField(**payload["field"])
    tracker = MinimalTemporalIdentityTracker(**tracking_config)
    memory = MinimalPrototypeMemory(**memory_config)

    target_metadata = _load_target_metadata()
    rows: list[dict[str, Any]] = []
    frame_debug: dict[int, dict[str, Any]] = {}
    prev_memory_output = None

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

        gt_box = _gt_box(current_frame, TARGET_GT_OBJECT_ID)
        frame_surface = _build_frame_surface(tracking_output, memory_output)
        assignment_map: dict[int, tuple[Any, Any]] = {}
        for tracking_assignment, prototype_assignment in zip(tracking_output.assignments, memory_output.assignments):
            assignment_map[int(tracking_assignment.proposal_index)] = (tracking_assignment, prototype_assignment)

        frame_rows: list[dict[str, Any]] = []
        for routing_row in tracking_output.routing_debug_rows:
            proposal_id = int(routing_row["proposal_id"])
            proposal_box = tuple(int(v) for v in routing_row["proposal_box"])
            tracking_assignment, prototype_assignment = assignment_map.get(proposal_id, (None, None))
            final_track_id = None if tracking_assignment is None else int(tracking_assignment.track_id)
            final_lineage_id = None if tracking_assignment is None else tracking_assignment.final_lineage_id
            final_prototype_id = None if prototype_assignment is None else int(prototype_assignment.prototype_id)
            final_source = "none" if tracking_assignment is None else str(tracking_assignment.final_assignment_source)
            resurrection_candidate_seen = 0
            restore_attempted_after_reroute = 0
            same_track_after_reroute = 0
            if tracking_assignment is not None:
                resurrection_candidate_seen = int(
                    int(tracking_assignment.candidate_pool_size) > 0
                    or int(tracking_assignment.anchor_candidate_pool_size) > 0
                    or int(tracking_assignment.continuation_bank_size) > 0
                    or int(tracking_assignment.live_candidate_pool_size) > 0
                )
                restore_attempted_after_reroute = int(
                    bool(tracking_assignment.resurrection_attempted)
                    or bool(tracking_assignment.restore_attempted_from_anchor)
                    or bool(tracking_assignment.continuation_attempted)
                )
                same_track_after_reroute = int(
                    target_metadata["old_track_id"] is not None
                    and int(tracking_assignment.track_id) == int(target_metadata["old_track_id"])
                )
            frame_rows.append(
                {
                    "run_label": run_label,
                    "sequence_id": 0,
                    "event_id": TARGET_EVENT_ID,
                    "gt_object_id": TARGET_GT_OBJECT_ID,
                    "old_track_id": target_metadata["old_track_id"],
                    "old_lineage_id": target_metadata["old_lineage_id"],
                    "old_prototype_id": target_metadata["old_prototype_id"],
                    "frame_id": int(current_frame.frame_index),
                    "frame_local_offset": int(current_frame.frame_index - TARGET_FRAME),
                    "proposal_id": proposal_id,
                    "proposal_box": proposal_box,
                    "proposal_score": float(routing_row["proposal_score"]),
                    "proposal_iou_to_gt": float(_iou(proposal_box, gt_box)),
                    "proposal_proto_hint": routing_row.get("proposal_proto_hint"),
                    "proposal_lineage_hint_topk": _json_compact(routing_row["proposal_lineage_hint_topk"]),
                    "topk_active_candidates": _json_compact(routing_row["active_candidates_topk"]),
                    "tentative_active_track_id": routing_row.get("preempting_active_track_id"),
                    "tentative_lineage_id": routing_row.get("tentative_lineage_id"),
                    "tentative_assignment_source": routing_row.get("tentative_assignment_source"),
                    "active_claim_confidence": routing_row.get("active_claim_confidence"),
                    "topk_recovery_candidates": _json_compact(routing_row["recovery_candidates_topk"]),
                    "best_recovery_lineage_id": routing_row.get("best_recovery_lineage_id"),
                    "best_recovery_source_type": routing_row.get("best_recovery_source_type"),
                    "recovery_claim_confidence": routing_row.get("recovery_claim_confidence"),
                    "cross_lineage_preemption_flag": int(bool(routing_row["cross_lineage_preemption_flag"])),
                    "preemption_reason": routing_row.get("preemption_reason"),
                    "routing_margin": routing_row.get("routing_margin"),
                    "routing_arbitration_triggered": int(bool(routing_row["routing_arbitration_triggered"])),
                    "was_rerouted": int(bool(routing_row["was_rerouted"])),
                    "preempting_active_track_id": routing_row.get("preempting_active_track_id"),
                    "preempting_active_lineage_id": routing_row.get("preempting_active_lineage_id"),
                    "final_assignment_source": final_source,
                    "final_lineage_id": final_lineage_id,
                    "final_track_id": final_track_id,
                    "final_prototype_id": final_prototype_id,
                    "resurrection_attempted": 0 if tracking_assignment is None else int(bool(tracking_assignment.resurrection_attempted)),
                    "restore_attempted_after_reroute": restore_attempted_after_reroute,
                    "resurrection_candidate_seen": resurrection_candidate_seen,
                    "candidate_pool_size": 0 if tracking_assignment is None else int(tracking_assignment.candidate_pool_size),
                    "anchor_candidate_pool_size": 0 if tracking_assignment is None else int(tracking_assignment.anchor_candidate_pool_size),
                    "continuation_bank_size_runtime": 0 if tracking_assignment is None else int(tracking_assignment.continuation_bank_size),
                    "anchor_success": 0 if tracking_assignment is None else int(bool(tracking_assignment.anchor_success)),
                    "continuation_success": 0 if tracking_assignment is None else int(bool(tracking_assignment.continuation_success)),
                    "same_track_after_reroute": same_track_after_reroute,
                    **frame_surface,
                    "gt_box": gt_box,
                }
            )

        if gt_box is not None and frame_rows:
            best_row = max(frame_rows, key=lambda item: float(item["proposal_iou_to_gt"]))
            for row in frame_rows:
                row["is_target_proposal"] = int(row is best_row and float(row["proposal_iou_to_gt"]) > 0.0)
        else:
            for row in frame_rows:
                row["is_target_proposal"] = 0

        include_frame = TARGET_FRAME - WINDOW_LEFT <= int(current_frame.frame_index) <= TARGET_FRAME + WINDOW_RIGHT
        for row in frame_rows:
            if include_frame or int(row["cross_lineage_preemption_flag"]) == 1 or int(row["was_rerouted"]) == 1:
                rows.append(row)

        if include_frame:
            frame_debug[int(current_frame.frame_index)] = {
                "frame": current_frame.frame.copy(),
                "gt_box": gt_box,
                "rows": [dict(item) for item in frame_rows],
            }

        prev_memory_output = memory_output

    return {
        "label": run_label,
        "rows": rows,
        "frame_debug": frame_debug,
    }


def _target_row(rows: list[dict[str, Any]], *, frame_id: int = TARGET_FRAME) -> dict[str, Any] | None:
    frame_rows = [row for row in rows if int(row["frame_id"]) == int(frame_id)]
    target_rows = [row for row in frame_rows if int(row["is_target_proposal"]) == 1]
    if target_rows:
        return target_rows[0]
    if not frame_rows:
        return None
    return max(frame_rows, key=lambda item: float(item["proposal_iou_to_gt"]))


def _write_design_notes(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Phase 3D Stage A.3 Design Notes",
                "",
                "- `active_match` is treated as a tentative claim for routing audit.",
                "- cross-lineage preemption is detected when the tentative active lineage conflicts with a recovery lineage that still has visible recovery surface.",
                "- rerouted proposals are converted into `rerouted_to_resurrection` candidates before memory + resurrection consume them.",
                "- no anchor redesign, temp-attach redesign, or promotion logic is changed in Stage A.3.",
            ]
        ),
        encoding="utf-8",
    )


def _write_routing_audit(
    path: Path,
    *,
    baseline_target: dict[str, Any] | None,
    minimal_target: dict[str, Any] | None,
    forced_target_summary: dict[str, Any],
    forced_visibility_summary: dict[str, Any],
) -> None:
    lines = [
        "# Phase 3D Stage A.3 Routing Audit",
        "",
        "## Target Event",
        "",
        f"- `event_id = {TARGET_EVENT_ID}`",
        f"- `target_gt_object_id = {TARGET_GT_OBJECT_ID}`",
        f"- `target_frame = {TARGET_FRAME}`",
        "",
        "## Baseline Audit",
        "",
    ]
    if baseline_target is None:
        lines.append("- no target proposal row captured")
    else:
        lines.extend(
            [
                f"- tentative active lineage: `{baseline_target['tentative_lineage_id']}`",
                f"- best recovery lineage: `{baseline_target['best_recovery_lineage_id']}`",
                f"- `cross_lineage_preemption_flag = {int(baseline_target['cross_lineage_preemption_flag'])}`",
                f"- final assignment source: `{baseline_target['final_assignment_source']}`",
                f"- final lineage: `{baseline_target['final_lineage_id']}`",
            ]
        )
    lines.extend(["", "## Minimal Routing Policy", ""])
    if minimal_target is None:
        lines.append("- no target proposal row captured")
    else:
        lines.extend(
            [
                f"- `routing_arbitration_triggered = {int(minimal_target['routing_arbitration_triggered'])}`",
                f"- `was_rerouted = {int(minimal_target['was_rerouted'])}`",
                f"- final assignment source: `{minimal_target['final_assignment_source']}`",
                f"- final lineage: `{minimal_target['final_lineage_id']}`",
                f"- `restore_attempted_after_reroute = {int(minimal_target['restore_attempted_after_reroute'])}`",
                f"- `resurrection_candidate_seen = {int(minimal_target['resurrection_candidate_seen'])}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Forced Probes",
            "",
            f"- forced target reroute final lineage: `{forced_target_summary['final_assignment_lineage']}`",
            f"- forced target reroute final source: `{forced_target_summary['final_assignment_source']}`",
            f"- forced target reroute `restore_attempted_after_reroute = {int(forced_target_summary['restore_attempted_after_reroute'])}`",
            f"- forced target reroute `resurrection_candidate_seen = {int(forced_target_summary['resurrection_candidate_seen'])}`",
            f"- forced visibility total flagged proposals: `{forced_visibility_summary['cross_lineage_flagged_proposals']}`",
            f"- forced visibility rerouted proposals: `{forced_visibility_summary['rerouted_proposals']}`",
            f"- forced visibility proposals that reached resurrection source: `{forced_visibility_summary['proposals_reaching_resurrection']}`",
            "",
            "## Reading",
            "",
            "1. Stage A.3 treats cross-lineage active matches as tentative instead of irreversible.",
            "2. Reroute now exposes those proposals to the existing resurrection consumer path.",
            "3. The next gate is no longer surface evaporation; it is whether the rerouted proposal lands on the intended recovery lineage consistently enough to replace preemptive active claims.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_recommendation(path: Path, *, minimal_target: dict[str, Any] | None, forced_target_summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 3D Stage A.3 Recommendation",
        "",
        "Do not enter Stage B.",
        "",
        "Stage A.3 should be judged on routing only:",
        "- whether cross-lineage preemption is now visible as a tentative state",
        "- whether reroute exposes the proposal to resurrection consume",
        "- whether final assignment can move off the original preempting active lineage",
        "",
    ]
    if minimal_target is not None:
        lines.append(
            f"Current minimal policy target result: `was_rerouted={int(minimal_target['was_rerouted'])}`, `final_source={minimal_target['final_assignment_source']}`, `final_lineage={minimal_target['final_lineage_id']}`."
        )
    lines.append(
        f"Forced reroute probe: `restore_attempted_after_reroute={int(forced_target_summary['restore_attempted_after_reroute'])}`, `final_lineage={forced_target_summary['final_assignment_lineage']}`."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_frame990_strip(path: Path, *, run_payload: dict[str, Any]) -> None:
    frame_debug = run_payload["frame_debug"]
    frame_ids = list(range(TARGET_FRAME - 4, TARGET_FRAME + 5))
    ncols = len(frame_ids)
    fig, axes = plt.subplots(1, ncols, figsize=(2.5 * ncols, 4.2), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    for axis, frame_id in zip(axes, frame_ids):
        debug = frame_debug.get(frame_id)
        axis.axis("off")
        if debug is None:
            continue
        axis.imshow(debug["frame"])
        _draw_box(axis, debug["gt_box"], color="#22c55e", label="GT")
        target_rows = [row for row in debug["rows"] if int(row["is_target_proposal"]) == 1]
        target_row = target_rows[0] if target_rows else None
        if target_row is not None:
            _draw_box(axis, tuple(target_row["proposal_box"]), color="#06b6d4", label="proposal")
            text = [
                f"f={frame_id}",
                f"tentL={target_row['tentative_lineage_id']}",
                f"recL={target_row['best_recovery_lineage_id']}",
                f"preempt={target_row['cross_lineage_preemption_flag']}",
                f"reroute={target_row['was_rerouted']}",
                f"final={target_row['final_assignment_source']}",
            ]
        else:
            text = [f"f={frame_id}", "no target proposal"]
        axis.text(
            4,
            6,
            "\n".join(text),
            va="top",
            ha="left",
            fontsize=7,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
        )
    Image.fromarray(_figure_to_array(fig)).save(path)


def _render_preemption_gallery(path: Path, *, rows: list[dict[str, Any]], frame_debug: dict[int, dict[str, Any]]) -> None:
    flagged = [row for row in rows if int(row["cross_lineage_preemption_flag"]) == 1]
    flagged.sort(key=lambda item: (-float(item["proposal_iou_to_gt"]), int(item["frame_id"])))
    selected = flagged[:6]
    if not selected:
        return
    ncols = min(3, len(selected))
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    for axis in axes_array.ravel():
        axis.axis("off")
    for axis, row in zip(axes_array.ravel(), selected):
        debug = frame_debug.get(int(row["frame_id"]))
        if debug is None:
            continue
        axis.imshow(debug["frame"])
        _draw_box(axis, debug["gt_box"], color="#22c55e", label="GT")
        _draw_box(axis, tuple(row["proposal_box"]), color="#06b6d4", label="proposal")
        axis.text(
            4,
            6,
            "\n".join(
                [
                    f"f={row['frame_id']} p={row['proposal_id']}",
                    f"activeL={row['tentative_lineage_id']}",
                    f"recoveryL={row['best_recovery_lineage_id']}",
                    f"rerouted={row['was_rerouted']}",
                    f"final={row['final_assignment_source']}",
                ]
            ),
            va="top",
            ha="left",
            fontsize=7,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
        )
    Image.fromarray(_figure_to_array(fig)).save(path)


def _render_routing_timeline(path: Path, *, baseline_rows: list[dict[str, Any]], minimal_rows: list[dict[str, Any]]) -> None:
    baseline_target = [row for row in baseline_rows if int(row["is_target_proposal"]) == 1]
    minimal_target = [row for row in minimal_rows if int(row["is_target_proposal"]) == 1]
    frames = sorted({int(row["frame_id"]) for row in baseline_target} | {int(row["frame_id"]) for row in minimal_target})
    if not frames:
        return
    baseline_by_frame = {int(row["frame_id"]): row for row in baseline_target}
    minimal_by_frame = {int(row["frame_id"]): row for row in minimal_target}

    fig, axes = plt.subplots(2, 1, figsize=(10.4, 5.8), sharex=True, constrained_layout=True)
    axes[0].plot(
        frames,
        [baseline_by_frame.get(frame, {}).get("tentative_lineage_id", np.nan) for frame in frames],
        marker="o",
        label="baseline tentative active lineage",
        color="#2563eb",
    )
    axes[0].plot(
        frames,
        [baseline_by_frame.get(frame, {}).get("best_recovery_lineage_id", np.nan) for frame in frames],
        marker="s",
        label="baseline best recovery lineage",
        color="#f59e0b",
    )
    axes[0].plot(
        frames,
        [minimal_by_frame.get(frame, {}).get("final_lineage_id", np.nan) for frame in frames],
        marker="^",
        label="minimal final lineage",
        color="#10b981",
    )
    axes[0].set_ylabel("lineage id")
    axes[0].legend(loc="upper right")

    axes[1].step(
        frames,
        [int(bool(minimal_by_frame.get(frame, {}).get("was_rerouted", 0))) for frame in frames],
        where="mid",
        label="was rerouted",
        color="#8b5cf6",
    )
    axes[1].step(
        frames,
        [
            1
            if str(minimal_by_frame.get(frame, {}).get("final_assignment_source", "")).startswith("resurrection_")
            else 0
            for frame in frames
        ],
        where="mid",
        label="reaches resurrection",
        color="#ef4444",
    )
    axes[1].set_ylabel("routing state")
    axes[1].set_xlabel("frame")
    axes[1].legend(loc="upper right")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _forced_target_summary(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "target_event_id": TARGET_EVENT_ID,
            "target_frame": TARGET_FRAME,
            "proposal_seen": 0,
            "was_rerouted": 0,
            "restore_attempted_after_reroute": 0,
            "resurrection_candidate_seen": 0,
            "final_assignment_lineage": None,
            "final_assignment_source": "none",
            "same_track_after_reroute": 0,
        }
    return {
        "target_event_id": TARGET_EVENT_ID,
        "target_frame": TARGET_FRAME,
        "proposal_seen": 1,
        "was_rerouted": int(row["was_rerouted"]),
        "restore_attempted_after_reroute": int(row["restore_attempted_after_reroute"]),
        "resurrection_candidate_seen": int(row["resurrection_candidate_seen"]),
        "final_assignment_lineage": row["final_lineage_id"],
        "final_assignment_source": row["final_assignment_source"],
        "same_track_after_reroute": int(row["same_track_after_reroute"]),
    }


def _forced_visibility_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flagged = [row for row in rows if int(row["cross_lineage_preemption_flag"]) == 1]
    return {
        "cross_lineage_flagged_proposals": len(flagged),
        "rerouted_proposals": sum(int(row["was_rerouted"]) for row in flagged),
        "proposals_reaching_resurrection": sum(
            int(str(row["final_assignment_source"]).startswith("resurrection_")) for row in flagged
        ),
        "proposals_with_recovery_surface_seen": sum(int(row["resurrection_candidate_seen"]) for row in flagged),
        "proposals_finalizing_on_best_recovery_lineage": sum(
            int(
                row["final_lineage_id"] is not None
                and row["best_recovery_lineage_id"] is not None
                and int(row["final_lineage_id"]) == int(row["best_recovery_lineage_id"])
            )
            for row in flagged
        ),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _run_trace(
        config_path=config_path,
        seed=args.seed,
        run_label="baseline_audit",
        tracking_patch={"enable_phase3d_routing_repair": False},
    )
    minimal = _run_trace(
        config_path=config_path,
        seed=args.seed,
        run_label="minimal_policy",
        tracking_patch={"enable_phase3d_routing_repair": True},
    )
    forced_target = _run_trace(
        config_path=config_path,
        seed=args.seed,
        run_label="forced_target_reroute",
        tracking_patch={
            "enable_phase3d_routing_repair": True,
            "debug_force_reroute_frame": TARGET_FRAME,
        },
    )
    forced_visibility = _run_trace(
        config_path=config_path,
        seed=args.seed,
        run_label="forced_visibility",
        tracking_patch={
            "enable_phase3d_routing_repair": True,
            "debug_force_visibility_for_all_cross_lineage": True,
        },
    )

    all_rows = baseline["rows"] + minimal["rows"] + forced_target["rows"] + forced_visibility["rows"]
    _write_csv(output_dir / "phase3d_stagea3_routing_trace.csv", all_rows)

    forced_target_row = _target_row(forced_target["rows"])
    forced_target_summary = _forced_target_summary(forced_target_row)
    forced_visibility_summary = _forced_visibility_summary(forced_visibility["rows"])
    (output_dir / "phase3d_stagea3_forced_reroute_summary.json").write_text(
        json.dumps(forced_target_summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "phase3d_stagea3_forced_visibility_summary.json").write_text(
        json.dumps(forced_visibility_summary, indent=2),
        encoding="utf-8",
    )

    baseline_target = _target_row(baseline["rows"])
    minimal_target = _target_row(minimal["rows"])
    _write_design_notes(output_dir / "phase3d_stagea3_design_notes.md")
    _write_routing_audit(
        output_dir / "phase3d_stagea3_routing_audit.md",
        baseline_target=baseline_target,
        minimal_target=minimal_target,
        forced_target_summary=forced_target_summary,
        forced_visibility_summary=forced_visibility_summary,
    )
    _write_recommendation(
        output_dir / "phase3d_stagea3_recommendation.md",
        minimal_target=minimal_target,
        forced_target_summary=forced_target_summary,
    )

    _render_frame990_strip(output_dir / "frame990_routing_strip.png", run_payload=forced_target)
    _render_preemption_gallery(
        output_dir / "cross_lineage_preemption_gallery.png",
        rows=forced_visibility["rows"],
        frame_debug=forced_visibility["frame_debug"],
    )
    _render_routing_timeline(
        output_dir / "routing_decision_timeline.png",
        baseline_rows=baseline["rows"],
        minimal_rows=minimal["rows"],
    )

    print(f"saved_trace={output_dir / 'phase3d_stagea3_routing_trace.csv'}")
    print(f"saved_audit={output_dir / 'phase3d_stagea3_routing_audit.md'}")
    print(f"saved_design={output_dir / 'phase3d_stagea3_design_notes.md'}")
    print(f"saved_forced_target={output_dir / 'phase3d_stagea3_forced_reroute_summary.json'}")
    print(f"saved_forced_visibility={output_dir / 'phase3d_stagea3_forced_visibility_summary.json'}")
    print(f"saved_strip={output_dir / 'frame990_routing_strip.png'}")
    print(f"saved_gallery={output_dir / 'cross_lineage_preemption_gallery.png'}")
    print(f"saved_timeline={output_dir / 'routing_decision_timeline.png'}")
    print(f"saved_recommendation={output_dir / 'phase3d_stagea3_recommendation.md'}")


if __name__ == "__main__":
    main()
