"""Refresh Phase 3D visuals with core-recovery filtering."""

from __future__ import annotations

import argparse
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

from experiments.phase3d_utils import _build_run_frame_maps, _figure_to_array, _to_rgb  # noqa: E402
from experiments.run_phase3d_stage_a1_wiring_audit import _evaluate_track_c  # noqa: E402


VERY_SMALL_IOU = 0.05
GT_COLOR = "#22c55e"
HEAD_COLOR = "#2563eb"
ATTACH_COLOR = "#06b6d4"
TEMP_ATTACH_COLOR = "#a855f7"
PROMOTION_COLOR = "#ffffff"
OLD_ID_COLOR = "#facc15"
LINEAGE_CANDIDATE_COLOR = "#38bdf8"
GT_OVERLAP_COLOR = "#86efac"
FULL_DEBUG_PROPOSAL_COLOR = "#94a3b8"
FULL_DEBUG_PROPOSAL_EDGE = "#f59e0b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh Phase 3D visualization outputs.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--output-dir", default="results/phase3d")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _xywh_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
    bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def _target_gt_box(frame_record, gt_object_id: int) -> tuple[float, float, float, float] | None:
    for instance_id, gt_box in zip(getattr(frame_record, "instance_ids", []), getattr(frame_record, "gt_boxes", [])):
        if int(instance_id) == int(gt_object_id):
            return tuple(float(value) for value in gt_box)
    return None


def _box_xywh(box: tuple[Any, Any, Any, Any]) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in box)


def _build_frame_context(
    frame_record,
    instance_row: dict[str, Any] | None,
    event_row: dict[str, Any],
) -> dict[str, Any]:
    gt_object_id = event_row.get("gt_object_id")
    target_gt_box = None if gt_object_id is None else _target_gt_box(frame_record, int(gt_object_id))
    raw_proposals = list(getattr(frame_record.objectness_output, "proposals", []))
    raw_count = len(raw_proposals)

    tracking_assignments = list(getattr(frame_record.tracking_output, "assignments", []))
    memory_assignments = list(getattr(frame_record.memory_output, "assignments", []))
    track_by_id = {int(assignment.track_id): assignment for assignment in tracking_assignments}
    track_by_proposal = {int(assignment.proposal_index): assignment for assignment in tracking_assignments}
    memory_by_track = {int(assignment.track_id): assignment for assignment in memory_assignments}
    memory_by_proposal: dict[int, Any] = {}
    for assignment in memory_assignments:
        track_assignment = track_by_id.get(int(assignment.track_id))
        if track_assignment is not None:
            memory_by_proposal[int(track_assignment.proposal_index)] = assignment

    target_lineage_id = (
        None
        if event_row.get("matched_lineage_id") is None and event_row.get("old_lineage_id") is None
        else int(event_row.get("matched_lineage_id") if event_row.get("matched_lineage_id") is not None else event_row.get("old_lineage_id"))
    )
    target_track_id = None if instance_row is None or instance_row.get("track_id") is None else int(instance_row["track_id"])
    current_head_id = None
    if instance_row is not None and instance_row.get("current_head_prototype_id") is not None:
        current_head_id = int(instance_row["current_head_prototype_id"])
    elif event_row.get("current_head_prototype_id") is not None:
        current_head_id = int(event_row["current_head_prototype_id"])
    attach_target_type = "none" if instance_row is None else str(instance_row.get("recovery_attach_target", "none"))
    attach_target_id = None if instance_row is None or instance_row.get("recovery_attach_target_id") is None else int(instance_row["recovery_attach_target_id"])
    promotion_candidate_id = None
    if instance_row is not None and instance_row.get("promotion_candidate_id") is not None:
        promotion_candidate_id = int(instance_row["promotion_candidate_id"])

    relevant_indices: set[int] = set()
    matched_lineage_indices: set[int] = set()
    attach_indices: set[int] = set()
    head_indices: set[int] = set()
    old_identity_indices: set[int] = set()
    promotion_indices: set[int] = set()
    gt_overlap_indices: set[int] = set()

    for proposal_index, proposal in enumerate(raw_proposals):
        proposal_box = _box_xywh(proposal.box)
        track_assignment = track_by_proposal.get(proposal_index)
        memory_assignment = memory_by_proposal.get(proposal_index)
        lineage_match = False
        if target_lineage_id is not None:
            if memory_assignment is not None and int(memory_assignment.lineage_id) == int(target_lineage_id):
                lineage_match = True
            elif track_assignment is not None:
                linked_lineage_id = getattr(track_assignment, "linked_lineage_id", None)
                pre_memory_lineage_id = getattr(track_assignment, "pre_memory_linked_lineage_id", None)
                lineage_match = (
                    linked_lineage_id is not None and int(linked_lineage_id) == int(target_lineage_id)
                ) or (
                    pre_memory_lineage_id is not None and int(pre_memory_lineage_id) == int(target_lineage_id)
                )
        if lineage_match:
            matched_lineage_indices.add(proposal_index)
            relevant_indices.add(proposal_index)

        if target_gt_box is not None and _xywh_iou(target_gt_box, proposal_box) > VERY_SMALL_IOU:
            gt_overlap_indices.add(proposal_index)
            relevant_indices.add(proposal_index)

        if track_assignment is not None and event_row.get("old_track_id") is not None and int(track_assignment.track_id) == int(event_row["old_track_id"]):
            old_identity_indices.add(proposal_index)
            relevant_indices.add(proposal_index)
        if memory_assignment is not None and event_row.get("old_prototype_id") is not None and int(memory_assignment.prototype_id) == int(event_row["old_prototype_id"]):
            old_identity_indices.add(proposal_index)
            relevant_indices.add(proposal_index)

        if memory_assignment is not None and current_head_id is not None and int(memory_assignment.prototype_id) == int(current_head_id):
            head_indices.add(proposal_index)
            relevant_indices.add(proposal_index)

        if memory_assignment is not None and promotion_candidate_id is not None and int(memory_assignment.prototype_id) == int(promotion_candidate_id):
            promotion_indices.add(proposal_index)
            relevant_indices.add(proposal_index)

        if attach_target_type == "temporary_attach_slot" and target_track_id is not None and track_assignment is not None and int(track_assignment.track_id) == int(target_track_id):
            attach_indices.add(proposal_index)
            relevant_indices.add(proposal_index)
        elif attach_target_id is not None and memory_assignment is not None and int(memory_assignment.prototype_id) == int(attach_target_id):
            attach_indices.add(proposal_index)
            relevant_indices.add(proposal_index)

    if target_track_id is not None:
        target_track_assignment = track_by_id.get(target_track_id)
        if target_track_assignment is not None:
            relevant_indices.add(int(target_track_assignment.proposal_index))
            if attach_target_type == "temporary_attach_slot":
                attach_indices.add(int(target_track_assignment.proposal_index))

    hidden_count = max(0, raw_count - len(relevant_indices))
    matched_lineage_id = None if instance_row is None else instance_row.get("matched_lineage_id")
    attach_branch_entered = False if instance_row is None else bool(instance_row.get("attach_branch_entered", False))
    is_non_core_frame = matched_lineage_id is None and not attach_branch_entered
    is_core_frame = not is_non_core_frame and (target_gt_box is not None or len(relevant_indices) > 0)

    return {
        "target_gt_box": target_gt_box,
        "raw_proposals": raw_proposals,
        "relevant_indices": relevant_indices,
        "matched_lineage_indices": matched_lineage_indices,
        "attach_indices": attach_indices,
        "head_indices": head_indices,
        "old_identity_indices": old_identity_indices,
        "promotion_indices": promotion_indices,
        "gt_overlap_indices": gt_overlap_indices,
        "raw_proposal_count": raw_count,
        "relevant_candidate_count": len(relevant_indices),
        "matched_lineage_candidate_count": len(matched_lineage_indices),
        "unrelated_proposals_hidden_count": hidden_count,
        "is_non_core_frame": is_non_core_frame,
        "is_core_frame": is_core_frame,
        "attach_target_type": attach_target_type,
        "current_head_id": current_head_id,
    }


def _draw_box(axis, box, *, edgecolor: str, linewidth: float, linestyle: str = "-", alpha: float = 1.0, label: str | None = None) -> None:
    x, y, w, h = _box_xywh(box)
    axis.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            lw=linewidth,
            ec=edgecolor,
            linestyle=linestyle,
            alpha=alpha,
        )
    )
    if label:
        axis.text(
            x,
            max(2.0, y - 4.0),
            label,
            fontsize=7,
            color=edgecolor,
            bbox={"facecolor": "black", "alpha": 0.7, "pad": 1.5},
        )


def _draw_frame(
    axis,
    frame_record,
    instance_row: dict[str, Any] | None,
    event_row: dict[str, Any],
    *,
    view_mode: str,
) -> None:
    frame = frame_record.frame
    if frame is None:
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
    axis.imshow(_to_rgb(frame))
    axis.axis("off")

    context = _build_frame_context(frame_record, instance_row, event_row)
    raw_proposals = context["raw_proposals"]
    relevant_indices = context["relevant_indices"]

    if view_mode == "full_debug_view":
        for proposal_index, proposal in enumerate(raw_proposals):
            if proposal_index in relevant_indices:
                continue
            _draw_box(
                axis,
                proposal.box,
                edgecolor=FULL_DEBUG_PROPOSAL_EDGE,
                linewidth=0.9,
                linestyle="-",
                alpha=0.35,
            )
            _draw_box(
                axis,
                proposal.box,
                edgecolor=FULL_DEBUG_PROPOSAL_COLOR,
                linewidth=2.2,
                linestyle="-",
                alpha=0.15,
            )

    if context["target_gt_box"] is not None:
        _draw_box(axis, context["target_gt_box"], edgecolor=GT_COLOR, linewidth=2.0, label="GT")

    for proposal_index in sorted(relevant_indices):
        proposal = raw_proposals[proposal_index]
        box = proposal.box
        if proposal_index in context["matched_lineage_indices"]:
            _draw_box(axis, box, edgecolor=LINEAGE_CANDIDATE_COLOR, linewidth=1.1, linestyle="--", label="L")
        if proposal_index in context["old_identity_indices"]:
            _draw_box(axis, box, edgecolor=OLD_ID_COLOR, linewidth=1.4, linestyle="--", label="old")
        if proposal_index in context["head_indices"]:
            _draw_box(axis, box, edgecolor=HEAD_COLOR, linewidth=2.0, label="head")
        if proposal_index in context["attach_indices"]:
            attach_color = TEMP_ATTACH_COLOR if context["attach_target_type"] == "temporary_attach_slot" else ATTACH_COLOR
            attach_label = "temp" if context["attach_target_type"] == "temporary_attach_slot" else "attach"
            _draw_box(axis, box, edgecolor=attach_color, linewidth=2.2, label=attach_label)
        if proposal_index in context["promotion_indices"]:
            _draw_box(axis, box, edgecolor=PROMOTION_COLOR, linewidth=1.8, linestyle=":", label="promo")
        if (
            proposal_index in context["gt_overlap_indices"]
            and proposal_index not in context["matched_lineage_indices"]
            and proposal_index not in context["attach_indices"]
            and proposal_index not in context["head_indices"]
        ):
            _draw_box(axis, box, edgecolor=GT_OVERLAP_COLOR, linewidth=1.0, linestyle="--", label="gt-ov")

    frame_id = int(frame_record.frame_index)
    matched_lineage_id = None if instance_row is None else instance_row.get("matched_lineage_id")
    attach_branch = False if instance_row is None else bool(instance_row.get("attach_branch_entered", False))
    promotion_pending = False if instance_row is None else bool(instance_row.get("promotion_pending_flag", False))
    info_lines = [
        f"{view_mode} f={frame_id} obj={event_row.get('gt_object_id', '?')}",
        f"matchedL={matched_lineage_id} attachBranch={'yes' if attach_branch else 'no'}",
        f"head={context['current_head_id']} attach={context['attach_target_type']} promoPending={'yes' if promotion_pending else 'no'}",
        f"raw={context['raw_proposal_count']} relevant={context['relevant_candidate_count']}",
        f"matchedL_cand={context['matched_lineage_candidate_count']} hidden={context['unrelated_proposals_hidden_count'] if view_mode == 'core_recovery_view' else 0}",
    ]
    if view_mode == "full_debug_view" and context["is_non_core_frame"]:
        info_lines.append("upstream false-proposal frame, not core attach case")

    axis.text(
        4,
        6,
        "\n".join(info_lines),
        va="top",
        ha="left",
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.72, "pad": 3},
    )


def _select_core_event(event_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        row
        for row in event_rows
        if row.get("matched_lineage_id") is not None and int(bool(row.get("attach_state_written", 0))) == 1
    ]
    temp_attach_candidates = [row for row in candidates if str(row.get("recovery_attach_target", "none")) == "temporary_attach_slot"]
    if temp_attach_candidates:
        return temp_attach_candidates[0]
    if candidates:
        return candidates[0]
    raise RuntimeError("No matched-lineage attach_state_written event found for core visualization.")


def _select_core_frames(run_payload: dict[str, Any], event_row: dict[str, Any], *, desired_frames: int = 20) -> list[int]:
    frame_lookup, instance_lookup = _build_run_frame_maps(run_payload["run"])
    center = int(event_row["reappear_frame"])
    gt_object_id = int(event_row["gt_object_id"])
    candidate_ids = [frame_id for frame_id in sorted(frame_lookup) if center <= frame_id <= center + 120]
    core_frames: list[int] = []
    for frame_id in candidate_ids:
        instance_row = instance_lookup.get(frame_id, {}).get(gt_object_id)
        if instance_row is None:
            continue
        context = _build_frame_context(frame_lookup[frame_id], instance_row, event_row)
        matched_lineage_id = instance_row.get("matched_lineage_id")
        attach_branch = bool(instance_row.get("attach_branch_entered", False))
        if context["is_core_frame"] and (matched_lineage_id is not None or attach_branch):
            core_frames.append(frame_id)
        if len(core_frames) >= desired_frames:
            break
    if not core_frames:
        raise RuntimeError("No core frames found for matched-lineage event.")
    return core_frames


def _save_strip(run_payload: dict[str, Any], event_row: dict[str, Any], frame_ids: list[int], path: Path) -> None:
    frame_lookup, instance_lookup = _build_run_frame_maps(run_payload["run"])
    ncols = min(5, max(1, len(frame_ids)))
    nrows = int(np.ceil(len(frame_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    for axis in axes_array.ravel():
        axis.axis("off")
    for axis, frame_id in zip(axes_array.ravel(), frame_ids):
        _draw_frame(
            axis,
            frame_lookup[frame_id],
            instance_lookup.get(frame_id, {}).get(int(event_row["gt_object_id"])),
            event_row,
            view_mode="core_recovery_view",
        )
    Image.fromarray(_figure_to_array(fig)).save(path)


def _save_gif(run_payload: dict[str, Any], event_row: dict[str, Any], frame_ids: list[int], path: Path) -> None:
    frame_lookup, instance_lookup = _build_run_frame_maps(run_payload["run"])
    images: list[Image.Image] = []
    for frame_id in frame_ids:
        fig, axis = plt.subplots(1, 1, figsize=(5.8, 5.4), constrained_layout=True)
        _draw_frame(
            axis,
            frame_lookup[frame_id],
            instance_lookup.get(frame_id, {}).get(int(event_row["gt_object_id"])),
            event_row,
            view_mode="core_recovery_view",
        )
        images.append(Image.fromarray(_figure_to_array(fig)))
    images[0].save(path, save_all=True, append_images=images[1:], duration=180, loop=0)


def _make_event_stub(frame_id: int, gt_object_id: int, instance_row: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "gt_object_id": gt_object_id,
        "reappear_frame": frame_id,
        "matched_lineage_id": None if instance_row is None else instance_row.get("matched_lineage_id"),
        "old_lineage_id": None if instance_row is None else instance_row.get("prototype_lineage_id"),
        "old_track_id": None if instance_row is None else instance_row.get("track_id"),
        "old_prototype_id": None if instance_row is None else instance_row.get("selected_prototype_id"),
        "current_head_prototype_id": None if instance_row is None else instance_row.get("current_head_prototype_id"),
    }


def _select_fp_frames(run_payload: dict[str, Any], *, max_frames: int = 9) -> list[int]:
    frame_lookup, instance_lookup = _build_run_frame_maps(run_payload["run"])
    scored_frames: list[tuple[int, int]] = []
    for frame_id in sorted(frame_lookup):
        instance_rows = instance_lookup.get(frame_id, {})
        if not instance_rows:
            continue
        max_raw = 0
        is_fp_frame = False
        for gt_object_id, instance_row in instance_rows.items():
            event_stub = _make_event_stub(frame_id, int(gt_object_id), instance_row)
            context = _build_frame_context(frame_lookup[frame_id], instance_row, event_stub)
            max_raw = max(max_raw, int(context["raw_proposal_count"]))
            if context["is_non_core_frame"] and context["raw_proposal_count"] > 0:
                is_fp_frame = True
        if is_fp_frame:
            scored_frames.append((frame_id, max_raw))
    scored_frames.sort(key=lambda item: item[1], reverse=True)
    return [frame_id for frame_id, _ in scored_frames[:max_frames]]


def _save_fp_gallery(run_payload: dict[str, Any], path: Path) -> None:
    frame_lookup, instance_lookup = _build_run_frame_maps(run_payload["run"])
    frame_ids = _select_fp_frames(run_payload)
    if not frame_ids:
        raise RuntimeError("No false-proposal frames found for full_debug_view gallery.")
    ncols = min(3, len(frame_ids))
    nrows = int(np.ceil(len(frame_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.8 * nrows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    for axis in axes_array.ravel():
        axis.axis("off")
    for axis, frame_id in zip(axes_array.ravel(), frame_ids):
        instance_rows = instance_lookup.get(frame_id, {})
        if instance_rows:
            chosen_instance = max(
                instance_rows.values(),
                key=lambda row: len(getattr(frame_lookup[frame_id].objectness_output, "proposals", [])),
            )
        else:
            chosen_instance = None
        chosen_gt_object_id = (
            next((int(obj_id) for obj_id, row in instance_rows.items() if row is chosen_instance), -1)
            if chosen_instance is not None
            else -1
        )
        event_stub = _make_event_stub(frame_id, chosen_gt_object_id, chosen_instance)
        _draw_frame(
            axis,
            frame_lookup[frame_id],
            chosen_instance,
            event_stub,
            view_mode="full_debug_view",
        )
    Image.fromarray(_figure_to_array(fig)).save(path)


def _write_notes(path: Path) -> None:
    lines = [
        "# Visualization Mode Notes",
        "",
        "## Output Mapping",
        "",
        "- `matched_lineage_core_case_strip.png`: core recovery chain view.",
        "- `matched_lineage_core_case.gif`: core recovery chain view.",
        "- `proposal_fp_gallery.png`: full debug false-positive view.",
        "",
        "## Core Recovery View",
        "",
        "- mode: `core_recovery_view`",
        "- GT current target",
        "- old track / old prototype object when present",
        "- matched-lineage candidate boxes",
        "- selected attach target",
        "- current head",
        "- temp attach slot proxy box",
        "- promotion candidate",
        "",
        "The core recovery view hides unrelated raw proposals. A proposal is shown only if it overlaps the current GT target, belongs to the matched-lineage candidate set, is the selected attach target, is the current head, is the temp attach slot, or is the promotion pending candidate.",
        "",
        "## Full Debug View",
        "",
        "- mode: `full_debug_view`",
        "- all raw proposals remain visible",
        "- non-core frames are explicitly labeled `upstream false-proposal frame, not core attach case`",
        "",
        "## Interpretation Guardrail",
        "",
        "Current GIFs with many boxes do not prove that temporary attach is working. Most of those boxes are upstream raw proposals or false positives. The previous visualization did not filter candidates down to the current core re-entry event, so that GIF cannot be used as evidence that the recovery path is connected.",
        "",
        "Do not mix the core recovery view with the false-positive gallery when judging whether the attach path is wired. The first is for recovery-path inspection; the second is only for upstream proposal noise inspection.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_payload = _evaluate_track_c(
        args.config,
        seed=args.seed,
        run_label="forced_temp_attach_core_visual",
        tracking_patch={"debug_inherit_lineage_from_hint": True},
        memory_patch={"debug_force_temp_attach": True},
    )

    core_event = _select_core_event(run_payload["event_rows"])
    core_frame_ids = _select_core_frames(run_payload, core_event, desired_frames=20)

    _save_strip(run_payload, core_event, core_frame_ids, output_dir / "matched_lineage_core_case_strip.png")
    _save_gif(run_payload, core_event, core_frame_ids, output_dir / "matched_lineage_core_case.gif")
    _save_fp_gallery(run_payload, output_dir / "proposal_fp_gallery.png")
    _write_notes(output_dir / "visualization_mode_notes.md")

    print(f"saved_core_strip={output_dir / 'matched_lineage_core_case_strip.png'}")
    print(f"saved_core_gif={output_dir / 'matched_lineage_core_case.gif'}")
    print(f"saved_fp_gallery={output_dir / 'proposal_fp_gallery.png'}")
    print(f"saved_mode_notes={output_dir / 'visualization_mode_notes.md'}")


if __name__ == "__main__":
    main()
