"""Phase 2F-R Part B/C: proposal geometry repair and region-support comparison."""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase2fr_utils import (
    RegionDescriptor,
    box_iou,
    build_track_scenarios,
    copy_config,
    encode_objectness_sequence,
    load_config_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2F-R geometry repair comparison.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--output-dir", default="results/phase2fr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--track-a-length", type=int, default=240)
    parser.add_argument("--track-c-length", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = load_config_payload(args.config)
    scenarios = {scenario["name"]: scenario for scenario in build_track_scenarios(args.config)}
    selected_names = ["track_a_bridge", "track_c_long_horizon"]

    run_payloads: dict[str, dict[str, Any]] = {}
    for index, scenario_name in enumerate(selected_names):
        scenario_config = copy_config(scenarios[scenario_name]["config"])
        scenario_config.sequence_length = args.track_a_length if scenario_name == "track_a_bridge" else args.track_c_length
        run_payloads[scenario_name] = encode_objectness_sequence(
            config_payload=config_payload,
            scenario_name=scenario_name,
            scenario_config=scenario_config,
            seed=args.seed + 31 * index,
            sequence_id=0,
        )

    rows = _build_ablation_rows(run_payloads)
    ablation_path = output_dir / "proposal_geometry_ablation_summary.csv"
    _write_csv(ablation_path, rows)

    comparison_path = output_dir / "proposal_representation_comparison.png"
    gallery_path = output_dir / "proposal_refined_case_gallery.png"
    gif_path = output_dir / "proposal_mask_proxy_preview.gif"
    design_path = output_dir / "phase2fr_design_notes.md"
    summary_path = output_dir / "phase2fr_final_summary_v1.md"

    representative_frames = _select_representative_frames(run_payloads)
    _save_representation_comparison(comparison_path, representative_frames)
    _save_refined_case_gallery(gallery_path, representative_frames)
    _save_mask_proxy_preview(gif_path, representative_frames["track_c_long_horizon"])
    design_path.write_text(_build_design_notes(rows), encoding="utf-8")
    summary_path.write_text(_build_final_summary(rows), encoding="utf-8")

    print(f"saved_ablation={ablation_path}")
    print(f"saved_comparison={comparison_path}")
    print(f"saved_gallery={gallery_path}")
    print(f"saved_gif={gif_path}")
    print(f"saved_design={design_path}")
    print(f"saved_summary={summary_path}")


def _build_ablation_rows(run_payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_name, run_payload in run_payloads.items():
        metrics_by_method = {
            "raw_bbox_baseline": defaultdict(list),
            "phase2fr_refined": defaultdict(list),
        }
        frame_count = 0
        for frame_record in run_payload["frame_records"]:
            if not frame_record["gt_boxes"]:
                continue
            frame_count += 1
            old_proposals = _raw_proposals_from_components(frame_record["components"])
            new_proposals = _refined_proposals(frame_record["objectness_output"].proposals)
            gt_boxes = list(frame_record["gt_boxes"])
            gt_masks = [np.asarray(mask, dtype=bool) for mask in frame_record["gt_masks"]]
            for method_name, proposals in {
                "raw_bbox_baseline": old_proposals,
                "phase2fr_refined": new_proposals,
            }.items():
                frame_metrics = _compute_frame_metrics(
                    proposals=proposals,
                    gt_boxes=gt_boxes,
                    gt_masks=gt_masks,
                    frame_shape=frame_record["frame"].shape[:2],
                )
                for key, value in frame_metrics.items():
                    metrics_by_method[method_name][key].append(float(value))

        for method_name, metric_lists in metrics_by_method.items():
            row = {
                "scenario_name": scenario_name,
                "method": method_name,
                "frames": frame_count,
            }
            for key, values in metric_lists.items():
                row[key] = float(np.mean(values)) if values else 0.0
            rows.append(row)
    return rows


def _raw_proposals_from_components(components: list[RegionDescriptor]) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for descriptor in components:
        proposals.append(
            {
                "box": tuple(int(v) for v in descriptor.box),
                "support_box": tuple(int(v) for v in descriptor.box),
                "support_mask": descriptor.local_mask.astype(bool),
                "area": int(descriptor.region_area),
                "score": float(descriptor.proposal_score),
                "fill_ratio": float(descriptor.region_fill_ratio),
                "compactness": float(descriptor.region_compactness),
            }
        )
    return proposals


def _refined_proposals(proposals: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "box": tuple(int(v) for v in proposal.box),
            "support_box": tuple(int(v) for v in proposal.support_box),
            "support_mask": np.asarray(proposal.support_mask, dtype=bool),
            "area": int(proposal.area),
            "score": float(proposal.score),
            "fill_ratio": float(proposal.fill_ratio),
            "compactness": float(proposal.compactness),
            "raw_box": tuple(int(v) for v in proposal.raw_box),
        }
        for proposal in proposals
    ]


def _compute_frame_metrics(
    *,
    proposals: list[dict[str, Any]],
    gt_boxes: list[tuple[int, int, int, int]],
    gt_masks: list[np.ndarray],
    frame_shape: tuple[int, int],
) -> dict[str, float]:
    fp_count = 0
    bbox_tightness_values: list[float] = []
    compactness_values: list[float] = []
    fill_values: list[float] = []

    for proposal in proposals:
        bbox_area = max(1, (proposal["box"][2] - proposal["box"][0]) * (proposal["box"][3] - proposal["box"][1]))
        bbox_tightness_values.append(float(proposal["area"]) / bbox_area)
        compactness_values.append(float(proposal["compactness"]))
        fill_values.append(float(proposal["fill_ratio"]))
        max_iou = max((box_iou(proposal["box"], gt_box) for gt_box in gt_boxes), default=0.0)
        max_support_cover = max((_support_cover_ratio(proposal, gt_mask) for gt_mask in gt_masks), default=0.0)
        if max_iou < 0.10 and max_support_cover < 0.10:
            fp_count += 1

    gt_support_cover = _mean_best_cover(gt_masks, proposals, use_bbox=False, frame_shape=frame_shape)
    gt_bbox_cover = _mean_best_cover(gt_masks, proposals, use_bbox=True, frame_shape=frame_shape)
    gt_bbox_recall = _bbox_recall(gt_boxes, proposals)

    return {
        "avg_proposals_per_frame": float(len(proposals)),
        "fp_count_per_frame": float(fp_count),
        "region_compactness": float(np.mean(compactness_values)) if compactness_values else 0.0,
        "region_fill_ratio": float(np.mean(fill_values)) if fill_values else 0.0,
        "bbox_tightness": float(np.mean(bbox_tightness_values)) if bbox_tightness_values else 0.0,
        "gt_coverage_by_support_region": gt_support_cover,
        "gt_coverage_by_refined_bbox": gt_bbox_cover,
        "u_recall_bbox": gt_bbox_recall,
    }


def _support_cover_ratio(proposal: dict[str, Any], gt_mask: np.ndarray) -> float:
    support_box = proposal["support_box"]
    support_mask = proposal["support_mask"]
    x1, y1, x2, y2 = support_box
    gt_crop = gt_mask[y1:y2, x1:x2]
    if gt_crop.shape != support_mask.shape:
        return 0.0
    gt_area = int(gt_mask.sum())
    if gt_area <= 0:
        return 0.0
    intersection = int(np.logical_and(gt_crop, support_mask).sum())
    return float(intersection / gt_area)


def _bbox_cover_ratio(box: tuple[int, int, int, int], gt_mask: np.ndarray, frame_shape: tuple[int, int]) -> float:
    gt_area = int(gt_mask.sum())
    if gt_area <= 0:
        return 0.0
    x1, y1, x2, y2 = box
    bbox_mask = np.zeros(frame_shape, dtype=bool)
    bbox_mask[y1:y2, x1:x2] = True
    intersection = int(np.logical_and(bbox_mask, gt_mask).sum())
    return float(intersection / gt_area)


def _mean_best_cover(
    gt_masks: list[np.ndarray],
    proposals: list[dict[str, Any]],
    *,
    use_bbox: bool,
    frame_shape: tuple[int, int],
) -> float:
    if not gt_masks:
        return 1.0
    cover_values = []
    for gt_mask in gt_masks:
        if use_bbox:
            best = max((_bbox_cover_ratio(proposal["box"], gt_mask, frame_shape) for proposal in proposals), default=0.0)
        else:
            best = max((_support_cover_ratio(proposal, gt_mask) for proposal in proposals), default=0.0)
        cover_values.append(best)
    return float(np.mean(cover_values)) if cover_values else 0.0


def _bbox_recall(gt_boxes: list[tuple[int, int, int, int]], proposals: list[dict[str, Any]]) -> float:
    if not gt_boxes:
        return 1.0
    hits = 0
    for gt_box in gt_boxes:
        if max((box_iou(gt_box, proposal["box"]) for proposal in proposals), default=0.0) >= 0.5:
            hits += 1
    return float(hits / len(gt_boxes))


def _select_representative_frames(run_payloads: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for scenario_name, run_payload in run_payloads.items():
        best_payload: dict[str, Any] | None = None
        best_score = -1.0
        for frame_record in run_payload["frame_records"]:
            if not frame_record["gt_boxes"]:
                continue
            old_proposals = _raw_proposals_from_components(frame_record["components"])
            new_proposals = _refined_proposals(frame_record["objectness_output"].proposals)
            focus_case = _best_gt_alignment_case(
                old_proposals=old_proposals,
                new_proposals=new_proposals,
                gt_boxes=list(frame_record["gt_boxes"]),
                gt_masks=[np.asarray(mask, dtype=bool) for mask in frame_record["gt_masks"]],
                frame_shape=frame_record["frame"].shape[:2],
            )
            if focus_case is None:
                continue
            score = focus_case["selection_score"]
            if score > best_score:
                best_score = score
                best_payload = {
                    "scenario_name": scenario_name,
                    "frame_record": frame_record,
                    "old_proposals": old_proposals,
                    "new_proposals": new_proposals,
                    "run_payload": run_payload,
                    "focus_case": focus_case,
                }
        if best_payload is None:
            raise RuntimeError(f"No representative frame found for {scenario_name}.")
        selected[scenario_name] = best_payload
    return selected


def _save_representation_comparison(path: Path, selected: dict[str, dict[str, Any]]) -> None:
    scenario_names = ["track_a_bridge", "track_c_long_horizon"]
    fig, axes = plt.subplots(len(scenario_names), 5, figsize=(24, 9), constrained_layout=True)
    if len(scenario_names) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, scenario_name in enumerate(scenario_names):
        payload = selected[scenario_name]
        frame_record = payload["frame_record"]
        frame = frame_record["frame"]
        gt_boxes = frame_record["gt_boxes"]
        focus_case = payload["focus_case"]
        focus_gt_box = focus_case["gt_box"]
        focus_raw_proposals = [focus_case["raw_proposal"]] if focus_case["raw_proposal"] is not None else []
        focus_new_proposals = [focus_case["new_proposal"]] if focus_case["new_proposal"] is not None else []
        axes[row_index, 0].imshow(frame)
        axes[row_index, 0].set_title(f"{scenario_name}\nFrame {frame_record['frame_index']}: image")
        _draw_gt(axes[row_index, 0], [focus_gt_box])
        _set_focus_limits(axes[row_index, 0], focus_gt_box, focus_new_proposals or focus_raw_proposals)

        axes[row_index, 1].imshow(frame_record["objectness_output"].normalized_objectness, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[row_index, 1].set_title("raw heatmap")
        _set_focus_limits(axes[row_index, 1], focus_gt_box, focus_new_proposals or focus_raw_proposals)

        axes[row_index, 2].imshow(frame)
        axes[row_index, 2].set_title("raw proposals")
        _draw_gt(axes[row_index, 2], [focus_gt_box])
        _draw_raw_proposals(axes[row_index, 2], focus_raw_proposals)
        _set_focus_limits(axes[row_index, 2], focus_gt_box, focus_raw_proposals)

        axes[row_index, 3].imshow(frame)
        axes[row_index, 3].set_title("connected support regions")
        _draw_gt(axes[row_index, 3], [focus_gt_box])
        _draw_support_regions(axes[row_index, 3], focus_new_proposals)
        _set_focus_limits(axes[row_index, 3], focus_gt_box, focus_new_proposals)

        axes[row_index, 4].imshow(frame)
        axes[row_index, 4].set_title("refined bbox + support")
        _draw_gt(axes[row_index, 4], [focus_gt_box])
        _draw_support_regions(axes[row_index, 4], focus_new_proposals)
        _draw_refined_proposals(axes[row_index, 4], focus_new_proposals)
        _set_focus_limits(axes[row_index, 4], focus_gt_box, focus_new_proposals)

        for axis in axes[row_index]:
            axis.axis("off")

    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_refined_case_gallery(path: Path, selected: dict[str, dict[str, Any]]) -> None:
    cases = []
    for scenario_name, payload in selected.items():
        frame_record = payload["frame_record"]
        focus_case = payload["focus_case"]
        proposal = focus_case["new_proposal"]
        if proposal is None:
            continue
        raw_box = proposal.get("raw_box", proposal["box"])
        raw_area = max(1, (raw_box[2] - raw_box[0]) * (raw_box[3] - raw_box[1]))
        refined_area = max(1, (proposal["box"][2] - proposal["box"][0]) * (proposal["box"][3] - proposal["box"][1]))
        improvement = focus_case["selection_score"]
        cases.append((improvement, scenario_name, frame_record, proposal, focus_case["gt_box"]))
    cases.sort(key=lambda item: item[0], reverse=True)
    chosen = cases[:6]
    ncols = 3
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 5.5 * nrows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    for axis in axes_array.ravel():
        axis.axis("off")

    for axis, (improvement, scenario_name, frame_record, proposal, gt_box) in zip(axes_array.ravel(), chosen):
        axis.imshow(frame_record["frame"])
        _draw_gt(axis, [gt_box])
        _draw_single_support(axis, proposal, color=(0.05, 0.85, 0.85, 0.28))
        raw_box = proposal.get("raw_box", proposal["box"])
        axis.add_patch(Rectangle((raw_box[0], raw_box[1]), raw_box[2] - raw_box[0], raw_box[3] - raw_box[1], fill=False, lw=1.3, ec="#fb923c", linestyle="--"))
        axis.add_patch(Rectangle((proposal["box"][0], proposal["box"][1]), proposal["box"][2] - proposal["box"][0], proposal["box"][3] - proposal["box"][1], fill=False, lw=1.8, ec="#ffffff"))
        _set_focus_limits(axis, gt_box, [proposal])
        axis.set_title(
            f"{scenario_name} f={frame_record['frame_index']}\n"
            f"alignment gain={improvement:.3f} fill={proposal['fill_ratio']:.3f}",
            fontsize=9,
        )
        axis.axis("off")

    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_mask_proxy_preview(path: Path, track_c_payload: dict[str, Any]) -> None:
    run_payload = track_c_payload["run_payload"]
    target_frame = int(track_c_payload["frame_record"]["frame_index"])
    preview_records = [
        record
        for record in run_payload["frame_records"]
        if target_frame - 6 <= int(record["frame_index"]) <= target_frame + 6
    ]
    frames: list[Image.Image] = []
    for frame_record in preview_records:
        proposals = _refined_proposals(frame_record["objectness_output"].proposals)
        gt_boxes = list(frame_record["gt_boxes"])
        gt_masks = [np.asarray(mask, dtype=bool) for mask in frame_record["gt_masks"]]
        focus_case = _best_gt_alignment_case(
            old_proposals=[],
            new_proposals=proposals,
            gt_boxes=gt_boxes,
            gt_masks=gt_masks,
            frame_shape=frame_record["frame"].shape[:2],
        )
        if focus_case is None or focus_case["new_proposal"] is None:
            continue
        focus_gt_box = focus_case["gt_box"]
        focus_proposals = [focus_case["new_proposal"]]
        fig, axis = plt.subplots(1, 1, figsize=(5.8, 5.8))
        axis.imshow(frame_record["frame"])
        _draw_gt(axis, [focus_gt_box])
        _draw_support_regions(axis, focus_proposals)
        _draw_refined_proposals(axis, focus_proposals)
        _set_focus_limits(axis, focus_gt_box, focus_proposals)
        axis.set_title(f"track_c_long_horizon frame={frame_record['frame_index']}")
        axis.axis("off")
        frames.append(_figure_to_image(fig))
        plt.close(fig)

    if not frames:
        return
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=220, loop=0)


def _draw_gt(axis, gt_boxes: list[tuple[int, int, int, int]]) -> None:
    for box in gt_boxes:
        axis.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, lw=1.4, ec="#22c55e"))


def _draw_raw_proposals(axis, proposals: list[dict[str, Any]]) -> None:
    for proposal in proposals[:6]:
        box = proposal["box"]
        axis.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, lw=1.2, ec="#f59e0b"))


def _draw_refined_proposals(axis, proposals: list[dict[str, Any]]) -> None:
    for proposal in proposals[:6]:
        box = proposal["box"]
        axis.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, lw=1.6, ec="#ffffff"))


def _draw_support_regions(axis, proposals: list[dict[str, Any]]) -> None:
    palette = [
        (0.05, 0.85, 0.85, 0.28),
        (0.9, 0.3, 0.4, 0.24),
        (0.95, 0.8, 0.2, 0.24),
        (0.6, 0.4, 1.0, 0.24),
    ]
    for proposal_index, proposal in enumerate(proposals[:4]):
        _draw_single_support(axis, proposal, color=palette[proposal_index % len(palette)])


def _draw_single_support(axis, proposal: dict[str, Any], color: tuple[float, float, float, float]) -> None:
    support_box = proposal["support_box"]
    support_mask = proposal["support_mask"]
    x1, y1, x2, y2 = support_box
    overlay = np.zeros((support_mask.shape[0], support_mask.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = support_mask.astype(np.float32) * color[0]
    overlay[..., 1] = support_mask.astype(np.float32) * color[1]
    overlay[..., 2] = support_mask.astype(np.float32) * color[2]
    overlay[..., 3] = support_mask.astype(np.float32) * color[3]
    axis.imshow(overlay, extent=(x1, x2, y2, y1))


def _best_gt_alignment_case(
    *,
    old_proposals: list[dict[str, Any]],
    new_proposals: list[dict[str, Any]],
    gt_boxes: list[tuple[int, int, int, int]],
    gt_masks: list[np.ndarray],
    frame_shape: tuple[int, int],
) -> dict[str, Any] | None:
    best_case: dict[str, Any] | None = None
    best_score = -1.0
    for gt_box, gt_mask in zip(gt_boxes, gt_masks):
        best_new = max(
            new_proposals,
            key=lambda proposal: max(
                box_iou(proposal["box"], gt_box),
                _support_cover_ratio(proposal, gt_mask),
            ),
            default=None,
        )
        if best_new is None:
            continue
        best_old = max((box_iou(proposal["box"], gt_box) for proposal in old_proposals), default=0.0)
        new_iou = box_iou(best_new["box"], gt_box)
        new_support_cover = _support_cover_ratio(best_new, gt_mask)
        new_bbox_cover = _bbox_cover_ratio(best_new["box"], gt_mask, frame_shape)
        selection_score = 0.90 * new_support_cover + 0.50 * new_bbox_cover + 0.30 * max(0.0, new_iou - best_old)
        if selection_score > best_score:
            best_score = selection_score
            raw_match = max(old_proposals, key=lambda proposal: box_iou(proposal["box"], gt_box), default=None)
            best_case = {
                "gt_box": gt_box,
                "gt_mask": gt_mask,
                "new_proposal": best_new,
                "raw_proposal": raw_match,
                "selection_score": float(selection_score),
            }
    return best_case


def _set_focus_limits(axis, gt_box: tuple[int, int, int, int], proposals: list[dict[str, Any]], margin: int = 16) -> None:
    boxes = [gt_box] + [proposal["box"] for proposal in proposals]
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    axis.set_xlim(max(0, x1 - margin), x2 + margin)
    axis.set_ylim(y2 + margin, max(0, y1 - margin))


def _figure_to_image(fig) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No geometry repair rows collected.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_design_notes(rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Phase 2F-R Design Notes",
            "",
            "## Part A conclusion carried into Part B",
            "",
            "Field-side background response remains the dominant false-positive source, but proposal geometry is still too crude to ignore. The main repair in this pass is not threshold tightening; it is proposal representation repair.",
            "",
            "## Proposal path change",
            "",
            "- old path: binary objectness mask -> connected component -> coarse enclosing bbox",
            "- new path: binary objectness mask -> connected support region -> support refinement -> refined bbox",
            "",
            "## New proposal representation",
            "",
            "- each proposal now carries a support region mask",
            "- each proposal keeps both raw component bbox and refined bbox",
            "- ranking is quality-aware instead of score-only, using compactness, fill ratio, aspect penalty, and boundary penalty",
            "",
            "## Evaluation slice used here",
            "",
            "This comparison run uses shortened Track A / Track C bridge-synthetic sequences as a fast geometry check. The purpose is localization representation repair, not long-horizon tracking evaluation.",
        ]
    )


def _build_final_summary(rows: list[dict[str, Any]]) -> str:
    by_key = {(row["scenario_name"], row["method"]): row for row in rows}
    lines = [
        "# Phase 2F-R Final Summary v1",
        "",
        "This update extends Phase 2F-R from Part A attribution into Part B/C proposal geometry repair.",
        "",
    ]
    for scenario_name in ["track_a_bridge", "track_c_long_horizon"]:
        old_row = by_key[(scenario_name, "raw_bbox_baseline")]
        new_row = by_key[(scenario_name, "phase2fr_refined")]
        lines.extend(
            [
                f"## {scenario_name}",
                "",
                f"- bbox_tightness: {old_row['bbox_tightness']:.4f} -> {new_row['bbox_tightness']:.4f}",
                f"- gt_coverage_by_support_region: {old_row['gt_coverage_by_support_region']:.4f} -> {new_row['gt_coverage_by_support_region']:.4f}",
                f"- gt_coverage_by_refined_bbox: {old_row['gt_coverage_by_refined_bbox']:.4f} -> {new_row['gt_coverage_by_refined_bbox']:.4f}",
                f"- fp_count_per_frame: {old_row['fp_count_per_frame']:.4f} -> {new_row['fp_count_per_frame']:.4f}",
                f"- region_fill_ratio: {old_row['region_fill_ratio']:.4f} -> {new_row['region_fill_ratio']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Readout",
            "",
            "The proposal path now carries region support explicitly instead of collapsing everything into a coarse connected-component box. This is the intended direction for moving from block-like synthetic boxes toward object-aligned candidates that can later transfer to real-object detection.",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
