"""Audit LaSOT objectness heatmap/threshold support at re-entry targets.

This diagnostic stays below proposal ranking. GT boxes are used only to inspect
whether the bottom-up objectness heatmap and binary mask provide enough support
inside the target region.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.external.lasot_adapter import LaSOTAdapter  # noqa: E402
from nops_owr.encoder.spike_encoder import MinimalSpikeEncoder  # noqa: E402
from nops_owr.evaluation.external_event_windows import (  # noqa: E402
    collect_lasot_reentry_events,
    frame_gt_box,
    frame_is_visible,
    frame_phase,
    load_rgb_frame,
    make_event_window,
    scale_box,
    sequence_category,
)
from nops_owr.evaluation.objectness_profiles import build_objectness_from_profile  # noqa: E402
from nops_owr.evaluation.reentry_audit import bbox_iou, gap_bucket  # noqa: E402


FRAME_FIELDS = [
    "sequence_id",
    "event_id",
    "category",
    "frame_idx",
    "phase",
    "gt_visible",
    "gt_box_area_ratio",
    "proposal_count",
    "best_proposal_iou",
    "target_heatmap_mean",
    "target_heatmap_p90",
    "target_heatmap_max",
    "target_threshold_mean",
    "target_threshold_min",
    "target_margin_mean",
    "target_margin_p90",
    "target_margin_max",
    "target_binary_coverage",
    "target_positive_pixel_count",
    "overlap_component_count",
    "largest_overlap_component_area",
    "largest_overlap_component_gt_coverage",
    "largest_overlap_component_iou",
    "component_fragmentation_ratio",
    "support_failure_type",
    "gap_bucket",
]


def run_audit(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_support_map_audit",
    max_events: int = 20,
    min_gap: int = 8,
    pre_context: int = 80,
    post_context: int = 20,
    objectness_profile: str = "A8_quantile_q050_component_props48",
    max_image_side: int = 160,
    category_filter: str = "",
    sequence_filter: str = "",
    frame_stride: int = 1,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapter = LaSOTAdapter(root)
    events = collect_lasot_reentry_events(
        adapter,
        min_gap=min_gap,
        category_filter=category_filter,
        sequence_filter=sequence_filter,
        max_events=max_events,
    )
    encoder = MinimalSpikeEncoder()
    objectness = build_objectness_from_profile(objectness_profile)
    rows: list[dict[str, Any]] = []
    frames_by_sequence: dict[str, list[Any]] = {}
    skipped: Counter[str] = Counter()
    evaluated_events = 0
    for event in events:
        frames = frames_by_sequence.setdefault(event.sequence_id, list(adapter.iter_frames(event.sequence_id)))
        window = make_event_window(
            frames,
            event,
            pre_context=pre_context,
            post_context=post_context,
            frame_stride=frame_stride,
        )
        if window is None:
            skipped["invalid_window"] += 1
            continue
        evaluated_events += 1
        objectness.reset()
        prev_image = None
        for frame in window.frames:
            current_image, scale_x, scale_y = load_rgb_frame(frame.frame_path, max_image_side=max_image_side)
            if prev_image is None:
                prev_image = current_image
            encoding = encoder.encode(prev_image, current_image)
            output = objectness.compute(encoding)
            gt_box = scale_box(frame_gt_box(frame), scale_x, scale_y)
            visible = frame_is_visible(frame) and gt_box is not None
            rows.append(
                _frame_row(
                    sequence_id=str(event.sequence_id),
                    event_id=f"{event.sequence_id}:{event.disappear_frame}:{event.reappear_frame}",
                    category=sequence_category(str(event.sequence_id)),
                    frame_idx=int(frame.frame_idx),
                    phase=frame_phase(frame, event),
                    gt_visible=visible,
                    gt_box=gt_box,
                    output=output,
                    gap=int(event.gap_length),
                )
            )
            prev_image = current_image

    summary = _summary(
        rows,
        candidate_events=len(events),
        evaluated_events=evaluated_events,
        skipped=skipped,
        root=str(root),
        objectness_profile=objectness_profile,
    )
    _write_csv(out / "support_map_frames.csv", rows, FRAME_FIELDS)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _frame_row(
    *,
    sequence_id: str,
    event_id: str,
    category: str,
    frame_idx: int,
    phase: str,
    gt_visible: bool,
    gt_box: tuple[float, float, float, float] | None,
    output: Any,
    gap: int,
) -> dict[str, Any]:
    frame_shape = output.heatmap.shape
    best_iou = max((bbox_iou(proposal.box, gt_box) for proposal in output.proposals), default=0.0)
    if not gt_visible or gt_box is None:
        return _empty_row(
            sequence_id=sequence_id,
            event_id=event_id,
            category=category,
            frame_idx=frame_idx,
            phase=phase,
            gt_visible=gt_visible,
            frame_shape=frame_shape,
            proposal_count=len(output.proposals),
            best_iou=0.0,
            gap=gap,
        )
    crop_box = _clip_float_box(gt_box, frame_shape)
    heat = _crop(output.heatmap, crop_box)
    threshold = _crop(output.threshold_map, crop_box)
    binary = _crop(output.binary_mask, crop_box).astype(bool)
    margin = heat - threshold
    component_stats = _component_overlap_stats(output.binary_mask.astype(bool), crop_box)
    positive_count = int(binary.sum())
    coverage = positive_count / max(1.0, float(binary.size))
    failure = _support_failure_type(
        positive_count=positive_count,
        coverage=coverage,
        margin_max=float(np.max(margin)) if margin.size else 0.0,
        largest_iou=component_stats["largest_overlap_component_iou"],
        best_proposal_iou=best_iou,
    )
    return {
        "sequence_id": sequence_id,
        "event_id": event_id,
        "category": category,
        "frame_idx": int(frame_idx),
        "phase": phase,
        "gt_visible": int(gt_visible),
        "gt_box_area_ratio": _box_area_ratio(gt_box, frame_shape),
        "proposal_count": len(output.proposals),
        "best_proposal_iou": best_iou,
        "target_heatmap_mean": _mean_array(heat),
        "target_heatmap_p90": _quantile(heat, 0.90),
        "target_heatmap_max": _max_array(heat),
        "target_threshold_mean": _mean_array(threshold),
        "target_threshold_min": _min_array(threshold),
        "target_margin_mean": _mean_array(margin),
        "target_margin_p90": _quantile(margin, 0.90),
        "target_margin_max": _max_array(margin),
        "target_binary_coverage": coverage,
        "target_positive_pixel_count": positive_count,
        **component_stats,
        "support_failure_type": failure,
        "gap_bucket": gap_bucket(gap),
    }


def _empty_row(
    *,
    sequence_id: str,
    event_id: str,
    category: str,
    frame_idx: int,
    phase: str,
    gt_visible: bool,
    frame_shape: tuple[int, int],
    proposal_count: int,
    best_iou: float,
    gap: int,
) -> dict[str, Any]:
    return {
        "sequence_id": sequence_id,
        "event_id": event_id,
        "category": category,
        "frame_idx": int(frame_idx),
        "phase": phase,
        "gt_visible": int(gt_visible),
        "gt_box_area_ratio": 0.0,
        "proposal_count": int(proposal_count),
        "best_proposal_iou": float(best_iou),
        "target_heatmap_mean": 0.0,
        "target_heatmap_p90": 0.0,
        "target_heatmap_max": 0.0,
        "target_threshold_mean": 0.0,
        "target_threshold_min": 0.0,
        "target_margin_mean": 0.0,
        "target_margin_p90": 0.0,
        "target_margin_max": 0.0,
        "target_binary_coverage": 0.0,
        "target_positive_pixel_count": 0,
        "overlap_component_count": 0,
        "largest_overlap_component_area": 0,
        "largest_overlap_component_gt_coverage": 0.0,
        "largest_overlap_component_iou": 0.0,
        "component_fragmentation_ratio": 0.0,
        "support_failure_type": "not_visible" if not gt_visible else "no_gt_box",
        "gap_bucket": gap_bucket(gap),
    }


def _component_overlap_stats(binary_mask: np.ndarray, gt_box: tuple[int, int, int, int]) -> dict[str, Any]:
    height, width = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    gx1, gy1, gx2, gy2 = gt_box
    gt_area = max(1, (gx2 - gx1) * (gy2 - gy1))
    components = []
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not binary_mask[y, x]:
                continue
            pixels = _flood(binary_mask, visited, y, x)
            xs = [px for _, px in pixels]
            ys = [py for py, _ in pixels]
            box = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
            overlap_pixels = sum(1 for py, px in pixels if gx1 <= px < gx2 and gy1 <= py < gy2)
            if overlap_pixels > 0:
                components.append((len(pixels), overlap_pixels, box))
    if not components:
        return {
            "overlap_component_count": 0,
            "largest_overlap_component_area": 0,
            "largest_overlap_component_gt_coverage": 0.0,
            "largest_overlap_component_iou": 0.0,
            "component_fragmentation_ratio": 0.0,
        }
    largest = max(components, key=lambda item: item[1])
    total_overlap = sum(item[1] for item in components)
    return {
        "overlap_component_count": len(components),
        "largest_overlap_component_area": int(largest[0]),
        "largest_overlap_component_gt_coverage": float(largest[1] / gt_area),
        "largest_overlap_component_iou": bbox_iou(largest[2], gt_box),
        "component_fragmentation_ratio": 0.0 if total_overlap <= 0 else float(1.0 - (largest[1] / total_overlap)),
    }


def _flood(binary_mask: np.ndarray, visited: np.ndarray, start_y: int, start_x: int) -> list[tuple[int, int]]:
    height, width = binary_mask.shape
    queue = [(start_y, start_x)]
    visited[start_y, start_x] = True
    pixels: list[tuple[int, int]] = []
    while queue:
        y, x = queue.pop()
        pixels.append((y, x))
        for ny in range(max(0, y - 1), min(height, y + 2)):
            for nx in range(max(0, x - 1), min(width, x + 2)):
                if visited[ny, nx] or not binary_mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))
    return pixels


def _support_failure_type(
    *,
    positive_count: int,
    coverage: float,
    margin_max: float,
    largest_iou: float,
    best_proposal_iou: float,
) -> str:
    if best_proposal_iou >= 0.25:
        return "proposal_recalled"
    if positive_count <= 0:
        return "threshold_no_positive_pixels" if margin_max > -0.02 else "heatmap_low_response"
    if coverage < 0.03:
        return "sparse_binary_support"
    if largest_iou < 0.10:
        return "fragmented_or_misaligned_support"
    return "component_not_promoted_to_proposal"


def _summary(
    rows: list[dict[str, Any]],
    *,
    candidate_events: int,
    evaluated_events: int,
    skipped: Counter[str],
    root: str,
    objectness_profile: str,
) -> dict[str, Any]:
    visible = [row for row in rows if int(row["gt_visible"])]
    reappear = [row for row in visible if row["phase"] == "reappear"]
    failures = Counter(str(row["support_failure_type"]) for row in reappear)
    return {
        "dataset_name": "lasot",
        "root": root,
        "objectness_profile": objectness_profile,
        "total_candidate_events": int(candidate_events),
        "evaluated_event_count": int(evaluated_events),
        "skipped_event_count": int(sum(skipped.values())),
        "skip_reasons": dict(skipped),
        "visible_frame_count": len(visible),
        "reappear_frame_count": len(reappear),
        "proposal_recall_at_reappear": _mean([float(row["best_proposal_iou"]) >= 0.25 for row in reappear]),
        "target_binary_coverage_reappear": _mean([row["target_binary_coverage"] for row in reappear]),
        "target_margin_mean_reappear": _mean([row["target_margin_mean"] for row in reappear]),
        "target_margin_max_reappear": _mean([row["target_margin_max"] for row in reappear]),
        "target_positive_pixel_rate_reappear": _mean([int(row["target_positive_pixel_count"]) > 0 for row in reappear]),
        "largest_component_iou_reappear": _mean([row["largest_overlap_component_iou"] for row in reappear]),
        "component_fragmentation_ratio_reappear": _mean([row["component_fragmentation_ratio"] for row in reappear]),
        "support_failure_counts": dict(failures),
        "dominant_support_failure": failures.most_common(1)[0][0] if failures else "no_reappear_rows",
        "gap_bucket_metrics": _group_by(reappear, "gap_bucket"),
        "category_metrics": _group_by(reappear, "category"),
    }


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        name: {
            "count": float(len(group)),
            "proposal_recall": _mean([float(row["best_proposal_iou"]) >= 0.25 for row in group]),
            "binary_coverage": _mean([row["target_binary_coverage"] for row in group]),
            "margin_max": _mean([row["target_margin_max"] for row in group]),
        }
        for name, group in sorted(grouped.items())
    }


def _clip_float_box(box: tuple[float, float, float, float], frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = frame_shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, max(0, width - 1)))
    y1 = max(0, min(y1, max(0, height - 1)))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return (x1, y1, x2, y2)


def _crop(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return array[y1:y2, x1:x2]


def _box_area_ratio(box: tuple[float, float, float, float], frame_shape: tuple[int, int]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, float(frame_shape[0] * frame_shape[1]))


def _mean_array(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.mean(values))


def _max_array(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.max(values))


def _min_array(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.min(values))


def _quantile(values: np.ndarray, q: float) -> float:
    return 0.0 if values.size == 0 else float(np.quantile(values, q))


def _mean(values: list[Any]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# LaSOT Support Map Audit",
        "",
        f"- objectness_profile: `{summary['objectness_profile']}`",
        f"- evaluated_event_count: {summary['evaluated_event_count']}",
        f"- proposal_recall_at_reappear: {summary['proposal_recall_at_reappear']:.4f}",
        f"- target_binary_coverage_reappear: {summary['target_binary_coverage_reappear']:.4f}",
        f"- target_margin_mean_reappear: {summary['target_margin_mean_reappear']:.4f}",
        f"- target_margin_max_reappear: {summary['target_margin_max_reappear']:.4f}",
        f"- largest_component_iou_reappear: {summary['largest_component_iou_reappear']:.4f}",
        f"- dominant_support_failure: `{summary['dominant_support_failure']}`",
        "",
        "| failure | count |",
        "|---|---:|",
    ]
    for key, count in sorted(summary["support_failure_counts"].items()):
        lines.append(f"| {key} | {count} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_support_map_audit")
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--pre-context", type=int, default=80)
    parser.add_argument("--post-context", type=int, default=20)
    parser.add_argument("--objectness-profile", default="A8_quantile_q050_component_props48")
    parser.add_argument("--max-image-side", type=int, default=160)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--frame-stride", type=int, default=1)
    summary = run_audit(**vars(parser.parse_args()))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
