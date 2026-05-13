"""Event-centered LaSOT objectness/proposal recall audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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
from nops_owr.evaluation.reentry_audit import bbox_iou, gap_bucket  # noqa: E402
from nops_owr.objectness.field import MinimalObjectnessField  # noqa: E402


FRAME_FIELDS = [
    "sequence_id",
    "event_id",
    "category",
    "frame_idx",
    "phase",
    "gt_visible",
    "proposal_count",
    "best_iou",
    "recalled_025",
    "recalled_050",
    "target_area_ratio",
    "gap_bucket",
]


def run_audit(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_objectness_recall_audit",
    max_events: int = 50,
    min_gap: int = 8,
    pre_context: int = 80,
    post_context: int = 20,
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
    objectness = MinimalObjectnessField(tau_obj=0.35, threshold_mode="fixed", min_area=16, max_proposals=8)
    rows: list[dict[str, Any]] = []
    evaluated_events = 0
    skipped = Counter()
    frames_by_sequence: dict[str, list[Any]] = {}
    for event_index, event in enumerate(events):
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
        objectness.reset()
        evaluated_events += 1
        prev_image = None
        for frame in window.frames:
            current_image, scale_x, scale_y = load_rgb_frame(frame.frame_path, max_image_side=max_image_side)
            if prev_image is None:
                prev_image = current_image
            encoding = encoder.encode(prev_image, current_image)
            output = objectness.compute(encoding)
            gt_box = scale_box(frame_gt_box(frame), scale_x, scale_y)
            best_iou = max((bbox_iou(proposal.box, gt_box) for proposal in output.proposals), default=0.0)
            visible = frame_is_visible(frame) and gt_box is not None
            target_area = 0.0
            if gt_box is not None:
                x1, y1, x2, y2 = gt_box
                target_area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, current_image.shape[0] * current_image.shape[1])
            rows.append(
                {
                    "sequence_id": event.sequence_id,
                    "event_id": f"{event.sequence_id}:{event.disappear_frame}:{event.reappear_frame}",
                    "category": sequence_category(event.sequence_id),
                    "frame_idx": frame.frame_idx,
                    "phase": frame_phase(frame, event),
                    "gt_visible": int(visible),
                    "proposal_count": len(output.proposals),
                    "best_iou": best_iou if visible else 0.0,
                    "recalled_025": int(visible and best_iou >= 0.25),
                    "recalled_050": int(visible and best_iou >= 0.50),
                    "target_area_ratio": target_area,
                    "gap_bucket": gap_bucket(event.gap_length),
                }
            )
            prev_image = current_image
    summary = _summary(
        rows,
        candidate_events=len(events),
        evaluated_events=evaluated_events,
        skipped=skipped,
        root=str(root),
    )
    _write_csv(out / "objectness_frame_recall.csv", rows, FRAME_FIELDS)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _summary(
    rows: list[dict[str, Any]],
    *,
    candidate_events: int,
    evaluated_events: int,
    skipped: Counter[str],
    root: str,
) -> dict[str, Any]:
    visible_rows = [row for row in rows if int(row["gt_visible"])]
    pre_rows = [row for row in visible_rows if row["phase"] == "pre_visible"]
    reappear_rows = [row for row in visible_rows if row["phase"] == "reappear"]
    category = _group_recall(visible_rows, "category")
    gap = _group_recall(visible_rows, "gap_bucket")
    return {
        "dataset_name": "lasot",
        "root": root,
        "total_candidate_events": int(candidate_events),
        "evaluated_event_count": int(evaluated_events),
        "skipped_event_count": int(sum(skipped.values())),
        "skip_reasons": dict(skipped),
        "visible_frame_count": len(visible_rows),
        "proposal_recall_iou_025": _mean([row["recalled_025"] for row in visible_rows]),
        "proposal_recall_iou_050": _mean([row["recalled_050"] for row in visible_rows]),
        "mean_best_iou": _mean([row["best_iou"] for row in visible_rows]),
        "mean_proposal_count": _mean([row["proposal_count"] for row in rows]),
        "recall_before_disappear": _mean([row["recalled_025"] for row in pre_rows]),
        "recall_at_reappear": _mean([row["recalled_025"] for row in reappear_rows]),
        "recall_by_category": category,
        "recall_by_gap_bucket": gap,
    }


def _group_recall(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {
        key: {
            "count": float(len(group)),
            "recall_iou_025": _mean([row["recalled_025"] for row in group]),
            "mean_best_iou": _mean([row["best_iou"] for row in group]),
        }
        for key, group in sorted(grouped.items())
    }


def _mean(values: list[Any]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _report(summary: dict[str, Any]) -> str:
    return (
        "# LaSOT Objectness Recall Audit\n\n"
        f"- evaluated_event_count: {summary['evaluated_event_count']}\n"
        f"- proposal_recall_iou_025: {summary['proposal_recall_iou_025']:.4f}\n"
        f"- proposal_recall_iou_050: {summary['proposal_recall_iou_050']:.4f}\n"
        f"- recall_before_disappear: {summary['recall_before_disappear']:.4f}\n"
        f"- recall_at_reappear: {summary['recall_at_reappear']:.4f}\n"
        f"- mean_best_iou: {summary['mean_best_iou']:.4f}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_objectness_recall_audit")
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--pre-context", type=int, default=80)
    parser.add_argument("--post-context", type=int, default=20)
    parser.add_argument("--max-image-side", type=int, default=160)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()
    summary = run_audit(
        root=args.root,
        output_dir=args.output_dir,
        max_events=args.max_events,
        min_gap=args.min_gap,
        pre_context=args.pre_context,
        post_context=args.post_context,
        max_image_side=args.max_image_side,
        category_filter=args.category_filter,
        sequence_filter=args.sequence_filter,
        frame_stride=args.frame_stride,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
