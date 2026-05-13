"""Event-centered LaSOT cognitive re-entry evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.external.lasot_adapter import LaSOTAdapter  # noqa: E402
from experiments.run_lasot_cognitive_reentry_eval import (  # noqa: E402
    FRAME_FIELDNAMES,
    REENTRY_FIELDNAMES,
    _build_loop,
    _matched_object,
    _reentry_row,
)
from nops_owr.evaluation.external_event_windows import (  # noqa: E402
    collect_lasot_reentry_events,
    frame_gt_box,
    load_rgb_frame,
    make_event_window,
    scale_box,
    sequence_category,
)
from nops_owr.evaluation.reentry_audit import find_target_episode_from_bundles, summarize_reentry_rows, write_reentry_report  # noqa: E402


EVENT_REENTRY_FIELDS = [
    "event_id",
    "window_start_frame",
    "window_end_frame",
    "pre_visible_frame_count",
    "invisible_gap_frame_count",
    "post_visible_frame_count",
    "frame_stride",
    "objectness_recall_before_disappear",
    "objectness_recall_at_reappear",
    "target_episode_written_before_disappear",
] + REENTRY_FIELDNAMES

EVENT_FRAME_FIELDS = ["event_id", "window_start_frame", "window_end_frame"] + FRAME_FIELDNAMES


def run_eval(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_event_window_eval",
    max_events: int = 50,
    min_gap: int = 8,
    pre_context: int = 80,
    post_context: int = 20,
    category_filter: str = "",
    sequence_filter: str = "",
    max_image_side: int = 160,
    strict_min_iou: float = 0.25,
    frame_stride: int = 1,
    objectness_profile: str = "A0_current_fixed_tau035_area16_props8",
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
    frames_by_sequence: dict[str, list[Any]] = {}
    reentry_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    evaluated = 0

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
        evaluated += 1
        event_id = _event_id(event)
        loop = _build_loop(objectness_profile=objectness_profile)
        prev_image = None
        pre_visible_matched = 0
        pre_visible_count = 0
        reappear_recalled = 0
        target_episode_written_before_disappear = 0
        for frame in window.frames:
            current_image, scale_x, scale_y = load_rgb_frame(frame.frame_path, max_image_side=max_image_side)
            if prev_image is None:
                prev_image = current_image
            gt_box = scale_box(frame_gt_box(frame), scale_x, scale_y)
            scaled_boxes = [] if gt_box is None else [gt_box]
            result = loop.step(
                prev_image,
                current_image,
                int(frame.frame_idx),
                ground_truth={
                    # LaSOT is single-target; numeric id 1 is an eval-local target id.
                    "boxes": scaled_boxes,
                    "instance_ids": [1 for _ in scaled_boxes],
                    "concept_ids": [1 for _ in scaled_boxes],
                },
            )
            row = {
                "event_id": event_id,
                "window_start_frame": window.window_start_frame,
                "window_end_frame": window.window_end_frame,
                "dataset_name": "lasot",
                "sequence_id": event.sequence_id,
                "category": sequence_category(event.sequence_id),
                "frame_index": frame.frame_idx,
                "object_file_count": len(result.object_files),
                "attended_object_count": len(result.attended_object_files),
                "decision_count": len(result.recognition_decisions),
                "memory_context_used": int(result.memory_context_used),
                "active_episode_count": result.active_episode_count,
            }
            frame_rows.append(row)
            if gt_box is not None and int(frame.frame_idx) <= int(event.disappear_frame):
                pre_visible_count += 1
                matched_object, _ = _matched_object(result, 1, gt_box, strict_min_iou)
                pre_visible_matched += int(matched_object is not None)
                if _episode_written_before_disappear(loop, int(event.disappear_frame)):
                    target_episode_written_before_disappear = 1
            if int(frame.frame_idx) == int(event.reappear_frame):
                matched_object, _ = _matched_object(result, 1, gt_box, strict_min_iou)
                reappear_recalled = int(matched_object is not None)
                reentry = _reentry_row(
                    event=event,
                    result=result,
                    loop=loop,
                    gt_box=gt_box,
                    strict_min_iou=strict_min_iou,
                    oracle_gt_box_eval_only=False,
                )
                reentry.update(
                    {
                        "event_id": event_id,
                        "window_start_frame": window.window_start_frame,
                        "window_end_frame": window.window_end_frame,
                        "pre_visible_frame_count": window.pre_visible_frame_count,
                        "invisible_gap_frame_count": window.invisible_gap_frame_count,
                        "post_visible_frame_count": window.post_visible_frame_count,
                        "frame_stride": frame_stride,
                        "objectness_recall_before_disappear": _safe_div(pre_visible_matched, pre_visible_count),
                        "objectness_recall_at_reappear": reappear_recalled,
                        "target_episode_written_before_disappear": target_episode_written_before_disappear,
                    }
                )
                reentry_rows.append(reentry)
            prev_image = current_image

    _write_csv(out / "event_window_reentry_events.csv", reentry_rows, EVENT_REENTRY_FIELDS)
    _write_csv(out / "event_window_frame_metrics.csv", frame_rows, EVENT_FRAME_FIELDS)
    summary = summarize_reentry_rows(
        reentry_rows,
        dataset_name="lasot",
        sequence_count=len({event.sequence_id for event in events}),
        evaluated_sequence_count=len({row["sequence_id"] for row in reentry_rows}),
        extra={
            "total_candidate_events": len(events),
            "evaluated_event_count": evaluated,
            "skipped_event_count": int(sum(skipped.values())),
            "skip_reasons": dict(skipped),
            "proposal_recall_before_disappear": _mean([row["objectness_recall_before_disappear"] for row in reentry_rows]),
            "target_episode_written_before_disappear_rate": _mean(
                [row["target_episode_written_before_disappear"] for row in reentry_rows]
            ),
            "pre_context": int(pre_context),
            "post_context": int(post_context),
            "frame_stride": int(frame_stride),
            "objectness_profile": objectness_profile,
        },
    )
    (out / "event_window_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_reentry_report(out / "event_window_report.md", summary, reentry_rows, title="LaSOT Event-window Re-entry Eval")
    return summary


def _episode_written_before_disappear(loop: Any, disappear_frame: int) -> bool:
    episode = find_target_episode_from_bundles(loop.episodic_memory.bundles, 1, disappear_frame + 1)
    return episode is not None and int(getattr(episode, "frame_start", 10**9)) <= int(disappear_frame)


def _event_id(event: Any) -> str:
    return f"{event.sequence_id}:{event.disappear_frame}:{event.reappear_frame}"


def _safe_div(num: int | float, den: int | float) -> float:
    return 0.0 if float(den) == 0.0 else float(num) / float(den)


def _mean(values: list[Any]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_event_window_eval")
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--pre-context", type=int, default=80)
    parser.add_argument("--post-context", type=int, default=20)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--max-image-side", type=int, default=160)
    parser.add_argument("--strict-min-iou", type=float, default=0.25)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--objectness-profile", default="A0_current_fixed_tau035_area16_props8")
    args = parser.parse_args()
    summary = run_eval(
        root=args.root,
        output_dir=args.output_dir,
        max_events=args.max_events,
        min_gap=args.min_gap,
        pre_context=args.pre_context,
        post_context=args.post_context,
        category_filter=args.category_filter,
        sequence_filter=args.sequence_filter,
        max_image_side=args.max_image_side,
        strict_min_iou=args.strict_min_iou,
        frame_stride=args.frame_stride,
        objectness_profile=args.objectness_profile,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
