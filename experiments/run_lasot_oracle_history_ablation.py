"""Event-centered LaSOT oracle-history upper-bound ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.external.lasot_adapter import LaSOTAdapter  # noqa: E402
from experiments.run_lasot_cognitive_reentry_eval import _build_loop, _oracle_object_file, _reentry_row  # noqa: E402
from nops_owr.evaluation.external_event_windows import (  # noqa: E402
    collect_lasot_reentry_events,
    frame_gt_box,
    frame_is_visible,
    load_rgb_frame,
    make_event_window,
    scale_box,
)
from nops_owr.evaluation.reentry_audit import summarize_reentry_rows, write_reentry_report  # noqa: E402


def run_ablation(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_oracle_history_ablation",
    max_events: int = 50,
    min_gap: int = 8,
    pre_context: int = 80,
    post_context: int = 20,
    category_filter: str = "",
    sequence_filter: str = "",
    max_image_side: int = 160,
    strict_min_iou: float = 0.25,
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
    frames_by_sequence: dict[str, list[Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for mode in ("normal", "oracle_reappear_only", "oracle_history_and_reappear"):
        rows: list[dict[str, Any]] = []
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
                continue
            evaluated += 1
            row = _run_event_mode(
                window=window,
                mode=mode,
                max_image_side=max_image_side,
                strict_min_iou=strict_min_iou,
            )
            if row is not None:
                rows.append(row)
        summary = summarize_reentry_rows(
            rows,
            dataset_name="lasot",
            sequence_count=len({event.sequence_id for event in events}),
            evaluated_sequence_count=len({row["sequence_id"] for row in rows}),
            extra={
                "mode": mode,
                "total_candidate_events": len(events),
                "evaluated_event_count": evaluated,
                "gt_used_for_offline_diagnostic_only": int(mode != "normal"),
            },
        )
        summaries[mode] = summary
        (out / f"{mode}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_reentry_report(out / f"{mode}_report.md", summary, rows, title=f"LaSOT {mode} Re-entry Audit")

    comparison = {
        "normal_target_episode_top5_rate": summaries["normal"].get("target_episode_top5_rate", 0.0),
        "oracle_reappear_only_target_episode_top5_rate": summaries["oracle_reappear_only"].get("target_episode_top5_rate", 0.0),
        "oracle_history_and_reappear_target_episode_top5_rate": summaries["oracle_history_and_reappear"].get("target_episode_top5_rate", 0.0),
        "normal_no_object_file_matched": summaries["normal"].get("failure_buckets", {}).get("no_object_file_matched", 0),
        "oracle_reappear_only_no_object_file_matched": summaries["oracle_reappear_only"].get("failure_buckets", {}).get("no_object_file_matched", 0),
        "oracle_history_and_reappear_no_object_file_matched": summaries["oracle_history_and_reappear"].get("failure_buckets", {}).get("no_object_file_matched", 0),
        "normal_false_resurrection_rate": summaries["normal"].get("false_resurrection_rate_at_reentry", 0.0),
        "oracle_reappear_only_false_resurrection_rate": summaries["oracle_reappear_only"].get("false_resurrection_rate_at_reentry", 0.0),
        "oracle_history_and_reappear_false_resurrection_rate": summaries["oracle_history_and_reappear"].get("false_resurrection_rate_at_reentry", 0.0),
    }
    (out / "oracle_history_ablation_report.md").write_text(_comparison_report(comparison), encoding="utf-8")
    return {"summaries": summaries, "comparison": comparison}


def _run_event_mode(
    *,
    window: Any,
    mode: str,
    max_image_side: int,
    strict_min_iou: float,
) -> dict[str, Any] | None:
    loop = _build_loop()
    prev_image = None
    oracle_episode_id: int | None = None
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
                "boxes": scaled_boxes,
                "instance_ids": [1 for _ in scaled_boxes],
                "concept_ids": [1 for _ in scaled_boxes],
            },
        )
        if mode == "oracle_history_and_reappear" and gt_box is not None and frame_is_visible(frame):
            oracle_object = _oracle_object_file(result.encoding, gt_box, int(frame.frame_idx), str(window.event.sequence_id))
            if int(frame.frame_idx) < int(window.event.reappear_frame):
                oracle_episode_id = loop.episodic_memory.write_or_extend_episode(
                    oracle_object,
                    frame_index=int(frame.frame_idx),
                    track_id=-1,
                    prototype_id=-1,
                    concept_id=-1,
                    source_state="oracle_history_eval_only",
                    active_episode_id=oracle_episode_id,
                    metadata={"gt_instance_id": 1, "oracle_history_eval_only": 1},
                )
                if int(frame.frame_idx) == int(window.event.disappear_frame):
                    loop.episodic_memory.close_episode(
                        oracle_episode_id,
                        int(frame.frame_idx),
                        close_reason="oracle_history_disappear_eval_only",
                    )
            elif int(frame.frame_idx) == int(window.event.reappear_frame):
                row = _reentry_row(
                    event=window.event,
                    result=result,
                    loop=loop,
                    gt_box=gt_box,
                    strict_min_iou=strict_min_iou,
                    oracle_gt_box_eval_only=True,
                    oracle_force_query=True,
                )
                row["mode"] = mode
                return row
        elif int(frame.frame_idx) == int(window.event.reappear_frame):
            row = _reentry_row(
                event=window.event,
                result=result,
                loop=loop,
                gt_box=gt_box,
                strict_min_iou=strict_min_iou,
                oracle_gt_box_eval_only=(mode == "oracle_reappear_only"),
                oracle_force_query=(mode == "oracle_reappear_only"),
            )
            row["mode"] = mode
            return row
        prev_image = current_image
    return None


def _comparison_report(comparison: dict[str, Any]) -> str:
    return (
        "# LaSOT Oracle History Upper-bound Ablation\n\n"
        "GT is used for offline diagnostic only, not online model behavior. "
        "`oracle_history_and_reappear` writes target GT-box object files during visible "
        "pre-gap frames and queries with the GT box at reappearance.\n\n"
        f"- normal_target_episode_top5_rate: {comparison['normal_target_episode_top5_rate']}\n"
        f"- oracle_reappear_only_target_episode_top5_rate: {comparison['oracle_reappear_only_target_episode_top5_rate']}\n"
        f"- oracle_history_and_reappear_target_episode_top5_rate: {comparison['oracle_history_and_reappear_target_episode_top5_rate']}\n"
        f"- normal_no_object_file_matched: {comparison['normal_no_object_file_matched']}\n"
        f"- oracle_reappear_only_no_object_file_matched: {comparison['oracle_reappear_only_no_object_file_matched']}\n"
        f"- oracle_history_and_reappear_no_object_file_matched: {comparison['oracle_history_and_reappear_no_object_file_matched']}\n"
        f"- normal_false_resurrection_rate: {comparison['normal_false_resurrection_rate']}\n"
        f"- oracle_reappear_only_false_resurrection_rate: {comparison['oracle_reappear_only_false_resurrection_rate']}\n"
        f"- oracle_history_and_reappear_false_resurrection_rate: {comparison['oracle_history_and_reappear_false_resurrection_rate']}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_oracle_history_ablation")
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--pre-context", type=int, default=80)
    parser.add_argument("--post-context", type=int, default=20)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--max-image-side", type=int, default=160)
    parser.add_argument("--strict-min-iou", type=float, default=0.25)
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()
    result = run_ablation(
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
    )
    print(json.dumps(result["comparison"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
