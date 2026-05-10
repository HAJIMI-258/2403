"""Compare normal LaSOT cognitive eval against eval-only GT-box oracle audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_lasot_cognitive_reentry_eval import run_eval  # noqa: E402


def run_ablation(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_oracle_ablation",
    max_sequences: int = 5,
    max_frames: int = 300,
    min_gap: int = 8,
    category_filter: str = "",
    sequence_filter: str = "",
    strict_min_iou: float = 0.25,
    max_image_side: int = 160,
    start_index: int = 0,
    frame_stride: int = 1,
) -> dict[str, object]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    normal_summary = run_eval(
        root=root,
        output_dir=out / "normal",
        max_sequences=max_sequences,
        max_frames=max_frames,
        min_gap=min_gap,
        category_filter=category_filter,
        sequence_filter=sequence_filter,
        strict_min_iou=strict_min_iou,
        max_image_side=max_image_side,
        start_index=start_index,
        frame_stride=frame_stride,
        oracle_gt_box_eval_only=False,
    )
    oracle_summary = run_eval(
        root=root,
        output_dir=out / "oracle_gt_box_eval_only",
        max_sequences=max_sequences,
        max_frames=max_frames,
        min_gap=min_gap,
        category_filter=category_filter,
        sequence_filter=sequence_filter,
        strict_min_iou=strict_min_iou,
        max_image_side=max_image_side,
        start_index=start_index,
        frame_stride=frame_stride,
        oracle_gt_box_eval_only=True,
    )
    comparison = {
        "dataset_name": "lasot",
        "normal_reentry_event_count": normal_summary.get("reentry_event_count", 0),
        "oracle_reentry_event_count": oracle_summary.get("reentry_event_count", 0),
        "normal_no_object_file_matched": normal_summary.get("failure_buckets", {}).get("no_object_file_matched", 0),
        "oracle_no_object_file_matched": oracle_summary.get("failure_buckets", {}).get("no_object_file_matched", 0),
        "normal_target_episode_top5_rate": normal_summary.get("target_episode_top5_rate", 0.0),
        "oracle_target_episode_top5_rate": oracle_summary.get("target_episode_top5_rate", 0.0),
        "normal_same_instance_recall_at_reentry": normal_summary.get("same_instance_recall_at_reentry", 0.0),
        "oracle_same_instance_recall_at_reentry": oracle_summary.get("same_instance_recall_at_reentry", 0.0),
        "normal_false_resurrection_rate_at_reentry": normal_summary.get("false_resurrection_rate_at_reentry", 0.0),
        "oracle_false_resurrection_rate_at_reentry": oracle_summary.get("false_resurrection_rate_at_reentry", 0.0),
    }
    (out / "normal_summary.json").write_text(json.dumps(normal_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "oracle_summary.json").write_text(json.dumps(oracle_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "oracle_ablation_report.md").write_text(_report(comparison), encoding="utf-8")
    return comparison


def _report(comparison: dict[str, object]) -> str:
    return (
        "# LaSOT Oracle GT-box Ablation\n\n"
        "The oracle mode is eval-only and does not alter the main VisualCognitiveLoop. "
        "It diagnoses whether failures come from objectness/proposal matching or from "
        "memory write/retrieval/decision after the object is visible.\n\n"
        f"- normal_no_object_file_matched: {comparison['normal_no_object_file_matched']}\n"
        f"- oracle_no_object_file_matched: {comparison['oracle_no_object_file_matched']}\n"
        f"- normal_target_episode_top5_rate: {comparison['normal_target_episode_top5_rate']}\n"
        f"- oracle_target_episode_top5_rate: {comparison['oracle_target_episode_top5_rate']}\n"
        f"- normal_same_instance_recall_at_reentry: {comparison['normal_same_instance_recall_at_reentry']}\n"
        f"- oracle_same_instance_recall_at_reentry: {comparison['oracle_same_instance_recall_at_reentry']}\n"
        f"- normal_false_resurrection_rate_at_reentry: {comparison['normal_false_resurrection_rate_at_reentry']}\n"
        f"- oracle_false_resurrection_rate_at_reentry: {comparison['oracle_false_resurrection_rate_at_reentry']}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_oracle_ablation")
    parser.add_argument("--max-sequences", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--strict-min-iou", type=float, default=0.25)
    parser.add_argument("--max-image-side", type=int, default=160)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()
    comparison = run_ablation(
        root=args.root,
        output_dir=args.output_dir,
        max_sequences=args.max_sequences,
        max_frames=args.max_frames,
        min_gap=args.min_gap,
        category_filter=args.category_filter,
        sequence_filter=args.sequence_filter,
        strict_min_iou=args.strict_min_iou,
        max_image_side=args.max_image_side,
        start_index=args.start_index,
        frame_stride=args.frame_stride,
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
