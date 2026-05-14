"""Audit LaSOT target recall by proposal source."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.external.lasot_adapter import LaSOTAdapter  # noqa: E402
from experiments.run_lasot_cognitive_reentry_eval import _build_loop, _matched_object  # noqa: E402
from nops_owr.evaluation.external_event_windows import (  # noqa: E402
    collect_lasot_reentry_events,
    frame_gt_box,
    load_rgb_frame,
    make_event_window,
    scale_box,
    sequence_category,
)
from nops_owr.evaluation.reentry_audit import bbox_iou, find_target_episode_from_bundles  # noqa: E402


SOURCE_FIELDS = [
    "source",
    "proposal_count",
    "matched_target_count",
    "matched_at_reappear_count",
    "mean_best_iou",
    "recall_at_reappear",
    "attention_selected_count",
    "target_episode_top5_when_source_matched",
    "mean_matched_proposal_index",
    "mean_matched_quality_score",
    "mean_matched_source_score",
    "mean_matched_attention_score",
    "mean_matched_attention_rank",
]

DETAIL_FIELDS = [
    "event_id",
    "source",
    "object_file_id",
    "proposal_index",
    "matched",
    "iou",
    "quality_score",
    "source_score",
    "attention_score",
    "attention_rank",
    "attended",
    "target_top5",
]


def run_audit(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_proposal_source_recall_audit",
    max_events: int = 20,
    min_gap: int = 8,
    pre_context: int = 80,
    post_context: int = 20,
    max_image_side: int = 160,
    strict_min_iou: float = 0.25,
    frame_stride: int = 2,
    category_filter: str = "",
    sequence_filter: str = "",
    objectness_profile: str = "A5_quantile_q060_k000_area8_props24",
    attention_profile: str = "A4_recall_max16",
    memory_guided_profile: str = "M2_closed_episode_global_windows_k16",
    memory_guided_attention: int = 1,
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
    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    detail_rows: list[dict[str, Any]] = []
    frames_by_sequence: dict[str, list[Any]] = {}
    evaluated = 0
    for event in events:
        frames = frames_by_sequence.setdefault(event.sequence_id, list(adapter.iter_frames(event.sequence_id)))
        window = make_event_window(frames, event, pre_context=pre_context, post_context=post_context, frame_stride=frame_stride)
        if window is None:
            continue
        evaluated += 1
        loop = _build_loop(
            objectness_profile=objectness_profile,
            attention_profile=attention_profile,
            memory_guided_profile=memory_guided_profile,
            memory_guided_attention=bool(memory_guided_attention),
        )
        prev_image = None
        for frame in window.frames:
            current_image, scale_x, scale_y = load_rgb_frame(frame.frame_path, max_image_side=max_image_side)
            if prev_image is None:
                prev_image = current_image
            gt_box = scale_box(frame_gt_box(frame), scale_x, scale_y)
            boxes = [] if gt_box is None else [gt_box]
            result = loop.step(
                prev_image,
                current_image,
                int(frame.frame_idx),
                ground_truth={"boxes": boxes, "instance_ids": [1 for _ in boxes], "concept_ids": [1 for _ in boxes]},
            )
            if int(frame.frame_idx) == int(event.reappear_frame):
                _accumulate_sources(
                    source_rows,
                    detail_rows,
                    event=event,
                    result=result,
                    loop=loop,
                    gt_box=gt_box,
                    strict_min_iou=strict_min_iou,
                )
            prev_image = current_image

    summaries = [_source_summary(source, rows, evaluated_event_count=evaluated) for source, rows in sorted(source_rows.items())]
    compact = {
        "stage": "LASOT_PROPOSAL_SOURCE_RECALL_AUDIT",
        "evaluated_event_count": evaluated,
        "source_count": len(summaries),
        "objectness_profile": objectness_profile,
        "attention_profile": attention_profile,
        "memory_guided_profile": memory_guided_profile,
        "memory_guided_attention": int(memory_guided_attention),
        "source_results": {row["source"]: row for row in summaries},
    }
    _write_csv(out / "source_summary.csv", summaries, SOURCE_FIELDS)
    _write_csv(out / "source_detail.csv", detail_rows, DETAIL_FIELDS)
    (out / "source_summary.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(_report(compact, summaries), encoding="utf-8")
    return compact


def _accumulate_sources(
    source_rows: dict[str, list[dict[str, Any]]],
    detail_rows: list[dict[str, Any]],
    *,
    event: Any,
    result: Any,
    loop: Any,
    gt_box: tuple[float, float, float, float] | None,
    strict_min_iou: float,
) -> None:
    event_id = f"{event.sequence_id}:{event.disappear_frame}:{event.reappear_frame}"
    target_episode = find_target_episode_from_bundles(loop.episodic_memory.bundles, 1, int(result.frame_index))
    attended_ids = {item.object_file_id for item in result.attended_object_files}
    scored = _attention_scores(loop, result.object_files)
    ranks = {object_file.object_file_id: rank for rank, (_, object_file) in enumerate(scored, start=1)}
    scores = {object_file.object_file_id: score for score, object_file in scored}
    for object_file in result.object_files:
        source = str(object_file.proposal_source)
        iou = bbox_iou(object_file.box, gt_box)
        matched = iou >= strict_min_iou
        retrievals = result.episodic_retrievals.get(object_file.object_file_id, [])
        target_rank = 0
        if target_episode is not None:
            for candidate in retrievals:
                if int(candidate.bundle.episode_id) == int(target_episode.episode_id):
                    target_rank = int(candidate.rank)
                    break
        row = {
            "event_id": event_id,
            "source": source,
            "object_file_id": object_file.object_file_id,
            "proposal_index": int(object_file.proposal_index),
            "matched": int(matched),
            "iou": float(iou),
            "quality_score": float(object_file.quality_score),
            "source_score": float(object_file.proposal_source_score),
            "attention_score": float(scores.get(object_file.object_file_id, 0.0)),
            "attention_rank": int(ranks.get(object_file.object_file_id, 0)),
            "attended": int(object_file.object_file_id in attended_ids),
            "target_top5": int(matched and 1 <= target_rank <= 5),
        }
        source_rows[source].append(row)
        detail_rows.append(row)


def _attention_scores(loop: Any, object_files: list[Any]) -> list[tuple[float, Any]]:
    scored = []
    for object_file in object_files:
        salience = float(object_file.metadata.get("memory_salience", 0.0))
        scored.append((float(loop.attention_gate.score(object_file, salience)), object_file))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def _source_summary(source: str, rows: list[dict[str, Any]], *, evaluated_event_count: int) -> dict[str, Any]:
    matched = [row for row in rows if int(row["matched"])]
    matched_events = {str(row["event_id"]) for row in matched}
    return {
        "source": source,
        "proposal_count": len(rows),
        "matched_target_count": len(matched),
        "matched_at_reappear_count": len(matched_events),
        "mean_best_iou": max((float(row["iou"]) for row in rows), default=0.0),
        "recall_at_reappear": 0.0 if evaluated_event_count <= 0 else len(matched_events) / float(evaluated_event_count),
        "attention_selected_count": sum(int(row["attended"]) for row in matched),
        "target_episode_top5_when_source_matched": sum(int(row["target_top5"]) for row in matched),
        "mean_matched_proposal_index": _mean([row["proposal_index"] for row in matched]),
        "mean_matched_quality_score": _mean([row["quality_score"] for row in matched]),
        "mean_matched_source_score": _mean([row["source_score"] for row in matched]),
        "mean_matched_attention_score": _mean([row["attention_score"] for row in matched]),
        "mean_matched_attention_rank": _mean([row["attention_rank"] for row in matched]),
    }


def _mean(values: list[Any]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _report(compact: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# LaSOT Proposal Source Recall Audit",
        "",
        f"- evaluated_event_count: {compact['evaluated_event_count']}",
        "",
        "| source | proposals | matched | recall@reappear | attended_matches | mean_attn_rank | target_top5_matches |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['source']} | {row['proposal_count']} | {row['matched_target_count']} | "
            f"{float(row['recall_at_reappear']):.4f} | {row['attention_selected_count']} | "
            f"{float(row['mean_matched_attention_rank']):.2f} | "
            f"{row['target_episode_top5_when_source_matched']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_proposal_source_recall_audit")
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--pre-context", type=int, default=80)
    parser.add_argument("--post-context", type=int, default=20)
    parser.add_argument("--max-image-side", type=int, default=160)
    parser.add_argument("--strict-min-iou", type=float, default=0.25)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--objectness-profile", default="A5_quantile_q060_k000_area8_props24")
    parser.add_argument("--attention-profile", default="A4_recall_max16")
    parser.add_argument("--memory-guided-profile", default="M2_closed_episode_global_windows_k16")
    parser.add_argument("--memory-guided-attention", type=int, default=1)
    args = parser.parse_args()
    summary = run_audit(**vars(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
