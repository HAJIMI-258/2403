"""Audit memory-guided re-entry search on LaSOT event windows.

This is an eval-only diagnostic. It does not alter VisualCognitiveLoop online
decisions. GT is used only to audit whether generated search ObjectFiles cover
the target and whether the old target episode appears in retrieval top-k.
"""

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
from experiments.run_lasot_cognitive_reentry_eval import _build_loop, _matched_object  # noqa: E402
from nops_owr.evaluation.external_event_windows import (  # noqa: E402
    collect_lasot_reentry_events,
    frame_gt_box,
    load_rgb_frame,
    make_event_window,
    scale_box,
    sequence_category,
)
from nops_owr.evaluation.memory_guided_search import (  # noqa: E402
    MemoryGuidedSearchConfig,
    build_memory_guided_object_files,
)
from nops_owr.evaluation.reentry_audit import bbox_iou, find_target_episode_from_bundles, gap_bucket  # noqa: E402
from nops_owr.memory.retrieval_context import RetrievalContext  # noqa: E402


ROW_FIELDS = [
    "event_id",
    "sequence_id",
    "category",
    "disappear_frame",
    "reappear_frame",
    "gap_length",
    "gap_bucket",
    "normal_matched",
    "normal_best_iou",
    "normal_object_file_count",
    "target_episode_exists",
    "target_episode_id",
    "memory_search_candidate_count",
    "memory_search_best_iou",
    "memory_search_recall",
    "online_selected_search_iou",
    "online_selected_search_recall",
    "online_selected_source_episode_id",
    "online_selected_source_gt_eval_only",
    "best_iou_source_episode_id",
    "best_iou_source_gt_eval_only",
    "best_iou_target_rank",
    "best_iou_target_score",
    "best_iou_top1_score",
    "best_iou_top1_margin",
    "online_selected_target_rank",
    "online_selected_target_score",
    "online_selected_top1_score",
    "online_selected_top1_margin",
    "failure_bucket",
]


def run_audit(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_memory_guided_search_audit",
    max_events: int = 20,
    min_gap: int = 8,
    pre_context: int = 80,
    post_context: int = 20,
    category_filter: str = "",
    sequence_filter: str = "",
    max_image_side: int = 160,
    strict_min_iou: float = 0.25,
    frame_stride: int = 2,
    objectness_profile: str = "A5_quantile_q060_k000_area8_props24",
    attention_profile: str = "A4_recall_max16",
    max_search_candidates: int = 32,
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
    rows: list[dict[str, Any]] = []
    skipped = Counter()

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
        loop = _build_loop(objectness_profile=objectness_profile, attention_profile=attention_profile)
        prev_image = None
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
            if int(frame.frame_idx) == int(event.reappear_frame):
                rows.append(
                    _audit_row(
                        event=event,
                        result=result,
                        loop=loop,
                        gt_box=gt_box,
                        strict_min_iou=strict_min_iou,
                        max_search_candidates=max_search_candidates,
                    )
                )
            prev_image = current_image

    summary = _summary(
        rows,
        total_candidate_events=len(events),
        skipped=skipped,
        objectness_profile=objectness_profile,
        attention_profile=attention_profile,
        frame_stride=frame_stride,
        max_search_candidates=max_search_candidates,
    )
    _write_csv(out / "memory_guided_search_events.csv", rows, ROW_FIELDS)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _audit_row(
    *,
    event: Any,
    result: Any,
    loop: Any,
    gt_box: tuple[float, float, float, float] | None,
    strict_min_iou: float,
    max_search_candidates: int,
) -> dict[str, Any]:
    normal_matched, normal_iou = _matched_object(result, 1, gt_box, strict_min_iou)
    target_episode = find_target_episode_from_bundles(loop.episodic_memory.bundles, 1, int(event.reappear_frame))
    search_objects = build_memory_guided_object_files(
        encoding=result.encoding,
        heatmap=result.objectness_output.heatmap,
        bundles=loop.episodic_memory.bundles,
        frame_index=int(event.reappear_frame),
        config=MemoryGuidedSearchConfig(max_candidates=max_search_candidates),
    )
    best_iou_object = None
    best_iou = 0.0
    for object_file in search_objects:
        iou = bbox_iou(object_file.box, gt_box)
        if iou > best_iou:
            best_iou = iou
            best_iou_object = object_file
    online_selected = search_objects[0] if search_objects else None
    online_iou = 0.0 if online_selected is None else bbox_iou(online_selected.box, gt_box)
    target_episode_id = None if target_episode is None else int(target_episode.episode_id)
    best_iou_retrieval = _retrieval_audit(best_iou_object, loop, int(event.reappear_frame), target_episode_id)
    online_retrieval = _retrieval_audit(online_selected, loop, int(event.reappear_frame), target_episode_id)
    failure = _failure_bucket(
        normal_matched=normal_matched is not None,
        target_episode=target_episode is not None,
        search_recall=best_iou >= strict_min_iou,
        online_recall=online_iou >= strict_min_iou,
        online_target_rank=int(online_retrieval["target_rank"]),
        best_iou_target_rank=int(best_iou_retrieval["target_rank"]),
    )
    return {
        "event_id": f"{event.sequence_id}:{event.disappear_frame}:{event.reappear_frame}",
        "sequence_id": event.sequence_id,
        "category": sequence_category(event.sequence_id),
        "disappear_frame": int(event.disappear_frame),
        "reappear_frame": int(event.reappear_frame),
        "gap_length": int(event.gap_length),
        "gap_bucket": gap_bucket(int(event.gap_length)),
        "normal_matched": int(normal_matched is not None),
        "normal_best_iou": float(normal_iou),
        "normal_object_file_count": len(result.object_files),
        "target_episode_exists": int(target_episode is not None),
        "target_episode_id": "" if target_episode is None else int(target_episode.episode_id),
        "memory_search_candidate_count": len(search_objects),
        "memory_search_best_iou": float(best_iou),
        "memory_search_recall": int(best_iou >= strict_min_iou),
        "online_selected_search_iou": float(online_iou),
        "online_selected_search_recall": int(online_iou >= strict_min_iou),
        "online_selected_source_episode_id": _source_episode_id(online_selected),
        "online_selected_source_gt_eval_only": _source_gt(online_selected),
        "best_iou_source_episode_id": _source_episode_id(best_iou_object),
        "best_iou_source_gt_eval_only": _source_gt(best_iou_object),
        "best_iou_target_rank": int(best_iou_retrieval["target_rank"]),
        "best_iou_target_score": float(best_iou_retrieval["target_score"]),
        "best_iou_top1_score": float(best_iou_retrieval["top1_score"]),
        "best_iou_top1_margin": float(best_iou_retrieval["top1_margin"]),
        "online_selected_target_rank": int(online_retrieval["target_rank"]),
        "online_selected_target_score": float(online_retrieval["target_score"]),
        "online_selected_top1_score": float(online_retrieval["top1_score"]),
        "online_selected_top1_margin": float(online_retrieval["top1_margin"]),
        "failure_bucket": failure,
    }


def _retrieval_audit(object_file: Any, loop: Any, frame_index: int, target_episode_id: int | None) -> dict[str, float | int]:
    if object_file is None:
        return {"target_rank": 0, "target_score": 0.0, "top1_score": 0.0, "top1_margin": 0.0}
    context = RetrievalContext(
        frame_index=int(frame_index),
        mode="reentry",
        min_reentry_gap=8,
        prefer_closed_episodes=True,
        suppress_active_conflicts=True,
    )
    retrievals = loop.episodic_memory.retrieve(object_file, top_k=5, context=context)
    target_rank = 0
    target_score = 0.0
    for candidate in retrievals:
        if target_episode_id is not None and int(candidate.bundle.episode_id) == int(target_episode_id):
            target_rank = int(candidate.rank)
            target_score = float(candidate.score)
            break
    top1 = retrievals[0] if retrievals else None
    return {
        "target_rank": target_rank,
        "target_score": target_score,
        "top1_score": 0.0 if top1 is None else float(top1.score),
        "top1_margin": 0.0 if top1 is None else float(top1.margin_to_next),
    }


def _failure_bucket(
    *,
    normal_matched: bool,
    target_episode: bool,
    search_recall: bool,
    online_recall: bool,
    online_target_rank: int,
    best_iou_target_rank: int,
) -> str:
    if normal_matched:
        return "normal_proposal_already_matched"
    if not target_episode:
        return "target_episode_missing"
    if not search_recall:
        return "memory_search_no_covering_candidate"
    if not online_recall:
        return "memory_search_has_candidate_but_ranking_misses"
    if online_target_rank == 1:
        return "memory_search_online_success"
    if 1 < online_target_rank <= 5:
        return "memory_search_retrieval_low_rank"
    if 1 <= best_iou_target_rank <= 5:
        return "search_candidate_good_but_retrieval_or_selection_misses"
    return "memory_search_retrieval_miss"


def _summary(
    rows: list[dict[str, Any]],
    *,
    total_candidate_events: int,
    skipped: Counter[str],
    objectness_profile: str,
    attention_profile: str,
    frame_stride: int,
    max_search_candidates: int,
) -> dict[str, Any]:
    count = len(rows)
    buckets = Counter(str(row["failure_bucket"]) for row in rows)
    return {
        "stage": "LASOT_MEMORY_GUIDED_SEARCH_AUDIT",
        "total_candidate_events": int(total_candidate_events),
        "evaluated_event_count": int(count),
        "skipped_event_count": int(sum(skipped.values())),
        "skip_reasons": dict(skipped),
        "normal_proposal_recall": _mean([row["normal_matched"] for row in rows]),
        "memory_search_recall": _mean([row["memory_search_recall"] for row in rows]),
        "online_selected_search_recall": _mean([row["online_selected_search_recall"] for row in rows]),
        "target_episode_presence_rate": _mean([row["target_episode_exists"] for row in rows]),
        "best_iou_target_top5_rate": _mean([1 <= int(row["best_iou_target_rank"]) <= 5 for row in rows]),
        "online_selected_target_top5_rate": _mean([1 <= int(row["online_selected_target_rank"]) <= 5 for row in rows]),
        "mean_memory_search_best_iou": _mean([row["memory_search_best_iou"] for row in rows]),
        "mean_online_selected_iou": _mean([row["online_selected_search_iou"] for row in rows]),
        "mean_candidate_count": _mean([row["memory_search_candidate_count"] for row in rows]),
        "failure_buckets": dict(buckets),
        "objectness_profile": objectness_profile,
        "attention_profile": attention_profile,
        "frame_stride": int(frame_stride),
        "max_search_candidates": int(max_search_candidates),
        "next_recommendation": _recommendation(buckets),
    }


def _recommendation(buckets: Counter[str]) -> str:
    if buckets.get("memory_search_no_covering_candidate", 0) > 0:
        return "memory search windows still do not cover target; improve search region generation"
    if buckets.get("memory_search_has_candidate_but_ranking_misses", 0) > 0:
        return "memory search creates covering candidates but ranking misses; improve search candidate selection"
    if buckets.get("memory_search_retrieval_miss", 0) > 0:
        return "memory search covers target but retrieval misses; inspect episodic signatures"
    return "memory-guided search diagnostic ready; compare against normal event-window eval"


def _source_episode_id(object_file: Any) -> str:
    if object_file is None:
        return ""
    return str(object_file.metadata.get("source_episode_id", ""))


def _source_gt(object_file: Any) -> str:
    if object_file is None:
        return ""
    value = object_file.metadata.get("source_episode_gt_instance_id_eval_only")
    return "" if value is None else str(value)


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


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# LaSOT Memory-guided Search Audit",
        "",
        "This diagnostic uses old episodic memory and current heatmap peaks to build eval-only search ObjectFiles.",
        "GT is used only to audit target coverage and target episode retrieval.",
        "",
        f"- evaluated_event_count: {summary['evaluated_event_count']}",
        f"- normal_proposal_recall: {summary['normal_proposal_recall']:.4f}",
        f"- memory_search_recall: {summary['memory_search_recall']:.4f}",
        f"- online_selected_search_recall: {summary['online_selected_search_recall']:.4f}",
        f"- best_iou_target_top5_rate: {summary['best_iou_target_top5_rate']:.4f}",
        f"- online_selected_target_top5_rate: {summary['online_selected_target_top5_rate']:.4f}",
        "",
        "## Failure Buckets",
        "",
        "| bucket | count |",
        "|---|---:|",
    ]
    for bucket, count in summary["failure_buckets"].items():
        lines.append(f"| {bucket} | {count} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_memory_guided_search_audit")
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--pre-context", type=int, default=80)
    parser.add_argument("--post-context", type=int, default=20)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--max-image-side", type=int, default=160)
    parser.add_argument("--strict-min-iou", type=float, default=0.25)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--objectness-profile", default="A5_quantile_q060_k000_area8_props24")
    parser.add_argument("--attention-profile", default="A4_recall_max16")
    parser.add_argument("--max-search-candidates", type=int, default=32)
    args = parser.parse_args()
    summary = run_audit(**vars(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
