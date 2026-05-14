"""Audit LaSOT component proposal ranking and support refinement.

GT boxes are used only to locate the best matching proposal for offline
diagnostics. No GT signal is used to generate, sort, or refine proposals.
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
from nops_owr.encoder.spike_encoder import MinimalSpikeEncoder  # noqa: E402
from nops_owr.evaluation.component_ranking_profiles import proposal_rank_score  # noqa: E402
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
from nops_owr.evaluation.reentry_audit import bbox_iou  # noqa: E402
from nops_owr.objectness.field import Proposal  # noqa: E402


FRAME_FIELDS = [
    "sequence_id",
    "event_id",
    "category",
    "frame_idx",
    "phase",
    "gt_visible",
    "gt_box_area_ratio",
    "proposal_count",
    "best_iou",
    "best_iou_rank_by_quality",
    "best_iou_rank_by_score",
    "best_iou_rank_by_area",
    "best_iou_rank_by_raw_area",
    "best_iou_rank_by_component_score_candidate",
    "best_proposal_source",
    "best_box",
    "best_raw_box",
    "best_support_box",
    "best_area",
    "best_raw_area",
    "best_score",
    "best_quality_score",
    "best_fill_ratio",
    "best_compactness",
    "best_boundary_smoothness",
    "best_near_boundary",
    "best_aspect_ratio",
    "raw_box_iou",
    "refined_box_iou",
    "support_box_iou",
    "refinement_delta_iou",
    "quality_minus_score",
    "failure_note",
]


def run_audit(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_component_ranking_audit",
    max_events: int = 50,
    min_gap: int = 8,
    pre_context: int = 80,
    post_context: int = 20,
    objectness_profile: str = "A8_quantile_q050_component_props48",
    max_image_side: int = 160,
    strict_min_iou: float = 0.25,
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
                    frame_shape=output.heatmap.shape,
                    proposals=list(output.proposals),
                    strict_min_iou=strict_min_iou,
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
        strict_min_iou=strict_min_iou,
    )
    _write_csv(out / "component_ranking_frames.csv", rows, FRAME_FIELDS)
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
    frame_shape: tuple[int, int],
    proposals: list[Proposal],
    strict_min_iou: float,
) -> dict[str, Any]:
    gt_area_ratio = _box_area_ratio(gt_box, frame_shape)
    best = _best_iou_proposal(proposals, gt_box)
    best_iou = 0.0 if best is None else bbox_iou(best.box, gt_box)
    raw_iou = 0.0 if best is None else bbox_iou(best.raw_box, gt_box)
    support_iou = 0.0 if best is None else bbox_iou(best.support_box, gt_box)
    failure_note = _failure_note(best_iou, raw_iou, best, strict_min_iou)
    return {
        "sequence_id": sequence_id,
        "event_id": event_id,
        "category": category,
        "frame_idx": int(frame_idx),
        "phase": phase,
        "gt_visible": int(gt_visible),
        "gt_box_area_ratio": gt_area_ratio,
        "proposal_count": len(proposals),
        "best_iou": best_iou if gt_visible else 0.0,
        "best_iou_rank_by_quality": _rank_of(proposals, best, key=lambda proposal: proposal.quality_score),
        "best_iou_rank_by_score": _rank_of(proposals, best, key=lambda proposal: proposal.score),
        "best_iou_rank_by_area": _rank_of(proposals, best, key=lambda proposal: proposal.area),
        "best_iou_rank_by_raw_area": _rank_of(proposals, best, key=lambda proposal: proposal.raw_area),
        "best_iou_rank_by_component_score_candidate": _rank_of(
            proposals,
            best,
            key=lambda proposal: proposal_rank_score(proposal, "R5_boundary_tolerant", frame_shape),
        ),
        "best_proposal_source": "" if best is None else best.source,
        "best_box": "" if best is None else _fmt_box(best.box),
        "best_raw_box": "" if best is None else _fmt_box(best.raw_box),
        "best_support_box": "" if best is None else _fmt_box(best.support_box),
        "best_area": 0 if best is None else int(best.area),
        "best_raw_area": 0 if best is None else int(best.raw_area),
        "best_score": 0.0 if best is None else float(best.score),
        "best_quality_score": 0.0 if best is None else float(best.quality_score),
        "best_fill_ratio": 0.0 if best is None else float(best.fill_ratio),
        "best_compactness": 0.0 if best is None else float(best.compactness),
        "best_boundary_smoothness": 0.0 if best is None else float(best.boundary_smoothness),
        "best_near_boundary": 0 if best is None else int(best.near_boundary),
        "best_aspect_ratio": 0.0 if best is None else _aspect_ratio(best.box),
        "raw_box_iou": raw_iou if gt_visible else 0.0,
        "refined_box_iou": best_iou if gt_visible else 0.0,
        "support_box_iou": support_iou if gt_visible else 0.0,
        "refinement_delta_iou": (best_iou - raw_iou) if gt_visible else 0.0,
        "quality_minus_score": 0.0 if best is None else float(best.quality_score - best.score),
        "failure_note": failure_note if gt_visible else "not_visible",
    }


def _best_iou_proposal(proposals: list[Proposal], gt_box: tuple[float, float, float, float] | None) -> Proposal | None:
    if gt_box is None or not proposals:
        return None
    return max(proposals, key=lambda proposal: bbox_iou(proposal.box, gt_box))


def _rank_of(proposals: list[Proposal], target: Proposal | None, *, key: Any) -> int:
    if target is None:
        return 0
    ordered = sorted(proposals, key=key, reverse=True)
    for rank, proposal in enumerate(ordered, start=1):
        if proposal is target:
            return rank
    return 0


def _failure_note(best_iou: float, raw_iou: float, best: Proposal | None, strict_min_iou: float) -> str:
    if best is None or best_iou < float(strict_min_iou):
        if raw_iou >= float(strict_min_iou):
            return "refinement_hurts"
        return "detection_support_missing"
    if _aspect_ratio(best.box) >= 3.0:
        return "high_aspect_target"
    if int(best.near_boundary):
        return "near_boundary_target"
    return "matched"


def _summary(
    rows: list[dict[str, Any]],
    *,
    candidate_events: int,
    evaluated_events: int,
    skipped: Counter[str],
    root: str,
    objectness_profile: str,
    strict_min_iou: float,
) -> dict[str, Any]:
    visible = [row for row in rows if int(row["gt_visible"])]
    reappear = [row for row in visible if row["phase"] == "reappear"]
    dominant = _dominant_failure_mode(reappear, strict_min_iou)
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
        "recall_at_reappear_iou025": _mean([float(row["best_iou"]) >= 0.25 for row in reappear]),
        "mean_best_iou_reappear": _mean([row["best_iou"] for row in reappear]),
        "mean_best_quality_rank_reappear": _mean([row["best_iou_rank_by_quality"] for row in reappear if int(row["best_iou_rank_by_quality"]) > 0]),
        "mean_best_score_rank_reappear": _mean([row["best_iou_rank_by_score"] for row in reappear if int(row["best_iou_rank_by_score"]) > 0]),
        "mean_refinement_delta_iou_reappear": _mean([row["refinement_delta_iou"] for row in reappear]),
        "fraction_refinement_hurts_iou": _mean([float(row["refinement_delta_iou"]) < -1e-6 for row in reappear]),
        "fraction_best_near_boundary": _mean([row["best_near_boundary"] for row in reappear]),
        "fraction_best_high_aspect_ratio": _mean([float(row["best_aspect_ratio"]) >= 3.0 for row in reappear]),
        "dominant_failure_mode": dominant,
    }


def _dominant_failure_mode(rows: list[dict[str, Any]], strict_min_iou: float) -> str:
    if not rows:
        return "no_reappear_rows"
    counts: Counter[str] = Counter()
    for row in rows:
        best_iou = float(row["best_iou"])
        raw_iou = float(row["raw_box_iou"])
        rank = int(row["best_iou_rank_by_quality"])
        if best_iou < float(strict_min_iou):
            counts["refinement_hurts" if raw_iou >= float(strict_min_iou) else "detection_support_missing"] += 1
        elif rank > 16:
            counts["quality_misranking"] += 1
        else:
            counts["mixed"] += 1
    return counts.most_common(1)[0][0]


def _box_area_ratio(box: tuple[float, float, float, float] | None, frame_shape: tuple[int, int]) -> float:
    if box is None:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, float(frame_shape[0] * frame_shape[1]))


def _aspect_ratio(box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    ratio = width / height
    return float(max(ratio, 1.0 / max(ratio, 1e-6)))


def _fmt_box(box: tuple[int, int, int, int]) -> str:
    return ",".join(str(int(v)) for v in box)


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
    return (
        "# LaSOT Component Ranking Audit\n\n"
        f"- objectness_profile: `{summary['objectness_profile']}`\n"
        f"- evaluated_event_count: {summary['evaluated_event_count']}\n"
        f"- recall_at_reappear_iou025: {summary['recall_at_reappear_iou025']:.4f}\n"
        f"- mean_best_quality_rank_reappear: {summary['mean_best_quality_rank_reappear']:.2f}\n"
        f"- mean_refinement_delta_iou_reappear: {summary['mean_refinement_delta_iou_reappear']:.4f}\n"
        f"- fraction_refinement_hurts_iou: {summary['fraction_refinement_hurts_iou']:.4f}\n"
        f"- fraction_best_high_aspect_ratio: {summary['fraction_best_high_aspect_ratio']:.4f}\n"
        f"- fraction_best_near_boundary: {summary['fraction_best_near_boundary']:.4f}\n"
        f"- dominant_failure_mode: `{summary['dominant_failure_mode']}`\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_component_ranking_audit")
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--pre-context", type=int, default=80)
    parser.add_argument("--post-context", type=int, default=20)
    parser.add_argument("--objectness-profile", default="A8_quantile_q050_component_props48")
    parser.add_argument("--max-image-side", type=int, default=160)
    parser.add_argument("--strict-min-iou", type=float, default=0.25)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--frame-stride", type=int, default=1)
    summary = run_audit(**vars(parser.parse_args()))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
