"""Sweep attention profiles on LaSOT event windows.

This script isolates the layer after proposal generation: if the target object
file exists, does the sparse attention gate select it for memory retrieval?
GT boxes are used only to identify which object file overlaps the target.
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
from nops_owr.cognition.object_file import ObjectFileBuilder  # noqa: E402
from nops_owr.encoder.spike_encoder import MinimalSpikeEncoder  # noqa: E402
from nops_owr.evaluation.attention_profiles import attention_profile_specs, build_attention_from_profile  # noqa: E402
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
from nops_owr.evaluation.objectness_eval_adapter import ProfiledObjectnessField  # noqa: E402
from nops_owr.evaluation.reentry_audit import bbox_iou, gap_bucket  # noqa: E402


SUMMARY_FIELDS = [
    "attention_profile",
    "evaluated_event_count",
    "target_object_file_presence_rate",
    "attention_recall_given_object_file",
    "attention_recall_at_reappear",
    "attention_recall_before_disappear",
    "mean_target_attention_rank",
    "mean_attended_count",
    "selected_as_best",
]

TRACE_FIELDS = [
    "attention_profile",
    "sequence_id",
    "event_id",
    "category",
    "frame_idx",
    "phase",
    "gt_visible",
    "target_object_file_present",
    "target_best_iou",
    "target_attention_score",
    "target_attention_rank",
    "target_attended",
    "object_file_count",
    "attended_count",
    "gap_bucket",
]


def run_sweep(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_attention_profile_sweep",
    max_events: int = 10,
    min_gap: int = 8,
    pre_context: int = 80,
    post_context: int = 20,
    max_image_side: int = 160,
    category_filter: str = "",
    sequence_filter: str = "",
    frame_stride: int = 2,
    objectness_profile: str = "A5_quantile_q060_k000_area8_props24",
    strict_min_iou: float = 0.25,
    component_ranking_profile: str = "R0_current_quality",
    support_box_profile: str = "B0_refined_box_current",
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
    builder = ObjectFileBuilder()
    profile_names = [str(profile["profile_name"]) for profile in attention_profile_specs()]
    rows_by_profile: dict[str, list[dict[str, Any]]] = {profile: [] for profile in profile_names}
    skipped = Counter()
    frames_by_sequence: dict[str, list[Any]] = {}
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
        encoded = _object_files_for_window(
            window.frames,
            encoder=encoder,
            builder=builder,
            objectness_profile=objectness_profile,
            component_ranking_profile=component_ranking_profile,
            support_box_profile=support_box_profile,
            max_image_side=max_image_side,
        )
        for attention_profile in profile_names:
            gate = build_attention_from_profile(attention_profile)
            for row in encoded:
                target_object_file = row["target_object_file"]
                target_score = 0.0
                target_rank = 0
                target_attended = 0
                attended = gate.select(row["object_files"])
                if target_object_file is not None:
                    scored = sorted(
                        [(gate.score(object_file), object_file) for object_file in row["object_files"]],
                        key=lambda item: item[0],
                        reverse=True,
                    )
                    for rank, (score, object_file) in enumerate(scored, start=1):
                        if object_file.object_file_id == target_object_file.object_file_id:
                            target_score = float(score)
                            target_rank = rank
                            break
                    target_attended = int(any(item.object_file_id == target_object_file.object_file_id for item in attended))
                rows_by_profile[attention_profile].append(
                    {
                        "attention_profile": attention_profile,
                        "sequence_id": event.sequence_id,
                        "event_id": f"{event.sequence_id}:{event.disappear_frame}:{event.reappear_frame}",
                        "category": sequence_category(event.sequence_id),
                        "frame_idx": row["frame_idx"],
                        "phase": frame_phase(row["frame"], event),
                        "gt_visible": int(row["visible"]),
                        "target_object_file_present": int(target_object_file is not None),
                        "target_best_iou": float(row["target_best_iou"]),
                        "target_attention_score": target_score,
                        "target_attention_rank": target_rank,
                        "target_attended": target_attended,
                        "object_file_count": len(row["object_files"]),
                        "attended_count": len(attended),
                        "gap_bucket": gap_bucket(event.gap_length),
                    }
                )

    summaries = [
        _profile_summary(name, rows, evaluated_events=evaluated_events)
        for name, rows in rows_by_profile.items()
    ]
    best_profile = _select_best(summaries)
    for row in summaries:
        row["selected_as_best"] = int(row["attention_profile"] == best_profile)
    trace_rows = [row for rows in rows_by_profile.values() for row in rows]
    compact = {
        "stage": "LASOT_ATTENTION_PROFILE_SWEEP",
        "objectness_profile": objectness_profile,
        "component_ranking_profile": component_ranking_profile,
        "support_box_profile": support_box_profile,
        "total_candidate_events": len(events),
        "evaluated_event_count": evaluated_events,
        "skipped_event_count": int(sum(skipped.values())),
        "skip_reasons": dict(skipped),
        "best_attention_profile": best_profile,
        "best_profile_summary": next((row for row in summaries if row["attention_profile"] == best_profile), {}),
        "next_recommendation": "run event-window eval with selected objectness+attention profiles before touching recognizer",
    }
    _write_csv(out / "attention_profile_summary.csv", summaries, SUMMARY_FIELDS)
    _write_csv(out / "attention_profile_trace.csv", trace_rows, TRACE_FIELDS)
    (out / "summary.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(_report(compact, summaries), encoding="utf-8")
    return compact


def _object_files_for_window(
    frames: list[Any],
    *,
    encoder: MinimalSpikeEncoder,
    builder: ObjectFileBuilder,
    objectness_profile: str,
    component_ranking_profile: str,
    support_box_profile: str,
    max_image_side: int,
) -> list[dict[str, Any]]:
    objectness = ProfiledObjectnessField(
        build_objectness_from_profile(objectness_profile),
        component_ranking_profile=component_ranking_profile,
        support_box_profile=support_box_profile,
    )
    objectness.reset()
    rows: list[dict[str, Any]] = []
    prev_image = None
    for frame in frames:
        current_image, scale_x, scale_y = load_rgb_frame(frame.frame_path, max_image_side=max_image_side)
        if prev_image is None:
            prev_image = current_image
        encoding = encoder.encode(prev_image, current_image)
        output = objectness.compute(encoding)
        object_files = builder.build(output, encoding, int(frame.frame_idx), current_image)
        gt_box = scale_box(frame_gt_box(frame), scale_x, scale_y)
        visible = frame_is_visible(frame) and gt_box is not None
        best = None
        best_iou = 0.0
        if visible:
            for object_file in object_files:
                iou = bbox_iou(object_file.box, gt_box)
                if iou > best_iou:
                    best = object_file
                    best_iou = iou
        rows.append(
            {
                "frame": frame,
                "frame_idx": int(frame.frame_idx),
                "visible": bool(visible),
                "object_files": object_files,
                "target_object_file": best if best_iou >= 0.25 else None,
                "target_best_iou": best_iou,
            }
        )
        prev_image = current_image
    return rows


def _profile_summary(profile: str, rows: list[dict[str, Any]], *, evaluated_events: int) -> dict[str, Any]:
    visible = [row for row in rows if int(row["gt_visible"])]
    with_object = [row for row in visible if int(row["target_object_file_present"])]
    pre = [row for row in with_object if row["phase"] == "pre_visible"]
    reappear = [row for row in visible if row["phase"] == "reappear"]
    ranks = [int(row["target_attention_rank"]) for row in with_object if int(row["target_attention_rank"]) > 0]
    return {
        "attention_profile": profile,
        "evaluated_event_count": int(evaluated_events),
        "target_object_file_presence_rate": _mean([row["target_object_file_present"] for row in visible]),
        "attention_recall_given_object_file": _mean([row["target_attended"] for row in with_object]),
        "attention_recall_at_reappear": _mean([row["target_attended"] for row in reappear]),
        "attention_recall_before_disappear": _mean([row["target_attended"] for row in pre]),
        "mean_target_attention_rank": _mean(ranks),
        "mean_attended_count": _mean([row["attended_count"] for row in rows]),
        "selected_as_best": 0,
    }


def _select_best(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    best = max(
        rows,
        key=lambda row: (
            float(row["attention_recall_at_reappear"]),
            float(row["attention_recall_given_object_file"]),
            -float(row["mean_attended_count"]),
        ),
    )
    return str(best["attention_profile"])


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


def _report(compact: dict[str, Any], summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# LaSOT Attention Profile Sweep",
        "",
        f"- objectness_profile: `{compact['objectness_profile']}`",
        f"- component_ranking_profile: `{compact.get('component_ranking_profile', 'R0_current_quality')}`",
        f"- support_box_profile: `{compact.get('support_box_profile', 'B0_refined_box_current')}`",
        f"- evaluated_event_count: {compact['evaluated_event_count']}",
        f"- best_attention_profile: `{compact['best_attention_profile']}`",
        "",
        "| profile | reappear_attention | recall_given_object | pre_attention | mean_rank | mean_attended |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['attention_profile']} | {float(row['attention_recall_at_reappear']):.4f} | "
            f"{float(row['attention_recall_given_object_file']):.4f} | "
            f"{float(row['attention_recall_before_disappear']):.4f} | "
            f"{float(row['mean_target_attention_rank']):.2f} | {float(row['mean_attended_count']):.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_attention_profile_sweep")
    parser.add_argument("--max-events", type=int, default=10)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--pre-context", type=int, default=80)
    parser.add_argument("--post-context", type=int, default=20)
    parser.add_argument("--max-image-side", type=int, default=160)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--objectness-profile", default="A5_quantile_q060_k000_area8_props24")
    parser.add_argument("--component-ranking-profile", default="R0_current_quality")
    parser.add_argument("--support-box-profile", default="B0_refined_box_current")
    args = parser.parse_args()
    summary = run_sweep(
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
        objectness_profile=args.objectness_profile,
        component_ranking_profile=args.component_ranking_profile,
        support_box_profile=args.support_box_profile,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
