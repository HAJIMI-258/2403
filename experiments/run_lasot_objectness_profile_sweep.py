"""Sweep GT-free objectness profiles on LaSOT event windows.

This script is an evaluation gate, not a model merge. It uses LaSOT GT boxes
only to audit proposal recall after each profile has produced proposals from
the online-visible frames.
"""

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
from nops_owr.encoder.spike_encoder import MinimalSpikeEncoder, SpikeEncoding  # noqa: E402
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
from nops_owr.evaluation.objectness_profiles import (  # noqa: E402
    build_objectness_from_profile,
    objectness_profile_specs,
)
from nops_owr.evaluation.reentry_audit import bbox_iou, gap_bucket  # noqa: E402


PROFILE_SUMMARY_FIELDS = [
    "profile_name",
    "evaluated_event_count",
    "visible_frame_count",
    "proposal_recall_iou_025",
    "proposal_recall_iou_050",
    "recall_before_disappear",
    "recall_at_reappear",
    "mean_best_iou",
    "mean_proposal_count",
    "selected_as_best",
]

FRAME_FIELDS = [
    "profile_name",
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
    "gap_bucket",
]


def run_sweep(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_objectness_profile_sweep",
    max_events: int = 20,
    min_gap: int = 8,
    pre_context: int = 80,
    post_context: int = 20,
    max_image_side: int = 160,
    category_filter: str = "",
    sequence_filter: str = "",
    frame_stride: int = 2,
    profile_filter: str = "",
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
    requested_profiles = _parse_profile_filter(profile_filter)
    all_profiles = objectness_profile_specs()
    known_profiles = {str(profile["profile_name"]) for profile in all_profiles}
    missing_profiles = sorted(requested_profiles - known_profiles)
    if missing_profiles:
        raise ValueError(f"Unknown objectness profile(s): {', '.join(missing_profiles)}")
    profiles = [
        profile
        for profile in all_profiles
        if not requested_profiles or str(profile["profile_name"]) in requested_profiles
    ]
    rows_by_profile: dict[str, list[dict[str, Any]]] = {profile["profile_name"]: [] for profile in profiles}
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
        encoded_rows = _encode_window(window.frames, max_image_side=max_image_side)
        for profile in profiles:
            objectness = build_objectness_from_profile(str(profile["profile_name"]))
            profile_name = str(profile["profile_name"])
            objectness.reset()
            for encoded in encoded_rows:
                output = objectness.compute(encoded["encoding"])
                gt_box = encoded["gt_box"]
                visible = bool(encoded["visible"] and gt_box is not None)
                best_iou = max((bbox_iou(proposal.box, gt_box) for proposal in output.proposals), default=0.0)
                rows_by_profile[profile_name].append(
                    {
                        "profile_name": profile_name,
                        "sequence_id": event.sequence_id,
                        "event_id": f"{event.sequence_id}:{event.disappear_frame}:{event.reappear_frame}",
                        "category": sequence_category(event.sequence_id),
                        "frame_idx": encoded["frame_idx"],
                        "phase": frame_phase(encoded["frame"], event),
                        "gt_visible": int(visible),
                        "proposal_count": len(output.proposals),
                        "best_iou": best_iou if visible else 0.0,
                        "recalled_025": int(visible and best_iou >= 0.25),
                        "recalled_050": int(visible and best_iou >= 0.50),
                        "gap_bucket": gap_bucket(event.gap_length),
                    }
                )

    profile_summaries = [
        _profile_summary(profile_name, rows, evaluated_events=evaluated_events)
        for profile_name, rows in rows_by_profile.items()
    ]
    best_profile = _select_best_profile(profile_summaries)
    for row in profile_summaries:
        row["selected_as_best"] = int(row["profile_name"] == best_profile)
    frame_rows = [row for rows in rows_by_profile.values() for row in rows]
    compact = {
        "stage": "LASOT_OBJECTNESS_PROFILE_SWEEP",
        "total_candidate_events": len(events),
        "evaluated_event_count": evaluated_events,
        "skipped_event_count": int(sum(skipped.values())),
        "skip_reasons": dict(skipped),
        "best_profile": best_profile,
        "best_profile_summary": next((row for row in profile_summaries if row["profile_name"] == best_profile), {}),
        "profile_count": len(profile_summaries),
        "profile_filter": profile_filter,
        "next_recommendation": "run event-window eval with selected profile if recall_at_reappear improves cleanly",
    }
    _write_csv(out / "profile_summary.csv", profile_summaries, PROFILE_SUMMARY_FIELDS)
    _write_csv(out / "profile_frame_recall.csv", frame_rows, FRAME_FIELDS)
    (out / "summary.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(_report(compact, profile_summaries), encoding="utf-8")
    return compact


def _encode_window(frames: list[Any], *, max_image_side: int) -> list[dict[str, Any]]:
    encoder = MinimalSpikeEncoder()
    rows: list[dict[str, Any]] = []
    prev_image = None
    for frame in frames:
        current_image, scale_x, scale_y = load_rgb_frame(frame.frame_path, max_image_side=max_image_side)
        if prev_image is None:
            prev_image = current_image
        encoding = encoder.encode(prev_image, current_image)
        rows.append(
            {
                "frame": frame,
                "frame_idx": frame.frame_idx,
                "visible": frame_is_visible(frame),
                "gt_box": scale_box(frame_gt_box(frame), scale_x, scale_y),
                "encoding": encoding,
            }
        )
        prev_image = current_image
    return rows


def _profile_summary(profile_name: str, rows: list[dict[str, Any]], *, evaluated_events: int) -> dict[str, Any]:
    visible = [row for row in rows if int(row["gt_visible"])]
    pre = [row for row in visible if row["phase"] == "pre_visible"]
    reappear = [row for row in visible if row["phase"] == "reappear"]
    return {
        "profile_name": profile_name,
        "evaluated_event_count": int(evaluated_events),
        "visible_frame_count": len(visible),
        "proposal_recall_iou_025": _mean([row["recalled_025"] for row in visible]),
        "proposal_recall_iou_050": _mean([row["recalled_050"] for row in visible]),
        "recall_before_disappear": _mean([row["recalled_025"] for row in pre]),
        "recall_at_reappear": _mean([row["recalled_025"] for row in reappear]),
        "mean_best_iou": _mean([row["best_iou"] for row in visible]),
        "mean_proposal_count": _mean([row["proposal_count"] for row in rows]),
        "selected_as_best": 0,
    }


def _select_best_profile(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    best = max(
        rows,
        key=lambda row: (
            float(row["recall_at_reappear"]),
            float(row["proposal_recall_iou_025"]),
            float(row["mean_best_iou"]),
            -float(row["mean_proposal_count"]),
        ),
    )
    return str(best["profile_name"])


def _parse_profile_filter(profile_filter: str) -> set[str]:
    if not profile_filter:
        return set()
    return {item.strip() for item in profile_filter.split(",") if item.strip()}


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


def _report(compact: dict[str, Any], profile_summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# LaSOT Objectness Profile Sweep",
        "",
        "GT boxes are used only to audit proposal recall. Profiles do not use GT online.",
        "",
        f"- evaluated_event_count: {compact['evaluated_event_count']}",
        f"- best_profile: `{compact['best_profile']}`",
        "",
        "| profile | reappear@0.25 | visible@0.25 | visible@0.50 | mean_iou | mean_props |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in profile_summaries:
        lines.append(
            f"| {row['profile_name']} | {float(row['recall_at_reappear']):.4f} | "
            f"{float(row['proposal_recall_iou_025']):.4f} | {float(row['proposal_recall_iou_050']):.4f} | "
            f"{float(row['mean_best_iou']):.4f} | {float(row['mean_proposal_count']):.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_objectness_profile_sweep")
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--pre-context", type=int, default=80)
    parser.add_argument("--post-context", type=int, default=20)
    parser.add_argument("--max-image-side", type=int, default=160)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument(
        "--profile-filter",
        default="",
        help="Comma-separated objectness profile names to run. Empty runs all profiles.",
    )
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
        profile_filter=args.profile_filter,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
