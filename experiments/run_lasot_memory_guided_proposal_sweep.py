"""Sweep memory-guided proposal profiles on LaSOT event windows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_lasot_event_window_eval import run_eval  # noqa: E402
from nops_owr.evaluation.memory_guided_profiles import memory_guided_profile_specs  # noqa: E402


SUMMARY_FIELDS = [
    "memory_guided_profile",
    "evaluated_event_count",
    "proposal_recall_at_reentry",
    "attention_recall_at_reentry",
    "target_episode_presence_rate",
    "target_episode_top5_rate",
    "same_instance_recall_at_reentry",
    "false_resurrection_rate_at_reentry",
    "no_object_file_matched_count",
    "attention_missed_object_count",
    "target_in_topk_but_low_rank_count",
    "mean_memory_guided_proposal_count",
    "mean_proposal_count",
    "selected_as_best",
]


def run_sweep(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_memory_guided_proposal_sweep",
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
    memory_guided_attention: int = 1,
    profile_filter: str = "",
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    requested_profiles = _parse_profile_filter(profile_filter)
    all_profiles = memory_guided_profile_specs()
    known_profiles = {str(profile["profile_name"]) for profile in all_profiles}
    missing_profiles = sorted(requested_profiles - known_profiles)
    if missing_profiles:
        raise ValueError(f"Unknown profile(s): {', '.join(missing_profiles)}")
    profiles = [
        profile
        for profile in all_profiles
        if not requested_profiles or str(profile["profile_name"]) in requested_profiles
    ]
    for profile in profiles:
        profile_name = str(profile["profile_name"])
        subdir = out / profile_name
        summary = run_eval(
            root=root,
            output_dir=subdir,
            max_events=max_events,
            min_gap=min_gap,
            pre_context=pre_context,
            post_context=post_context,
            category_filter=category_filter,
            sequence_filter=sequence_filter,
            max_image_side=max_image_side,
            strict_min_iou=strict_min_iou,
            frame_stride=frame_stride,
            objectness_profile=objectness_profile,
            attention_profile=attention_profile,
            memory_guided_profile=profile_name,
            memory_guided_attention=memory_guided_attention,
        )
        buckets = summary.get("failure_buckets", {}) or {}
        rows.append(
            {
                "memory_guided_profile": profile_name,
                "evaluated_event_count": int(summary.get("evaluated_event_count", summary.get("reentry_event_count", 0))),
                "proposal_recall_at_reentry": float(summary.get("proposal_recall_at_reentry", 0.0)),
                "attention_recall_at_reentry": float(summary.get("attention_recall_at_reentry", 0.0)),
                "target_episode_presence_rate": float(summary.get("target_episode_presence_rate", 0.0)),
                "target_episode_top5_rate": float(summary.get("target_episode_top5_rate", 0.0)),
                "same_instance_recall_at_reentry": float(summary.get("same_instance_recall_at_reentry", 0.0)),
                "false_resurrection_rate_at_reentry": float(summary.get("false_resurrection_rate_at_reentry", 0.0)),
                "no_object_file_matched_count": int(buckets.get("no_object_file_matched", 0)),
                "attention_missed_object_count": int(buckets.get("attention_missed_object", 0)),
                "target_in_topk_but_low_rank_count": int(buckets.get("target_in_topk_but_low_rank", 0)),
                "mean_memory_guided_proposal_count": float(summary.get("mean_memory_guided_proposal_count", 0.0)),
                "mean_proposal_count": float(summary.get("mean_proposal_count", 0.0)),
                "selected_as_best": 0,
            }
        )
    best = _select_best(rows)
    for row in rows:
        row["selected_as_best"] = int(row["memory_guided_profile"] == best)
    compact = {
        "stage": "LASOT_MEMORY_GUIDED_PROPOSAL_SWEEP",
        "evaluated_event_count": rows[0]["evaluated_event_count"] if rows else 0,
        "objectness_profile": objectness_profile,
        "attention_profile": attention_profile,
        "memory_guided_attention": int(memory_guided_attention),
        "profile_filter": profile_filter,
        "best_profile": best,
        "best_profile_summary": next((row for row in rows if row["memory_guided_profile"] == best), {}),
        "profile_count": len(rows),
        "next_recommendation": "use source-level audit before merging any memory-guided proposal profile",
    }
    _write_csv(out / "profile_summary.csv", rows, SUMMARY_FIELDS)
    (out / "summary.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(_report(compact, rows), encoding="utf-8")
    return compact


def _select_best(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    best = max(
        rows,
        key=lambda row: (
            -float(row["false_resurrection_rate_at_reentry"]),
            float(row["target_episode_top5_rate"]),
            float(row["proposal_recall_at_reentry"]),
            -float(row["no_object_file_matched_count"]),
            -float(row["mean_proposal_count"]),
        ),
    )
    return str(best["memory_guided_profile"])


def _parse_profile_filter(profile_filter: str) -> set[str]:
    if not profile_filter:
        return set()
    return {item.strip() for item in profile_filter.split(",") if item.strip()}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _report(compact: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# LaSOT Memory-guided Proposal Sweep",
        "",
        "Memory-guided profiles are GT-free at runtime. GT is used only by the underlying event-window audit.",
        "",
        f"- best_profile: `{compact['best_profile']}`",
        "",
        "| profile | proposal@reentry | attention@reentry | target@top5 | recall | false_res | no_object | mean_props |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['memory_guided_profile']} | {float(row['proposal_recall_at_reentry']):.4f} | "
            f"{float(row['attention_recall_at_reentry']):.4f} | {float(row['target_episode_top5_rate']):.4f} | "
            f"{float(row['same_instance_recall_at_reentry']):.4f} | "
            f"{float(row['false_resurrection_rate_at_reentry']):.4f} | "
            f"{int(row['no_object_file_matched_count'])} | {float(row['mean_proposal_count']):.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_memory_guided_proposal_sweep")
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
    parser.add_argument("--memory-guided-attention", type=int, default=1)
    parser.add_argument(
        "--profile-filter",
        default="",
        help="Comma-separated memory-guided profile names to run. Empty runs all profiles.",
    )
    args = parser.parse_args()
    summary = run_sweep(**vars(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
