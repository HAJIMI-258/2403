"""Sweep GT-free proposal box/refinement profiles on LaSOT event windows."""

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
from nops_owr.evaluation.support_refinement_profiles import support_refinement_profile_specs  # noqa: E402


SUMMARY_FIELDS = [
    "support_box_profile",
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
    "mean_proposal_count",
    "selected_as_best",
]


def run_sweep(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_support_refinement_sweep",
    max_events: int = 20,
    min_gap: int = 8,
    pre_context: int = 80,
    post_context: int = 20,
    objectness_profile: str = "A8_quantile_q050_component_props48",
    attention_profile: str = "A10_source_spatial_diverse_max16",
    component_ranking_profile: str = "R0_current_quality",
    max_image_side: int = 160,
    frame_stride: int = 1,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for profile in support_refinement_profile_specs():
        summary = run_eval(
            root=root,
            output_dir=out / profile,
            max_events=max_events,
            min_gap=min_gap,
            pre_context=pre_context,
            post_context=post_context,
            max_image_side=max_image_side,
            frame_stride=frame_stride,
            objectness_profile=objectness_profile,
            attention_profile=attention_profile,
            component_ranking_profile=component_ranking_profile,
            support_box_profile=profile,
        )
        rows.append(_row(profile, summary))
    best = _select_best(rows)
    for row in rows:
        row["selected_as_best"] = int(row["support_box_profile"] == best)
    compact = {
        "stage": "LASOT_SUPPORT_REFINEMENT_SWEEP",
        "objectness_profile": objectness_profile,
        "attention_profile": attention_profile,
        "component_ranking_profile": component_ranking_profile,
        "best_support_box_profile": best,
        "best_profile_summary": next((row for row in rows if row["support_box_profile"] == best), {}),
        "profiles": rows,
    }
    _write_csv(out / "support_refinement_summary.csv", rows, SUMMARY_FIELDS)
    (out / "summary.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(_report(compact, rows), encoding="utf-8")
    return compact


def _row(profile: str, summary: dict[str, Any]) -> dict[str, Any]:
    buckets = summary.get("failure_buckets", {}) or {}
    return {
        "support_box_profile": profile,
        "evaluated_event_count": summary.get("evaluated_event_count", summary.get("reentry_event_count", 0)),
        "proposal_recall_at_reentry": summary.get("proposal_recall_at_reentry", 0.0),
        "attention_recall_at_reentry": summary.get("attention_recall_at_reentry", 0.0),
        "target_episode_presence_rate": summary.get("target_episode_presence_rate", 0.0),
        "target_episode_top5_rate": summary.get("target_episode_top5_rate", 0.0),
        "same_instance_recall_at_reentry": summary.get("same_instance_recall_at_reentry", 0.0),
        "false_resurrection_rate_at_reentry": summary.get("false_resurrection_rate_at_reentry", 0.0),
        "no_object_file_matched_count": buckets.get("no_object_file_matched", 0),
        "attention_missed_object_count": buckets.get("attention_missed_object", 0),
        "target_in_topk_but_low_rank_count": buckets.get("target_in_topk_but_low_rank", 0),
        "mean_proposal_count": summary.get("mean_proposal_count", 0.0),
        "selected_as_best": 0,
    }


def _select_best(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    best = max(
        rows,
        key=lambda row: (
            -float(row["false_resurrection_rate_at_reentry"]),
            float(row["proposal_recall_at_reentry"]),
            float(row["attention_recall_at_reentry"]),
            float(row["target_episode_top5_rate"]),
            -float(row["no_object_file_matched_count"]),
        ),
    )
    return str(best["support_box_profile"])


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _report(compact: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# LaSOT Support Refinement Sweep",
        "",
        f"- objectness_profile: `{compact['objectness_profile']}`",
        f"- attention_profile: `{compact['attention_profile']}`",
        f"- component_ranking_profile: `{compact['component_ranking_profile']}`",
        f"- best_support_box_profile: `{compact['best_support_box_profile']}`",
        "",
        "| profile | proposal | attention | top5 | same_instance | false_res | no_object |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['support_box_profile']} | {float(row['proposal_recall_at_reentry']):.4f} | "
            f"{float(row['attention_recall_at_reentry']):.4f} | {float(row['target_episode_top5_rate']):.4f} | "
            f"{float(row['same_instance_recall_at_reentry']):.4f} | {float(row['false_resurrection_rate_at_reentry']):.4f} | "
            f"{int(row['no_object_file_matched_count'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_support_refinement_sweep")
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--pre-context", type=int, default=80)
    parser.add_argument("--post-context", type=int, default=20)
    parser.add_argument("--objectness-profile", default="A8_quantile_q050_component_props48")
    parser.add_argument("--attention-profile", default="A10_source_spatial_diverse_max16")
    parser.add_argument("--component-ranking-profile", default="R0_current_quality")
    parser.add_argument("--max-image-side", type=int, default=160)
    parser.add_argument("--frame-stride", type=int, default=1)
    summary = run_sweep(**vars(parser.parse_args()))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
