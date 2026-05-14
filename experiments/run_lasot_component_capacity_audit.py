"""Sweep high-capacity component-only objectness profiles on LaSOT events.

This audit isolates top-k truncation from genuine support failure. If recall
improves when only max_proposals/threshold capacity increases, the target was
likely present in low-ranked components. If it does not, the bottom-up support
map itself is missing the target.
"""

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

from experiments.run_lasot_component_ranking_audit import run_audit  # noqa: E402


SUMMARY_FIELDS = [
    "objectness_profile",
    "evaluated_event_count",
    "recall_at_reappear_iou025",
    "mean_best_iou_reappear",
    "mean_best_quality_rank_reappear",
    "mean_best_score_rank_reappear",
    "mean_proposal_count_reappear",
    "fraction_refinement_hurts_iou",
    "fraction_best_near_boundary",
    "fraction_best_high_aspect_ratio",
    "dominant_failure_mode",
    "selected_as_best",
]


def run_capacity_audit(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_component_capacity_audit",
    max_events: int = 20,
    min_gap: int = 8,
    pre_context: int = 80,
    post_context: int = 20,
    profiles: str = "A8_quantile_q050_component_props48,A10_quantile_q050_component_props96,A11_quantile_q045_component_props128",
    max_image_side: int = 160,
    category_filter: str = "",
    sequence_filter: str = "",
    frame_stride: int = 1,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile_names = [item.strip() for item in profiles.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    for profile in profile_names:
        summary = run_audit(
            root=root,
            output_dir=out / profile,
            max_events=max_events,
            min_gap=min_gap,
            pre_context=pre_context,
            post_context=post_context,
            objectness_profile=profile,
            max_image_side=max_image_side,
            category_filter=category_filter,
            sequence_filter=sequence_filter,
            frame_stride=frame_stride,
        )
        rows.append(_row(profile, summary))
    best = _select_best(rows)
    baseline = rows[0] if rows else {}
    for row in rows:
        row["selected_as_best"] = int(row["objectness_profile"] == best)
    compact = {
        "stage": "LASOT_COMPONENT_CAPACITY_AUDIT",
        "profiles": rows,
        "best_objectness_profile": best,
        "best_profile_summary": next((row for row in rows if row["objectness_profile"] == best), {}),
        "baseline_profile": baseline.get("objectness_profile", ""),
        "baseline_recall_at_reappear_iou025": baseline.get("recall_at_reappear_iou025", 0.0),
        "capacity_improves_recall": int(
            any(
                float(row["recall_at_reappear_iou025"]) > float(baseline.get("recall_at_reappear_iou025", 0.0))
                for row in rows[1:]
            )
        ),
        "next_recommendation": _recommendation(rows),
    }
    _write_csv(out / "capacity_summary.csv", rows, SUMMARY_FIELDS)
    (out / "summary.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(_report(compact, rows), encoding="utf-8")
    return compact


def _row(profile: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "objectness_profile": profile,
        "evaluated_event_count": summary.get("evaluated_event_count", 0),
        "recall_at_reappear_iou025": summary.get("recall_at_reappear_iou025", 0.0),
        "mean_best_iou_reappear": summary.get("mean_best_iou_reappear", 0.0),
        "mean_best_quality_rank_reappear": summary.get("mean_best_quality_rank_reappear", 0.0),
        "mean_best_score_rank_reappear": summary.get("mean_best_score_rank_reappear", 0.0),
        "mean_proposal_count_reappear": summary.get("mean_proposal_count_reappear", 0.0),
        "fraction_refinement_hurts_iou": summary.get("fraction_refinement_hurts_iou", 0.0),
        "fraction_best_near_boundary": summary.get("fraction_best_near_boundary", 0.0),
        "fraction_best_high_aspect_ratio": summary.get("fraction_best_high_aspect_ratio", 0.0),
        "dominant_failure_mode": summary.get("dominant_failure_mode", ""),
        "selected_as_best": 0,
    }


def _select_best(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    best = max(
        rows,
        key=lambda row: (
            float(row["recall_at_reappear_iou025"]),
            float(row["mean_best_iou_reappear"]),
            -float(row["mean_proposal_count_reappear"]),
        ),
    )
    return str(best["objectness_profile"])


def _recommendation(rows: list[dict[str, Any]]) -> str:
    if len(rows) < 2:
        return "run more capacity profiles"
    baseline = float(rows[0]["recall_at_reappear_iou025"])
    best = max(float(row["recall_at_reappear_iou025"]) for row in rows)
    if best > baseline:
        return "top-k truncation contributes; evaluate calibrated ranking before increasing default capacity"
    return "capacity does not recover targets; inspect bottom-up support map / thresholding"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _report(compact: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# LaSOT Component Capacity Audit",
        "",
        f"- best_objectness_profile: `{compact['best_objectness_profile']}`",
        f"- capacity_improves_recall: {compact['capacity_improves_recall']}",
        f"- next_recommendation: {compact['next_recommendation']}",
        "",
        "| profile | recall@reappear | mean_iou | quality_rank | score_rank | mean_props | dominant_failure |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['objectness_profile']} | {float(row['recall_at_reappear_iou025']):.4f} | "
            f"{float(row['mean_best_iou_reappear']):.4f} | {float(row['mean_best_quality_rank_reappear']):.2f} | "
            f"{float(row['mean_best_score_rank_reappear']):.2f} | {float(row['mean_proposal_count_reappear']):.2f} | "
            f"{row['dominant_failure_mode']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_component_capacity_audit")
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--pre-context", type=int, default=80)
    parser.add_argument("--post-context", type=int, default=20)
    parser.add_argument(
        "--profiles",
        default="A8_quantile_q050_component_props48,A10_quantile_q050_component_props96,A11_quantile_q045_component_props128",
    )
    parser.add_argument("--max-image-side", type=int, default=160)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--frame-stride", type=int, default=1)
    summary = run_capacity_audit(**vars(parser.parse_args()))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
