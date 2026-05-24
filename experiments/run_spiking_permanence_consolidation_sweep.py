"""Sweep pre-disappearance capsule consolidation observations.

The sweep tests whether multiple stable online observations improve a fixed-size
capsule without increasing the number of long-term memory slots.
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

from experiments.run_spiking_morph_permanence_eval import run_eval  # noqa: E402


SUMMARY_FIELDS = [
    "context_observations_per_object",
    "match_profile",
    "same_object_threshold",
    "same_object_margin_threshold",
    "false_resurrection_risk_threshold",
    "same_instance_reentry_recall",
    "false_resurrection_rate",
    "accepted_reentry_decision_count",
    "false_resurrection_count",
    "top1_true_capsule_rate",
    "true_capsule_top3_rate",
    "true_capsule_top5_rate",
    "top1_true_but_not_accepted_rate",
    "mean_score_gap_top1_minus_true",
    "bytes_per_capsule",
    "mean_spike_density",
    "memory_growth_rate",
    "selected_as_best_safe",
]


def run_sweep(
    output_dir: str | Path = "results/spiking_permanence_consolidation_sweep",
    seed: int = 7,
    object_count: int = 16,
    events_per_object: int = 4,
    max_capsules: int = 32,
    spike_dim: int = 128,
    match_profile: str = "hash_chroma_deform",
    same_object_threshold: float = 0.90,
    same_object_margin_threshold: float = 0.14,
    false_resurrection_risk_threshold: float = 0.25,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    observation_counts = [1, 2, 4, 6, 8]
    rows: list[dict[str, Any]] = []
    for count in observation_counts:
        summary = run_eval(
            output_dir=out / f"context_{count}",
            seed=seed,
            object_count=object_count,
            events_per_object=events_per_object,
            max_capsules=max_capsules,
            spike_dim=spike_dim,
            match_profile=match_profile,
            same_object_threshold=same_object_threshold,
            same_object_margin_threshold=same_object_margin_threshold,
            false_resurrection_risk_threshold=false_resurrection_risk_threshold,
            context_observations_per_object=count,
        )
        rows.append({field: summary.get(field, 0.0) for field in SUMMARY_FIELDS} | {"selected_as_best_safe": 0})
    best = _select_best(rows)
    if best is not None:
        best["selected_as_best_safe"] = 1
    summary_json = {
        "config_count": len(rows),
        "best_safe_config": dict(best or {}),
        "baseline_context_1": dict(next((row for row in rows if int(row["context_observations_per_object"]) == 1), {})),
        "main_diagnosis": _diagnosis(rows, best),
    }
    _write_csv(out / "consolidation_summary.csv", rows, SUMMARY_FIELDS)
    (out / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(_report(summary_json), encoding="utf-8")
    return summary_json


def _select_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    nonzero = [row for row in rows if float(row["same_instance_reentry_recall"]) > 0.0]
    candidates = nonzero if nonzero else rows
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            float(row["false_resurrection_rate"]),
            -float(row["same_instance_reentry_recall"]),
            -float(row["top1_true_capsule_rate"]),
            float(row["context_observations_per_object"]),
        ),
    )


def _diagnosis(rows: list[dict[str, Any]], best: dict[str, Any] | None) -> str:
    baseline = next((row for row in rows if int(row["context_observations_per_object"]) == 1), None)
    if baseline is None or best is None:
        return "missing_consolidation_rows"
    recall_gain = float(best["same_instance_reentry_recall"]) - float(baseline["same_instance_reentry_recall"])
    false_delta = float(best["false_resurrection_rate"]) - float(baseline["false_resurrection_rate"])
    if recall_gain > 0.0 and false_delta <= 0.0:
        return "multi_observation_consolidation_improves_safe_recall"
    if recall_gain > 0.0:
        return "multi_observation_consolidation_improves_recall_with_false_tradeoff"
    return "multi_observation_consolidation_no_gain"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _report(summary: dict[str, Any]) -> str:
    best = summary.get("best_safe_config", {})
    baseline = summary.get("baseline_context_1", {})
    return (
        "# Spiking Permanence Consolidation Sweep\n\n"
        f"- baseline_recall: {float(baseline.get('same_instance_reentry_recall', 0.0)):.4f}\n"
        f"- baseline_false_resurrection: {float(baseline.get('false_resurrection_rate', 0.0)):.4f}\n"
        f"- best_context_observations: {best.get('context_observations_per_object', '')}\n"
        f"- best_recall: {float(best.get('same_instance_reentry_recall', 0.0)):.4f}\n"
        f"- best_false_resurrection: {float(best.get('false_resurrection_rate', 0.0)):.4f}\n"
        f"- main_diagnosis: {summary.get('main_diagnosis', '')}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/spiking_permanence_consolidation_sweep")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--object-count", type=int, default=16)
    parser.add_argument("--events-per-object", type=int, default=4)
    parser.add_argument("--max-capsules", type=int, default=32)
    parser.add_argument("--spike-dim", type=int, default=128)
    parser.add_argument("--match-profile", default="hash_chroma_deform")
    parser.add_argument("--same-object-threshold", type=float, default=0.90)
    parser.add_argument("--same-object-margin-threshold", type=float, default=0.14)
    parser.add_argument("--false-resurrection-risk-threshold", type=float, default=0.25)
    summary = run_sweep(**vars(parser.parse_args()))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
