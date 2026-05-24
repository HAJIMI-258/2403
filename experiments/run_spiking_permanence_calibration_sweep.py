"""Threshold calibration sweep for spiking object permanence.

This script does not add model capacity. It runs the same bounded capsule
memory under different permanence decision thresholds to expose the
recall/false-resurrection tradeoff.
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
    "config_id",
    "same_object_threshold",
    "same_object_margin_threshold",
    "false_resurrection_risk_threshold",
    "same_instance_reentry_recall",
    "false_resurrection_rate",
    "accepted_reentry_decision_count",
    "false_resurrection_count",
    "top1_true_capsule_rate",
    "top1_true_but_not_accepted_rate",
    "uncertain_hold_rate",
    "false_resurrection_risk_decision_rate",
    "mean_top1_margin",
    "mean_false_resurrection_risk",
    "bytes_per_capsule",
    "mean_spike_density",
    "memory_growth_rate",
    "selected_as_best_safe",
]


def run_sweep(
    output_dir: str | Path = "results/spiking_permanence_calibration_sweep",
    seed: int = 7,
    object_count: int = 16,
    events_per_object: int = 4,
    max_capsules: int = 32,
    spike_dim: int = 128,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    thresholds = [0.72, 0.80, 0.86, 0.90, 0.92]
    margins = [0.04, 0.06, 0.10]
    risk_thresholds = [0.30, 0.35, 0.45]
    rows: list[dict[str, Any]] = []
    config_index = 0
    for same_threshold in thresholds:
        for margin_threshold in margins:
            for risk_threshold in risk_thresholds:
                config_id = f"C{config_index:03d}"
                config_dir = out / config_id
                summary = run_eval(
                    output_dir=config_dir,
                    seed=seed,
                    object_count=object_count,
                    events_per_object=events_per_object,
                    max_capsules=max_capsules,
                    spike_dim=spike_dim,
                    same_object_threshold=same_threshold,
                    same_object_margin_threshold=margin_threshold,
                    false_resurrection_risk_threshold=risk_threshold,
                )
                rows.append(
                    {
                        "config_id": config_id,
                        **{field: summary.get(field, 0.0) for field in SUMMARY_FIELDS if field != "config_id"},
                        "selected_as_best_safe": 0,
                    }
                )
                config_index += 1

    best = _select_best_safe(rows)
    if best is not None:
        best["selected_as_best_safe"] = 1
    strict_lowest_false = min(rows, key=lambda row: (float(row["false_resurrection_rate"]), -float(row["same_instance_reentry_recall"]))) if rows else {}
    summary_json = {
        "config_count": len(rows),
        "best_safe_config": dict(best or {}),
        "best_recall_config": dict(max(rows, key=lambda row: (float(row["same_instance_reentry_recall"]), -float(row["false_resurrection_rate"]))) if rows else {}),
        "lowest_false_resurrection_config": dict(strict_lowest_false),
    }
    _write_csv(out / "sweep_summary.csv", rows, SUMMARY_FIELDS)
    (out / "sweep_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "sweep_report.md").write_text(_report(summary_json), encoding="utf-8")
    return summary_json


def _select_best_safe(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    # Safety first, but avoid selecting a trivial no-acceptance configuration as
    # the actionable best setting. Strict zero-false configs are still reported
    # separately as lowest_false_resurrection_config.
    nonzero_recall = [row for row in rows if float(row["same_instance_reentry_recall"]) > 0.0]
    candidates = nonzero_recall if nonzero_recall else rows
    return min(
        candidates,
        key=lambda row: (
            float(row["false_resurrection_rate"]),
            -float(row["same_instance_reentry_recall"]),
            -float(row["top1_true_capsule_rate"]),
            float(row["mean_spike_density"]),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _report(summary: dict[str, Any]) -> str:
    best_safe = summary.get("best_safe_config", {})
    best_recall = summary.get("best_recall_config", {})
    return (
        "# Spiking Permanence Calibration Sweep\n\n"
        f"- config_count: {summary.get('config_count', 0)}\n"
        f"- best_safe_config: {best_safe.get('config_id', '')}\n"
        f"- best_safe_recall: {float(best_safe.get('same_instance_reentry_recall', 0.0)):.4f}\n"
        f"- best_safe_false_resurrection: {float(best_safe.get('false_resurrection_rate', 0.0)):.4f}\n"
        f"- best_recall_config: {best_recall.get('config_id', '')}\n"
        f"- best_recall: {float(best_recall.get('same_instance_reentry_recall', 0.0)):.4f}\n"
        f"- best_recall_false_resurrection: {float(best_recall.get('false_resurrection_rate', 0.0)):.4f}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/spiking_permanence_calibration_sweep")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--object-count", type=int, default=16)
    parser.add_argument("--events-per-object", type=int, default=4)
    parser.add_argument("--max-capsules", type=int, default=32)
    parser.add_argument("--spike-dim", type=int, default=128)
    summary = run_sweep(**vars(parser.parse_args()))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
