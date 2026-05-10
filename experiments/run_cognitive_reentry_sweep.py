"""Small diagnostic sweep for re-entry recognition gates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_cognitive_reentry_eval import run_eval  # noqa: E402


def run_sweep(
    sequences: int = 3,
    max_frames: int = 100,
    seeds: str = "41,42,43",
    output_dir: str | Path = "results/cognitive_reentry_sweep",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    seed_values = [int(value.strip()) for value in seeds.split(",") if value.strip()]
    rows: list[dict[str, Any]] = []
    config_id = 0
    for margin_threshold in (0.04, 0.08, 0.12):
        for reentry_threshold in (0.72, 0.76, 0.80):
            config_id += 1
            run_dir = output_path / f"config_{config_id:02d}"
            summary = run_eval(
                sequences=sequences,
                max_frames=max_frames,
                output_dir=run_dir,
                min_gap=8,
                seeds=seed_values,
                recognizer_kwargs={
                    "same_instance_margin_threshold": margin_threshold,
                    "reentry_same_instance_threshold": reentry_threshold,
                    "active_conflict_block": True,
                },
            )
            rows.append(
                {
                    "config_id": config_id,
                    "same_instance_margin_threshold": margin_threshold,
                    "reentry_same_instance_threshold": reentry_threshold,
                    "active_conflict_block": 1,
                    "min_gap": 8,
                    "success_rate": summary["long_gap_reentry_success_rate"],
                    "false_resurrection_rate": summary["false_resurrection_rate_at_reentry"],
                    "same_instance_precision_at_reentry": summary["same_instance_precision_at_reentry"],
                    "same_instance_recall_at_reentry": summary["same_instance_recall_at_reentry"],
                    "unresolved_but_target_in_topk_rate": summary["unresolved_but_target_in_topk_rate"],
                    "target_episode_top5_rate": summary["target_episode_top5_rate"],
                    "reentry_event_count": summary["reentry_event_count"],
                    "benchmark_status": summary["benchmark_status"],
                    "actual_reentry_event_count": summary["actual_reentry_event_count"],
                    "benchmark_valid": int(bool(summary["benchmark_valid"])),
                    "failure_buckets": json.dumps(summary["failure_buckets"], sort_keys=True),
                }
            )

    _write_csv(output_path / "sweep_summary.csv", rows)
    payload = {
        "config_count": len(rows),
        "rows": rows,
        "lowest_false_resurrection_with_nonzero_recall": _select_low_false_nonzero_recall(rows),
    }
    (output_path / "sweep_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_path / "sweep_report.md").write_text(_report(payload), encoding="utf-8")
    return payload


def _select_low_false_nonzero_recall(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if float(row["same_instance_recall_at_reentry"]) > 0.0]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            float(row["false_resurrection_rate"]),
            -float(row["same_instance_recall_at_reentry"]),
            -float(row["same_instance_precision_at_reentry"]),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _report(payload: dict[str, Any]) -> str:
    selected = payload["lowest_false_resurrection_with_nonzero_recall"]
    selected_text = "none" if selected is None else json.dumps(selected, indent=2)
    return (
        "# Cognitive Re-entry Sweep\n\n"
        "This sweep is diagnostic. It reports tradeoffs between false resurrection, "
        "re-entry recall, and unresolved target-in-top-k cases; it is not a final score claim.\n\n"
        f"Lowest false resurrection with nonzero recall:\n\n```json\n{selected_text}\n```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="41,42,43")
    parser.add_argument("--output-dir", type=str, default="results/cognitive_reentry_sweep")
    args = parser.parse_args()
    payload = run_sweep(
        sequences=args.sequences,
        max_frames=args.max_frames,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
