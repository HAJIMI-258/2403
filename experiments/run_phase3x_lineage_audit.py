"""Run the Phase 3X lineage / continuation audit summary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3x_utils import (
    TRACK_C_NAME,
    classify_phase3x_failure_stage,
    infer_primary_loss_stage,
    plot_continuation_lifecycle,
    plot_prototype_lineage_timeline,
    plot_track_c_failure_stage,
    summarize_phase3x_audit,
    write_phase3x_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 3X final audit summary.")
    parser.add_argument("--output-dir", default="results/phase3x")
    parser.add_argument("--artifact-version", default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_rows = load_csv_rows(output_dir / f"reentry_event_trace_{args.artifact_version}.csv")
    lifecycle_rows = load_csv_rows(output_dir / f"continuation_lifecycle_{args.artifact_version}.csv")
    write_rows = load_csv_rows(output_dir / f"continuation_write_{args.artifact_version}.csv")
    prototype_rows = load_csv_rows(output_dir / f"prototype_lineage_rows_{args.artifact_version}.csv")

    for row in trace_rows:
        row["failure_stage"] = classify_phase3x_failure_stage(row)

    summary = summarize_phase3x_audit(trace_rows, write_rows)
    summary["answers"] = build_answers(summary)

    summary_path = output_dir / f"phase3x_final_audit_summary_{args.artifact_version}.json"
    write_phase3x_json(summary_path, summary)

    plot_prototype_lineage_timeline(prototype_rows, output_dir / "prototype_lineage_timeline_v1.png")
    plot_continuation_lifecycle(lifecycle_rows, output_dir / "continuation_lifecycle_v1.png")
    plot_track_c_failure_stage(trace_rows, output_dir / "track_c_event_failure_stage_v1.png")

    print(f"saved_summary={summary_path}")
    print(f"saved_lineage_plot={output_dir / 'prototype_lineage_timeline_v1.png'}")
    print(f"saved_lifecycle_plot={output_dir / 'continuation_lifecycle_v1.png'}")
    print(f"saved_failure_plot={output_dir / 'track_c_event_failure_stage_v1.png'}")
    print(
        "track_c_audit="
        f"write_success={float(summary['continuation_write_success_rate']):.4f}, "
        f"survival={float(summary['continuation_survival_until_concept_recovery_rate']):.4f}, "
        f"lineage_mismatch={float(summary['concept_recovered_but_lineage_mismatch_rate']):.4f}, "
        f"same_lineage={float(summary['same_lineage_prototype_reentry_recovery']):.4f}, "
        f"access|same_lineage={float(summary['continuation_bank_access_rate_given_same_lineage']):.4f}, "
        f"primary_loss_stage={summary['primary_loss_stage']}"
    )


def build_answers(summary: dict[str, object]) -> dict[str, object]:
    same_proto = float(summary.get("track_c", {}).get("same_prototype_reentry_recovery", 0.0))  # type: ignore[arg-type]
    same_lineage = float(summary.get("same_lineage_prototype_reentry_recovery", 0.0))
    delta = same_lineage - same_proto
    return {
        "continuation_write_success_rate": float(summary.get("continuation_write_success_rate", 0.0)),
        "continuation_survival_until_concept_recovery_rate": float(
            summary.get("continuation_survival_until_concept_recovery_rate", 0.0)
        ),
        "concept_recovered_but_lineage_mismatch_rate": float(
            summary.get("concept_recovered_but_lineage_mismatch_rate", 0.0)
        ),
        "lineage_aware_same_prototype_recovery_delta": float(delta),
        "continuation_loss_stage": str(summary.get("primary_loss_stage", infer_primary_loss_stage(summary, []))),
    }


def load_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for row in reader:
            parsed: dict[str, object] = dict(row)
            for key, value in list(parsed.items()):
                if value in ("", None):
                    parsed[key] = None
                    continue
                if key.endswith("_id") or key.endswith("_frame") or key.endswith("_count") or key in {
                    "event_id",
                    "instance_id",
                    "gap_length",
                    "proposal_detected",
                    "concept_recovered",
                    "same_prototype_id",
                    "same_lineage_id",
                    "same_track",
                    "new_track_created",
                    "new_prototype_created",
                    "continuation_written_before",
                    "continuation_alive_at_reentry",
                    "continuation_bank_nonempty",
                    "continuation_attempted",
                    "continuation_success",
                    "prototype_matched_continuation_count",
                    "lineage_matched_continuation_count",
                    "alive_same_lineage_continuation_count",
                    "alive_same_prototype_continuation_count",
                    "write_success",
                    "is_alive",
                    "age_since_write",
                    "age_since_last_seen",
                }:
                    parsed[key] = int(float(value))
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
            rows.append(parsed)
    return rows


if __name__ == "__main__":
    main()
