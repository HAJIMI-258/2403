"""Build Phase 3R.2 gap-bucket summaries and plots from re-entry audit outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3r2_utils import load_csv_rows, plot_candidate_pool, plot_reentry_vs_gap
from experiments.phase3r_utils import summarize_gap_buckets, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 3R.2 gap-bucket plots.")
    parser.add_argument("--output-dir", default="results/phase3r2", help="Directory for artifacts.")
    parser.add_argument(
        "--events-csv",
        default="results/phase3r2/reentry_events_v2.csv",
        help="Path to reentry_events_v2.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _coerce_rows(load_csv_rows(args.events_csv))
    gap_rows = summarize_gap_buckets(rows)

    gap_csv = output_dir / "reentry_gap_bucket_summary_v2.csv"
    figure_gap = output_dir / "reentry_vs_gap_v2.png"
    figure_pool = output_dir / "resurrection_candidate_pool_v1.png"

    write_csv(gap_csv, gap_rows)
    plot_reentry_vs_gap(gap_rows, figure_gap)
    plot_candidate_pool(rows, figure_pool)

    print(f"saved_gap_csv={gap_csv}")
    print(f"saved_gap_plot={figure_gap}")
    print(f"saved_pool_plot={figure_pool}")


def _coerce_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    coerced: list[dict[str, object]] = []
    int_keys = {
        "event_id",
        "instance_id",
        "old_track_id",
        "old_prototype_id",
        "disappear_frame",
        "reappear_frame",
        "gap_length",
        "proposal_detected",
        "matched_same_track",
        "matched_same_prototype",
        "new_track_created",
        "new_prototype_created",
        "reactivation_attempted",
        "concept_only_recovery",
        "concept_recovered",
        "same_track_after_concept_recovery",
        "idsw_after_reentry_window",
        "candidate_pool_size",
        "resurrection_attempted",
        "resurrection_success",
        "best_candidate_gap",
    }
    float_keys = {
        "reactivation_cost",
        "prototype_similarity",
        "position_error",
        "objectness_at_reentry",
        "resurrection_cost_best",
    }
    for row in rows:
        parsed = dict(row)
        for key in int_keys:
            value = parsed.get(key)
            if value in ("", None):
                parsed[key] = 0
            else:
                parsed[key] = int(float(value))
        for key in float_keys:
            value = parsed.get(key)
            parsed[key] = None if value in ("", None) else float(value)
        coerced.append(parsed)
    return coerced


if __name__ == "__main__":
    main()
