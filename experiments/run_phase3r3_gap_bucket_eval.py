"""Build Phase 3R.3 gap-bucket summaries and plots from re-entry audit outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3r2_utils import load_csv_rows
from experiments.phase3r3_utils import coerce_reentry_rows, plot_candidate_pool_nonempty, plot_reentry_vs_gap_v3
from experiments.phase3r_utils import summarize_gap_buckets, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 3R.3 gap-bucket plots.")
    parser.add_argument("--output-dir", default="results/phase3r3", help="Directory for artifacts.")
    parser.add_argument(
        "--events-csv",
        default="results/phase3r3/reentry_events_v3.csv",
        help="Path to reentry_events_v3.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = coerce_reentry_rows(load_csv_rows(args.events_csv))
    gap_rows = summarize_gap_buckets(rows)

    gap_csv = output_dir / "reentry_gap_bucket_summary_v3.csv"
    figure_gap = output_dir / "reentry_vs_gap_v3.png"
    figure_pool = output_dir / "candidate_pool_nonempty_v1.png"

    write_csv(gap_csv, gap_rows)
    plot_reentry_vs_gap_v3(gap_rows, figure_gap)
    plot_candidate_pool_nonempty(rows, figure_pool)

    print(f"saved_gap_csv={gap_csv}")
    print(f"saved_gap_plot={figure_gap}")
    print(f"saved_pool_plot={figure_pool}")


if __name__ == "__main__":
    main()
