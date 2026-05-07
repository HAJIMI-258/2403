"""Build Phase 3S gap-bucket summaries and plots from re-entry audit outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3r2_utils import load_csv_rows
from experiments.phase3r_utils import summarize_gap_buckets, write_csv
from experiments.phase3s_utils import (
    coerce_reentry_rows,
    plot_continuation_bank_nonempty,
    plot_reentry_vs_gap_v4,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 3S gap-bucket plots.")
    parser.add_argument("--output-dir", default="results/phase3s", help="Directory for artifacts.")
    parser.add_argument(
        "--events-csv",
        default="results/phase3s/reentry_events_v4.csv",
        help="Path to reentry_events_v4.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = coerce_reentry_rows(load_csv_rows(args.events_csv))
    gap_rows = summarize_gap_buckets(rows)

    gap_csv = output_dir / "reentry_gap_bucket_summary_v4.csv"
    figure_gap = output_dir / "reentry_vs_gap_v4.png"
    figure_bank = output_dir / "continuation_bank_nonempty_v1.png"

    write_csv(gap_csv, gap_rows)
    plot_reentry_vs_gap_v4(gap_rows, figure_gap)
    plot_continuation_bank_nonempty(rows, figure_bank)

    print(f"saved_gap_csv={gap_csv}")
    print(f"saved_gap_plot={figure_gap}")
    print(f"saved_bank_plot={figure_bank}")


if __name__ == "__main__":
    main()
