"""Summarize Phase 3R re-entry events by gap bucket and plot recovery/fragmentation curves."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3r_utils import summarize_gap_buckets, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 3R gap-bucket summaries.")
    parser.add_argument("--events-csv", default="results/phase3r/reentry_events_v1.csv", help="Event csv path.")
    parser.add_argument("--output-dir", default="results/phase3r", help="Directory for outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = _read_csv(args.events_csv)
    summary_rows = summarize_gap_buckets(events)
    summary_path = output_dir / "reentry_gap_bucket_summary_v1.csv"
    reentry_plot_path = output_dir / "reentry_vs_gap_v1.png"
    pfr_plot_path = output_dir / "pfr_vs_gap_v1.png"

    write_csv(summary_path, summary_rows)
    _save_reentry_plot(summary_rows, reentry_plot_path)
    _save_fragmentation_plot(summary_rows, pfr_plot_path)

    print(f"saved_summary={summary_path}")
    print(f"saved_reentry_plot={reentry_plot_path}")
    print(f"saved_fragmentation_plot={pfr_plot_path}")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _track_c_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [row for row in summary_rows if row["scenario_name"] == "track_c_long_horizon"]
    return rows or summary_rows


def _save_reentry_plot(summary_rows: list[dict[str, object]], path: Path) -> None:
    rows = _track_c_rows(summary_rows)
    buckets = [str(row["gap_bucket"]) for row in rows]
    same_track = [float(row["same_track_recovery_rate"]) for row in rows]
    same_proto = [float(row["same_prototype_recovery_rate"]) for row in rows]

    x = np.arange(len(buckets))
    width = 0.34
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - width / 2, same_track, width=width, color="tab:blue", label="same-track")
    axis.bar(x + width / 2, same_proto, width=width, color="tab:orange", label="same-prototype")
    axis.set_title("Track C Re-entry Recovery vs Gap Bucket")
    axis.set_xticks(x)
    axis.set_xticklabels(buckets)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("recovery rate")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_fragmentation_plot(summary_rows: list[dict[str, object]], path: Path) -> None:
    rows = _track_c_rows(summary_rows)
    buckets = [str(row["gap_bucket"]) for row in rows]
    fragment = [float(row["reentry_fragmentation_rate"]) for row in rows]
    new_proto = [float(row["new_prototype_rate"]) for row in rows]

    x = np.arange(len(buckets))
    width = 0.34
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - width / 2, fragment, width=width, color="tab:red", label="fragmentation proxy")
    axis.bar(x + width / 2, new_proto, width=width, color="tab:purple", label="new prototype rate")
    axis.set_title("Track C Prototype Fragmentation vs Gap Bucket")
    axis.set_xticks(x)
    axis.set_xticklabels(buckets)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("rate")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
