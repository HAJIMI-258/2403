"""Run Phase 3X strict vs lineage-aware evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3r_utils import write_csv
from experiments.phase3x_utils import build_lineage_eval_rows, plot_strict_vs_lineage_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 3X lineage-aware evaluation summary.")
    parser.add_argument("--output-dir", default="results/phase3x")
    parser.add_argument("--events-csv", default="")
    parser.add_argument("--artifact-version", default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events_csv = Path(args.events_csv) if args.events_csv else output_dir / f"reentry_event_trace_{args.artifact_version}.csv"
    rows = load_csv_rows(events_csv)
    eval_rows = build_lineage_eval_rows(rows)

    summary_path = output_dir / f"phase3x_lineage_eval_summary_{args.artifact_version}.csv"
    figure_path = output_dir / "strict_vs_lineage_eval_v1.png"
    write_csv(summary_path, eval_rows)
    plot_strict_vs_lineage_eval(eval_rows, figure_path)

    print(f"saved_summary={summary_path}")
    print(f"saved_figure={figure_path}")


def load_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for row in reader:
            parsed = dict(row)
            for key in [
                "event_id",
                "instance_id",
                "old_track_id",
                "old_prototype_id",
                "old_lineage_id",
                "reappear_frame",
                "proposal_detected",
                "concept_recovered",
                "matched_prototype_id",
                "matched_lineage_id",
                "same_prototype_id",
                "same_lineage_id",
                "continuation_written_before",
                "continuation_alive_at_reentry",
                "continuation_owner_prototype_id",
                "continuation_owner_lineage_id",
                "continuation_bank_nonempty",
                "continuation_attempted",
                "continuation_success",
                "same_track",
                "new_track_created",
                "new_prototype_created",
                "prototype_matched_continuation_count",
                "lineage_matched_continuation_count",
                "alive_same_lineage_continuation_count",
                "alive_same_prototype_continuation_count",
                "best_continuation_age",
            ]:
                if key in parsed:
                    parsed[key] = None if parsed[key] in ("", None) else int(float(parsed[key]))
            rows.append(parsed)
    return rows


if __name__ == "__main__":
    main()
