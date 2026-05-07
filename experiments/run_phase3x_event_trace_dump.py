"""Dump Phase 3X re-entry event traces and continuation lifecycle rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3r_utils import write_csv
from experiments.phase3x_utils import (
    build_phase3x_event_trace,
    collect_phase3x_audit_rows,
    evaluate_phase3x_runs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump Phase 3X event trace rows.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--output-dir", default="results/phase3x")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-version", default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = evaluate_phase3x_runs(args.config, seed=args.seed)
    event_rows, write_rows, lifecycle_rows, prototype_rows = collect_phase3x_audit_rows(bundle["runs"])
    trace_rows = build_phase3x_event_trace(event_rows, write_rows, lifecycle_rows)

    trace_path = output_dir / f"reentry_event_trace_{args.artifact_version}.csv"
    lifecycle_path = output_dir / f"continuation_lifecycle_{args.artifact_version}.csv"
    write_path = output_dir / f"continuation_write_{args.artifact_version}.csv"
    proto_path = output_dir / f"prototype_lineage_rows_{args.artifact_version}.csv"

    write_csv(trace_path, trace_rows)
    write_csv(lifecycle_path, lifecycle_rows)
    write_csv(write_path, write_rows)
    write_csv(proto_path, prototype_rows)

    print(f"saved_trace={trace_path}")
    print(f"saved_lifecycle={lifecycle_path}")
    print(f"saved_write={write_path}")
    print(f"saved_prototype_rows={proto_path}")
    print(f"trace_events={len(trace_rows)}")


if __name__ == "__main__":
    main()
