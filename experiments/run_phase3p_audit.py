"""Run Phase 3P Stage A audit on top of the current Phase 3L baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3p_utils import (
    TRACK_C_NAME,
    evaluate_phase3p_bundle,
    save_phase3p_stage_a_outputs,
)
from experiments.phase3r_utils import extract_reentry_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3P Stage A audit.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--output-dir", default="results/phase3p")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = evaluate_phase3p_bundle(
        args.config,
        seed=args.seed,
        scenario_names=[TRACK_C_NAME],
    )
    run = bundle["runs"][0]
    event_rows, _ = extract_reentry_events(run["scenario_name"], run["sequence"], run["result"])
    prototype_lineage_rows = []
    for frame_record in run["result"].frame_records:
        prototype_lineage_rows.extend(getattr(frame_record.memory_output, "prototype_lineage_rows", []))
    payload = save_phase3p_stage_a_outputs(
        output_dir=args.output_dir,
        event_rows=event_rows,
        prototype_lineage_rows=prototype_lineage_rows,
    )
    print(f"saved_event_audit={Path(args.output_dir) / 'phase3p_event_audit.csv'}")
    print(f"saved_lineage_aggregate={Path(args.output_dir) / 'phase3p_lineage_aggregate.csv'}")
    print(f"saved_summary={Path(args.output_dir) / 'phase3p_audit_summary.md'}")
    print(f"saved_top_cases={Path(args.output_dir) / 'phase3p_top_failure_cases.md'}")
    print(f"dominant_action_bucket={payload['dominant_bucket']}")


if __name__ == "__main__":
    main()
