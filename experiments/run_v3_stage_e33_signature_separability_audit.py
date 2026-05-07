from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v3_stage_e33_cue_signature_repair import run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run v3 Stage E3.3 signature separability audit only.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--event-audit", default="results/v3_e1/stage_E1_event_audit_v1.csv")
    p.add_argument("--cross-run-alignment", default="results/v3_e2rm/stage_E2R_cross_run_target_alignment_v1.csv")
    p.add_argument("--e2c-negative-events", default="results/v3_e2c/stage_E2C_negative_control_events_v1.csv")
    p.add_argument("--output-dir", default="results/v3_e33")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Use the full shared E3.3 runner so audit files and ablation summaries are
    # generated from the same event pass and cannot drift in metric semantics.
    args.audit_only = False
    run(args)


if __name__ == "__main__":
    main()
