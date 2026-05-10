from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1k_windowed_render_cache import write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AZ milestone freeze.")
    p.add_argument("--core1am", default="results/core1am/stage_CORE1AM_compact_for_gpt_v1.json")
    p.add_argument("--core1au", default="results/core1au/stage_CORE1AU_compact_for_gpt_v1.json")
    p.add_argument("--core1av", default="results/core1av/stage_CORE1AV_compact_for_gpt_v1.json")
    p.add_argument("--core1ay", default="results/core1ay/stage_CORE1AY_compact_for_gpt_v1.json")
    p.add_argument("--output-dir", default="results/core1az")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    core1am = read_json(Path(args.core1am))
    core1au = read_json(Path(args.core1au))
    core1av = read_json(Path(args.core1av))
    core1ay = read_json(Path(args.core1ay))

    rows = [
        {
            "finding": "learned_metric",
            "status": "rejected",
            "evidence": f"control_significance_passed={core1am['control_significance_passed']}",
            "next_action": "do_not_integrate",
        },
        {
            "finding": "bounded_wait_uncertainty_policy",
            "status": "accepted_experimental_disabled",
            "evidence": f"precision={core1av['six_sequence_policy_precision']}, coverage={core1av['six_sequence_policy_coverage']}",
            "next_action": "use_as_disabled_eval_flag",
        },
        {
            "finding": "parameter_sensitivity",
            "status": "passed",
            "evidence": f"eligible_configs={core1ay['eligible_config_count']}",
            "next_action": "keep_canonical_threshold_0.0194_horizon_10",
        },
    ]
    compact = {
        "stage": "CORE-1AZ",
        "artifact_version": args.artifact_version,
        "core1_milestone_frozen": 1,
        "learned_metric_integrated": 0,
        "uncertainty_policy_experimental_available": 1,
        "uncertainty_policy_default_enabled": 0,
        "best_confirmed_query_count": core1av["six_sequence_query_count"],
        "best_confirmed_policy_precision": core1av["six_sequence_policy_precision"],
        "best_confirmed_policy_coverage": core1av["six_sequence_policy_coverage"],
        "best_confirmed_false_old_recall_reduction": core1av["six_sequence_false_old_recall_reduction"],
        "parameter_sensitivity_passed": core1ay["parameter_sensitivity_passed"],
        "next_core_stage": "CORE-2 object-file consolidation and uncertainty-driven evidence handling",
        "passed_minimum": 1,
    }
    report = f"""# CORE-1AZ Milestone Freeze

CORE-1 is frozen as a memory-decision milestone.

## Decisions

- Learned metric integrated: 0
- Uncertainty policy available: 1
- Default enabled: 0
- Best confirmed query count: {core1av['six_sequence_query_count']}
- Best confirmed policy precision: {float(core1av['six_sequence_policy_precision']):.4f}
- Best confirmed policy coverage: {float(core1av['six_sequence_policy_coverage']):.4f}
- False-old recall reduction: {core1av['six_sequence_false_old_recall_reduction']}
- Parameter sensitivity passed: {core1ay['parameter_sensitivity_passed']}

Next core stage: {compact['next_core_stage']}
"""
    prefix = "stage_CORE1AZ_"
    write_csv(out_dir / f"{prefix}decision_table_{args.artifact_version}.csv", rows, ["finding", "status", "evidence", "next_action"])
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
