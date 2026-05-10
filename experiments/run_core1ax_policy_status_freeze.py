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
    p = argparse.ArgumentParser(description="CORE-1AX policy status freeze.")
    p.add_argument("--core1am", default="results/core1am/stage_CORE1AM_compact_for_gpt_v1.json")
    p.add_argument("--core1ap", default="results/core1ap/stage_CORE1AP_compact_for_gpt_v1.json")
    p.add_argument("--core1as", default="results/core1as/stage_CORE1AS_compact_for_gpt_v1.json")
    p.add_argument("--core1av", default="results/core1av/stage_CORE1AV_compact_for_gpt_v1.json")
    p.add_argument("--core1aw", default="results/core1aw/stage_CORE1AW_compact_for_gpt_v1.json")
    p.add_argument("--output-dir", default="results/core1ax")
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
    core1ap = read_json(Path(args.core1ap))
    core1as = read_json(Path(args.core1as))
    core1av = read_json(Path(args.core1av))
    core1aw = read_json(Path(args.core1aw))

    rows = [
        {
            "component": "learned_metric",
            "status": "rejected_for_integration",
            "evidence": f"control_significance_passed={core1am['control_significance_passed']}",
            "default_enabled": 0,
        },
        {
            "component": "uncertainty_margin_gate",
            "status": "passed_split_gate",
            "evidence": f"split_false_suppressed={core1ap['split_false_suppressed_count']}",
            "default_enabled": 0,
        },
        {
            "component": "bounded_wait_release",
            "status": "passed_policy_audit",
            "evidence": f"resolved={core1as['resolved_correct_count']}, wrong={core1as['released_wrong_count']}",
            "default_enabled": 0,
        },
        {
            "component": "broader_sequence_regression",
            "status": "passed",
            "evidence": f"six_seq_false_reduction={core1av['six_sequence_false_old_recall_reduction']}",
            "default_enabled": 0,
        },
        {
            "component": "policy_flag_harness",
            "status": "passed_experimental_disabled",
            "evidence": f"enabled_precision={core1aw['enabled_precision']}, enabled_coverage={core1aw['enabled_coverage']}",
            "default_enabled": core1aw["default_policy_enabled"],
        },
    ]
    compact = {
        "stage": "CORE-1AX",
        "artifact_version": args.artifact_version,
        "learned_metric_integrated": 0,
        "uncertainty_policy_available": 1,
        "uncertainty_policy_default_enabled": 0,
        "policy_flag_harness_passed": core1aw["policy_flag_harness_passed"],
        "six_sequence_policy_precision": core1av["six_sequence_policy_precision"],
        "six_sequence_policy_coverage": core1av["six_sequence_policy_coverage"],
        "six_sequence_false_old_recall_reduction": core1av["six_sequence_false_old_recall_reduction"],
        "safe_for_default_enable": 0,
        "safe_for_experimental_eval_flag": 1,
        "next_recommendation": "CORE-1AY broader seed/config regression before default enablement",
        "passed_minimum": 1,
    }
    report = f"""# CORE-1AX Policy Status Freeze

This stage freezes the current CORE-1 decision: descriptor metric integration is rejected, while bounded-wait uncertainty handling is available as an experimental disabled-by-default evaluation flag.

## Result

- Learned metric integrated: 0
- Uncertainty policy available: 1
- Default enabled: 0
- Six-sequence policy precision: {float(core1av['six_sequence_policy_precision']):.4f}
- Six-sequence policy coverage: {float(core1av['six_sequence_policy_coverage']):.4f}
- Six-sequence false-old reduction: {core1av['six_sequence_false_old_recall_reduction']}
- Safe for experimental eval flag: 1
- Safe for default enable: 0

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AX_"
    write_csv(out_dir / f"{prefix}integration_status_{args.artifact_version}.csv", rows, ["component", "status", "evidence", "default_enabled"])
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
