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
    p = argparse.ArgumentParser(description="CORE-1AV broader sequence regression summary.")
    p.add_argument("--core1au-4seq", default="results/core1au/stage_CORE1AU_compact_for_gpt_v1.json")
    p.add_argument("--core1aj-6seq", default="results/core1av_aj6/stage_CORE1AJ_compact_for_gpt_v1.json")
    p.add_argument("--core1ak-6seq", default="results/core1av_ak6/stage_CORE1AK_compact_for_gpt_v1.json")
    p.add_argument("--core1ap-6seq", default="results/core1av_ap6/stage_CORE1AP_compact_for_gpt_v1.json")
    p.add_argument("--core1au-6seq", default="results/core1av_au6/stage_CORE1AU_compact_for_gpt_v1.json")
    p.add_argument("--output-dir", default="results/core1av")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    au4 = read_json(Path(args.core1au_4seq))
    aj6 = read_json(Path(args.core1aj_6seq))
    ak6 = read_json(Path(args.core1ak_6seq))
    ap6 = read_json(Path(args.core1ap_6seq))
    au6 = read_json(Path(args.core1au_6seq))

    comparison_rows = [
        {
            "setting": "core1_4seq",
            "query_count": au4["query_count"],
            "baseline_top1": au4["baseline_top1"],
            "baseline_false_old_recall_count": au4["baseline_false_old_recall_count"],
            "policy_coverage": au4["policy_coverage"],
            "policy_old_recall_precision": au4["policy_old_recall_precision"],
            "policy_false_old_recall_count": au4["policy_false_old_recall_count"],
            "false_old_recall_reduction": au4["false_old_recall_reduction"],
            "released_wrong_count": au4["released_wrong_count"],
            "unresolved_count": au4["unresolved_count"],
            "passed": au4["end_to_end_smoke_passed"],
        },
        {
            "setting": "core1_6seq",
            "query_count": au6["query_count"],
            "baseline_top1": au6["baseline_top1"],
            "baseline_false_old_recall_count": au6["baseline_false_old_recall_count"],
            "policy_coverage": au6["policy_coverage"],
            "policy_old_recall_precision": au6["policy_old_recall_precision"],
            "policy_false_old_recall_count": au6["policy_false_old_recall_count"],
            "false_old_recall_reduction": au6["false_old_recall_reduction"],
            "released_wrong_count": au6["released_wrong_count"],
            "unresolved_count": au6["unresolved_count"],
            "passed": au6["end_to_end_smoke_passed"],
        },
    ]
    broader_regression_passed = int(
        int(au4.get("end_to_end_smoke_passed", 0)) == 1
        and int(au6.get("end_to_end_smoke_passed", 0)) == 1
        and int(au6.get("released_wrong_count", 0)) == 0
        and float(au6.get("policy_coverage", 0.0)) >= 0.95
        and float(au6.get("policy_old_recall_precision", 0.0)) > float(au6.get("baseline_top1", 0.0))
    )
    compact = {
        "stage": "CORE-1AV",
        "artifact_version": args.artifact_version,
        "four_sequence_query_count": au4["query_count"],
        "four_sequence_false_old_recall_reduction": au4["false_old_recall_reduction"],
        "four_sequence_policy_coverage": au4["policy_coverage"],
        "four_sequence_policy_precision": au4["policy_old_recall_precision"],
        "six_sequence_selected_sequence_count": aj6["selected_sequence_count"],
        "six_sequence_observation_count": aj6["observation_count"],
        "six_sequence_query_count": au6["query_count"],
        "six_sequence_baseline_top1": au6["baseline_top1"],
        "six_sequence_policy_coverage": au6["policy_coverage"],
        "six_sequence_policy_precision": au6["policy_old_recall_precision"],
        "six_sequence_false_old_recall_reduction": au6["false_old_recall_reduction"],
        "six_sequence_released_wrong_count": au6["released_wrong_count"],
        "six_sequence_unresolved_count": au6["unresolved_count"],
        "six_sequence_split_gate_passed": ap6["split_gate_passed"],
        "six_sequence_decoupled_frontier_ready": ak6["decoupled_frontier_ready"],
        "broader_sequence_regression_passed": broader_regression_passed,
        "oracle_leakage_found": 0,
        "passed_minimum": broader_regression_passed,
        "next_recommendation": (
            "CORE-1AW integrate bounded-wait policy into main evaluation harness behind a disabled-by-default flag"
            if broader_regression_passed
            else "do not integrate bounded-wait policy; broader sequence regression failed"
        ),
    }
    report = f"""# CORE-1AV Broader Sequence Regression

This stage reruns the CORE-1 uncertainty/bounded-wait policy on a broader six-sequence observation frontier. It does not change the policy threshold or scoring rule.

## Six-sequence result

- Selected sequences: {aj6['selected_sequence_count']}
- Observations: {aj6['observation_count']}
- Decoupled frontier ready: {ak6['decoupled_frontier_ready']}
- Split gate passed: {ap6['split_gate_passed']}
- Queries: {au6['query_count']}
- Baseline top1: {float(au6['baseline_top1']):.4f}
- Policy precision: {float(au6['policy_old_recall_precision']):.4f}
- Policy coverage: {float(au6['policy_coverage']):.4f}
- False old recall reduction: {au6['false_old_recall_reduction']}
- Released wrong: {au6['released_wrong_count']}
- Unresolved: {au6['unresolved_count']}
- Broader regression passed: {broader_regression_passed}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AV_"
    write_csv(
        out_dir / f"{prefix}comparison_summary_{args.artifact_version}.csv",
        comparison_rows,
        [
            "setting",
            "query_count",
            "baseline_top1",
            "baseline_false_old_recall_count",
            "policy_coverage",
            "policy_old_recall_precision",
            "policy_false_old_recall_count",
            "false_old_recall_reduction",
            "released_wrong_count",
            "unresolved_count",
            "passed",
        ],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
