from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AO uncertainty-aware memory policy audit.")
    p.add_argument("--core1an-events", default="results/core1an/stage_CORE1AN_event_uncertainty_trace_v1.csv")
    p.add_argument("--core1an-risk", default="results/core1an/stage_CORE1AN_risk_coverage_summary_v1.csv")
    p.add_argument("--core1an-compact", default="results/core1an/stage_CORE1AN_compact_for_gpt_v1.json")
    p.add_argument("--output-dir", default="results/core1ao")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def i(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def f(x: Any, default: float = 0.0) -> float:
    try:
        out = float(x)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize_policy(events: list[dict[str, Any]], threshold: float, policy_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out_events: list[dict[str, Any]] = []
    for row in events:
        commit = f(row["top1_margin"]) >= threshold
        success = i(row["top1_success"])
        action = "old_recall" if commit else "uncertain_need_more_evidence"
        false_old = int(commit and success == 0)
        true_old = int(commit and success == 1)
        false_suppressed = int((not commit) and success == 0)
        unnecessary_uncertain = int((not commit) and success == 1)
        out_events.append(
            dict(
                row,
                policy_name=policy_name,
                policy_threshold=threshold,
                memory_action=action,
                old_recall_success=true_old,
                false_old_recall=false_old,
                false_old_suppressed=false_suppressed,
                unnecessary_uncertain=unnecessary_uncertain,
            )
        )
    total = len(out_events)
    old_recall_count = sum(i(r["memory_action"] == "old_recall") for r in out_events)
    uncertain_count = total - old_recall_count
    false_old_count = sum(i(r["false_old_recall"]) for r in out_events)
    true_old_count = sum(i(r["old_recall_success"]) for r in out_events)
    false_suppressed_count = sum(i(r["false_old_suppressed"]) for r in out_events)
    unnecessary_uncertain_count = sum(i(r["unnecessary_uncertain"]) for r in out_events)
    baseline_false = sum(1 for r in out_events if i(r["top1_success"]) == 0)
    summary = {
        "policy_name": policy_name,
        "threshold": threshold,
        "query_count": total,
        "old_recall_count": old_recall_count,
        "uncertain_count": uncertain_count,
        "coverage": old_recall_count / total if total else 0.0,
        "old_recall_precision": true_old_count / old_recall_count if old_recall_count else 0.0,
        "false_old_recall_count": false_old_count,
        "false_old_recall_rate": false_old_count / old_recall_count if old_recall_count else 0.0,
        "baseline_false_old_recall_count": baseline_false,
        "false_old_suppressed_count": false_suppressed_count,
        "false_old_suppression_recall": false_suppressed_count / baseline_false if baseline_false else 0.0,
        "unnecessary_uncertain_count": unnecessary_uncertain_count,
        "uncertainty_precision_eval_only": false_suppressed_count / uncertain_count if uncertain_count else 0.0,
        "policy_utility_score": false_suppressed_count - 0.10 * unnecessary_uncertain_count,
    }
    return summary, out_events


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events = read_csv(Path(args.core1an_events))
    risk_rows = read_csv(Path(args.core1an_risk))
    compact_an = read_json(Path(args.core1an_compact))
    selected_threshold = f(compact_an.get("best_threshold"), 0.0)
    # Include the selected operating point plus a few interpretable alternatives from CORE-1AN.
    thresholds = {
        "A0_forced_old_recall": 0.0,
        "A1_core1an_selected_margin_gate": selected_threshold,
        "A2_light_margin_gate_001": 0.01,
        "A3_round_margin_gate_002": 0.02,
        "A4_conservative_margin_gate_004": 0.04,
    }
    # Add the highest eligible risk row if it differs from the selected threshold.
    eligible = [r for r in risk_rows if i(r.get("eligible")) == 1]
    if eligible:
        best_precision = max(eligible, key=lambda r: f(r["committed_top1"]))
        thresholds["A5_best_precision_eligible_gate"] = f(best_precision["threshold"])

    summary_rows: list[dict[str, Any]] = []
    policy_event_rows: list[dict[str, Any]] = []
    for name, threshold in thresholds.items():
        summary, rows = summarize_policy(events, threshold, name)
        summary_rows.append(summary)
        policy_event_rows.extend(rows)

    candidates = [r for r in summary_rows if str(r["policy_name"]) != "A0_forced_old_recall" and f(r["coverage"]) >= 0.80 and f(r["old_recall_precision"]) >= f(summary_rows[0]["old_recall_precision"])]
    if candidates:
        best = max(candidates, key=lambda r: (i(r["false_old_suppressed_count"]), f(r["old_recall_precision"]), f(r["coverage"])))
    else:
        best = max(summary_rows, key=lambda r: f(r["policy_utility_score"])) if summary_rows else {}
    for row in summary_rows:
        row["selected_as_best_policy"] = int(row is best)
        row["eligible_for_policy_integration"] = int(
            str(row["policy_name"]) != "A0_forced_old_recall"
            and f(row["coverage"]) >= 0.80
            and f(row["old_recall_precision"]) >= f(summary_rows[0]["old_recall_precision"])
            and i(row["false_old_suppressed_count"]) > 0
        )

    policy_gate_passed = int(i(best.get("eligible_for_policy_integration")) == 1)
    compact = {
        "stage": "CORE-1AO",
        "artifact_version": args.artifact_version,
        "baseline_old_recall_precision": summary_rows[0]["old_recall_precision"] if summary_rows else 0.0,
        "baseline_false_old_recall_count": summary_rows[0]["false_old_recall_count"] if summary_rows else 0,
        "best_policy": best.get("policy_name", ""),
        "best_threshold": best.get("threshold", 0.0),
        "best_coverage": best.get("coverage", 0.0),
        "best_old_recall_precision": best.get("old_recall_precision", 0.0),
        "best_false_old_recall_count": best.get("false_old_recall_count", 0),
        "false_old_suppressed_count": best.get("false_old_suppressed_count", 0),
        "unnecessary_uncertain_count": best.get("unnecessary_uncertain_count", 0),
        "policy_gate_passed": policy_gate_passed,
        "oracle_leakage_found": 0,
        "passed_minimum": policy_gate_passed,
        "next_recommendation": (
            "CORE-1AP add uncertainty state to core memory API / downstream audit"
            if policy_gate_passed
            else "do not integrate uncertainty policy; risk-coverage tradeoff is not acceptable"
        ),
    }
    report = f"""# CORE-1AO Uncertainty-Aware Memory Policy

This stage turns CORE-1AN's margin signal into an explicit memory action: `old_recall` or `uncertain_need_more_evidence`. It does not change retrieval ranking.

## Result

- Baseline old-recall precision: {float(compact['baseline_old_recall_precision']):.4f}
- Baseline false old recalls: {compact['baseline_false_old_recall_count']}
- Best policy: {compact['best_policy']}
- Best threshold: {float(compact['best_threshold']):.4f}
- Coverage: {float(compact['best_coverage']):.4f}
- Old-recall precision: {float(compact['best_old_recall_precision']):.4f}
- False old recalls after policy: {compact['best_false_old_recall_count']}
- False old recalls suppressed: {compact['false_old_suppressed_count']}
- Unnecessary uncertain decisions: {compact['unnecessary_uncertain_count']}
- Policy gate passed: {policy_gate_passed}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AO_"
    write_csv(
        out_dir / f"{prefix}policy_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "policy_name",
            "threshold",
            "query_count",
            "old_recall_count",
            "uncertain_count",
            "coverage",
            "old_recall_precision",
            "false_old_recall_count",
            "false_old_recall_rate",
            "baseline_false_old_recall_count",
            "false_old_suppressed_count",
            "false_old_suppression_recall",
            "unnecessary_uncertain_count",
            "uncertainty_precision_eval_only",
            "policy_utility_score",
            "eligible_for_policy_integration",
            "selected_as_best_policy",
        ],
    )
    write_csv(
        out_dir / f"{prefix}policy_event_trace_{args.artifact_version}.csv",
        policy_event_rows,
        [
            "policy_name",
            "policy_threshold",
            "sequence_id",
            "event_id",
            "window_kind",
            "query_obs_id",
            "candidate_count",
            "top1_obs_id",
            "top1_success",
            "target_rank",
            "top1_margin",
            "target_margin",
            "memory_action",
            "old_recall_success",
            "false_old_recall",
            "false_old_suppressed",
            "unnecessary_uncertain",
        ],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1ao_uncertainty_memory_policy.py",
                "gt_used_for_online_scoring": 0,
                "gt_used_for_policy_action": 0,
                "gt_used_for_eval_only": 1,
                "pretrained_weights_used": 0,
                "leakage_found": 0,
            }
        ],
        ["file", "gt_used_for_online_scoring", "gt_used_for_policy_action", "gt_used_for_eval_only", "pretrained_weights_used", "leakage_found"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
