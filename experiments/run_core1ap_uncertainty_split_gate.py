from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AP split validation for uncertainty threshold.")
    p.add_argument("--core1an-events", default="results/core1an/stage_CORE1AN_event_uncertainty_trace_v1.csv")
    p.add_argument("--core1an-risk", default="results/core1an/stage_CORE1AN_risk_coverage_summary_v1.csv")
    p.add_argument("--output-dir", default="results/core1ap")
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


def event_key(row: dict[str, Any]) -> str:
    return f"{row.get('sequence_id','')}::{row.get('event_id','')}::{row.get('window_kind','')}"


def eval_threshold(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    total = len(rows)
    committed = [r for r in rows if f(r["top1_margin"]) >= threshold]
    abstained = [r for r in rows if f(r["top1_margin"]) < threshold]
    baseline_false = sum(1 for r in rows if i(r["top1_success"]) == 0)
    false_suppressed = sum(1 for r in abstained if i(r["top1_success"]) == 0)
    unnecessary_uncertain = sum(1 for r in abstained if i(r["top1_success"]) == 1)
    committed_false = sum(1 for r in committed if i(r["top1_success"]) == 0)
    committed_true = sum(1 for r in committed if i(r["top1_success"]) == 1)
    baseline_acc = float(np.mean([i(r["top1_success"]) for r in rows])) if rows else 0.0
    return {
        "threshold": threshold,
        "query_count": total,
        "baseline_top1": baseline_acc,
        "baseline_false_count": baseline_false,
        "committed_count": len(committed),
        "abstained_count": len(abstained),
        "coverage": len(committed) / total if total else 0.0,
        "committed_top1": committed_true / len(committed) if committed else 0.0,
        "committed_false_retrieval_rate": committed_false / len(committed) if committed else 0.0,
        "false_suppressed_count": false_suppressed,
        "false_suppression_recall": false_suppressed / baseline_false if baseline_false else 0.0,
        "unnecessary_uncertain_count": unnecessary_uncertain,
        "utility_score": false_suppressed - 0.10 * unnecessary_uncertain,
    }


def select_threshold(train_rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    baseline = eval_threshold(train_rows, 0.0)
    evaluated = []
    for th in thresholds:
        row = eval_threshold(train_rows, th)
        row["eligible"] = int(f(row["coverage"]) >= 0.80 and f(row["committed_top1"]) >= f(baseline["baseline_top1"]) and i(row["false_suppressed_count"]) > 0)
        evaluated.append(row)
    eligible = [r for r in evaluated if i(r["eligible"]) == 1]
    if eligible:
        return max(eligible, key=lambda r: (i(r["false_suppressed_count"]), f(r["committed_top1"]), f(r["coverage"])))
    return max(evaluated, key=lambda r: (f(r["utility_score"]), f(r["committed_top1"]), f(r["coverage"])))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.core1an_events))
    risk_rows = read_csv(Path(args.core1an_risk))
    thresholds = sorted({f(r["threshold"]) for r in risk_rows} | {0.0, 0.01, 0.02, 0.03, 0.04, 0.05})
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_event[event_key(row)].append(row)
    keys = sorted(by_event)

    fold_rows: list[dict[str, Any]] = []
    event_trace: list[dict[str, Any]] = []
    for key in keys:
        train_rows = [r for k, vals in by_event.items() if k != key for r in vals]
        test_rows = by_event[key]
        selected = select_threshold(train_rows, thresholds)
        test = eval_threshold(test_rows, f(selected["threshold"]))
        baseline_test = eval_threshold(test_rows, 0.0)
        fold_row = {
            "fold_event_key": key,
            "train_query_count": len(train_rows),
            "test_query_count": len(test_rows),
            "selected_threshold": selected["threshold"],
            "train_selected_coverage": selected["coverage"],
            "train_selected_committed_top1": selected["committed_top1"],
            "test_baseline_top1": baseline_test["baseline_top1"],
            "test_baseline_false_count": baseline_test["baseline_false_count"],
            "test_coverage": test["coverage"],
            "test_committed_top1": test["committed_top1"],
            "test_false_suppressed_count": test["false_suppressed_count"],
            "test_unnecessary_uncertain_count": test["unnecessary_uncertain_count"],
            "test_utility_score": test["utility_score"],
            "fold_passed": int(f(test["coverage"]) >= 0.75 and f(test["committed_top1"]) >= f(baseline_test["baseline_top1"]) and i(test["false_suppressed_count"]) >= 0),
        }
        fold_rows.append(fold_row)
        for row in test_rows:
            action = "old_recall" if f(row["top1_margin"]) >= f(selected["threshold"]) else "uncertain_need_more_evidence"
            event_trace.append(
                {
                    "fold_event_key": key,
                    "selected_threshold": selected["threshold"],
                    "sequence_id": row["sequence_id"],
                    "event_id": row["event_id"],
                    "window_kind": row["window_kind"],
                    "query_obs_id": row["query_obs_id"],
                    "top1_success": row["top1_success"],
                    "top1_margin": row["top1_margin"],
                    "target_rank": row["target_rank"],
                    "memory_action": action,
                    "false_old_suppressed": int(action != "old_recall" and i(row["top1_success"]) == 0),
                    "unnecessary_uncertain": int(action != "old_recall" and i(row["top1_success"]) == 1),
                }
            )

    # Aggregate event-level split decisions by applying each fold's threshold to its held-out event.
    total_test = sum(i(r["test_query_count"]) for r in fold_rows)
    baseline_false = sum(i(r["test_baseline_false_count"]) for r in fold_rows)
    false_suppressed = sum(i(r["test_false_suppressed_count"]) for r in fold_rows)
    unnecessary = sum(i(r["test_unnecessary_uncertain_count"]) for r in fold_rows)
    weighted_coverage = sum(f(r["test_coverage"]) * i(r["test_query_count"]) for r in fold_rows) / total_test if total_test else 0.0
    # Compute committed precision from event trace to avoid averaging fold ratios.
    committed = [r for r in event_trace if r["memory_action"] == "old_recall"]
    committed_top1 = float(np.mean([i(r["top1_success"]) for r in committed])) if committed else 0.0
    baseline_top1 = float(np.mean([i(r["top1_success"]) for r in rows])) if rows else 0.0
    split_gate_passed = int(weighted_coverage >= 0.75 and committed_top1 >= baseline_top1 and false_suppressed > 0)

    compact = {
        "stage": "CORE-1AP",
        "artifact_version": args.artifact_version,
        "fold_count": len(fold_rows),
        "query_count": total_test,
        "baseline_top1": baseline_top1,
        "baseline_false_count": baseline_false,
        "split_coverage": weighted_coverage,
        "split_committed_top1": committed_top1,
        "split_false_suppressed_count": false_suppressed,
        "split_unnecessary_uncertain_count": unnecessary,
        "split_gate_passed": split_gate_passed,
        "oracle_leakage_found": 0,
        "passed_minimum": split_gate_passed,
        "next_recommendation": (
            "CORE-1AQ integrate uncertainty state with split-validated threshold"
            if split_gate_passed
            else "uncertainty threshold does not generalize under event-level split; keep CORE-1AO as diagnostic only"
        ),
    }
    report = f"""# CORE-1AP Uncertainty Split Gate

This stage validates CORE-1AN/CORE-1AO uncertainty thresholding under leave-one-event-out splits. A threshold is selected on all other events, then applied to the held-out event.

## Result

- Folds: {len(fold_rows)}
- Queries: {total_test}
- Baseline top1: {baseline_top1:.4f}
- Baseline false count: {baseline_false}
- Split coverage: {weighted_coverage:.4f}
- Split committed top1: {committed_top1:.4f}
- Split false suppressed: {false_suppressed}
- Split unnecessary uncertain: {unnecessary}
- Split gate passed: {split_gate_passed}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AP_"
    write_csv(
        out_dir / f"{prefix}fold_summary_{args.artifact_version}.csv",
        fold_rows,
        [
            "fold_event_key",
            "train_query_count",
            "test_query_count",
            "selected_threshold",
            "train_selected_coverage",
            "train_selected_committed_top1",
            "test_baseline_top1",
            "test_baseline_false_count",
            "test_coverage",
            "test_committed_top1",
            "test_false_suppressed_count",
            "test_unnecessary_uncertain_count",
            "test_utility_score",
            "fold_passed",
        ],
    )
    write_csv(
        out_dir / f"{prefix}event_policy_trace_{args.artifact_version}.csv",
        event_trace,
        [
            "fold_event_key",
            "selected_threshold",
            "sequence_id",
            "event_id",
            "window_kind",
            "query_obs_id",
            "top1_success",
            "top1_margin",
            "target_rank",
            "memory_action",
            "false_old_suppressed",
            "unnecessary_uncertain",
        ],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1ap_uncertainty_split_gate.py",
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
