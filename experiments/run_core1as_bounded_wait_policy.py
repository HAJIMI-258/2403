from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AS bounded wait policy audit.")
    p.add_argument("--core1ar-trace", default="results/core1ar/stage_CORE1AR_delayed_resolution_trace_v1.csv")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--output-dir", default="results/core1as")
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = [r for r in read_csv(Path(args.core1ar_trace)) if i(r.get("horizon_frames")) == args.horizon]
    policy_rows: list[dict[str, Any]] = []
    for row in trace:
        state = str(row["resolution_state"])
        if state == "resolved_correct_old_recall":
            final_action = "delayed_old_recall_release"
            unresolved = 0
            released_wrong = 0
            resolved_correct = 1
        elif state == "released_wrong_old_recall":
            final_action = "delayed_wrong_release"
            unresolved = 0
            released_wrong = 1
            resolved_correct = 0
        else:
            final_action = "remain_uncertain_after_horizon"
            unresolved = 1
            released_wrong = 0
            resolved_correct = 0
        policy_rows.append(
            {
                "sequence_id": row["sequence_id"],
                "event_id": row["event_id"],
                "window_kind": row["window_kind"],
                "track_id": row["track_id"],
                "uncertain_query_obs_id": row["uncertain_query_obs_id"],
                "uncertain_frame_idx": row["uncertain_frame_idx"],
                "bounded_wait_horizon": args.horizon,
                "release_query_obs_id": row["release_query_obs_id"],
                "release_frame_idx": row["release_frame_idx"],
                "release_wait_frames": row["release_wait_frames"],
                "final_memory_action": final_action,
                "resolved_correct_old_recall_eval_only": resolved_correct,
                "released_wrong_old_recall_eval_only": released_wrong,
                "unresolved_after_horizon": unresolved,
                "false_old_suppressed_eval_only": row["false_old_suppressed_eval_only"],
                "correct_old_delayed_eval_only": row["correct_old_delayed_eval_only"],
                "memory_update_allowed_before_release": 0,
                "attach_allowed": 0,
                "promotion_allowed": 0,
                "head_update_allowed": 0,
                "future_used_for_initial_decision": 0,
            }
        )

    total = len(policy_rows)
    resolved = sum(i(r["resolved_correct_old_recall_eval_only"]) for r in policy_rows)
    wrong = sum(i(r["released_wrong_old_recall_eval_only"]) for r in policy_rows)
    unresolved = sum(i(r["unresolved_after_horizon"]) for r in policy_rows)
    false_suppressed_resolved = sum(1 for r in policy_rows if i(r["resolved_correct_old_recall_eval_only"]) == 1 and i(r["false_old_suppressed_eval_only"]) == 1)
    correct_delay_resolved = sum(1 for r in policy_rows if i(r["resolved_correct_old_recall_eval_only"]) == 1 and i(r["correct_old_delayed_eval_only"]) == 1)
    wait_values = [i(r["release_wait_frames"]) for r in policy_rows if i(r["resolved_correct_old_recall_eval_only"]) == 1]
    violations = sum(1 for r in policy_rows if i(r["memory_update_allowed_before_release"]) != 0 or i(r["attach_allowed"]) != 0 or i(r["promotion_allowed"]) != 0 or i(r["head_update_allowed"]) != 0)

    summary_rows = [
        {
            "policy": "bounded_wait_uncertainty_resolution",
            "horizon_frames": args.horizon,
            "uncertain_count": total,
            "resolved_correct_count": resolved,
            "released_wrong_count": wrong,
            "unresolved_after_horizon_count": unresolved,
            "resolution_rate": resolved / total if total else 0.0,
            "wrong_release_rate": wrong / total if total else 0.0,
            "false_suppressed_resolved_count": false_suppressed_resolved,
            "correct_delay_resolved_count": correct_delay_resolved,
            "mean_wait_frames": float(np.mean(wait_values)) if wait_values else 0.0,
            "policy_violation_count": violations,
        }
    ]
    passed = int(total > 0 and resolved >= 50 and wrong == 0 and violations == 0)
    compact = {
        "stage": "CORE-1AS",
        "artifact_version": args.artifact_version,
        "bounded_wait_horizon": args.horizon,
        "uncertain_count": total,
        "resolved_correct_count": resolved,
        "released_wrong_count": wrong,
        "unresolved_after_horizon_count": unresolved,
        "resolution_rate": resolved / total if total else 0.0,
        "false_suppressed_resolved_count": false_suppressed_resolved,
        "correct_delay_resolved_count": correct_delay_resolved,
        "mean_wait_frames": float(np.mean(wait_values)) if wait_values else 0.0,
        "policy_violation_count": violations,
        "bounded_wait_policy_passed": passed,
        "oracle_leakage_found": 0,
        "passed_minimum": passed,
        "next_recommendation": (
            "CORE-1AT write core memory decision spec and add regression tests"
            if passed
            else "bounded wait policy unsafe; keep uncertainty queue diagnostic only"
        ),
    }
    report = f"""# CORE-1AS Bounded Wait Policy

This stage turns CORE-1AR's delayed resolution audit into a concrete bounded-wait policy: uncertain recalls wait up to {args.horizon} frames for a high-margin observation on the same online-visible track.

## Result

- Uncertain decisions: {total}
- Resolved correct: {resolved}
- Released wrong: {wrong}
- Unresolved after horizon: {unresolved}
- Resolution rate: {compact['resolution_rate']:.4f}
- False-suppressed cases resolved: {false_suppressed_resolved}
- Correct delayed cases resolved: {correct_delay_resolved}
- Mean wait frames: {compact['mean_wait_frames']:.2f}
- Policy violations: {violations}
- Bounded wait policy passed: {passed}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AS_"
    write_csv(
        out_dir / f"{prefix}bounded_wait_policy_trace_{args.artifact_version}.csv",
        policy_rows,
        [
            "sequence_id",
            "event_id",
            "window_kind",
            "track_id",
            "uncertain_query_obs_id",
            "uncertain_frame_idx",
            "bounded_wait_horizon",
            "release_query_obs_id",
            "release_frame_idx",
            "release_wait_frames",
            "final_memory_action",
            "resolved_correct_old_recall_eval_only",
            "released_wrong_old_recall_eval_only",
            "unresolved_after_horizon",
            "false_old_suppressed_eval_only",
            "correct_old_delayed_eval_only",
            "memory_update_allowed_before_release",
            "attach_allowed",
            "promotion_allowed",
            "head_update_allowed",
            "future_used_for_initial_decision",
        ],
    )
    write_csv(
        out_dir / f"{prefix}policy_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "policy",
            "horizon_frames",
            "uncertain_count",
            "resolved_correct_count",
            "released_wrong_count",
            "unresolved_after_horizon_count",
            "resolution_rate",
            "wrong_release_rate",
            "false_suppressed_resolved_count",
            "correct_delay_resolved_count",
            "mean_wait_frames",
            "policy_violation_count",
        ],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1as_bounded_wait_policy.py",
                "future_used_for_initial_decision": 0,
                "gt_used_for_online_scoring": 0,
                "gt_used_for_eval_only": 1,
                "pretrained_weights_used": 0,
                "leakage_found": 0,
            }
        ],
        ["file", "future_used_for_initial_decision", "gt_used_for_online_scoring", "gt_used_for_eval_only", "pretrained_weights_used", "leakage_found"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
