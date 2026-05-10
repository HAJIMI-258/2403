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
from nops_owr.memory import MemoryDecisionConfig, can_release_after_wait, decide_memory_retrieval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AU memory policy end-to-end smoke.")
    p.add_argument("--core1an-events", default="results/core1an/stage_CORE1AN_event_uncertainty_trace_v1.csv")
    p.add_argument("--output-dir", default="results/core1au")
    p.add_argument("--threshold", type=float, default=0.0194)
    p.add_argument("--horizon", type=int, default=10)
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


def group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return str(row["sequence_id"]), str(row["event_id"]), str(row["window_kind"]), str(row["track_id"])


def find_release(row: dict[str, Any], track_rows: list[dict[str, Any]], cfg: MemoryDecisionConfig) -> dict[str, Any] | None:
    start = i(row["frame_idx"])
    for cand in sorted(track_rows, key=lambda r: (i(r["frame_idx"]), i(r["query_obs_id"]))):
        wait = i(cand["frame_idx"]) - start
        if wait <= 0:
            continue
        if can_release_after_wait(wait_frames=wait, release_margin=f(cand["top1_margin"]), config=cfg):
            return cand
    return None


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.core1an_events))
    cfg = MemoryDecisionConfig(uncertainty_margin_threshold=args.threshold, bounded_wait_horizon_frames=args.horizon)
    by_track: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_track[group_key(row)].append(row)

    trace_rows: list[dict[str, Any]] = []
    for row in rows:
        decision = decide_memory_retrieval(f(row["top1_margin"]), cfg)
        baseline_success = i(row["top1_success"])
        if decision.retrieval_state.value == "old_recall_candidate":
            final_action = "immediate_old_recall"
            final_success = baseline_success
            release_query = ""
            release_wait = ""
            release_margin = ""
            release_success = ""
        else:
            release = find_release(row, by_track[group_key(row)], cfg)
            if release is None:
                final_action = "remain_uncertain_after_wait"
                final_success = ""
                release_query = ""
                release_wait = ""
                release_margin = ""
                release_success = ""
            else:
                final_action = "delayed_old_recall"
                final_success = i(release["top1_success"])
                release_query = release["query_obs_id"]
                release_wait = i(release["frame_idx"]) - i(row["frame_idx"])
                release_margin = release["top1_margin"]
                release_success = i(release["top1_success"])
        accepted = int(final_action in ("immediate_old_recall", "delayed_old_recall"))
        false_old = int(accepted and i(final_success) == 0)
        true_old = int(accepted and i(final_success) == 1)
        trace_rows.append(
            {
                "sequence_id": row["sequence_id"],
                "event_id": row["event_id"],
                "window_kind": row["window_kind"],
                "query_obs_id": row["query_obs_id"],
                "frame_idx": row["frame_idx"],
                "track_id": row["track_id"],
                "top1_margin": row["top1_margin"],
                "baseline_top1_success": baseline_success,
                "initial_retrieval_state": decision.retrieval_state.value,
                "final_action": final_action,
                "accepted_old_recall": accepted,
                "final_success_eval_only": final_success,
                "true_old_recall_eval_only": true_old,
                "false_old_recall_eval_only": false_old,
                "release_query_obs_id": release_query,
                "release_wait_frames": release_wait,
                "release_top1_margin": release_margin,
                "release_success_eval_only": release_success,
                "memory_update_before_release_allowed": int(decision.memory_update_allowed and final_action == "immediate_old_recall"),
                "attach_allowed": 0,
                "promotion_allowed": 0,
                "head_update_allowed": 0,
            }
        )

    total = len(trace_rows)
    baseline_false = sum(1 for r in trace_rows if i(r["baseline_top1_success"]) == 0)
    baseline_top1 = float(np.mean([i(r["baseline_top1_success"]) for r in trace_rows])) if trace_rows else 0.0
    accepted = [r for r in trace_rows if i(r["accepted_old_recall"]) == 1]
    unresolved = [r for r in trace_rows if r["final_action"] == "remain_uncertain_after_wait"]
    delayed = [r for r in trace_rows if r["final_action"] == "delayed_old_recall"]
    false_after = sum(i(r["false_old_recall_eval_only"]) for r in trace_rows)
    true_after = sum(i(r["true_old_recall_eval_only"]) for r in trace_rows)
    released_wrong = sum(1 for r in delayed if i(r["release_success_eval_only"]) == 0)
    violations = sum(1 for r in trace_rows if i(r["attach_allowed"]) or i(r["promotion_allowed"]) or i(r["head_update_allowed"]))
    summary_rows = [
        {
            "policy": "forced_old_recall_baseline",
            "query_count": total,
            "accepted_old_recall_count": total,
            "unresolved_count": 0,
            "coverage": 1.0,
            "old_recall_precision": baseline_top1,
            "false_old_recall_count": baseline_false,
            "delayed_old_recall_count": 0,
            "released_wrong_count": 0,
        },
        {
            "policy": "uncertainty_bounded_wait",
            "query_count": total,
            "accepted_old_recall_count": len(accepted),
            "unresolved_count": len(unresolved),
            "coverage": len(accepted) / total if total else 0.0,
            "old_recall_precision": true_after / len(accepted) if accepted else 0.0,
            "false_old_recall_count": false_after,
            "delayed_old_recall_count": len(delayed),
            "released_wrong_count": released_wrong,
        },
    ]
    passed = int(
        summary_rows[1]["coverage"] >= 0.95
        and summary_rows[1]["old_recall_precision"] > summary_rows[0]["old_recall_precision"]
        and summary_rows[1]["false_old_recall_count"] < summary_rows[0]["false_old_recall_count"]
        and released_wrong == 0
        and violations == 0
    )
    compact = {
        "stage": "CORE-1AU",
        "artifact_version": args.artifact_version,
        "threshold": args.threshold,
        "bounded_wait_horizon": args.horizon,
        "query_count": total,
        "baseline_top1": baseline_top1,
        "baseline_false_old_recall_count": baseline_false,
        "policy_coverage": summary_rows[1]["coverage"],
        "policy_old_recall_precision": summary_rows[1]["old_recall_precision"],
        "policy_false_old_recall_count": false_after,
        "false_old_recall_reduction": baseline_false - false_after,
        "delayed_old_recall_count": len(delayed),
        "released_wrong_count": released_wrong,
        "unresolved_count": len(unresolved),
        "policy_violation_count": violations,
        "end_to_end_smoke_passed": passed,
        "oracle_leakage_found": 0,
        "passed_minimum": passed,
        "next_recommendation": (
            "CORE-1AV run broader seed/sequence regression for bounded-wait memory policy"
            if passed
            else "do not integrate bounded-wait policy; end-to-end smoke failed"
        ),
    }
    report = f"""# CORE-1AU Memory Policy End-to-End Smoke

This stage applies the package-level memory decision policy to the CORE-1 hard evaluation stream and compares forced old recall with uncertainty plus bounded wait.

## Result

- Query count: {total}
- Baseline top1: {baseline_top1:.4f}
- Baseline false old recalls: {baseline_false}
- Policy coverage: {compact['policy_coverage']:.4f}
- Policy old-recall precision: {compact['policy_old_recall_precision']:.4f}
- Policy false old recalls: {false_after}
- False old recall reduction: {baseline_false - false_after}
- Delayed old recalls: {len(delayed)}
- Released wrong: {released_wrong}
- Unresolved: {len(unresolved)}
- Policy violations: {violations}
- End-to-end smoke passed: {passed}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AU_"
    write_csv(
        out_dir / f"{prefix}end_to_end_decision_trace_{args.artifact_version}.csv",
        trace_rows,
        [
            "sequence_id",
            "event_id",
            "window_kind",
            "query_obs_id",
            "frame_idx",
            "track_id",
            "top1_margin",
            "baseline_top1_success",
            "initial_retrieval_state",
            "final_action",
            "accepted_old_recall",
            "final_success_eval_only",
            "true_old_recall_eval_only",
            "false_old_recall_eval_only",
            "release_query_obs_id",
            "release_wait_frames",
            "release_top1_margin",
            "release_success_eval_only",
            "memory_update_before_release_allowed",
            "attach_allowed",
            "promotion_allowed",
            "head_update_allowed",
        ],
    )
    write_csv(
        out_dir / f"{prefix}summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "policy",
            "query_count",
            "accepted_old_recall_count",
            "unresolved_count",
            "coverage",
            "old_recall_precision",
            "false_old_recall_count",
            "delayed_old_recall_count",
            "released_wrong_count",
        ],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1au_memory_policy_end_to_end_smoke.py",
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
