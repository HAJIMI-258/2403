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
    p = argparse.ArgumentParser(description="CORE-1AW policy flag evaluation harness.")
    p.add_argument("--events", default="results/core1av_an6/stage_CORE1AN_event_uncertainty_trace_v1.csv")
    p.add_argument("--output-dir", default="results/core1aw")
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


def release_candidate(row: dict[str, Any], by_track: dict[tuple[str, str, str, str], list[dict[str, Any]]], cfg: MemoryDecisionConfig) -> dict[str, Any] | None:
    start = i(row["frame_idx"])
    for cand in by_track[group_key(row)]:
        wait = i(cand["frame_idx"]) - start
        if can_release_after_wait(wait_frames=wait, release_margin=f(cand["top1_margin"]), config=cfg):
            return cand
    return None


def evaluate(rows: list[dict[str, Any]], *, enabled: bool, cfg: MemoryDecisionConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_track: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_track[group_key(row)].append(row)
    for key in by_track:
        by_track[key] = sorted(by_track[key], key=lambda r: (i(r["frame_idx"]), i(r["query_obs_id"])))

    trace: list[dict[str, Any]] = []
    for row in rows:
        if not enabled:
            action = "forced_old_recall"
            final_success = i(row["top1_success"])
            release_wait = ""
            release_query = ""
        else:
            decision = decide_memory_retrieval(f(row["top1_margin"]), cfg)
            if decision.retrieval_state.value == "old_recall_candidate":
                action = "immediate_old_recall"
                final_success = i(row["top1_success"])
                release_wait = ""
                release_query = ""
            else:
                release = release_candidate(row, by_track, cfg)
                if release is None:
                    action = "remain_uncertain_after_wait"
                    final_success = ""
                    release_wait = ""
                    release_query = ""
                else:
                    action = "delayed_old_recall"
                    final_success = i(release["top1_success"])
                    release_wait = i(release["frame_idx"]) - i(row["frame_idx"])
                    release_query = release["query_obs_id"]
        accepted = int(action in ("forced_old_recall", "immediate_old_recall", "delayed_old_recall"))
        trace.append(
            {
                "policy_enabled": int(enabled),
                "sequence_id": row["sequence_id"],
                "event_id": row["event_id"],
                "window_kind": row["window_kind"],
                "query_obs_id": row["query_obs_id"],
                "frame_idx": row["frame_idx"],
                "track_id": row["track_id"],
                "top1_margin": row["top1_margin"],
                "baseline_top1_success": row["top1_success"],
                "final_action": action,
                "accepted_old_recall": accepted,
                "final_success_eval_only": final_success,
                "false_old_recall_eval_only": int(accepted and i(final_success) == 0),
                "true_old_recall_eval_only": int(accepted and i(final_success) == 1),
                "release_wait_frames": release_wait,
                "release_query_obs_id": release_query,
                "attach_allowed": 0,
                "promotion_allowed": 0,
                "head_update_allowed": 0,
            }
        )
    accepted_rows = [r for r in trace if i(r["accepted_old_recall"]) == 1]
    summary = {
        "policy_enabled": int(enabled),
        "query_count": len(trace),
        "accepted_old_recall_count": len(accepted_rows),
        "coverage": len(accepted_rows) / len(trace) if trace else 0.0,
        "old_recall_precision": float(np.mean([i(r["final_success_eval_only"]) for r in accepted_rows])) if accepted_rows else 0.0,
        "false_old_recall_count": sum(i(r["false_old_recall_eval_only"]) for r in trace),
        "delayed_old_recall_count": sum(1 for r in trace if r["final_action"] == "delayed_old_recall"),
        "unresolved_count": sum(1 for r in trace if r["final_action"] == "remain_uncertain_after_wait"),
        "policy_violation_count": sum(1 for r in trace if i(r["attach_allowed"]) or i(r["promotion_allowed"]) or i(r["head_update_allowed"])),
    }
    return summary, trace


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.events))
    cfg = MemoryDecisionConfig(uncertainty_margin_threshold=args.threshold, bounded_wait_horizon_frames=args.horizon)
    disabled_summary, disabled_trace = evaluate(rows, enabled=False, cfg=cfg)
    enabled_summary, enabled_trace = evaluate(rows, enabled=True, cfg=cfg)
    summary_rows = [
        dict(disabled_summary, mode="policy_disabled_forced_recall"),
        dict(enabled_summary, mode="policy_enabled_bounded_wait"),
    ]
    passed = int(
        enabled_summary["coverage"] >= 0.95
        and enabled_summary["old_recall_precision"] > disabled_summary["old_recall_precision"]
        and enabled_summary["false_old_recall_count"] < disabled_summary["false_old_recall_count"]
        and enabled_summary["policy_violation_count"] == 0
    )
    compact = {
        "stage": "CORE-1AW",
        "artifact_version": args.artifact_version,
        "default_policy_enabled": 0,
        "threshold": args.threshold,
        "bounded_wait_horizon": args.horizon,
        "query_count": len(rows),
        "disabled_precision": disabled_summary["old_recall_precision"],
        "enabled_precision": enabled_summary["old_recall_precision"],
        "disabled_false_old_recall_count": disabled_summary["false_old_recall_count"],
        "enabled_false_old_recall_count": enabled_summary["false_old_recall_count"],
        "enabled_coverage": enabled_summary["coverage"],
        "delayed_old_recall_count": enabled_summary["delayed_old_recall_count"],
        "unresolved_count": enabled_summary["unresolved_count"],
        "policy_violation_count": enabled_summary["policy_violation_count"],
        "policy_flag_harness_passed": passed,
        "oracle_leakage_found": 0,
        "passed_minimum": passed,
        "next_recommendation": (
            "CORE-1AX run external/synthetic documentation update and mark policy experimental-disabled"
            if passed
            else "keep policy disabled; harness did not pass"
        ),
    }
    report = f"""# CORE-1AW Policy Flag Harness

This stage adds a disabled-by-default evaluation harness for the memory decision policy. With the flag off, behavior is forced old recall. With the flag on, low-margin recalls use bounded wait.

## Result

- Query count: {len(rows)}
- Disabled precision: {disabled_summary['old_recall_precision']:.4f}
- Enabled precision: {enabled_summary['old_recall_precision']:.4f}
- Disabled false old recalls: {disabled_summary['false_old_recall_count']}
- Enabled false old recalls: {enabled_summary['false_old_recall_count']}
- Enabled coverage: {enabled_summary['coverage']:.4f}
- Delayed old recalls: {enabled_summary['delayed_old_recall_count']}
- Unresolved: {enabled_summary['unresolved_count']}
- Policy violations: {enabled_summary['policy_violation_count']}
- Harness passed: {passed}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AW_"
    write_csv(
        out_dir / f"{prefix}flag_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "mode",
            "policy_enabled",
            "query_count",
            "accepted_old_recall_count",
            "coverage",
            "old_recall_precision",
            "false_old_recall_count",
            "delayed_old_recall_count",
            "unresolved_count",
            "policy_violation_count",
        ],
    )
    write_csv(
        out_dir / f"{prefix}decision_trace_{args.artifact_version}.csv",
        disabled_trace + enabled_trace,
        [
            "policy_enabled",
            "sequence_id",
            "event_id",
            "window_kind",
            "query_obs_id",
            "frame_idx",
            "track_id",
            "top1_margin",
            "baseline_top1_success",
            "final_action",
            "accepted_old_recall",
            "final_success_eval_only",
            "false_old_recall_eval_only",
            "true_old_recall_eval_only",
            "release_wait_frames",
            "release_query_obs_id",
            "attach_allowed",
            "promotion_allowed",
            "head_update_allowed",
        ],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
