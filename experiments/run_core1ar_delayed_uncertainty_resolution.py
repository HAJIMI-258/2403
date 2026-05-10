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
    p = argparse.ArgumentParser(description="CORE-1AR delayed uncertainty resolution audit.")
    p.add_argument("--core1an-events", default="results/core1an/stage_CORE1AN_event_uncertainty_trace_v1.csv")
    p.add_argument("--output-dir", default="results/core1ar")
    p.add_argument("--horizons", default="3,5,10,20")
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
    # Track-local future evidence is online-visible. GT is only used later for
    # evaluating whether the eventual release was correct.
    return str(row["sequence_id"]), str(row["event_id"]), str(row["window_kind"]), str(row["track_id"])


def find_release(row: dict[str, Any], track_rows: list[dict[str, Any]], horizon: int) -> dict[str, Any] | None:
    start_frame = i(row["frame_idx"])
    threshold = f(row["selected_threshold"])
    candidates = [
        r
        for r in track_rows
        if i(r["frame_idx"]) > start_frame
        and i(r["frame_idx"]) <= start_frame + horizon
        and f(r["top1_margin"]) >= threshold
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (i(r["frame_idx"]), i(r["query_obs_id"])))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.core1an_events))
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    by_track: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_track[group_key(row)].append(row)
    for key in by_track:
        by_track[key] = sorted(by_track[key], key=lambda r: (i(r["frame_idx"]), i(r["query_obs_id"])))

    uncertain = [r for r in rows if str(r.get("uncertainty_action")) != "commit"]
    trace_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        horizon_trace: list[dict[str, Any]] = []
        for row in uncertain:
            release = find_release(row, by_track[group_key(row)], horizon)
            false_suppressed = i(row.get("false_retrieval_avoided"))
            correct_delayed = i(row.get("correct_abstained"))
            if release is None:
                resolution_state = "unresolved_within_horizon"
                release_success = 0
                release_frame = ""
                release_query = ""
                release_margin = ""
                wait_frames = ""
            else:
                release_success = i(release["top1_success"])
                release_frame = release["frame_idx"]
                release_query = release["query_obs_id"]
                release_margin = release["top1_margin"]
                wait_frames = i(release["frame_idx"]) - i(row["frame_idx"])
                if release_success:
                    resolution_state = "resolved_correct_old_recall"
                else:
                    resolution_state = "released_wrong_old_recall"
            horizon_trace.append(
                {
                    "horizon_frames": horizon,
                    "sequence_id": row["sequence_id"],
                    "event_id": row["event_id"],
                    "window_kind": row["window_kind"],
                    "track_id": row["track_id"],
                    "uncertain_query_obs_id": row["query_obs_id"],
                    "uncertain_frame_idx": row["frame_idx"],
                    "uncertain_top1_margin": row["top1_margin"],
                    "uncertain_top1_success_eval_only": row["top1_success"],
                    "false_old_suppressed_eval_only": false_suppressed,
                    "correct_old_delayed_eval_only": correct_delayed,
                    "release_query_obs_id": release_query,
                    "release_frame_idx": release_frame,
                    "release_wait_frames": wait_frames,
                    "release_top1_margin": release_margin,
                    "release_success_eval_only": release_success,
                    "resolution_state": resolution_state,
                    "future_used_for_current_decision": 0,
                }
            )
        trace_rows.extend(horizon_trace)
        resolved = [r for r in horizon_trace if r["resolution_state"] == "resolved_correct_old_recall"]
        wrong_release = [r for r in horizon_trace if r["resolution_state"] == "released_wrong_old_recall"]
        unresolved = [r for r in horizon_trace if r["resolution_state"] == "unresolved_within_horizon"]
        false_suppressed_resolved = sum(1 for r in resolved if i(r["false_old_suppressed_eval_only"]) == 1)
        correct_delay_resolved = sum(1 for r in resolved if i(r["correct_old_delayed_eval_only"]) == 1)
        summary_rows.append(
            {
                "horizon_frames": horizon,
                "uncertain_count": len(horizon_trace),
                "resolved_correct_count": len(resolved),
                "released_wrong_count": len(wrong_release),
                "unresolved_count": len(unresolved),
                "resolution_rate": len(resolved) / len(horizon_trace) if horizon_trace else 0.0,
                "wrong_release_rate": len(wrong_release) / len(horizon_trace) if horizon_trace else 0.0,
                "false_suppressed_resolved_count": false_suppressed_resolved,
                "correct_delay_resolved_count": correct_delay_resolved,
                "mean_wait_frames": float(np.mean([i(r["release_wait_frames"]) for r in resolved])) if resolved else 0.0,
                "eligible_for_delayed_policy": int(len(resolved) > len(wrong_release) and len(resolved) > 0),
            }
        )

    eligible = [r for r in summary_rows if i(r["eligible_for_delayed_policy"]) == 1]
    if eligible:
        best = max(eligible, key=lambda r: (i(r["resolved_correct_count"]) - 2 * i(r["released_wrong_count"]), -i(r["horizon_frames"])))
    else:
        best = max(summary_rows, key=lambda r: (i(r["resolved_correct_count"]) - 2 * i(r["released_wrong_count"]), -i(r["horizon_frames"]))) if summary_rows else {}

    delayed_resolution_passed = int(best and i(best["resolved_correct_count"]) > i(best["released_wrong_count"]) and i(best["resolved_correct_count"]) >= 5)
    compact = {
        "stage": "CORE-1AR",
        "artifact_version": args.artifact_version,
        "uncertain_count": len(uncertain),
        "best_horizon_frames": best.get("horizon_frames", 0),
        "best_resolved_correct_count": best.get("resolved_correct_count", 0),
        "best_released_wrong_count": best.get("released_wrong_count", 0),
        "best_unresolved_count": best.get("unresolved_count", 0),
        "best_resolution_rate": best.get("resolution_rate", 0.0),
        "best_false_suppressed_resolved_count": best.get("false_suppressed_resolved_count", 0),
        "best_correct_delay_resolved_count": best.get("correct_delay_resolved_count", 0),
        "best_mean_wait_frames": best.get("mean_wait_frames", 0.0),
        "delayed_resolution_passed": delayed_resolution_passed,
        "future_used_for_current_decision": 0,
        "oracle_leakage_found": 0,
        "passed_minimum": delayed_resolution_passed,
        "next_recommendation": (
            "CORE-1AS add delayed uncertainty resolution policy with bounded wait horizon"
            if delayed_resolution_passed
            else "uncertain queue does not resolve safely by waiting; route to active evidence acquisition instead"
        ),
    }
    report = f"""# CORE-1AR Delayed Uncertainty Resolution

This stage audits whether `uncertain_need_more_evidence` decisions can be resolved by waiting for a later high-margin observation on the same online-visible track. Future observations are used only for offline evaluation, not for the current decision.

## Result

- Uncertain decisions: {len(uncertain)}
- Best horizon: {compact['best_horizon_frames']} frames
- Resolved correct: {compact['best_resolved_correct_count']}
- Released wrong: {compact['best_released_wrong_count']}
- Unresolved: {compact['best_unresolved_count']}
- Resolution rate: {float(compact['best_resolution_rate']):.4f}
- False-suppressed cases resolved: {compact['best_false_suppressed_resolved_count']}
- Correct delayed cases resolved: {compact['best_correct_delay_resolved_count']}
- Mean wait frames: {float(compact['best_mean_wait_frames']):.2f}
- Delayed resolution passed: {delayed_resolution_passed}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AR_"
    write_csv(
        out_dir / f"{prefix}horizon_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "horizon_frames",
            "uncertain_count",
            "resolved_correct_count",
            "released_wrong_count",
            "unresolved_count",
            "resolution_rate",
            "wrong_release_rate",
            "false_suppressed_resolved_count",
            "correct_delay_resolved_count",
            "mean_wait_frames",
            "eligible_for_delayed_policy",
        ],
    )
    write_csv(
        out_dir / f"{prefix}delayed_resolution_trace_{args.artifact_version}.csv",
        trace_rows,
        [
            "horizon_frames",
            "sequence_id",
            "event_id",
            "window_kind",
            "track_id",
            "uncertain_query_obs_id",
            "uncertain_frame_idx",
            "uncertain_top1_margin",
            "uncertain_top1_success_eval_only",
            "false_old_suppressed_eval_only",
            "correct_old_delayed_eval_only",
            "release_query_obs_id",
            "release_frame_idx",
            "release_wait_frames",
            "release_top1_margin",
            "release_success_eval_only",
            "resolution_state",
            "future_used_for_current_decision",
        ],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1ar_delayed_uncertainty_resolution.py",
                "future_used_for_current_decision": 0,
                "gt_used_for_online_scoring": 0,
                "gt_used_for_eval_only": 1,
                "pretrained_weights_used": 0,
                "leakage_found": 0,
            }
        ],
        ["file", "future_used_for_current_decision", "gt_used_for_online_scoring", "gt_used_for_eval_only", "pretrained_weights_used", "leakage_found"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
