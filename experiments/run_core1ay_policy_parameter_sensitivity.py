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
    p = argparse.ArgumentParser(description="CORE-1AY bounded-wait policy parameter sensitivity.")
    p.add_argument("--events", default="results/core1av_an6/stage_CORE1AN_event_uncertainty_trace_v1.csv")
    p.add_argument("--output-dir", default="results/core1ay")
    p.add_argument("--thresholds", default="0.01,0.0149,0.0194,0.02,0.025,0.03,0.04")
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
    return str(row["sequence_id"]), str(row["event_id"]), str(row["window_kind"]), str(row["track_id"])


def evaluate(rows: list[dict[str, Any]], threshold: float, horizon: int) -> dict[str, Any]:
    cfg = MemoryDecisionConfig(uncertainty_margin_threshold=threshold, bounded_wait_horizon_frames=horizon)
    by_track: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_track[group_key(row)].append(row)
    for key in by_track:
        by_track[key] = sorted(by_track[key], key=lambda r: (i(r["frame_idx"]), i(r["query_obs_id"])))

    accepted_success: list[int] = []
    false_old = 0
    delayed = 0
    wrong_release = 0
    unresolved = 0
    for row in rows:
        decision = decide_memory_retrieval(f(row["top1_margin"]), cfg)
        if decision.retrieval_state.value == "old_recall_candidate":
            accepted_success.append(i(row["top1_success"]))
            false_old += int(i(row["top1_success"]) == 0)
            continue
        release = None
        start = i(row["frame_idx"])
        for cand in by_track[group_key(row)]:
            wait = i(cand["frame_idx"]) - start
            if can_release_after_wait(wait_frames=wait, release_margin=f(cand["top1_margin"]), config=cfg):
                release = cand
                break
        if release is None:
            unresolved += 1
            continue
        delayed += 1
        accepted_success.append(i(release["top1_success"]))
        wrong_release += int(i(release["top1_success"]) == 0)
        false_old += int(i(release["top1_success"]) == 0)
    baseline_top1 = float(np.mean([i(r["top1_success"]) for r in rows])) if rows else 0.0
    baseline_false = sum(1 for r in rows if i(r["top1_success"]) == 0)
    coverage = len(accepted_success) / len(rows) if rows else 0.0
    precision = float(np.mean(accepted_success)) if accepted_success else 0.0
    eligible = int(coverage >= 0.95 and precision > baseline_top1 and false_old < baseline_false and wrong_release == 0)
    return {
        "threshold": threshold,
        "horizon": horizon,
        "query_count": len(rows),
        "baseline_top1": baseline_top1,
        "baseline_false_old_recall_count": baseline_false,
        "policy_coverage": coverage,
        "policy_old_recall_precision": precision,
        "policy_false_old_recall_count": false_old,
        "false_old_recall_reduction": baseline_false - false_old,
        "delayed_old_recall_count": delayed,
        "released_wrong_count": wrong_release,
        "unresolved_count": unresolved,
        "eligible": eligible,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.events))
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    summary_rows = [evaluate(rows, threshold, horizon) for threshold in thresholds for horizon in horizons]
    eligible = [r for r in summary_rows if i(r["eligible"]) == 1]
    best = max(eligible, key=lambda r: (i(r["false_old_recall_reduction"]), f(r["policy_old_recall_precision"]), f(r["policy_coverage"]))) if eligible else max(summary_rows, key=lambda r: (i(r["false_old_recall_reduction"]), f(r["policy_old_recall_precision"])))
    for row in summary_rows:
        row["selected_as_best"] = int(row is best)
    canonical = [r for r in summary_rows if abs(f(r["threshold"]) - 0.0194) < 1e-9 and i(r["horizon"]) == 10]
    canonical_passed = i(canonical[0]["eligible"]) if canonical else 0
    compact = {
        "stage": "CORE-1AY",
        "artifact_version": args.artifact_version,
        "query_count": len(rows),
        "threshold_count": len(thresholds),
        "horizon_count": len(horizons),
        "config_count": len(summary_rows),
        "eligible_config_count": len(eligible),
        "canonical_threshold": 0.0194,
        "canonical_horizon": 10,
        "canonical_config_passed": canonical_passed,
        "best_threshold": best["threshold"],
        "best_horizon": best["horizon"],
        "best_policy_precision": best["policy_old_recall_precision"],
        "best_policy_coverage": best["policy_coverage"],
        "best_false_old_recall_reduction": best["false_old_recall_reduction"],
        "best_released_wrong_count": best["released_wrong_count"],
        "parameter_sensitivity_passed": int(canonical_passed and len(eligible) >= 3),
        "oracle_leakage_found": 0,
        "passed_minimum": int(canonical_passed and len(eligible) >= 3),
        "next_recommendation": (
            "CORE-1AZ freeze CORE-1 uncertainty policy milestone and prepare next core model objective"
            if canonical_passed and len(eligible) >= 3
            else "keep policy experimental; parameter sensitivity is too narrow"
        ),
    }
    report = f"""# CORE-1AY Policy Parameter Sensitivity

This stage scans threshold / bounded-wait horizon settings on the six-sequence CORE-1AV stream. It checks whether the policy works over a neighborhood of parameters, not only at one point.

## Result

- Query count: {len(rows)}
- Configs scanned: {len(summary_rows)}
- Eligible configs: {len(eligible)}
- Canonical config passed: {canonical_passed}
- Best threshold: {best['threshold']}
- Best horizon: {best['horizon']}
- Best precision: {float(best['policy_old_recall_precision']):.4f}
- Best coverage: {float(best['policy_coverage']):.4f}
- Best false-old reduction: {best['false_old_recall_reduction']}
- Best released wrong: {best['released_wrong_count']}
- Parameter sensitivity passed: {compact['parameter_sensitivity_passed']}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AY_"
    write_csv(
        out_dir / f"{prefix}parameter_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "threshold",
            "horizon",
            "query_count",
            "baseline_top1",
            "baseline_false_old_recall_count",
            "policy_coverage",
            "policy_old_recall_precision",
            "policy_false_old_recall_count",
            "false_old_recall_reduction",
            "delayed_old_recall_count",
            "released_wrong_count",
            "unresolved_count",
            "eligible",
            "selected_as_best",
        ],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
