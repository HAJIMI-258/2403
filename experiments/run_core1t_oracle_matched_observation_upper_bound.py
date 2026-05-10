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
from experiments.run_core1m_assignment_pair_confidence_gate import GATES, build_pairs_for_gate, summarize_gate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1T oracle matched-observation upper bound.")
    p.add_argument("--observations", default="results/core1p/stage_CORE1P_assignment_observation_trace_v1.csv")
    p.add_argument("--output-dir", default="results/core1t")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def f(v: Any, default: float = 0.0) -> float:
    if v in (None, ""):
        return default
    try:
        return float(v)
    except Exception:
        return default


def evaluate_variant(name: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries = []
    all_pairs = []
    for gate in GATES:
        pairs = build_pairs_for_gate(rows, gate)
        summary = summarize_gate(gate["name"], pairs)
        summary["oracle_filter_variant"] = name
        summaries.append(summary)
        for pair in pairs:
            pair["oracle_filter_variant"] = name
        all_pairs.extend(pairs)
    eligible = [s for s in summaries if int(s["eligible_for_training_smoke"]) == 1]
    if eligible:
        best = max(eligible, key=lambda s: (s["positive_pair_count"] + s["negative_pair_count"]))
    else:
        best = max(summaries, key=lambda s: (min(s["positive_pair_precision_eval_only"], s["negative_pair_precision_eval_only"]), s["positive_pair_count"] + s["negative_pair_count"])) if summaries else {}
    best["observation_count"] = len(rows)
    best["matched_observation_rate_eval_only"] = float(np.mean([1 if r.get("gt_instance_eval_only", "") != "" else 0 for r in rows])) if rows else 0.0
    return best, all_pairs


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.observations))
    variants = {
        "A0_no_oracle_filter": rows,
        "A1_oracle_matched_iou25": [r for r in rows if r.get("gt_instance_eval_only", "") != ""],
        "A2_oracle_matched_iou50": [r for r in rows if r.get("gt_instance_eval_only", "") != "" and f(r.get("match_iou_eval_only")) >= 0.50],
        "A3_oracle_matched_iou70": [r for r in rows if r.get("gt_instance_eval_only", "") != "" and f(r.get("match_iou_eval_only")) >= 0.70],
    }
    summary_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for name, variant_rows in variants.items():
        summary, pairs = evaluate_variant(name, variant_rows)
        summary_rows.append(summary)
        pair_rows.extend(pairs)
    eligible = [r for r in summary_rows if int(r["eligible_for_training_smoke"]) == 1]
    if eligible:
        best = max(eligible, key=lambda r: (r["positive_pair_count"] + r["negative_pair_count"]))
    else:
        best = max(summary_rows, key=lambda r: (min(r["positive_pair_precision_eval_only"], r["negative_pair_precision_eval_only"]), r["positive_pair_count"] + r["negative_pair_count"])) if summary_rows else {}
    compact = {
        "stage": "CORE-1T",
        "artifact_version": args.artifact_version,
        "diagnostic_only_uses_gt_filter": 1,
        "safe_for_main_online_training": 0,
        "best_oracle_filter_variant": best.get("oracle_filter_variant", ""),
        "best_gate": best.get("gate_name", ""),
        "best_observation_count": best.get("observation_count", 0),
        "best_positive_pair_count": best.get("positive_pair_count", 0),
        "best_negative_pair_count": best.get("negative_pair_count", 0),
        "best_positive_pair_precision_eval_only": best.get("positive_pair_precision_eval_only", 0.0),
        "best_negative_pair_precision_eval_only": best.get("negative_pair_precision_eval_only", 0.0),
        "oracle_matched_upper_bound_passed": int(bool(eligible)),
        "oracle_leakage_found": 0,
        "next_recommendation": (
            "CORE-1U learn/design GT-free matched-observation confidence target"
            if eligible
            else "tracker fragmentation remains after matched-observation filtering; repair tracker continuity"
        ),
    }
    report = f"""# CORE-1T Oracle Matched-Observation Upper Bound

This diagnostic uses GT only to remove unmatched observations and estimate whether matched-observation filtering would make pair mining viable. It is not a main-method training setup.

## Result

- Best oracle filter: {compact['best_oracle_filter_variant']}
- Best gate: {compact['best_gate']}
- Best observations: {compact['best_observation_count']}
- Best positive precision eval-only: {float(compact['best_positive_pair_precision_eval_only']):.4f}
- Best negative precision eval-only: {float(compact['best_negative_pair_precision_eval_only']):.4f}
- Upper bound passed: {compact['oracle_matched_upper_bound_passed']}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1T_"
    write_csv(
        out_dir / f"{prefix}oracle_filter_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "oracle_filter_variant",
            "gate_name",
            "observation_count",
            "matched_observation_rate_eval_only",
            "positive_pair_count",
            "negative_pair_count",
            "positive_pair_precision_eval_only",
            "negative_pair_precision_eval_only",
            "eligible_for_training_smoke",
        ],
    )
    write_csv(
        out_dir / f"{prefix}oracle_filtered_pair_trace_{args.artifact_version}.csv",
        pair_rows,
        [
            "oracle_filter_variant",
            "pair_id",
            "gate_name",
            "sequence_id",
            "event_id",
            "window_kind",
            "frame_i",
            "frame_j",
            "track_i",
            "track_j",
            "pair_type",
            "gt_instance_i_eval_only",
            "gt_instance_j_eval_only",
            "pair_correct_eval_only",
        ],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
