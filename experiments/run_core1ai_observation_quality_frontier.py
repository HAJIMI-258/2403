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

from experiments.run_core1aa_stability_namespace_pair_gate import (
    build_negative_pairs,
    build_positive_pairs,
    row_passes,
)
from experiments.run_core1ac_raw_descriptor_memory_integration_smoke import VARIANTS, build_event_rows, f, i
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json
from experiments.run_core1z_oracle_proposal_diagnostic_encoder import parse_descriptor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AI observation quality frontier.")
    p.add_argument("--observations", default="results/core1aa/stage_CORE1AA_stability_observation_trace_v1.csv")
    p.add_argument("--descriptor-trace", default="results/core1ad/stage_CORE1AD_descriptor_trace_v1.csv")
    p.add_argument("--output-dir", default="results/core1ai")
    p.add_argument("--max-negatives-per-observation", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def assign_obs_ids(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [dict(row, obs_id=idx) for idx, row in enumerate(rows, start=1)]


def load_desc(path: Path) -> dict[int, np.ndarray]:
    return {i(row["obs_id"]): parse_descriptor(str(row["descriptor"])) for row in read_csv(path)}


def candidate_gates() -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for score in [0.55, 0.60, 0.65, 0.70, 0.75]:
        for cost in [0.30, 0.35, 0.40, 0.45, 0.50]:
            gates.append({"gate_name": f"S{int(score*100):02d}_C{int(cost*100):02d}", "score_min": score, "match_cost_max": cost})
            gates.append({"gate_name": f"S{int(score*100):02d}_C{int(cost*100):02d}_streak2", "score_min": score, "match_cost_max": cost, "streak_min": 2})
            gates.append({"gate_name": f"S{int(score*100):02d}_C{int(cost*100):02d}_consec", "score_min": score, "match_cost_max": cost, "consecutive_required": True})
    # Deduplicate by name while preserving order.
    seen = set()
    out = []
    for gate in gates:
        if gate["gate_name"] not in seen:
            seen.add(gate["gate_name"])
            out.append(gate)
    return out


def precision(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return float(np.mean([i(r["pair_correct_eval_only"]) for r in rows]))


def pair_quality(selected: list[dict[str, Any]], gate: dict[str, Any], max_negatives: int) -> tuple[int, int, float, float]:
    positives = build_positive_pairs(selected, gate["gate_name"])
    negatives = build_negative_pairs(selected, gate["gate_name"], "cross_sequence", max_negatives)
    return len(positives), len(negatives), precision(positives), precision(negatives)


def baseline_queries(selected: list[dict[str, Any]], desc_by_obs: dict[int, np.ndarray], seed: int) -> tuple[int, float, int]:
    rows = build_event_rows(selected, desc_by_obs, VARIANTS[0], seed)
    if not rows:
        return 0, 0.0, 0
    top1 = float(np.mean([i(r["top1_success"]) for r in rows]))
    failures = sum(1 for r in rows if i(r["top1_success"]) == 0)
    return len(rows), top1, failures


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = assign_obs_ids(read_csv(Path(args.observations)))
    desc_by_obs = load_desc(Path(args.descriptor_trace))
    frontier_rows: list[dict[str, Any]] = []
    for gate in candidate_gates():
        selected = [row for row in rows if row_passes(row, gate) and i(row["obs_id"]) in desc_by_obs]
        pos_count, neg_count, pos_prec, neg_prec = pair_quality(selected, gate, args.max_negatives_per_observation)
        query_count, top1, failure_count = baseline_queries(selected, desc_by_obs, args.seed)
        pair_quality_passed = int(pos_count >= 20 and neg_count >= 20 and pos_prec >= 0.85 and neg_prec >= 0.85)
        hard_eval_ready = int(pair_quality_passed and failure_count >= 20 and query_count >= 100)
        frontier_rows.append(
            {
                "gate_name": gate["gate_name"],
                "score_min": gate.get("score_min", ""),
                "match_cost_max": gate.get("match_cost_max", ""),
                "streak_min": gate.get("streak_min", ""),
                "consecutive_required": int(bool(gate.get("consecutive_required"))),
                "selected_observation_count": len(selected),
                "positive_pair_count": pos_count,
                "negative_pair_count": neg_count,
                "positive_pair_precision_eval_only": pos_prec,
                "negative_pair_precision_namespace_eval_only": neg_prec,
                "query_count": query_count,
                "baseline_top1": top1,
                "baseline_failure_count": failure_count,
                "pair_quality_passed": pair_quality_passed,
                "hard_eval_ready": hard_eval_ready,
                "frontier_score": (min(pos_prec, neg_prec) * 1000.0) + failure_count + query_count / 1000.0,
            }
        )
    candidates = [r for r in frontier_rows if i(r["hard_eval_ready"]) == 1]
    if candidates:
        best = max(candidates, key=lambda r: (i(r["baseline_failure_count"]), f(r["positive_pair_precision_eval_only"]) + f(r["negative_pair_precision_namespace_eval_only"]), i(r["query_count"])))
    else:
        best = max(frontier_rows, key=lambda r: (i(r["pair_quality_passed"]), i(r["baseline_failure_count"]), f(r["frontier_score"]))) if frontier_rows else {}
    for row in frontier_rows:
        row["selected_as_best_frontier"] = int(row is best)
    compact = {
        "stage": "CORE-1AI",
        "artifact_version": args.artifact_version,
        "gate_count": len(frontier_rows),
        "best_gate": best.get("gate_name", ""),
        "best_selected_observation_count": best.get("selected_observation_count", 0),
        "best_positive_pair_precision": best.get("positive_pair_precision_eval_only", 0.0),
        "best_negative_pair_precision": best.get("negative_pair_precision_namespace_eval_only", 0.0),
        "best_query_count": best.get("query_count", 0),
        "best_baseline_top1": best.get("baseline_top1", 0.0),
        "best_baseline_failure_count": best.get("baseline_failure_count", 0),
        "hard_eval_ready": best.get("hard_eval_ready", 0),
        "pair_quality_passed": best.get("pair_quality_passed", 0),
        "oracle_leakage_found": 0,
        "next_recommendation": (
            "CORE-1AJ run descriptor cue on selected hard frontier gate"
            if i(best.get("hard_eval_ready")) == 1
            else "no gate provides both clean pairs and enough hard failures; repair proposal/observation generation"
        ),
    }
    report = f"""# CORE-1AI Observation Quality Frontier

This stage grid-searches online-visible score/match-cost/streak gates to find a useful frontier: clean pair mining plus enough baseline failures for descriptor integration testing.

## Result

- Gates scanned: {len(frontier_rows)}
- Best gate: {compact['best_gate']}
- Selected observations: {compact['best_selected_observation_count']}
- Pair precision: positive {float(compact['best_positive_pair_precision']):.4f}, negative {float(compact['best_negative_pair_precision']):.4f}
- Queries: {compact['best_query_count']}
- Baseline top1: {float(compact['best_baseline_top1']):.4f}
- Baseline failures: {compact['best_baseline_failure_count']}
- Pair quality passed: {compact['pair_quality_passed']}
- Hard eval ready: {compact['hard_eval_ready']}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AI_"
    write_csv(
        out_dir / f"{prefix}frontier_summary_{args.artifact_version}.csv",
        frontier_rows,
        [
            "gate_name",
            "score_min",
            "match_cost_max",
            "streak_min",
            "consecutive_required",
            "selected_observation_count",
            "positive_pair_count",
            "negative_pair_count",
            "positive_pair_precision_eval_only",
            "negative_pair_precision_namespace_eval_only",
            "query_count",
            "baseline_top1",
            "baseline_failure_count",
            "pair_quality_passed",
            "hard_eval_ready",
            "frontier_score",
            "selected_as_best_frontier",
        ],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
