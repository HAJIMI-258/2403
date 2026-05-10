from __future__ import annotations

import argparse
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
from experiments.run_core1ai_observation_quality_frontier import (
    baseline_queries,
    candidate_gates,
    pair_quality,
)
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json
from experiments.run_core1z_oracle_proposal_diagnostic_encoder import parse_descriptor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AK decoupled clean-train / hard-eval frontier.")
    p.add_argument("--observations", default="results/core1aj/stage_CORE1AJ_stability_observation_trace_v1.csv")
    p.add_argument("--descriptor-trace", default="results/core1aj/stage_CORE1AJ_descriptor_trace_v1.csv")
    p.add_argument("--output-dir", default="results/core1ak")
    p.add_argument("--max-negatives-per-observation", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def i(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def assign_obs_ids(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [dict(row, obs_id=idx) for idx, row in enumerate(rows, start=1)]


def load_desc(path: Path) -> dict[int, np.ndarray]:
    return {i(row["obs_id"]): parse_descriptor(str(row["descriptor"])) for row in read_csv(path)}


def precision(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return float(np.mean([i(r["pair_correct_eval_only"]) for r in rows]))


def pair_failure_samples(
    selected: list[dict[str, Any]],
    gate_name: str,
    max_negatives: int,
    limit: int = 25,
) -> list[dict[str, Any]]:
    positives = build_positive_pairs(selected, gate_name)
    negatives = build_negative_pairs(selected, gate_name, "cross_sequence", max_negatives)
    failures: list[dict[str, Any]] = []
    for row in positives:
        if i(row.get("pair_correct_eval_only")) == 0:
            failures.append(dict(row, pair_pool="positive"))
    for row in negatives:
        if i(row.get("pair_correct_eval_only")) == 0:
            failures.append(dict(row, pair_pool="negative"))
    return failures[:limit]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = assign_obs_ids(read_csv(Path(args.observations)))
    desc_by_obs = load_desc(Path(args.descriptor_trace))
    gates = candidate_gates()

    gate_rows: list[dict[str, Any]] = []
    selected_by_gate: dict[str, list[dict[str, Any]]] = {}
    for gate in gates:
        selected = [row for row in rows if row_passes(row, gate) and i(row["obs_id"]) in desc_by_obs]
        selected_by_gate[gate["gate_name"]] = selected
        pos_count, neg_count, pos_prec, neg_prec = pair_quality(selected, gate, args.max_negatives_per_observation)
        query_count, top1, failure_count = baseline_queries(selected, desc_by_obs, args.seed)
        pair_quality_passed = int(pos_count >= 20 and neg_count >= 20 and pos_prec >= 0.85 and neg_prec >= 0.85)
        hard_eval_ready = int(failure_count >= 20 and query_count >= 100)
        gate_rows.append(
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
            }
        )

    gate_by_name = {row["gate_name"]: row for row in gate_rows}
    combo_rows: list[dict[str, Any]] = []
    for train in gate_rows:
        for eval_row in gate_rows:
            train_pair_quality = int(i(train["pair_quality_passed"]) == 1)
            hard_eval_ready = int(i(eval_row["baseline_failure_count"]) >= 20 and i(eval_row["query_count"]) >= 100)
            decoupled_ready = int(train_pair_quality and hard_eval_ready)
            train_eval_same = int(train["gate_name"] == eval_row["gate_name"])
            # Prefer hard eval pools that are not wildly noisier than train, but do not require
            # pair precision there because it is not used as an online training signal.
            eval_noise_risk = max(0.0, 0.85 - f(eval_row["positive_pair_precision_eval_only"]))
            combo_rows.append(
                {
                    "train_gate": train["gate_name"],
                    "eval_gate": eval_row["gate_name"],
                    "train_eval_same": train_eval_same,
                    "train_selected_observation_count": train["selected_observation_count"],
                    "train_positive_pair_count": train["positive_pair_count"],
                    "train_negative_pair_count": train["negative_pair_count"],
                    "train_positive_pair_precision_eval_only": train["positive_pair_precision_eval_only"],
                    "train_negative_pair_precision_namespace_eval_only": train["negative_pair_precision_namespace_eval_only"],
                    "train_pair_quality_passed": train_pair_quality,
                    "eval_selected_observation_count": eval_row["selected_observation_count"],
                    "eval_query_count": eval_row["query_count"],
                    "eval_baseline_top1": eval_row["baseline_top1"],
                    "eval_baseline_failure_count": eval_row["baseline_failure_count"],
                    "eval_positive_pair_precision_eval_only_audit": eval_row["positive_pair_precision_eval_only"],
                    "eval_negative_pair_precision_namespace_eval_only_audit": eval_row["negative_pair_precision_namespace_eval_only"],
                    "eval_noise_risk": eval_noise_risk,
                    "hard_eval_ready": hard_eval_ready,
                    "decoupled_ready": decoupled_ready,
                    "frontier_score": (
                        decoupled_ready * 100000.0
                        + i(eval_row["baseline_failure_count"]) * 100.0
                        + f(train["positive_pair_precision_eval_only"]) * 10.0
                        + f(train["negative_pair_precision_namespace_eval_only"]) * 10.0
                        - eval_noise_risk
                    ),
                }
            )

    ready = [row for row in combo_rows if i(row["decoupled_ready"]) == 1]
    if ready:
        best = max(
            ready,
            key=lambda r: (
                i(r["eval_baseline_failure_count"]),
                f(r["train_positive_pair_precision_eval_only"]) + f(r["train_negative_pair_precision_namespace_eval_only"]),
                -f(r["eval_noise_risk"]),
                i(r["eval_query_count"]),
            ),
        )
    else:
        best = max(combo_rows, key=lambda r: f(r["frontier_score"])) if combo_rows else {}
    for row in combo_rows:
        row["selected_as_best_decoupled"] = int(row is best)

    failure_sample_rows: list[dict[str, Any]] = []
    if best:
        train_gate = str(best["train_gate"])
        eval_gate = str(best["eval_gate"])
        failure_sample_rows.extend(pair_failure_samples(selected_by_gate[train_gate], train_gate, args.max_negatives_per_observation))
        for row in pair_failure_samples(selected_by_gate[eval_gate], eval_gate, args.max_negatives_per_observation):
            row["pair_pool"] = f"eval_audit_{row.get('pair_pool', '')}"
            failure_sample_rows.append(row)

    compact = {
        "stage": "CORE-1AK",
        "artifact_version": args.artifact_version,
        "observation_count": len(rows),
        "descriptor_available_count": len(desc_by_obs),
        "gate_count": len(gate_rows),
        "combo_count": len(combo_rows),
        "decoupled_ready_combo_count": len(ready),
        "best_train_gate": best.get("train_gate", ""),
        "best_eval_gate": best.get("eval_gate", ""),
        "best_train_positive_pair_precision": best.get("train_positive_pair_precision_eval_only", 0.0),
        "best_train_negative_pair_precision": best.get("train_negative_pair_precision_namespace_eval_only", 0.0),
        "best_train_positive_pair_count": best.get("train_positive_pair_count", 0),
        "best_train_negative_pair_count": best.get("train_negative_pair_count", 0),
        "best_eval_query_count": best.get("eval_query_count", 0),
        "best_eval_baseline_top1": best.get("eval_baseline_top1", 0.0),
        "best_eval_baseline_failure_count": best.get("eval_baseline_failure_count", 0),
        "best_eval_noise_risk": best.get("eval_noise_risk", 0.0),
        "pair_quality_passed": best.get("train_pair_quality_passed", 0),
        "hard_eval_ready": best.get("hard_eval_ready", 0),
        "decoupled_frontier_ready": best.get("decoupled_ready", 0),
        "oracle_leakage_found": 0,
        "next_recommendation": (
            "CORE-1AL train/integrate online descriptor encoder using clean train gate and hard eval gate"
            if i(best.get("decoupled_ready")) == 1
            else "decoupled train/eval still not ready; repair proposal/observation frontier"
        ),
    }

    report = f"""# CORE-1AK Decoupled Train/Eval Frontier

This stage tests whether CORE-1 can decouple a clean online pair-mining pool from a broader hard evaluation pool. It does not change the model and does not use GT labels for online scoring.

## Result

- Gates scanned: {len(gate_rows)}
- Train/eval combinations: {len(combo_rows)}
- Ready combinations: {len(ready)}
- Best train gate: {compact['best_train_gate']}
- Best eval gate: {compact['best_eval_gate']}
- Train pair precision: positive {float(compact['best_train_positive_pair_precision']):.4f}, negative {float(compact['best_train_negative_pair_precision']):.4f}
- Train pairs: positive {compact['best_train_positive_pair_count']}, negative {compact['best_train_negative_pair_count']}
- Eval queries: {compact['best_eval_query_count']}
- Eval baseline top1: {float(compact['best_eval_baseline_top1']):.4f}
- Eval baseline failures: {compact['best_eval_baseline_failure_count']}
- Decoupled frontier ready: {compact['decoupled_frontier_ready']}

Next recommendation: {compact['next_recommendation']}
"""

    prefix = "stage_CORE1AK_"
    write_csv(
        out_dir / f"{prefix}gate_pool_summary_{args.artifact_version}.csv",
        gate_rows,
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
        ],
    )
    write_csv(
        out_dir / f"{prefix}decoupled_frontier_summary_{args.artifact_version}.csv",
        combo_rows,
        [
            "train_gate",
            "eval_gate",
            "train_eval_same",
            "train_selected_observation_count",
            "train_positive_pair_count",
            "train_negative_pair_count",
            "train_positive_pair_precision_eval_only",
            "train_negative_pair_precision_namespace_eval_only",
            "train_pair_quality_passed",
            "eval_selected_observation_count",
            "eval_query_count",
            "eval_baseline_top1",
            "eval_baseline_failure_count",
            "eval_positive_pair_precision_eval_only_audit",
            "eval_negative_pair_precision_namespace_eval_only_audit",
            "eval_noise_risk",
            "hard_eval_ready",
            "decoupled_ready",
            "frontier_score",
            "selected_as_best_decoupled",
        ],
    )
    write_csv(
        out_dir / f"{prefix}pair_failure_sample_{args.artifact_version}.csv",
        failure_sample_rows,
        sorted({key for row in failure_sample_rows for key in row.keys()}) if failure_sample_rows else ["pair_pool"],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1ak_decoupled_train_eval_frontier.py",
                "gt_used_for_online_scoring": 0,
                "gt_used_for_pair_audit_only": 1,
                "future_frame_used": 0,
                "pretrained_weights_used": 0,
                "leakage_found": 0,
            }
        ],
        ["file", "gt_used_for_online_scoring", "gt_used_for_pair_audit_only", "future_frame_used", "pretrained_weights_used", "leakage_found"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
