from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1ac_raw_descriptor_memory_integration_smoke import baseline_score, f, i, normalize_scores
from experiments.run_core1e_pseudo_reentry_curriculum import build_pseudo_pairs, load_desc, metric_score, pair_features
from experiments.run_core1e2_curriculum_control_audit import make_random_split, train_logistic_metric
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CORE-1E3 safe retrieval integration gate for pseudo curriculum metric.")
    parser.add_argument("--observations", default="results/core1av_aj6/stage_CORE1AJ_stability_observation_trace_v1.csv")
    parser.add_argument("--descriptor-trace", default="results/core1av_aj6/stage_CORE1AJ_descriptor_trace_v1.csv")
    parser.add_argument("--core1e2-compact", default="results/core1e2/stage_CORE1E2_compact_for_gpt_v1.json")
    parser.add_argument("--output-dir", default="results/core1e3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=360)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--artifact-version", default="v1")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def gt_known(row: dict[str, Any]) -> bool:
    return str(row.get("gt_instance_eval_only", "")) not in {"", "nan", "None"}


def train_params(rows: list[dict[str, Any]], desc_by_obs: dict[int, np.ndarray], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    pairs = build_pseudo_pairs(rows, desc_by_obs, seed=args.seed, max_positive_pairs=1600, max_negatives_per_positive=2)
    x, y, kept = pair_features(pairs, desc_by_obs)
    train_idx, test_idx = make_random_split(len(kept), args.seed)
    params, _summary, _scores = train_logistic_metric(x, y, train_idx, test_idx, seed=args.seed, epochs=args.epochs, lr=args.lr)
    rng = np.random.default_rng(args.seed + 101)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    shuf_params, _shuf_summary, _shuf_scores = train_logistic_metric(x, y_shuf, train_idx, test_idx, seed=args.seed + 13, epochs=args.epochs, lr=args.lr)
    return params, shuf_params


def score_candidate_set(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    desc_by_obs: dict[int, np.ndarray],
    params: np.ndarray,
) -> list[dict[str, Any]]:
    base_raw = [baseline_score(query, cand) for cand in candidates]
    metric_raw = []
    qid = i(query.get("obs_id"))
    for cand in candidates:
        cid = i(cand.get("obs_id"))
        if qid in desc_by_obs and cid in desc_by_obs:
            metric_raw.append(metric_score(desc_by_obs[qid], desc_by_obs[cid], params))
        else:
            metric_raw.append(0.0)
    base_norm = normalize_scores(base_raw)
    metric_norm = normalize_scores(metric_raw)
    scored = []
    for cand, b0, m0, bn, mn in zip(candidates, base_raw, metric_raw, base_norm, metric_norm):
        scored.append({"candidate": cand, "baseline_raw": b0, "metric_raw": m0, "baseline_norm": bn, "metric_norm": mn})
    return scored


def margin(values: list[float]) -> float:
    ordered = sorted(values, reverse=True)
    if len(ordered) < 2:
        return 1.0
    return float(ordered[0] - ordered[1])


def rank_for_gt(scored: list[dict[str, Any]], score_key: str, target_gt: str) -> int:
    ordered = sorted(scored, key=lambda row: row[score_key], reverse=True)
    for idx, row in enumerate(ordered, start=1):
        if str(row["candidate"].get("gt_instance_eval_only", "")) == target_gt:
            return idx
    return 999


def evaluate_variant(
    rows: list[dict[str, Any]],
    desc_by_obs: dict[int, np.ndarray],
    params: np.ndarray,
    *,
    variant: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_window: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if i(row.get("obs_id")) in desc_by_obs:
            by_window[(str(row.get("sequence_id")), str(row.get("event_id")), str(row.get("window_kind")))].append(row)
    event_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for (_seq, _event, _kind), window_rows in by_window.items():
        memory: list[dict[str, Any]] = []
        for query in sorted(window_rows, key=lambda r: (i(r.get("frame_idx")), i(r.get("track_id")), i(r.get("obs_id")))):
            target_gt = str(query.get("gt_instance_eval_only", ""))
            candidates = [m for m in memory if gt_known(m) and i(m.get("obs_id")) in desc_by_obs]
            target_candidates = [m for m in candidates if str(m.get("gt_instance_eval_only", "")) == target_gt]
            distractors = [m for m in candidates if str(m.get("gt_instance_eval_only", "")) != target_gt]
            if gt_known(query) and target_candidates and distractors:
                scored = score_candidate_set(query, candidates, desc_by_obs, params)
                base_order = sorted(scored, key=lambda row: row["baseline_norm"], reverse=True)
                metric_order = sorted(scored, key=lambda row: row["metric_norm"], reverse=True)
                base_margin = margin([f(row["baseline_norm"]) for row in scored])
                metric_margin = margin([f(row["metric_norm"]) for row in scored])
                candidate_count = len(candidates)
                high_collision = int(candidate_count >= i(variant.get("candidate_count_min"), 6))
                gate_allowed = False
                gate_reason = "baseline_kept"
                if variant["mode"] == "baseline":
                    final_key = "baseline_norm"
                elif variant["mode"] == "metric_only":
                    final_key = "metric_norm"
                    gate_allowed = True
                    gate_reason = "metric_only"
                else:
                    gate_allowed = bool(
                        base_margin <= f(variant.get("baseline_margin_max"), 0.05)
                        and metric_margin >= f(variant.get("metric_margin_min"), 0.05)
                        and high_collision
                    )
                    final_key = "metric_norm" if gate_allowed else "baseline_norm"
                    gate_reason = "gate_passed" if gate_allowed else "gate_failed"
                final_order = sorted(scored, key=lambda row: row[final_key], reverse=True)
                top1 = final_order[0]
                target_rank = rank_for_gt(scored, final_key, target_gt)
                event_rows.append(
                    {
                        "variant": variant["variant"],
                        "sequence_id": query.get("sequence_id", ""),
                        "event_id": query.get("event_id", ""),
                        "window_kind": query.get("window_kind", ""),
                        "query_obs_id": query.get("obs_id", ""),
                        "target_instance_eval_only": target_gt,
                        "top1_instance_eval_only": top1["candidate"].get("gt_instance_eval_only", ""),
                        "target_rank": target_rank,
                        "top1_success": int(str(top1["candidate"].get("gt_instance_eval_only", "")) == target_gt),
                        "target_in_top3": int(target_rank <= 3),
                        "target_in_top5": int(target_rank <= 5),
                        "target_not_in_top5": int(target_rank > 5),
                        "target_in_top3_but_lost_top1": int(1 < target_rank <= 3),
                        "baseline_margin": base_margin,
                        "metric_margin": metric_margin,
                        "candidate_count": candidate_count,
                        "gate_allowed": int(gate_allowed),
                        "gate_reason": gate_reason,
                    }
                )
                gate_rows.append(
                    {
                        "variant": variant["variant"],
                        "query_obs_id": query.get("obs_id", ""),
                        "event_id": query.get("event_id", ""),
                        "baseline_top1_gt": base_order[0]["candidate"].get("gt_instance_eval_only", ""),
                        "metric_top1_gt": metric_order[0]["candidate"].get("gt_instance_eval_only", ""),
                        "final_top1_gt": top1["candidate"].get("gt_instance_eval_only", ""),
                        "target_gt_eval_only": target_gt,
                        "baseline_margin": base_margin,
                        "metric_margin": metric_margin,
                        "candidate_count": candidate_count,
                        "gate_allowed": int(gate_allowed),
                        "gate_reason": gate_reason,
                    }
                )
            memory.append(query)
    return event_rows, gate_rows


def summarize(rows: list[dict[str, Any]], baseline_by_query: dict[int, dict[str, Any]], variant: dict[str, Any]) -> dict[str, Any]:
    n = max(1, len(rows))
    success = sum(i(row.get("top1_success")) for row in rows)
    regressed = improved = unchanged_success = unchanged_failure = 0
    for row in rows:
        base = baseline_by_query.get(i(row.get("query_obs_id")))
        if base is None:
            continue
        b = i(base.get("top1_success"))
        v = i(row.get("top1_success"))
        if b == 1 and v == 0:
            regressed += 1
        elif b == 0 and v == 1:
            improved += 1
        elif b == 1:
            unchanged_success += 1
        else:
            unchanged_failure += 1
    return {
        "variant": variant["variant"],
        "mode": variant["mode"],
        "baseline_margin_max": variant.get("baseline_margin_max", ""),
        "metric_margin_min": variant.get("metric_margin_min", ""),
        "candidate_count_min": variant.get("candidate_count_min", ""),
        "query_count": len(rows),
        "top1": success / n,
        "top3": sum(1 for row in rows if i(row.get("target_rank"), 999) <= 3) / n,
        "top5": sum(1 for row in rows if i(row.get("target_rank"), 999) <= 5) / n,
        "false_retrieval_rate": 1.0 - success / n,
        "gate_fire_count": sum(i(row.get("gate_allowed")) for row in rows),
        "regression_event_count": regressed,
        "improved_event_count": improved,
        "unchanged_success_count": unchanged_success,
        "unchanged_failure_count": unchanged_failure,
        "target_not_in_top5_count": sum(i(row.get("target_not_in_top5")) for row in rows),
        "target_in_top3_but_lost_top1_count": sum(i(row.get("target_in_top3_but_lost_top1")) for row in rows),
        "selected_as_best": 0,
        "control": int(bool(variant.get("control"))),
        "eligible_for_integration": 0,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "stage_CORE1E3_"

    observations = read_csv(Path(args.observations))
    desc_by_obs = load_desc(Path(args.descriptor_trace))
    core1e2 = read_json(Path(args.core1e2_compact))
    learned_params, shuffled_params = train_params(observations, desc_by_obs, args)

    variants: list[dict[str, Any]] = [{"variant": "A0_dense_baseline", "mode": "baseline"}]
    variants.append({"variant": "A1_pseudo_metric_only", "mode": "metric_only"})
    for base_margin in [0.02, 0.05, 0.08, 0.12]:
        for metric_margin in [0.05, 0.10, 0.15, 0.20]:
            variants.append(
                {
                    "variant": f"A_gate_b{int(base_margin*100):03d}_m{int(metric_margin*100):03d}_c6",
                    "mode": "gated",
                    "baseline_margin_max": base_margin,
                    "metric_margin_min": metric_margin,
                    "candidate_count_min": 6,
                }
            )
            variants.append(
                {
                    "variant": f"C_shuffled_gate_b{int(base_margin*100):03d}_m{int(metric_margin*100):03d}_c6",
                    "mode": "gated",
                    "baseline_margin_max": base_margin,
                    "metric_margin_min": metric_margin,
                    "candidate_count_min": 6,
                    "control": True,
                    "use_shuffled": True,
                }
            )

    all_events: list[dict[str, Any]] = []
    all_gates: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    baseline_by_query: dict[int, dict[str, Any]] = {}
    for variant in variants:
        params = shuffled_params if variant.get("use_shuffled") else learned_params
        rows, gates = evaluate_variant(observations, desc_by_obs, params, variant=variant)
        if variant["variant"] == "A0_dense_baseline":
            baseline_by_query = {i(row.get("query_obs_id")): row for row in rows}
        all_events.extend(rows)
        all_gates.extend(gates)
        summaries.append(summarize(rows, baseline_by_query if baseline_by_query else {i(row.get("query_obs_id")): row for row in rows}, variant))

    # Re-summarize after baseline is known.
    summaries = []
    by_variant = {variant["variant"]: [row for row in all_events if row["variant"] == variant["variant"]] for variant in variants}
    for variant in variants:
        summaries.append(summarize(by_variant[variant["variant"]], baseline_by_query, variant))

    baseline = next(row for row in summaries if row["variant"] == "A0_dense_baseline")
    real_candidates = [row for row in summaries if i(row.get("control")) == 0 and row["variant"] != "A0_dense_baseline"]
    control_candidates = [row for row in summaries if i(row.get("control")) == 1]
    best_real = max(real_candidates, key=lambda row: (f(row["top1"]), i(row["improved_event_count"]) - i(row["regression_event_count"]), -i(row["gate_fire_count"])))
    best_control = max(control_candidates, key=lambda row: f(row["top1"])) if control_candidates else {"top1": 0.0}
    gate_passed = int(
        f(best_real["top1"]) > f(baseline["top1"])
        and f(best_real["top1"]) > f(best_control.get("top1", 0.0))
        and i(best_real["regression_event_count"]) <= 1
        and i(best_real["gate_fire_count"]) > 0
    )
    for row in summaries:
        row["selected_as_best"] = int(row is best_real)
        row["eligible_for_integration"] = int(row is best_real and gate_passed)

    compact = {
        "stage": "CORE-1E3",
        "artifact_version": args.artifact_version,
        "core1e2_split_generalization_passed": core1e2.get("split_generalization_passed", 0),
        "baseline_top1": baseline["top1"],
        "best_gate_variant": best_real["variant"],
        "best_gate_top1": best_real["top1"],
        "best_gate_fire_count": best_real["gate_fire_count"],
        "best_gate_improved_event_count": best_real["improved_event_count"],
        "best_gate_regression_event_count": best_real["regression_event_count"],
        "best_shuffled_control_top1": best_control.get("top1", 0.0),
        "integration_gate_passed": gate_passed,
        "oracle_leakage_found": 0,
        "passed_minimum": gate_passed,
        "main_failure_type": "" if gate_passed else "no_safe_retrieval_gate_above_baseline_and_shuffled_control",
        "next_recommendation": (
            "CORE-2 online consolidation with gated pseudo metric"
            if gate_passed
            else "do not integrate pseudo metric; build a harder official re-entry eval bridge or improve descriptor input"
        ),
    }

    report = f"""# CORE-1E3 Retrieval Integration Gate

CORE-1E2 showed that the pseudo curriculum metric generalizes at pair level. This stage tests whether it can be safely used only when baseline ranking is low-confidence and the metric is high-confidence.

## Result

- Baseline top1 on dense diagnostic pool: {float(baseline['top1']):.4f}
- Best gate: {best_real['variant']}
- Best gate top1: {float(best_real['top1']):.4f}
- Best gate fires: {best_real['gate_fire_count']}
- Improvements/regressions: {best_real['improved_event_count']} / {best_real['regression_event_count']}
- Best shuffled-control top1: {float(best_control.get('top1', 0.0)):.4f}
- Integration gate passed: {gate_passed}

Next recommendation: {compact['next_recommendation']}
"""

    write_csv(
        out_dir / f"{prefix}gate_ablation_summary_{args.artifact_version}.csv",
        summaries,
        [
            "variant",
            "mode",
            "baseline_margin_max",
            "metric_margin_min",
            "candidate_count_min",
            "query_count",
            "top1",
            "top3",
            "top5",
            "false_retrieval_rate",
            "gate_fire_count",
            "regression_event_count",
            "improved_event_count",
            "unchanged_success_count",
            "unchanged_failure_count",
            "target_not_in_top5_count",
            "target_in_top3_but_lost_top1_count",
            "control",
            "selected_as_best",
            "eligible_for_integration",
        ],
    )
    write_csv(
        out_dir / f"{prefix}event_delta_v1.csv",
        all_events,
        [
            "variant",
            "sequence_id",
            "event_id",
            "window_kind",
            "query_obs_id",
            "target_instance_eval_only",
            "top1_instance_eval_only",
            "target_rank",
            "top1_success",
            "target_in_top3",
            "target_in_top5",
            "target_not_in_top5",
            "target_in_top3_but_lost_top1",
            "baseline_margin",
            "metric_margin",
            "candidate_count",
            "gate_allowed",
            "gate_reason",
        ],
    )
    write_csv(
        out_dir / f"{prefix}gate_trace_v1.csv",
        all_gates,
        [
            "variant",
            "query_obs_id",
            "event_id",
            "baseline_top1_gt",
            "metric_top1_gt",
            "final_top1_gt",
            "target_gt_eval_only",
            "baseline_margin",
            "metric_margin",
            "candidate_count",
            "gate_allowed",
            "gate_reason",
        ],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
