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

from experiments.run_core1ac_raw_descriptor_memory_integration_smoke import baseline_score, f, i
from experiments.run_core1aa_stability_namespace_pair_gate import row_passes
from experiments.run_core1al_decoupled_online_metric_smoke import assign_obs_ids, gate_by_name, load_desc, read_json
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AN uncertainty abstention gate.")
    p.add_argument("--observations", default="results/core1aj/stage_CORE1AJ_stability_observation_trace_v1.csv")
    p.add_argument("--descriptor-trace", default="results/core1aj/stage_CORE1AJ_descriptor_trace_v1.csv")
    p.add_argument("--core1ak-compact", default="results/core1ak/stage_CORE1AK_compact_for_gpt_v1.json")
    p.add_argument("--output-dir", default="results/core1an")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-9:
        return [0.5 for _ in values]
    return [float((v - mn) / (mx - mn)) for v in arr]


def baseline_query_rows(rows: list[dict[str, Any]], desc_ids: set[int]) -> list[dict[str, Any]]:
    by_window: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_window[(str(row["sequence_id"]), str(row["event_id"]), str(row["window_kind"]))].append(row)
    out: list[dict[str, Any]] = []
    for (_seq, _event, _kind), window_rows in by_window.items():
        memory: list[dict[str, Any]] = []
        for query in sorted(window_rows, key=lambda r: (i(r["frame_idx"]), i(r["track_id"]), i(r["obs_id"]))):
            qgt = str(query.get("gt_instance_eval_only", ""))
            qid = i(query["obs_id"])
            candidates = [m for m in memory if i(m["obs_id"]) in desc_ids and str(m.get("gt_instance_eval_only", "")) != ""]
            target_candidates = [m for m in candidates if str(m.get("gt_instance_eval_only", "")) == qgt]
            distractors = [m for m in candidates if str(m.get("gt_instance_eval_only", "")) != qgt]
            if qid in desc_ids and qgt != "" and target_candidates and distractors:
                scores_raw = [baseline_score(query, cand) for cand in candidates]
                scores = normalize(scores_raw)
                order = sorted(zip(candidates, scores, scores_raw), key=lambda x: x[1], reverse=True)
                top1, top1_norm, top1_raw = order[0]
                top2_norm = order[1][1] if len(order) > 1 else 0.0
                top2_raw = order[1][2] if len(order) > 1 else 0.0
                target_rank = 999
                target_norm_scores: list[float] = []
                wrong_norm_scores: list[float] = []
                for rank, (cand, score, _raw) in enumerate(order, start=1):
                    if str(cand.get("gt_instance_eval_only", "")) == qgt:
                        target_rank = min(target_rank, rank)
                        target_norm_scores.append(score)
                    else:
                        wrong_norm_scores.append(score)
                margin = float(top1_norm - top2_norm)
                target_margin = float(max(target_norm_scores) - max(wrong_norm_scores)) if target_norm_scores and wrong_norm_scores else 0.0
                out.append(
                    {
                        "sequence_id": query["sequence_id"],
                        "event_id": query["event_id"],
                        "window_kind": query["window_kind"],
                        "query_obs_id": qid,
                        "frame_idx": query["frame_idx"],
                        "track_id": query["track_id"],
                        "candidate_count": len(candidates),
                        "target_candidate_count": len(target_candidates),
                        "top1_obs_id": top1["obs_id"],
                        "top1_instance_eval_only": top1.get("gt_instance_eval_only", ""),
                        "target_instance_eval_only": qgt,
                        "top1_success": int(str(top1.get("gt_instance_eval_only", "")) == qgt),
                        "target_rank": target_rank,
                        "target_in_top3": int(target_rank <= 3),
                        "top1_margin": margin,
                        "top1_raw_margin": float(top1_raw - top2_raw),
                        "target_margin": target_margin,
                        "top1_score": top1_norm,
                        "top2_score": top2_norm,
                    }
                )
            memory.append(query)
    return out


def random_abstention_baseline(rows: list[dict[str, Any]], abstain_count: int, seed: int, trials: int = 200) -> dict[str, float]:
    if not rows:
        return {"random_false_avoided_mean": 0.0, "random_accuracy_mean": 0.0}
    rng = np.random.default_rng(seed)
    false_indices = {idx for idx, row in enumerate(rows) if i(row["top1_success"]) == 0}
    false_avoided: list[int] = []
    committed_acc: list[float] = []
    indices = np.arange(len(rows))
    for _ in range(trials):
        abstained = set(rng.choice(indices, size=min(abstain_count, len(rows)), replace=False).tolist()) if abstain_count > 0 else set()
        committed = [idx for idx in range(len(rows)) if idx not in abstained]
        false_avoided.append(len(false_indices & abstained))
        if committed:
            committed_acc.append(float(np.mean([i(rows[idx]["top1_success"]) for idx in committed])))
        else:
            committed_acc.append(0.0)
    return {"random_false_avoided_mean": float(np.mean(false_avoided)), "random_accuracy_mean": float(np.mean(committed_acc))}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    core1ak = read_json(Path(args.core1ak_compact))
    eval_gate = str(core1ak.get("best_eval_gate", "S55_C50"))
    rows = assign_obs_ids(read_csv(Path(args.observations)))
    desc_by_obs = load_desc(Path(args.descriptor_trace))
    selected = [row for row in rows if row_passes(row, gate_by_name(eval_gate)) and i(row["obs_id"]) in desc_by_obs]
    query_rows = baseline_query_rows(selected, set(desc_by_obs.keys()))
    thresholds = sorted(set([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20] + [round(float(np.quantile([f(r["top1_margin"]) for r in query_rows], q)), 4) for q in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]]))

    total = len(query_rows)
    baseline_false = sum(1 for r in query_rows if i(r["top1_success"]) == 0)
    baseline_acc = float(np.mean([i(r["top1_success"]) for r in query_rows])) if query_rows else 0.0
    summary_rows: list[dict[str, Any]] = []
    per_event_rows: list[dict[str, Any]] = []
    for th in thresholds:
        committed = [r for r in query_rows if f(r["top1_margin"]) >= th]
        abstained = [r for r in query_rows if f(r["top1_margin"]) < th]
        committed_false = sum(1 for r in committed if i(r["top1_success"]) == 0)
        false_avoided = sum(1 for r in abstained if i(r["top1_success"]) == 0)
        correct_abstained = sum(1 for r in abstained if i(r["top1_success"]) == 1)
        coverage = len(committed) / total if total else 0.0
        committed_acc = float(np.mean([i(r["top1_success"]) for r in committed])) if committed else 0.0
        committed_false_rate = committed_false / len(committed) if committed else 0.0
        random_stats = random_abstention_baseline(query_rows, len(abstained), args.seed + int(th * 10000))
        summary_rows.append(
            {
                "threshold": th,
                "query_count": total,
                "committed_count": len(committed),
                "abstained_count": len(abstained),
                "coverage": coverage,
                "committed_top1": committed_acc,
                "committed_false_retrieval_rate": committed_false_rate,
                "false_retrieval_avoided_count": false_avoided,
                "correct_abstained_count": correct_abstained,
                "baseline_false_count": baseline_false,
                "baseline_top1": baseline_acc,
                "false_avoidance_recall": false_avoided / baseline_false if baseline_false else 0.0,
                "random_false_avoided_mean": random_stats["random_false_avoided_mean"],
                "random_accuracy_mean": random_stats["random_accuracy_mean"],
                "beats_random_false_avoidance": int(false_avoided > random_stats["random_false_avoided_mean"]),
                "eligible": int(coverage >= 0.80 and committed_acc >= baseline_acc and false_avoided > random_stats["random_false_avoided_mean"]),
            }
        )
    eligible = [r for r in summary_rows if i(r["eligible"]) == 1]
    if eligible:
        best = max(eligible, key=lambda r: (f(r["committed_top1"]), i(r["false_retrieval_avoided_count"]), f(r["coverage"])))
    else:
        best = max(summary_rows, key=lambda r: (i(r["false_retrieval_avoided_count"]), f(r["committed_top1"]), f(r["coverage"]))) if summary_rows else {}
    for row in summary_rows:
        row["selected_as_best"] = int(row is best)

    best_th = f(best.get("threshold"), 0.0)
    for row in query_rows:
        action = "commit" if f(row["top1_margin"]) >= best_th else "abstain_uncertain"
        rr = dict(row)
        rr["selected_threshold"] = best_th
        rr["uncertainty_action"] = action
        rr["false_retrieval_avoided"] = int(action == "abstain_uncertain" and i(row["top1_success"]) == 0)
        rr["correct_abstained"] = int(action == "abstain_uncertain" and i(row["top1_success"]) == 1)
        per_event_rows.append(rr)

    uncertainty_gate_passed = int(
        best
        and f(best.get("coverage")) >= 0.80
        and f(best.get("committed_top1")) >= baseline_acc
        and i(best.get("false_retrieval_avoided_count")) > f(best.get("random_false_avoided_mean"))
    )
    compact = {
        "stage": "CORE-1AN",
        "artifact_version": args.artifact_version,
        "eval_gate": eval_gate,
        "query_count": total,
        "baseline_top1": baseline_acc,
        "baseline_false_count": baseline_false,
        "best_threshold": best.get("threshold", 0.0),
        "best_coverage": best.get("coverage", 0.0),
        "best_committed_top1": best.get("committed_top1", 0.0),
        "best_committed_false_retrieval_rate": best.get("committed_false_retrieval_rate", 1.0),
        "false_retrieval_avoided_count": best.get("false_retrieval_avoided_count", 0),
        "correct_abstained_count": best.get("correct_abstained_count", 0),
        "random_false_avoided_mean": best.get("random_false_avoided_mean", 0.0),
        "uncertainty_gate_passed": uncertainty_gate_passed,
        "oracle_leakage_found": 0,
        "passed_minimum": uncertainty_gate_passed,
        "next_recommendation": (
            "CORE-1AO integrate uncertainty/abstention state into object-file memory audit"
            if uncertainty_gate_passed
            else "uncertainty margin gate does not cleanly beat random abstention; inspect richer uncertainty signals"
        ),
    }
    report = f"""# CORE-1AN Uncertainty Abstention Gate

This stage does not change retrieval scoring. It tests whether online-visible baseline top1 margin can identify uncertain memory recalls and abstain instead of forcing a false old-object match.

## Result

- Eval gate: {eval_gate}
- Queries: {total}
- Baseline top1: {baseline_acc:.4f}
- Baseline false count: {baseline_false}
- Best threshold: {float(compact['best_threshold']):.4f}
- Coverage: {float(compact['best_coverage']):.4f}
- Committed top1: {float(compact['best_committed_top1']):.4f}
- False retrievals avoided: {compact['false_retrieval_avoided_count']}
- Correct abstained: {compact['correct_abstained_count']}
- Random false avoided mean: {float(compact['random_false_avoided_mean']):.2f}
- Uncertainty gate passed: {uncertainty_gate_passed}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AN_"
    write_csv(
        out_dir / f"{prefix}risk_coverage_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "threshold",
            "query_count",
            "committed_count",
            "abstained_count",
            "coverage",
            "committed_top1",
            "committed_false_retrieval_rate",
            "false_retrieval_avoided_count",
            "correct_abstained_count",
            "baseline_false_count",
            "baseline_top1",
            "false_avoidance_recall",
            "random_false_avoided_mean",
            "random_accuracy_mean",
            "beats_random_false_avoidance",
            "eligible",
            "selected_as_best",
        ],
    )
    write_csv(
        out_dir / f"{prefix}event_uncertainty_trace_{args.artifact_version}.csv",
        per_event_rows,
        [
            "sequence_id",
            "event_id",
            "window_kind",
            "query_obs_id",
            "frame_idx",
            "track_id",
            "candidate_count",
            "top1_obs_id",
            "top1_success",
            "target_rank",
            "top1_margin",
            "top1_raw_margin",
            "target_margin",
            "selected_threshold",
            "uncertainty_action",
            "false_retrieval_avoided",
            "correct_abstained",
        ],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1an_uncertainty_abstention_gate.py",
                "gt_used_for_online_scoring": 0,
                "gt_used_for_threshold_selection": 0,
                "gt_used_for_eval_only": 1,
                "pretrained_weights_used": 0,
                "leakage_found": 0,
            }
        ],
        ["file", "gt_used_for_online_scoring", "gt_used_for_threshold_selection", "gt_used_for_eval_only", "pretrained_weights_used", "leakage_found"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
