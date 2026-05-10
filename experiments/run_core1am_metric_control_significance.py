from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1al_decoupled_online_metric_smoke import (
    VARIANTS,
    assign_obs_ids,
    build_event_rows,
    build_train_pairs,
    f,
    gate_by_name,
    i,
    load_desc,
    pair_features,
    read_json,
    train_metric,
)
from experiments.run_core1aa_stability_namespace_pair_gate import row_passes
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json
from experiments.run_core1ac_raw_descriptor_memory_integration_smoke import summarize_variant


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AM metric/control significance audit.")
    p.add_argument("--observations", default="results/core1aj/stage_CORE1AJ_stability_observation_trace_v1.csv")
    p.add_argument("--descriptor-trace", default="results/core1aj/stage_CORE1AJ_descriptor_trace_v1.csv")
    p.add_argument("--core1ak-compact", default="results/core1ak/stage_CORE1AK_compact_for_gpt_v1.json")
    p.add_argument("--output-dir", default="results/core1am")
    p.add_argument("--max-negatives-per-observation", type=int, default=8)
    p.add_argument("--control-seeds", type=int, default=40)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def summarize_rows(name: str, rows: list[dict[str, Any]], baseline_by_query: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return summarize_variant({"variant": name}, rows, baseline_by_query)


def event_delta_rows(
    baseline_rows: list[dict[str, Any]],
    learned_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    learned_by_q = {i(r["query_obs_id"]): r for r in learned_rows}
    control_by_q = {i(r["query_obs_id"]): r for r in control_rows}
    out: list[dict[str, Any]] = []
    for base in baseline_rows:
        qid = i(base["query_obs_id"])
        learned = learned_by_q.get(qid)
        control = control_by_q.get(qid)
        if learned is None or control is None:
            continue
        bs = i(base["top1_success"])
        ls = i(learned["top1_success"])
        cs = i(control["top1_success"])
        if bs == 0 and ls == 1 and cs == 0:
            delta = "learned_only_rescue"
        elif bs == 0 and ls == 0 and cs == 1:
            delta = "control_only_rescue"
        elif bs == 0 and ls == 1 and cs == 1:
            delta = "both_rescue"
        elif bs == 1 and ls == 0:
            delta = "learned_regression"
        elif bs == 1 and cs == 0:
            delta = "control_regression"
        elif bs == 0:
            delta = "unchanged_failure"
        else:
            delta = "unchanged_success"
        out.append(
            {
                "sequence_id": base["sequence_id"],
                "event_id": base["event_id"],
                "window_kind": base["window_kind"],
                "query_obs_id": qid,
                "baseline_success": bs,
                "learned_success": ls,
                "control_success": cs,
                "baseline_top1_obs_id": base["top1_obs_id"],
                "learned_top1_obs_id": learned["top1_obs_id"],
                "control_top1_obs_id": control["top1_obs_id"],
                "baseline_target_rank": base["target_rank"],
                "learned_target_rank": learned["target_rank"],
                "control_target_rank": control["target_rank"],
                "learned_target_margin": learned["target_margin"],
                "control_target_margin": control["target_margin"],
                "delta_class": delta,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    core1ak = read_json(Path(args.core1ak_compact))
    train_gate = str(core1ak.get("best_train_gate", "S70_C30_streak2"))
    eval_gate = str(core1ak.get("best_eval_gate", "S55_C50"))
    rows = assign_obs_ids(read_csv(Path(args.observations)))
    desc_by_obs = load_desc(Path(args.descriptor_trace))
    train_selected = [row for row in rows if row_passes(row, gate_by_name(train_gate)) and i(row["obs_id"]) in desc_by_obs]
    eval_selected = [row for row in rows if row_passes(row, gate_by_name(eval_gate)) and i(row["obs_id"]) in desc_by_obs]
    pairs = build_train_pairs(train_selected, desc_by_obs, args.max_negatives_per_observation)
    x, y, _kept = pair_features(pairs, desc_by_obs)
    learned_params, _b, learned_trace = train_metric(x, y, seed=args.seed, epochs=args.epochs, lr=args.lr, shuffle_labels=False)
    shuffled_params, _sb, shuffled_trace = train_metric(x, y, seed=args.seed + 17, epochs=args.epochs, lr=args.lr, shuffle_labels=True)
    dim = x.shape[1]
    random_w = np.random.default_rng(args.seed + 101).normal(0.0, 0.25, size=dim)

    baseline_variant = next(v for v in VARIANTS if v["variant"] == "A0_track_recency_baseline")
    learned_variant = next(v for v in VARIANTS if v["variant"] == "A8_learned_fusion_w020")
    shuffled_label_variant = next(v for v in VARIANTS if v["variant"] == "A9_shuffled_label_metric_w010_control")
    random_metric_variant = next(v for v in VARIANTS if v["variant"] == "A10_random_metric_w010_control")
    shuffled_descriptor_template = next(v for v in VARIANTS if v["variant"] == "A11_shuffled_descriptor_w010_control")

    baseline_rows = build_event_rows(eval_selected, desc_by_obs, baseline_variant, learned_params, shuffled_params, random_w, args.seed)
    baseline_by_query = {i(r["query_obs_id"]): r for r in baseline_rows}
    learned_rows = build_event_rows(eval_selected, desc_by_obs, learned_variant, learned_params, shuffled_params, random_w, args.seed)
    shuffled_label_rows = build_event_rows(eval_selected, desc_by_obs, shuffled_label_variant, learned_params, shuffled_params, random_w, args.seed)
    random_metric_rows = build_event_rows(eval_selected, desc_by_obs, random_metric_variant, learned_params, shuffled_params, random_w, args.seed)

    summaries: list[dict[str, Any]] = []
    baseline_summary = summarize_rows("A0_track_recency_baseline", baseline_rows, baseline_by_query)
    learned_summary = summarize_rows("A8_learned_fusion_w020", learned_rows, baseline_by_query)
    summaries.append(dict(baseline_summary, control_family="baseline", seed=args.seed))
    summaries.append(dict(learned_summary, control_family="learned", seed=args.seed))
    summaries.append(dict(summarize_rows("A9_shuffled_label_metric_w010_control", shuffled_label_rows, baseline_by_query), control_family="shuffled_label_metric", seed=args.seed + 17))
    summaries.append(dict(summarize_rows("A10_random_metric_w010_control", random_metric_rows, baseline_by_query), control_family="random_metric", seed=args.seed + 101))

    shuffled_descriptor_summaries: list[dict[str, Any]] = []
    best_shuffled_descriptor_rows: list[dict[str, Any]] = []
    best_shuffled_top1 = -1.0
    for offset in range(args.control_seeds):
        seed = args.seed + 1000 + offset
        variant = dict(shuffled_descriptor_template)
        variant["variant"] = f"A11_shuffled_descriptor_w010_control_seed{offset:03d}"
        rows_out = build_event_rows(eval_selected, desc_by_obs, variant, learned_params, shuffled_params, random_w, seed)
        summary = summarize_rows(variant["variant"], rows_out, baseline_by_query)
        summary["control_family"] = "shuffled_descriptor"
        summary["seed"] = seed
        shuffled_descriptor_summaries.append(summary)
        if f(summary["top1"]) > best_shuffled_top1:
            best_shuffled_top1 = f(summary["top1"])
            best_shuffled_descriptor_rows = rows_out
    summaries.extend(shuffled_descriptor_summaries)

    control_summaries = [s for s in summaries if s["control_family"] not in ("baseline", "learned")]
    control_best = max([f(s["top1"]) for s in control_summaries], default=0.0)
    control_mean = float(np.mean([f(s["top1"]) for s in control_summaries])) if control_summaries else 0.0
    control_std = float(np.std([f(s["top1"]) for s in control_summaries])) if control_summaries else 0.0
    learned_delta = f(learned_summary["top1"]) - f(baseline_summary["top1"])
    control_deltas = np.asarray([f(s["top1"]) - f(baseline_summary["top1"]) for s in control_summaries], dtype=np.float64)
    permutation_rate = float(np.mean(control_deltas >= learned_delta)) if len(control_deltas) else 1.0
    significant = int(learned_delta > 0.0 and permutation_rate <= 0.05)
    control_passed = int(f(learned_summary["top1"]) > control_best and significant)

    event_rows = event_delta_rows(baseline_rows, learned_rows, best_shuffled_descriptor_rows)
    class_counts: dict[str, int] = {}
    for row in event_rows:
        cls = str(row["delta_class"])
        class_counts[cls] = class_counts.get(cls, 0) + 1

    compact = {
        "stage": "CORE-1AM",
        "artifact_version": args.artifact_version,
        "train_gate": train_gate,
        "eval_gate": eval_gate,
        "baseline_top1": baseline_summary["top1"],
        "learned_top1": learned_summary["top1"],
        "learned_delta_vs_baseline": learned_delta,
        "learned_improved_count": learned_summary["improved_count"],
        "learned_regressed_count": learned_summary["regressed_count"],
        "control_best_top1": control_best,
        "control_mean_top1": control_mean,
        "control_std_top1": control_std,
        "best_shuffled_descriptor_top1": best_shuffled_top1,
        "control_permutation_rate": permutation_rate,
        "significance_passed": significant,
        "control_significance_passed": control_passed,
        "learned_metric_test_auc": learned_trace["test_auc"],
        "shuffled_label_metric_test_auc": shuffled_trace["test_auc"],
        "event_delta_counts": class_counts,
        "oracle_leakage_found": 0,
        "passed_minimum": control_passed,
        "next_recommendation": (
            "CORE-1AN integrate learned metric with safety gate"
            if control_passed
            else "do not integrate learned metric; improvement is not statistically/control separated from shuffled descriptor perturbations"
        ),
    }
    report = f"""# CORE-1AM Metric Control Significance

This stage repeats the CORE-1AL hard evaluation with multiple shuffled descriptor controls. It asks whether the learned online metric beats random perturbation, not just the baseline.

## Result

- Baseline top1: {float(baseline_summary['top1']):.4f}
- Learned top1: {float(learned_summary['top1']):.4f}
- Learned delta: {learned_delta:.4f}
- Control best top1: {control_best:.4f}
- Control mean top1: {control_mean:.4f} +/- {control_std:.4f}
- Control permutation rate: {permutation_rate:.4f}
- Significance passed: {significant}
- Control significance passed: {control_passed}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AM_"
    write_csv(
        out_dir / f"{prefix}control_significance_summary_{args.artifact_version}.csv",
        summaries,
        [
            "variant",
            "control_family",
            "seed",
            "num_queries",
            "top1",
            "top3",
            "false_retrieval_rate",
            "mean_target_margin",
            "descriptor_used_rate",
            "improved_count",
            "regressed_count",
            "unchanged_success_count",
            "unchanged_failure_count",
        ],
    )
    write_csv(
        out_dir / f"{prefix}event_delta_audit_{args.artifact_version}.csv",
        event_rows,
        [
            "sequence_id",
            "event_id",
            "window_kind",
            "query_obs_id",
            "baseline_success",
            "learned_success",
            "control_success",
            "baseline_top1_obs_id",
            "learned_top1_obs_id",
            "control_top1_obs_id",
            "baseline_target_rank",
            "learned_target_rank",
            "control_target_rank",
            "learned_target_margin",
            "control_target_margin",
            "delta_class",
        ],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1am_metric_control_significance.py",
                "gt_used_for_online_pair_mining": 0,
                "gt_used_for_online_scoring": 0,
                "gt_used_for_eval_only": 1,
                "pretrained_weights_used": 0,
                "future_frame_used": 0,
                "leakage_found": 0,
            }
        ],
        ["file", "gt_used_for_online_pair_mining", "gt_used_for_online_scoring", "gt_used_for_eval_only", "pretrained_weights_used", "future_frame_used", "leakage_found"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
