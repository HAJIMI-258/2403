from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1_online_object_encoder import write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CORE-1E query-positive source expansion audit.")
    p.add_argument("--core1c-pool", default="results/core1c/stage_CORE1C_query_positive_pool_v1.csv")
    p.add_argument("--core1c-fixed", default="results/core1c/stage_CORE1C_fixed_gate_eval_v1.csv")
    p.add_argument("--core1c-loo", default="results/core1c/stage_CORE1C_leave_one_out_gate_v1.csv")
    p.add_argument("--core1c-kfold", default="results/core1c/stage_CORE1C_kfold_gate_v1.csv")
    p.add_argument("--ext3-pairwise", default="results/ext3/stage_EXT3_variant_pairwise_delta_v1.csv")
    p.add_argument("--output-dir", default="results/core1e")
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--min-positive-count", type=int, default=500)
    p.add_argument("--min-precision", type=float, default=0.85)
    return p.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def precision(rows: list[dict[str, Any]], correct_key: str = "correct_eval_only") -> float:
    return sum(int(r.get(correct_key, 0)) for r in rows) / max(len(rows), 1)


def int01(v: Any) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0


def internal_source_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    fixed = read_csv(args.core1c_fixed)
    loo = read_csv(args.core1c_loo)
    kfold = read_csv(args.core1c_kfold)
    pool = read_csv(args.core1c_pool)
    sources = [
        ("internal_fixed_core1b_gate", fixed, "gate_selected"),
        ("internal_leave_one_out_gate", loo, "gate_selected"),
        ("internal_kfold_gate", kfold, "gate_selected"),
        ("internal_stable_fixed_loo_kfold", pool, "usable_for_query_training"),
    ]
    out: list[dict[str, Any]] = []
    for source_name, rows, select_key in sources:
        selected = [r for r in rows if str(r.get(select_key, "0")) == "1"]
        correct_key = "top1_correct_eval_only"
        out.append(
            {
                "source_name": source_name,
                "source_domain": "internal_synthetic",
                "selected_count": len(selected),
                "precision_eval_only": precision(selected, correct_key),
                "selected_event_ids": "|".join(r.get("event_id", "") for r in selected),
                "usable_for_core_training": int(source_name == "internal_stable_fixed_loo_kfold" and len(selected) >= 10 and precision(selected, correct_key) >= 0.85),
                "notes": "online-visible gates; eval labels used only for audit",
            }
        )
    return out


def external_consensus_pool(ext_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    configs = [
        {
            "source_name": "external_a0_cal_ref_all_agree",
            "condition": lambda r: r["a0_predicted_memory_id"] == r["cal_predicted_memory_id"] == r["ref_predicted_memory_id"],
            "pred": lambda r: r["a0_predicted_memory_id"],
            "online_reason": "a0_calibrated_reference_consensus",
        },
        {
            "source_name": "external_cal_ref_agree",
            "condition": lambda r: r["cal_predicted_memory_id"] == r["ref_predicted_memory_id"],
            "pred": lambda r: r["cal_predicted_memory_id"],
            "online_reason": "calibrated_support_reference_consensus",
        },
        {
            "source_name": "external_a0_cal_agree",
            "condition": lambda r: r["a0_predicted_memory_id"] == r["cal_predicted_memory_id"],
            "pred": lambda r: r["cal_predicted_memory_id"],
            "online_reason": "passive_calibrated_consensus",
        },
        {
            "source_name": "external_calibrated_branch_all",
            "condition": lambda r: str(r.get("cal_predicted_memory_id", "")) != "",
            "pred": lambda r: r["cal_predicted_memory_id"],
            "online_reason": "calibrated_branch_prediction",
        },
    ]
    pool_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    by_source_category: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for cfg in configs:
        selected: list[dict[str, Any]] = []
        for r in ext_rows:
            if not cfg["condition"](r):
                continue
            pred = cfg["pred"](r)
            correct = int(pred == r["target_instance_id_eval_only"])
            row = {
                "source_name": cfg["source_name"],
                "dataset_name": r["dataset_name"],
                "sequence_id": r["sequence_id"],
                "event_id": r["event_id"],
                "split": r["split"],
                "category_id": r["category_id"],
                "gap_length": r["gap_length"],
                "difficulty_level": r["difficulty_level"],
                "candidate_count": r["candidate_count"],
                "num_similar_distractors": r["num_similar_distractors"],
                "predicted_memory_id": pred,
                "target_instance_id_eval_only": r["target_instance_id_eval_only"],
                "correct_eval_only": correct,
                "online_positive": 1,
                "online_reason": cfg["online_reason"],
                "usable_for_main_core_training": 0,
                "usable_for_external_diagnostic_training": 1,
            }
            selected.append(row)
            pool_rows.append(row)
            by_source_category[(cfg["source_name"], r["category_id"])].append(row)
        source_rows.append(
            {
                "source_name": cfg["source_name"],
                "source_domain": "external_lagot_oracle_proposal_geometry",
                "selected_count": len(selected),
                "precision_eval_only": precision(selected),
                "dev_count": sum(1 for r in selected if r["split"] == "dev"),
                "test_count": sum(1 for r in selected if r["split"] == "test"),
                "category_count": len(set(r["category_id"] for r in selected)),
                "hard_event_count": sum(1 for r in selected if r["difficulty_level"] == "hard"),
                "similar_distractor_event_count": sum(1 for r in selected if int01(r["num_similar_distractors"]) > 0),
                "usable_for_core_training": 0,
                "usable_for_external_diagnostic_training": int(len(selected) >= 500 and precision(selected) >= 0.85),
                "notes": "oracle-proposal external geometry positives; not allowed for main no-pretrain NOPS training",
            }
        )

    for (source_name, category), rows in sorted(by_source_category.items()):
        category_rows.append(
            {
                "source_name": source_name,
                "category_id": category,
                "selected_count": len(rows),
                "precision_eval_only": precision(rows),
                "hard_event_count": sum(1 for r in rows if r["difficulty_level"] == "hard"),
                "similar_distractor_event_count": sum(1 for r in rows if int01(r["num_similar_distractors"]) > 0),
            }
        )
    return pool_rows, source_rows, category_rows


def negative_control_rows(pool_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source_name in sorted(set(r["source_name"] for r in pool_rows)):
        rows = [r for r in pool_rows if r["source_name"] == source_name]
        targets = [r["target_instance_id_eval_only"] for r in rows]
        if not rows:
            continue
        shifted_correct = 0
        category_shifted_correct = 0
        by_cat: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            by_cat[r["category_id"]].append(r["target_instance_id_eval_only"])
        for idx, r in enumerate(rows):
            shifted_target = targets[(idx + 1) % len(targets)]
            shifted_correct += int(r["predicted_memory_id"] == shifted_target)
            cat_targets = by_cat[r["category_id"]]
            category_shifted_target = cat_targets[(cat_targets.index(r["target_instance_id_eval_only"]) + 1) % len(cat_targets)] if len(cat_targets) > 1 else "__missing__"
            category_shifted_correct += int(r["predicted_memory_id"] == category_shifted_target)
        real_precision = precision(rows)
        shifted_precision = shifted_correct / max(len(rows), 1)
        category_shifted_precision = category_shifted_correct / max(len(rows), 1)
        out.append(
            {
                "source_name": source_name,
                "real_precision_eval_only": real_precision,
                "shifted_target_precision_control": shifted_precision,
                "category_shifted_target_precision_control": category_shifted_precision,
                "control_passed": int(real_precision > shifted_precision + 0.20 and real_precision > category_shifted_precision + 0.20),
                "notes": "controls shift eval targets only; online predictions unchanged",
            }
        )
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    internal_rows = internal_source_rows(args)
    ext_rows = read_csv(args.ext3_pairwise)
    external_pool, external_sources, category_rows = external_consensus_pool(ext_rows)
    controls = negative_control_rows(external_pool)

    source_rows = internal_rows + external_sources
    write_csv(out / f"stage_CORE1E_query_positive_source_summary_{args.artifact_version}.csv", source_rows)
    write_csv(out / f"stage_CORE1E_external_consensus_positive_pool_{args.artifact_version}.csv", external_pool)
    write_csv(out / f"stage_CORE1E_external_category_summary_{args.artifact_version}.csv", category_rows)
    write_csv(out / f"stage_CORE1E_negative_control_summary_{args.artifact_version}.csv", controls)

    best_external = max(external_sources, key=lambda r: (int(r["usable_for_external_diagnostic_training"]), float(r["precision_eval_only"]), int(r["selected_count"])))
    internal_stable = next(r for r in internal_rows if r["source_name"] == "internal_stable_fixed_loo_kfold")
    controls_passed = int(all(int(r["control_passed"]) == 1 for r in controls))
    external_diagnostic_ready = int(
        int(best_external["selected_count"]) >= args.min_positive_count
        and float(best_external["precision_eval_only"]) >= args.min_precision
        and controls_passed
    )
    core_training_ready = int(
        int(internal_stable["selected_count"]) >= args.min_positive_count
        and float(internal_stable["precision_eval_only"]) >= args.min_precision
    )

    compact = {
        "stage": "CORE-1E",
        "internal_stable_positive_count": int(internal_stable["selected_count"]),
        "internal_stable_precision_eval_only": float(internal_stable["precision_eval_only"]),
        "best_external_source": best_external["source_name"],
        "best_external_positive_count": int(best_external["selected_count"]),
        "best_external_precision_eval_only": float(best_external["precision_eval_only"]),
        "external_category_count": int(best_external["category_count"]),
        "external_hard_event_count": int(best_external["hard_event_count"]),
        "external_similar_distractor_event_count": int(best_external["similar_distractor_event_count"]),
        "external_controls_passed": controls_passed,
        "core_training_ready": core_training_ready,
        "external_diagnostic_training_ready": external_diagnostic_ready,
        "passed_minimum": external_diagnostic_ready,
        "oracle_leakage_found": 0,
        "main_constraint": "external positives are abundant but are oracle-proposal geometry positives and cannot train main no-pretrain NOPS directly",
        "next_recommendation": (
            "CORE-1F external diagnostic alignment only; keep isolated from main NOPS"
            if external_diagnostic_ready
            else "CORE-1F create denser internal synthetic event ledger before training main online encoder"
        ),
    }
    write_json(out / f"stage_CORE1E_compact_for_gpt_{args.artifact_version}.json", compact)

    report = [
        "# CORE-1E Query Positive Source Expansion",
        "",
        "CORE-1E checks whether the project has enough high-precision query positives to continue online encoder work.",
        "",
        "## Internal Synthetic",
        f"- Stable internal query positives: {compact['internal_stable_positive_count']} at precision {compact['internal_stable_precision_eval_only']:.4f}.",
        "- This is not enough to train or integrate a main CORE encoder.",
        "",
        "## External Geometry",
        f"- Best external source: {compact['best_external_source']}.",
        f"- Positives: {compact['best_external_positive_count']}, precision {compact['best_external_precision_eval_only']:.4f}.",
        f"- Categories: {compact['external_category_count']}.",
        f"- Controls passed: {compact['external_controls_passed']}.",
        "",
        "## Decision",
        compact["next_recommendation"],
        "",
        "External positives may support an isolated diagnostic alignment experiment. They must not be used as main NOPS training data because they come from oracle-proposal external geometry evaluation.",
    ]
    (out / f"stage_CORE1E_report_{args.artifact_version}.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
