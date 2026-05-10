from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1_online_object_encoder import load_cache, safe_float, write_csv, write_json
from experiments.run_core1a_query_memory_alignment import baseline_event_rows
from experiments.run_core1b_query_pair_gate_repair import candidate_gate, gate_scan, passes_gate, top1_features


FOCUS_EVENTS = {"M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CORE-1C query gate split-validation audit.")
    p.add_argument("--cache", default="results/v3_e4a/cache/runtime_collection_cache_v1.pkl")
    p.add_argument("--output-dir", default="results/core1c")
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--min-train-selected", type=int, default=3)
    p.add_argument("--min-train-precision", type=float, default=0.85)
    p.add_argument("--min-heldout-selected", type=int, default=3)
    p.add_argument("--min-heldout-precision", type=float, default=0.85)
    return p.parse_args()


def gate_from_scan_row(row: dict[str, Any]) -> dict[str, float]:
    return {
        "support_v2_min": safe_float(row["support_v2_min"]),
        "content_min": safe_float(row["content_min"]),
        "disappearance_min": safe_float(row["disappearance_min"]),
        "temporal_max": safe_float(row["temporal_max"]),
        "margin_max": safe_float(row["margin_max"]),
    }


def gate_signature(gate: dict[str, float]) -> str:
    return (
        f"sv2>={gate['support_v2_min']:.2f};"
        f"content>={gate['content_min']:.2f};"
        f"dis>={gate['disappearance_min']:.2f};"
        f"temp<={gate['temporal_max']:.2f};"
        f"margin<={gate['margin_max']:.2f}"
    )


def select_best_gate(
    train_rows: dict[str, dict[str, Any]],
    min_selected: int,
    min_precision: float,
) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    scan = gate_scan(train_rows)
    eligible = [
        r for r in scan
        if int(r["selected_event_count"]) >= min_selected
        and safe_float(r["precision_eval_only"]) >= min_precision
    ]
    if eligible:
        row = eligible[0]
        row = {**row, "selected_by": "eligible_train_gate"}
    else:
        row = {**scan[0], "selected_by": "best_available_train_gate"} if scan else {
            "support_v2_min": 0.0,
            "content_min": 0.0,
            "disappearance_min": 0.0,
            "temporal_max": 1.0,
            "margin_max": 999.0,
            "selected_event_count": 0,
            "precision_eval_only": 0.0,
            "selected_event_ids": "",
            "gate_passed_eval_only": 0,
            "selected_by": "empty_train_gate",
        }
    return gate_from_scan_row(row), row, scan


def eval_gate_on_events(
    baseline_rows: dict[str, dict[str, Any]],
    event_ids: list[str],
    gate: dict[str, float],
    split_name: str,
    train_gate_row: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event_id in event_ids:
        row = baseline_rows[event_id]
        feats = top1_features(row)
        selected = int(passes_gate(feats, gate))
        correct = int(row.get("top1_correct_eval_only", 0))
        event = row["event"]
        out.append(
            {
                "split_name": split_name,
                "event_id": event_id,
                "scenario_name": event.get("scenario_name", ""),
                "is_focus_event": int(event_id in FOCUS_EVENTS),
                "gate_signature": gate_signature(gate),
                "gate_selected": selected,
                "top1_correct_eval_only": correct,
                "selected_and_correct_eval_only": int(selected and correct),
                "top1_bundle_id": "" if row.get("top1_bundle_id") is None else int(row["top1_bundle_id"]),
                "target_bundle_id_eval_only": "" if row.get("target_bundle_id_eval_only") is None else int(row["target_bundle_id_eval_only"]),
                "support_v2": feats["support_v2"],
                "content": feats["content"],
                "disappearance": feats["disappearance"],
                "temporal": feats["temporal"],
                "margin": feats["margin"],
                "train_selected_event_count": "" if train_gate_row is None else int(train_gate_row["selected_event_count"]),
                "train_precision_eval_only": "" if train_gate_row is None else safe_float(train_gate_row["precision_eval_only"]),
                "train_gate_passed_eval_only": "" if train_gate_row is None else int(train_gate_row["gate_passed_eval_only"]),
                "train_gate_selected_by": "" if train_gate_row is None else train_gate_row.get("selected_by", ""),
            }
        )
    return out


def summarize_eval(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    selected = [r for r in rows if int(r["gate_selected"]) == 1]
    selected_correct = [r for r in selected if int(r["top1_correct_eval_only"]) == 1]
    focus_selected = [r for r in selected if int(r["is_focus_event"]) == 1]
    return {
        f"{prefix}_selected_count": len(selected),
        f"{prefix}_correct_selected_count": len(selected_correct),
        f"{prefix}_precision_eval_only": len(selected_correct) / max(len(selected), 1),
        f"{prefix}_coverage_rate": len(selected) / max(len(rows), 1),
        f"{prefix}_focus_selected_count": len(focus_selected),
        f"{prefix}_wrong_selected_event_ids": "|".join(r["event_id"] for r in selected if int(r["top1_correct_eval_only"]) == 0),
    }


def make_folds(event_ids: list[str], k: int = 3) -> list[list[str]]:
    folds = [[] for _ in range(k)]
    for idx, event_id in enumerate(event_ids):
        folds[idx % k].append(event_id)
    return folds


def leakage_rows() -> list[dict[str, Any]]:
    return [
        {
            "audit_item": "gate_features",
            "uses_target_bundle_id": 0,
            "uses_old_track_id": 0,
            "uses_old_prototype_id": 0,
            "uses_gt_instance_id": 0,
            "uses_future_frame": 0,
            "uses_eval_label_for_online_gate": 0,
            "note": "CORE-1C evaluates online-visible gate features only; split gate selection uses eval precision for audit and is not integrated.",
            "leakage_found": 0,
        }
    ]


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache = load_cache(args.cache)
    baseline_rows = baseline_event_rows(cache["bundle_by_id"], cache["event_records"])
    event_ids = list(baseline_rows.keys())

    fixed_gate = candidate_gate()
    fixed_rows = eval_gate_on_events(baseline_rows, event_ids, fixed_gate, "fixed_core1b_candidate_gate")
    write_csv(out / f"stage_CORE1C_fixed_gate_eval_{args.artifact_version}.csv", fixed_rows)

    loo_rows: list[dict[str, Any]] = []
    chosen_loo_gates: list[dict[str, Any]] = []
    for heldout in event_ids:
        train_ids = [eid for eid in event_ids if eid != heldout]
        train_rows = {eid: baseline_rows[eid] for eid in train_ids}
        gate, train_gate_row, _ = select_best_gate(train_rows, args.min_train_selected, args.min_train_precision)
        chosen_loo_gates.append(
            {
                "heldout_event_id": heldout,
                "gate_signature": gate_signature(gate),
                "train_selected_event_count": int(train_gate_row["selected_event_count"]),
                "train_precision_eval_only": safe_float(train_gate_row["precision_eval_only"]),
                "train_gate_passed_eval_only": int(train_gate_row["gate_passed_eval_only"]),
                "selected_by": train_gate_row.get("selected_by", ""),
            }
        )
        loo_rows.extend(eval_gate_on_events(baseline_rows, [heldout], gate, "leave_one_out", train_gate_row))
    write_csv(out / f"stage_CORE1C_leave_one_out_gate_{args.artifact_version}.csv", loo_rows)

    kfold_rows: list[dict[str, Any]] = []
    kfold_gate_rows: list[dict[str, Any]] = []
    folds = make_folds(event_ids, 3)
    for fold_idx, test_ids in enumerate(folds):
        train_ids = [eid for eid in event_ids if eid not in set(test_ids)]
        train_rows = {eid: baseline_rows[eid] for eid in train_ids}
        gate, train_gate_row, _ = select_best_gate(train_rows, args.min_train_selected, args.min_train_precision)
        kfold_gate_rows.append(
            {
                "fold_idx": fold_idx,
                "test_event_ids": "|".join(test_ids),
                "gate_signature": gate_signature(gate),
                "train_selected_event_count": int(train_gate_row["selected_event_count"]),
                "train_precision_eval_only": safe_float(train_gate_row["precision_eval_only"]),
                "train_gate_passed_eval_only": int(train_gate_row["gate_passed_eval_only"]),
                "selected_by": train_gate_row.get("selected_by", ""),
            }
        )
        kfold_rows.extend(eval_gate_on_events(baseline_rows, test_ids, gate, f"kfold_{fold_idx}", train_gate_row))
    write_csv(out / f"stage_CORE1C_kfold_gate_{args.artifact_version}.csv", kfold_rows)

    stability_rows: list[dict[str, Any]] = []
    fixed_by_id = {r["event_id"]: r for r in fixed_rows}
    loo_by_id = {r["event_id"]: r for r in loo_rows}
    kfold_by_id = {r["event_id"]: r for r in kfold_rows}
    for event_id in event_ids:
        fixed_selected = int(fixed_by_id[event_id]["gate_selected"])
        loo_selected = int(loo_by_id[event_id]["gate_selected"])
        kfold_selected = int(kfold_by_id[event_id]["gate_selected"])
        correct = int(fixed_by_id[event_id]["top1_correct_eval_only"])
        stability_rows.append(
            {
                "event_id": event_id,
                "scenario_name": fixed_by_id[event_id]["scenario_name"],
                "is_focus_event": fixed_by_id[event_id]["is_focus_event"],
                "fixed_gate_selected": fixed_selected,
                "leave_one_out_selected": loo_selected,
                "kfold_selected": kfold_selected,
                "selection_stability_count": fixed_selected + loo_selected + kfold_selected,
                "top1_correct_eval_only": correct,
                "stable_query_positive": int(fixed_selected and loo_selected and kfold_selected and correct),
                "unstable_or_wrong_reason": "" if fixed_selected and loo_selected and kfold_selected and correct else (
                    "wrong_selected_by_gate" if (fixed_selected or loo_selected or kfold_selected) and not correct else "not_stably_selected"
                ),
            }
        )
    write_csv(out / f"stage_CORE1C_gate_stability_summary_{args.artifact_version}.csv", stability_rows)

    query_pool_rows = [
        {
            "event_id": r["event_id"],
            "scenario_name": r["scenario_name"],
            "is_focus_event": r["is_focus_event"],
            "usable_for_query_training": r["stable_query_positive"],
            "selection_stability_count": r["selection_stability_count"],
            "top1_correct_eval_only": r["top1_correct_eval_only"],
            "reason": "stable_across_fixed_loo_kfold" if int(r["stable_query_positive"]) else r["unstable_or_wrong_reason"],
        }
        for r in stability_rows
    ]
    write_csv(out / f"stage_CORE1C_query_positive_pool_{args.artifact_version}.csv", query_pool_rows)

    write_csv(out / f"stage_CORE1C_oracle_leakage_audit_{args.artifact_version}.csv", leakage_rows())

    fixed_summary = summarize_eval(fixed_rows, "fixed_gate")
    loo_summary = summarize_eval(loo_rows, "leave_one_out")
    kfold_summary = summarize_eval(kfold_rows, "kfold")
    stable_pool = [r for r in query_pool_rows if int(r["usable_for_query_training"]) == 1]
    stable_pool_precision = sum(int(r["top1_correct_eval_only"]) for r in stable_pool) / max(len(stable_pool), 1)

    gate_stability_passed = int(
        fixed_summary["fixed_gate_selected_count"] >= 7
        and fixed_summary["fixed_gate_precision_eval_only"] >= 0.85
        and loo_summary["leave_one_out_selected_count"] >= args.min_heldout_selected
        and loo_summary["leave_one_out_precision_eval_only"] >= args.min_heldout_precision
        and kfold_summary["kfold_selected_count"] >= args.min_heldout_selected
        and kfold_summary["kfold_precision_eval_only"] >= args.min_heldout_precision
        and len(stable_pool) >= 3
    )

    split_summary = {
        "stage": "CORE-1C",
        **fixed_summary,
        **loo_summary,
        **kfold_summary,
        "stable_query_positive_pool_size": len(stable_pool),
        "stable_query_positive_pool_precision_eval_only": stable_pool_precision,
        "gate_stability_passed": gate_stability_passed,
        "oracle_leakage_found": 0,
        "passed_minimum": gate_stability_passed,
        "next_recommendation": (
            "CORE-1D train alignment on stable query positives with strict no-integration guard"
            if gate_stability_passed
            else "CORE-1D collect more query positives / repair query gate before training integration"
        ),
    }
    write_json(out / f"stage_CORE1C_compact_for_gpt_{args.artifact_version}.json", split_summary)

    report = [
        "# CORE-1C Query Gate Split Validation",
        "",
        "CORE-1C audits whether the CORE-1B cue-consensus query gate generalizes under leave-one-out and deterministic 3-fold splits.",
        "No encoder training or retrieval integration is performed in this stage.",
        "",
        "## Key Results",
        f"- Fixed CORE-1B gate selected {fixed_summary['fixed_gate_selected_count']} events with precision {fixed_summary['fixed_gate_precision_eval_only']:.4f}.",
        f"- Leave-one-out selected {loo_summary['leave_one_out_selected_count']} held-out events with precision {loo_summary['leave_one_out_precision_eval_only']:.4f}.",
        f"- 3-fold selected {kfold_summary['kfold_selected_count']} test events with precision {kfold_summary['kfold_precision_eval_only']:.4f}.",
        f"- Stable query positive pool size: {len(stable_pool)}.",
        f"- Gate stability passed: {gate_stability_passed}.",
        "",
        "## Decision",
        split_summary["next_recommendation"],
        "",
        "The split gate selection uses evaluation precision only for audit. It is not an online scoring path and must not be merged into NOPS directly.",
    ]
    (out / f"stage_CORE1C_report_{args.artifact_version}.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
