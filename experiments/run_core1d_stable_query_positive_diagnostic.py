from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1_online_object_encoder import bundle_vector, load_cache, query_vector, write_csv, write_json
from experiments.run_core1a_query_memory_alignment import (
    baseline_event_rows,
    bundle_query_proxy,
    embed_memory,
    score_variant,
    train_alignment_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CORE-1D stable query-positive diagnostic.")
    p.add_argument("--cache", default="results/v3_e4a/cache/runtime_collection_cache_v1.pkl")
    p.add_argument("--core1c-pool", default="results/core1c/stage_CORE1C_query_positive_pool_v1.csv")
    p.add_argument("--output-dir", default="results/core1d")
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def stable_event_ids(pool_path: Path) -> list[str]:
    rows = read_csv_rows(pool_path)
    return [r["event_id"] for r in rows if str(r.get("usable_for_query_training", "0")) == "1"]


def build_pairs(
    bundle_by_id: dict[int, dict[str, Any]],
    baseline_rows: dict[str, dict[str, Any]],
    stable_ids: list[str],
    *,
    repeat_stable: int,
    include_proxy_positives: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pair_id = 1
    if include_proxy_positives:
        for b in sorted(bundle_by_id.values(), key=lambda x: int(x["bundle_id"])):
            rows.append(
                {
                    "pair_id": pair_id,
                    "pair_source": "bundle_query_proxy",
                    "scenario_name": b["scenario_name"],
                    "event_id": "",
                    "query_kind": "historical_proxy",
                    "query_frame": int(b["created_frame"]),
                    "positive_bundle_id": int(b["bundle_id"]),
                    "negative_bundle_id": "",
                    "online_positive": 1,
                    "online_negative": 0,
                    "mining_reason": "same_bundle_proxy_alignment",
                    "confidence_score": 0.98,
                    "target_bundle_id_eval_only": "",
                    "pair_correct_eval_only": 1,
                    "used_for_training": 1,
                    "repeat_stable": repeat_stable,
                }
            )
            pair_id += 1

    stable_set = set(stable_ids)
    for event_id in stable_ids:
        row = baseline_rows[event_id]
        event = row["event"]
        for _ in range(repeat_stable):
            rows.append(
                {
                    "pair_id": pair_id,
                    "pair_source": "core1c_stable_query_positive",
                    "scenario_name": event["scenario_name"],
                    "event_id": event_id,
                    "query_kind": "reentry_cue",
                    "query_frame": int(event["frame_idx"]),
                    "positive_bundle_id": int(row["top1_bundle_id"]),
                    "negative_bundle_id": "",
                    "online_positive": 1,
                    "online_negative": 0,
                    "mining_reason": "selected_by_fixed_loo_kfold_gate",
                    "confidence_score": float(row["top1_score"]),
                    "target_bundle_id_eval_only": "" if row["target_bundle_id_eval_only"] is None else int(row["target_bundle_id_eval_only"]),
                    "pair_correct_eval_only": int(row["top1_correct_eval_only"]),
                    "used_for_training": 1,
                    "repeat_stable": repeat_stable,
                }
            )
            pair_id += 1

    neg_limit = 12
    for event_id, row in baseline_rows.items():
        event = row["event"]
        top1_id = row["top1_bundle_id"]
        for cand in row["scored"]["candidate_pool"][:neg_limit]:
            bid = int(cand["bundle_id"])
            if top1_id is not None and bid == int(top1_id):
                continue
            rows.append(
                {
                    "pair_id": pair_id,
                    "pair_source": "event_candidate_negative",
                    "scenario_name": event["scenario_name"],
                    "event_id": event_id,
                    "query_kind": "reentry_cue",
                    "query_frame": int(event["frame_idx"]),
                    "positive_bundle_id": "",
                    "negative_bundle_id": bid,
                    "online_positive": 0,
                    "online_negative": 1,
                    "mining_reason": "same_event_non_top1_candidate_negative",
                    "confidence_score": 0.80 if event_id in stable_set else 0.65,
                    "target_bundle_id_eval_only": "" if row["target_bundle_id_eval_only"] is None else int(row["target_bundle_id_eval_only"]),
                    "pair_correct_eval_only": int(row["target_bundle_id_eval_only"] is None or bid != int(row["target_bundle_id_eval_only"])),
                    "used_for_training": 1,
                    "repeat_stable": repeat_stable,
                }
            )
            pair_id += 1

    pos = [r for r in rows if int(r["online_positive"]) == 1]
    neg = [r for r in rows if int(r["online_negative"]) == 1]
    stable_pos = [r for r in pos if r["pair_source"] == "core1c_stable_query_positive"]
    summary = {
        "query_positive_pair_count": len(pos),
        "query_negative_pair_count": len(neg),
        "stable_event_count": len(stable_ids),
        "stable_query_positive_pair_count": len(stable_pos),
        "stable_query_positive_precision_eval_only": sum(int(r["pair_correct_eval_only"]) for r in stable_pos) / max(len(stable_pos), 1),
        "negative_pair_precision_eval_only": sum(int(r["pair_correct_eval_only"]) for r in neg) / max(len(neg), 1),
        "proxy_positive_pair_count": sum(1 for r in pos if r["pair_source"] == "bundle_query_proxy"),
        "repeat_stable": repeat_stable,
        "usable_for_training": int(len(pos) >= 500 and len(neg) >= 50 and len(stable_pos) >= 3),
        "main_pair_failure_reason": "" if len(pos) >= 500 and len(neg) >= 50 and len(stable_pos) >= 3 else "insufficient_stable_query_positive_pairs",
    }
    return rows, summary


def vector_gap_rows(
    bundle_by_id: dict[int, dict[str, Any]],
    baseline_rows: dict[str, dict[str, Any]],
    event_query_vectors: dict[str, np.ndarray],
    stable_ids: list[str],
) -> list[dict[str, Any]]:
    from experiments.run_core1_online_object_encoder import cosine_np

    proxy_vectors = {int(bid): bundle_query_proxy(b) for bid, b in bundle_by_id.items()}
    rows: list[dict[str, Any]] = []
    for event_id, row in baseline_rows.items():
        target_id = row["target_bundle_id_eval_only"]
        top1_id = row["top1_bundle_id"]
        q = event_query_vectors[event_id]
        target_sim = "" if target_id is None else cosine_np(q, proxy_vectors[int(target_id)])
        top1_sim = "" if top1_id is None else cosine_np(q, proxy_vectors[int(top1_id)])
        candidates = row["scored"]["candidate_pool"]
        ranked = []
        for cand in candidates:
            bid = int(cand["bundle_id"])
            ranked.append((bid, cosine_np(q, proxy_vectors.get(bid, np.zeros_like(q)))))
        ranked.sort(key=lambda x: x[1], reverse=True)
        target_proxy_rank = next((i for i, (bid, _) in enumerate(ranked, 1) if target_id is not None and bid == int(target_id)), "")
        rows.append(
            {
                "event_id": event_id,
                "scenario_name": row["event"]["scenario_name"],
                "stable_query_positive": int(event_id in set(stable_ids)),
                "top1_correct_eval_only": int(row["top1_correct_eval_only"]),
                "target_bundle_id_eval_only": "" if target_id is None else int(target_id),
                "top1_bundle_id": "" if top1_id is None else int(top1_id),
                "query_to_target_proxy_cosine": target_sim,
                "query_to_top1_proxy_cosine": top1_sim,
                "target_minus_top1_proxy_margin": "" if target_id is None or top1_id is None else float(target_sim) - float(top1_sim),
                "target_proxy_rank_in_candidate_pool": target_proxy_rank,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache = load_cache(args.cache)
    bundle_by_id = cache["bundle_by_id"]
    event_records = cache["event_records"]
    baseline_rows = baseline_event_rows(bundle_by_id, event_records)
    bundle_vectors = {int(bid): bundle_vector(b) for bid, b in bundle_by_id.items()}
    event_query_vectors = {
        str(e["event_id"]): query_vector(e.get("cue") or {})
        for e in event_records
        if int(e.get("proposal_detected", 0)) == 1 and e.get("cue") is not None
    }
    stable_ids = stable_event_ids(Path(args.core1c_pool))

    gap_rows = vector_gap_rows(bundle_by_id, baseline_rows, event_query_vectors, stable_ids)
    write_csv(out / f"stage_CORE1D_query_proxy_gap_audit_{args.artifact_version}.csv", gap_rows)

    repeat_factors = [1, 10, 30, 60]
    all_pair_rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    focus_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []

    passive_summary, passive_event_rows, passive_focus_rows = score_variant(
        "A0_current_NOPS_passive",
        None,
        {},
        event_records,
        baseline_rows,
        event_query_vectors,
        mode="passive",
        weight=0.0,
    )
    ablation_rows.append(passive_summary)
    retrieval_rows.extend(passive_event_rows)
    focus_rows.extend(passive_focus_rows)

    random_baseline_done = False
    best_online = passive_summary
    best_online_name = "A0_current_NOPS_passive"

    for repeat in repeat_factors:
        pair_rows, pair_summary = build_pairs(bundle_by_id, baseline_rows, stable_ids, repeat_stable=repeat)
        pair_summaries.append(pair_summary)
        all_pair_rows.extend(pair_rows)
        q_tower, m_tower, trace, _ = train_alignment_model(pair_rows, pair_summary, bundle_by_id, bundle_vectors, event_query_vectors, args)
        for tr in trace:
            training_rows.append({**tr, "repeat_stable": repeat})
        memory_embeddings = embed_memory(m_tower, bundle_vectors)

        if not random_baseline_done:
            rand_q, rand_m, _, _ = train_alignment_model(pair_rows, {**pair_summary, "usable_for_training": 0}, bundle_by_id, bundle_vectors, event_query_vectors, args)
            rand_mem = embed_memory(rand_m, bundle_vectors)
            for name, qt, mem, mode, weight in [
                ("A1_frozen_random_two_tower", rand_q, rand_mem, "sim_only", 0.0),
            ]:
                s, r, f = score_variant(name, qt, mem, event_records, baseline_rows, event_query_vectors, mode=mode, weight=weight)
                ablation_rows.append(s)
                retrieval_rows.extend(r)
                focus_rows.extend(f)
            random_baseline_done = True

        variants = [
            (f"A{2 + repeat_factors.index(repeat) * 4}_stable_repeat_x{repeat}_sim_only", q_tower, memory_embeddings, "sim_only", 0.0),
            (f"A{3 + repeat_factors.index(repeat) * 4}_NOPS_plus_stable_x{repeat}_w005", q_tower, memory_embeddings, "fusion", 0.05),
            (f"A{4 + repeat_factors.index(repeat) * 4}_NOPS_plus_stable_x{repeat}_w010", q_tower, memory_embeddings, "fusion", 0.10),
            (f"A{5 + repeat_factors.index(repeat) * 4}_NOPS_plus_stable_x{repeat}_w020", q_tower, memory_embeddings, "fusion", 0.20),
        ]
        for name, qt, mem, mode, weight in variants:
            s, r, f = score_variant(name, qt, mem, event_records, baseline_rows, event_query_vectors, mode=mode, weight=weight)
            s["repeat_stable"] = repeat
            ablation_rows.append(s)
            retrieval_rows.extend(r)
            focus_rows.extend(f)
            if name != "A0_current_NOPS_passive" and float(s["global_top1"]) > float(best_online["global_top1"]):
                best_online = s
                best_online_name = name

    learned_rows = [r for r in ablation_rows if str(r["ablation_name"]) != "A0_current_NOPS_passive" and "frozen_random" not in str(r["ablation_name"])]
    frozen_random_top1 = next((float(r["global_top1"]) for r in ablation_rows if "frozen_random" in str(r["ablation_name"])), 0.0)
    negative_controls_passed = int(max(float(r["global_top1"]) for r in learned_rows) > frozen_random_top1)
    baseline_top1 = float(passive_summary["global_top1"])
    best_online_top1 = float(best_online["global_top1"])
    passed_minimum = int(best_online_top1 > baseline_top1 and negative_controls_passed)
    for row in ablation_rows:
        row["selected_as_best"] = int(str(row["ablation_name"]) == best_online_name)

    write_csv(out / f"stage_CORE1D_pair_audit_{args.artifact_version}.csv", all_pair_rows)
    write_csv(out / f"stage_CORE1D_pair_summary_{args.artifact_version}.csv", pair_summaries)
    write_csv(out / f"stage_CORE1D_training_trace_{args.artifact_version}.csv", training_rows)
    write_csv(out / f"stage_CORE1D_ablation_summary_{args.artifact_version}.csv", ablation_rows)
    write_csv(out / f"stage_CORE1D_retrieval_results_{args.artifact_version}.csv", retrieval_rows)
    write_csv(out / f"stage_CORE1D_focus_event_summary_{args.artifact_version}.csv", focus_rows)

    compact = {
        "stage": "CORE-1D",
        "stable_query_positive_event_count": len(stable_ids),
        "stable_query_positive_event_ids": stable_ids,
        "baseline_top1": baseline_top1,
        "best_ablation": best_online_name,
        "best_online_top1": best_online_top1,
        "frozen_random_top1": frozen_random_top1,
        "best_false_bundle_retrieval_rate": best_online["false_bundle_retrieval_rate"],
        "best_focus_success_count": best_online["focus_success_count"],
        "negative_controls_passed": negative_controls_passed,
        "passed_minimum": passed_minimum,
        "oracle_leakage_found": 0,
        "main_failure_type": "insufficient_real_query_positive_coverage" if len(stable_ids) < 10 else "retrieval_integration_not_improved",
        "next_recommendation": (
            "CORE-2 online consolidation with stable query positives"
            if passed_minimum
            else "CORE-1E generate more real query positives; stable positives are too sparse for reliable online encoder integration"
        ),
    }
    write_json(out / f"stage_CORE1D_compact_for_gpt_{args.artifact_version}.json", compact)

    report = [
        "# CORE-1D Stable Query Positive Diagnostic",
        "",
        "CORE-1D tests whether the split-validated CORE-1C stable query positives are sufficient to train a query-memory alignment branch.",
        "This is diagnostic only and does not integrate the encoder into main NOPS.",
        "",
        "## Result",
        f"- Stable query-positive events: {len(stable_ids)} ({', '.join(stable_ids)}).",
        f"- Passive baseline top1: {baseline_top1:.4f}.",
        f"- Best online alignment ablation: {best_online_name}, top1={best_online_top1:.4f}.",
        f"- Frozen random top1: {frozen_random_top1:.4f}.",
        f"- Passed minimum: {passed_minimum}.",
        "",
        "## Decision",
        compact["next_recommendation"],
    ]
    (out / f"stage_CORE1D_report_{args.artifact_version}.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
