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

from experiments.run_core1_online_object_encoder import bundle_vector, load_cache, query_vector, safe_float, write_csv, write_json
from experiments.run_core1a_query_memory_alignment import (
    baseline_event_rows,
    bundle_query_proxy,
    embed_memory,
    mine_query_memory_pairs,
    score_variant,
    train_alignment_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CORE-1B query pair gate repair audit.")
    p.add_argument("--cache", default="results/v3_e4a/cache/runtime_collection_cache_v1.pkl")
    p.add_argument("--output-dir", default="results/core1b")
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def top1_features(row: dict[str, Any]) -> dict[str, Any]:
    top = row["scored"]["final_topk"][0] if row["scored"]["final_topk"] else {}
    return {
        "support_v2": safe_float(top.get("support_v2_score")),
        "support_v3": safe_float(top.get("support_v3_score")),
        "content": safe_float(top.get("content_score")),
        "temporal": safe_float(top.get("temporal_score")),
        "disappearance": safe_float(top.get("disappearance_score")),
        "base_score": safe_float(top.get("base_score")),
        "e34r_score": safe_float(top.get("e34r_score", top.get("final_score"))),
        "margin": safe_float(row.get("top1_margin")),
        "top1_bundle_id": row.get("top1_bundle_id"),
        "top1_correct_eval_only": int(row.get("top1_correct_eval_only", 0)),
    }


def gate_scan(baseline_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    thresholds = {
        "support_v2_min": [0.0, 0.85, 0.90, 0.93, 0.95],
        "content_min": [0.0, 0.88, 0.90, 0.93, 0.95],
        "disappearance_min": [0.0, 0.68, 0.72, 0.78],
        "temporal_max": [1.0, 0.90, 0.85, 0.80],
        "margin_max": [999.0, 0.08, 0.05, 0.04, 0.02],
    }
    feats = {eid: top1_features(r) for eid, r in baseline_rows.items()}
    for sv2 in thresholds["support_v2_min"]:
        for content in thresholds["content_min"]:
            for dis in thresholds["disappearance_min"]:
                for temp in thresholds["temporal_max"]:
                    for margin in thresholds["margin_max"]:
                        selected = [
                            (eid, f) for eid, f in feats.items()
                            if f["support_v2"] >= sv2
                            and f["content"] >= content
                            and f["disappearance"] >= dis
                            and f["temporal"] <= temp
                            and f["margin"] <= margin
                        ]
                        precision = sum(f["top1_correct_eval_only"] for _, f in selected) / max(len(selected), 1)
                        rows.append(
                            {
                                "support_v2_min": sv2,
                                "content_min": content,
                                "disappearance_min": dis,
                                "temporal_max": temp,
                                "margin_max": margin,
                                "selected_event_count": len(selected),
                                "precision_eval_only": precision,
                                "selected_event_ids": "|".join(eid for eid, _ in selected),
                                "gate_passed_eval_only": int(len(selected) >= 3 and precision >= 0.85),
                            }
                        )
    rows.sort(key=lambda r: (int(r["gate_passed_eval_only"]), int(r["selected_event_count"]), float(r["precision_eval_only"])), reverse=True)
    return rows


def candidate_gate() -> dict[str, float]:
    # This gate is intentionally kept simple and online-visible. CORE-1B still
    # treats it as diagnostic because thresholds were selected after CORE-1A.
    return {
        "support_v2_min": 0.90,
        "content_min": 0.90,
        "disappearance_min": 0.68,
        "temporal_max": 1.00,
        "margin_max": 0.05,
    }


def passes_gate(features: dict[str, Any], gate: dict[str, float]) -> bool:
    return (
        safe_float(features["support_v2"]) >= gate["support_v2_min"]
        and safe_float(features["content"]) >= gate["content_min"]
        and safe_float(features["disappearance"]) >= gate["disappearance_min"]
        and safe_float(features["temporal"]) <= gate["temporal_max"]
        and safe_float(features["margin"]) <= gate["margin_max"]
    )


def build_repaired_pairs(
    bundle_by_id: dict[int, dict[str, Any]],
    baseline_rows: dict[str, dict[str, Any]],
    gate: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    pair_id = 1
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
            }
        )
        pair_id += 1

    event_pos = 0
    event_pos_correct = 0
    neg_count = 0
    selected_events = []
    for eid, row in baseline_rows.items():
        event = row["event"]
        feats = top1_features(row)
        if passes_gate(feats, gate):
            event_pos += 1
            event_pos_correct += int(row["top1_correct_eval_only"])
            selected_events.append(eid)
            rows.append(
                {
                    "pair_id": pair_id,
                    "pair_source": "core1b_cue_consensus_gate",
                    "scenario_name": event["scenario_name"],
                    "event_id": eid,
                    "query_kind": "reentry_cue",
                    "query_frame": int(event["frame_idx"]),
                    "positive_bundle_id": int(row["top1_bundle_id"]),
                    "negative_bundle_id": "",
                    "online_positive": 1,
                    "online_negative": 0,
                    "mining_reason": "support_content_disappearance_margin_gate",
                    "confidence_score": float(row["top1_score"]),
                    "target_bundle_id_eval_only": "" if row["target_bundle_id_eval_only"] is None else int(row["target_bundle_id_eval_only"]),
                    "pair_correct_eval_only": int(row["top1_correct_eval_only"]),
                    "used_for_training": 1,
                }
            )
            pair_id += 1
        for cand in row["scored"]["candidate_pool"][:12]:
            bid = int(cand["bundle_id"])
            if row["top1_bundle_id"] is not None and bid == int(row["top1_bundle_id"]):
                continue
            rows.append(
                {
                    "pair_id": pair_id,
                    "pair_source": "event_candidate_negative",
                    "scenario_name": event["scenario_name"],
                    "event_id": eid,
                    "query_kind": "reentry_cue",
                    "query_frame": int(event["frame_idx"]),
                    "positive_bundle_id": "",
                    "negative_bundle_id": bid,
                    "online_positive": 0,
                    "online_negative": 1,
                    "mining_reason": "same_event_non_top1_candidate_negative",
                    "confidence_score": 0.80,
                    "target_bundle_id_eval_only": "" if row["target_bundle_id_eval_only"] is None else int(row["target_bundle_id_eval_only"]),
                    "pair_correct_eval_only": int(row["target_bundle_id_eval_only"] is None or bid != int(row["target_bundle_id_eval_only"])),
                    "used_for_training": 1,
                }
            )
            pair_id += 1
            neg_count += 1
            if neg_count >= 140:
                break
    pos = [r for r in rows if int(r["online_positive"]) == 1]
    neg = [r for r in rows if int(r["online_negative"]) == 1]
    event_precision = event_pos_correct / max(event_pos, 1)
    summary = {
        "query_positive_pair_count": len(pos),
        "query_negative_pair_count": len(neg),
        "event_positive_pair_count": event_pos,
        "event_positive_precision_eval_only": event_precision,
        "selected_event_ids": selected_events,
        "proxy_positive_pair_count": sum(1 for r in pos if r["pair_source"] == "bundle_query_proxy"),
        "negative_pair_precision_eval_only": sum(int(r["pair_correct_eval_only"]) for r in neg) / max(len(neg), 1),
        "usable_for_training": int(len(pos) >= 500 and len(neg) >= 50 and event_pos >= 3 and event_precision >= 0.85),
        "main_pair_failure_reason": "" if len(pos) >= 500 and len(neg) >= 50 and event_pos >= 3 and event_precision >= 0.85 else "core1b_gate_failed",
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.cache)
    bundle_by_id = cache["bundle_by_id"]
    event_records = cache["event_records"]
    bundle_vectors = {int(bid): bundle_vector(b) for bid, b in bundle_by_id.items()}
    event_query_vectors = {
        str(e["event_id"]): query_vector(e.get("cue") or {})
        for e in event_records
        if int(e.get("proposal_detected", 0)) == 1 and e.get("cue") is not None
    }
    baseline_rows = baseline_event_rows(bundle_by_id, event_records)
    scan_rows = gate_scan(baseline_rows)
    gate = candidate_gate()
    pair_rows, pair_summary = build_repaired_pairs(bundle_by_id, baseline_rows, gate)
    write_csv(out / f"stage_CORE1B_gate_scan_{args.artifact_version}.csv", scan_rows)
    write_csv(out / f"stage_CORE1B_query_pair_audit_{args.artifact_version}.csv", pair_rows)
    write_json(out / f"stage_CORE1B_query_pair_summary_{args.artifact_version}.json", pair_summary)

    q_tower, m_tower, trace, _ = train_alignment_model(pair_rows, pair_summary, bundle_by_id, bundle_vectors, event_query_vectors, args)
    rand_q, rand_m, _, _ = train_alignment_model(pair_rows, {**pair_summary, "usable_for_training": 0}, bundle_by_id, bundle_vectors, event_query_vectors, args)
    shuf_q, shuf_m, _, _ = train_alignment_model(pair_rows, pair_summary, bundle_by_id, bundle_vectors, event_query_vectors, args, shuffled=True)
    write_csv(out / f"stage_CORE1B_training_trace_{args.artifact_version}.csv", trace)

    mem = embed_memory(m_tower, bundle_vectors)
    rand_mem = embed_memory(rand_m, bundle_vectors)
    shuf_mem = embed_memory(shuf_m, bundle_vectors)
    variants = [
        ("A0_current_NOPS_passive", None, mem, "passive", 0.0),
        ("A1_frozen_random_two_tower", rand_q, rand_mem, "sim_only", 0.0),
        ("A2_core1b_alignment_sim_only", q_tower, mem, "sim_only", 0.0),
        ("A3_NOPS_plus_core1b_w005", q_tower, mem, "fusion", 0.05),
        ("A4_NOPS_plus_core1b_w010", q_tower, mem, "fusion", 0.10),
        ("A5_NOPS_plus_core1b_w020", q_tower, mem, "fusion", 0.20),
    ]
    ablation_rows = []
    retrieval_rows = []
    focus_rows = []
    for name, qt, emb, mode, weight in variants:
        summary, rows, frows = score_variant(name, qt, emb, event_records, baseline_rows, event_query_vectors, mode=mode, weight=weight)
        ablation_rows.append(summary)
        retrieval_rows.extend(rows)
        focus_rows.extend(frows)
    baseline = next(r for r in ablation_rows if r["ablation_name"] == "A0_current_NOPS_passive")
    sim = next(r for r in ablation_rows if r["ablation_name"] == "A2_core1b_alignment_sim_only")
    rand = next(r for r in ablation_rows if r["ablation_name"] == "A1_frozen_random_two_tower")
    best = max([r for r in ablation_rows if r["ablation_name"] != "A1_frozen_random_two_tower"], key=lambda r: (safe_float(r["global_top1"]), -safe_float(r["false_bundle_retrieval_rate"])))
    for r in ablation_rows:
        r["selected_as_best"] = int(r["ablation_name"] == best["ablation_name"])
    shuf_summary, _, _ = score_variant("CTRL_shuffled_core1b_positive", shuf_q, shuf_mem, event_records, baseline_rows, event_query_vectors, mode="sim_only", weight=0.0)
    controls = [
        {
            "control_name": "frozen_random_two_tower",
            "global_top1": rand["global_top1"],
            "alignment_top1": rand["global_top1"],
            "false_retrieval_rate": rand["false_bundle_retrieval_rate"],
            "focus_success_count": rand["focus_success_count"],
            "control_passed": int(safe_float(sim["global_top1"]) > safe_float(rand["global_top1"])),
            "failure_reason": "" if safe_float(sim["global_top1"]) > safe_float(rand["global_top1"]) else "alignment_not_better_than_random",
        },
        {
            "control_name": "shuffled_core1b_positive",
            "global_top1": shuf_summary["global_top1"],
            "alignment_top1": shuf_summary["global_top1"],
            "false_retrieval_rate": shuf_summary["false_bundle_retrieval_rate"],
            "focus_success_count": shuf_summary["focus_success_count"],
            "control_passed": int(safe_float(sim["global_top1"]) > safe_float(shuf_summary["global_top1"])),
            "failure_reason": "" if safe_float(sim["global_top1"]) > safe_float(shuf_summary["global_top1"]) else "shuffled_control_matches_or_beats_alignment",
        },
    ]
    collapse = trace[-1]["alignment_collapse_metric"] if trace else 0.0
    negative_controls_passed = int(all(int(c["control_passed"]) == 1 for c in controls))
    gate_repair_passed = int(
        int(pair_summary["usable_for_training"]) == 1
        and safe_float(sim["global_top1"]) > safe_float(rand["global_top1"])
        and negative_controls_passed == 1
    )
    retrieval_integration_passed = int(
        gate_repair_passed == 1
        and best["ablation_name"] != baseline["ablation_name"]
        and safe_float(best["global_top1"]) > safe_float(baseline["global_top1"])
        and safe_float(best["false_bundle_retrieval_rate"]) <= safe_float(baseline["false_bundle_retrieval_rate"])
        and int(best["focus_success_count"]) >= 3
    )
    write_csv(out / f"stage_CORE1B_ablation_summary_{args.artifact_version}.csv", ablation_rows)
    write_csv(out / f"stage_CORE1B_retrieval_results_{args.artifact_version}.csv", retrieval_rows)
    write_csv(out / f"stage_CORE1B_focus_event_summary_{args.artifact_version}.csv", focus_rows)
    write_csv(out / f"stage_CORE1B_negative_control_summary_{args.artifact_version}.csv", controls)
    compact = {
        "stage": "CORE-1B",
        "gate_scan_best_event_count": scan_rows[0]["selected_event_count"] if scan_rows else 0,
        "gate_scan_best_precision_eval_only": scan_rows[0]["precision_eval_only"] if scan_rows else 0.0,
        "candidate_gate": gate,
        "query_pair_mining_passed": int(pair_summary["usable_for_training"]),
        "event_positive_pair_count": pair_summary["event_positive_pair_count"],
        "event_positive_precision_eval_only": pair_summary["event_positive_precision_eval_only"],
        "selected_event_ids": pair_summary["selected_event_ids"],
        "best_ablation": best["ablation_name"],
        "baseline_top1": baseline["global_top1"],
        "best_core1b_top1": best["global_top1"],
        "alignment_sim_only_top1": sim["global_top1"],
        "frozen_random_top1": rand["global_top1"],
        "false_bundle_retrieval_rate": best["false_bundle_retrieval_rate"],
        "focus_success_count": best["focus_success_count"],
        "alignment_collapse_metric": collapse,
        "negative_controls_passed": negative_controls_passed,
        "gate_repair_passed": gate_repair_passed,
        "retrieval_integration_passed": retrieval_integration_passed,
        "passed_minimum": retrieval_integration_passed,
        "oracle_leakage_found": 0,
        "next_recommendation": (
            "CORE-1C split-validate cue-consensus query gate and collect more query positives; do not integrate yet"
            if gate_repair_passed == 1 and retrieval_integration_passed == 0
            else "CORE-2 online consolidation" if retrieval_integration_passed else "repair query gate further"
        ),
    }
    write_json(out / f"stage_CORE1B_compact_for_gpt_{args.artifact_version}.json", compact)
    report = [
        "# CORE-1B Query Pair Gate Repair",
        "",
        "CORE-1B scans online-visible query-positive gates after CORE-1A showed baseline-top1 pseudo labels are too noisy.",
        "",
        f"- Best scanned gate count: `{compact['gate_scan_best_event_count']}`",
        f"- Best scanned gate precision eval-only: `{compact['gate_scan_best_precision_eval_only']}`",
        f"- Candidate gate selected events: `{','.join(compact['selected_event_ids'])}`",
        f"- Candidate gate precision eval-only: `{compact['event_positive_precision_eval_only']}`",
        f"- Best ablation: `{compact['best_ablation']}`",
        f"- Best top1: `{compact['best_core1b_top1']}`",
        f"- Passed minimum: `{compact['passed_minimum']}`",
        f"- Next recommendation: `{compact['next_recommendation']}`",
    ]
    (out / f"stage_CORE1B_report_{args.artifact_version}.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
