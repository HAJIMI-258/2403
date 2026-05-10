from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1_online_object_encoder import cosine_np, write_csv, write_json
from experiments.run_core1h_dense_diagnostic_encoder_upper_bound import memory_features, query_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CORE-1I dense diagnostic feature failure audit.")
    p.add_argument("--ledger", default="results/core1f/stage_CORE1F_dense_event_ledger_v1.csv")
    p.add_argument("--retrieval", default="results/core1h/stage_CORE1H_retrieval_results_v1.csv")
    p.add_argument("--output-dir", default="results/core1i")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def i(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def candidate_sets(rows: list[dict[str, str]], event: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    split = event["split"]
    seq = event["sequence_id"]
    concept = event["concept_id_eval_only"]
    split_rows = [r for r in rows if r["split"] == split]
    seq_rows = [r for r in split_rows if r["sequence_id"] == seq]
    seq_concept_rows = [r for r in seq_rows if r["concept_id_eval_only"] == concept]
    return {
        "same_sequence_same_concept": seq_concept_rows,
        "same_sequence_all": seq_rows,
        "same_split_all": split_rows,
    }


def rank_event(event: dict[str, str], candidates: list[dict[str, str]]) -> tuple[int | None, float, float, str]:
    q = query_features(event)
    scored = []
    for cand in candidates:
        scored.append((cand["event_id"], cosine_np(q, memory_features(cand)), cand))
    scored.sort(key=lambda x: x[1], reverse=True)
    rank = next((idx for idx, (eid, _, _) in enumerate(scored, 1) if eid == event["event_id"]), None)
    target_score = next((score for eid, score, _ in scored if eid == event["event_id"]), 0.0)
    top_score = scored[0][1] if scored else 0.0
    top_event = scored[0][0] if scored else ""
    return rank, target_score, top_score, top_event


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = [r for r in read_csv(args.ledger) if i(r.get("usable_real_gap")) == 1]

    rows = []
    taxonomy = Counter()
    for event in ledger:
        sets = candidate_sets(ledger, event)
        result: dict[str, Any] = {
            "event_id": event["event_id"],
            "split": event["split"],
            "sequence_id": event["sequence_id"],
            "concept_id": event["concept_id_eval_only"],
            "gap_length": event["gap_length"],
            "difficulty_level": event["difficulty_level"],
            "same_sequence_same_concept_candidate_count": len(sets["same_sequence_same_concept"]),
            "same_sequence_candidate_count": len(sets["same_sequence_all"]),
            "same_split_candidate_count": len(sets["same_split_all"]),
            "fast_planned_area_change_ratio": event["area_change_ratio"],
            "center_displacement": event["center_displacement"],
        }
        failure_reasons = []
        for name, candidates in sets.items():
            rank, target_score, top_score, top_event = rank_event(event, candidates)
            result[f"{name}_target_rank"] = "" if rank is None else rank
            result[f"{name}_top1"] = int(rank == 1)
            result[f"{name}_target_score"] = target_score
            result[f"{name}_top_score"] = top_score
            result[f"{name}_margin_target_minus_top"] = target_score - top_score
            result[f"{name}_top_event_id"] = top_event
        if i(result["same_sequence_same_concept_target_rank"], 9999) > 1:
            failure_reasons.append("same_concept_geometry_collision")
        if i(result["same_sequence_all_target_rank"], 9999) > 1 and i(result["same_sequence_same_concept_target_rank"], 9999) == 1:
            failure_reasons.append("different_concept_geometry_collision")
        if f(event["area_change_ratio"]) > 10:
            failure_reasons.append("fast_planned_area_degenerate")
        if f(event["center_displacement"]) > 180:
            failure_reasons.append("large_reentry_displacement")
        if i(result["same_split_candidate_count"]) > 100:
            failure_reasons.append("candidate_scope_too_broad")
        if not failure_reasons:
            failure_reasons.append("raw_geometry_candidate_ranking_ok")
        result["dominant_failure_reason"] = failure_reasons[0]
        for reason in failure_reasons:
            taxonomy[reason] += 1
        rows.append(result)

    summary_rows = []
    count_key = {
        "same_sequence_same_concept": "same_sequence_same_concept_candidate_count",
        "same_sequence_all": "same_sequence_candidate_count",
        "same_split_all": "same_split_candidate_count",
    }
    for scope in ("same_sequence_same_concept", "same_sequence_all", "same_split_all"):
        n = max(len(rows), 1)
        summary_rows.append(
            {
                "candidate_scope": scope,
                "top1": sum(i(r[f"{scope}_top1"]) for r in rows) / n,
                "mean_rank": sum(i(r[f"{scope}_target_rank"], 9999) for r in rows) / n,
                "mean_margin_target_minus_top": sum(f(r[f"{scope}_margin_target_minus_top"]) for r in rows) / n,
                "mean_candidate_count": sum(i(r[count_key[scope]]) for r in rows) / n,
            }
        )
    taxonomy_rows = [{"failure_reason": k, "count": v} for k, v in taxonomy.most_common()]
    write_csv(out / f"stage_CORE1I_feature_failure_audit_{args.artifact_version}.csv", rows)
    write_csv(out / f"stage_CORE1I_candidate_scope_summary_{args.artifact_version}.csv", summary_rows)
    write_csv(out / f"stage_CORE1I_failure_taxonomy_{args.artifact_version}.csv", taxonomy_rows)

    same_seq = next(r for r in summary_rows if r["candidate_scope"] == "same_sequence_all")
    same_concept = next(r for r in summary_rows if r["candidate_scope"] == "same_sequence_same_concept")
    same_split = next(r for r in summary_rows if r["candidate_scope"] == "same_split_all")
    compact = {
        "stage": "CORE-1I",
        "event_count": len(rows),
        "same_sequence_same_concept_top1": same_concept["top1"],
        "same_sequence_top1": same_seq["top1"],
        "same_split_top1": same_split["top1"],
        "same_sequence_mean_candidate_count": same_seq["mean_candidate_count"],
        "same_split_mean_candidate_count": same_split["mean_candidate_count"],
        "main_failure_counts": dict(taxonomy),
        "passed_minimum": int(same_seq["top1"] >= 0.50),
        "next_recommendation": (
            "CORE-1J render selected dense ledger windows and mine tracker-derived online pairs"
            if same_seq["top1"] >= 0.50
            else "CORE-1J repair dense ledger candidate scope / render selected windows; fast-planned geometry is not sufficient"
        ),
    }
    write_json(out / f"stage_CORE1I_compact_for_gpt_{args.artifact_version}.json", compact)
    report = [
        "# CORE-1I Dense Feature Failure Audit",
        "",
        "CORE-1I checks whether CORE-1H failed because of broad candidate scope or because fast-planned geometry is intrinsically weak.",
        "",
        "## Result",
        f"- Same-sequence same-concept top1: {same_concept['top1']:.4f}.",
        f"- Same-sequence top1: {same_seq['top1']:.4f}.",
        f"- Same-split top1: {same_split['top1']:.4f}.",
        f"- Main failure counts: {dict(taxonomy)}.",
        "",
        "## Decision",
        compact["next_recommendation"],
    ]
    (out / f"stage_CORE1I_report_{args.artifact_version}.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
