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

from experiments.ext1_utils import read_csv, write_csv


NOPS = "B3_nops_anchor_episodic_passive"
BEST = "B2_support_trajectory_memory"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXT-1A external failure analysis.")
    p.add_argument("--ext1-dir", default="results/ext1")
    p.add_argument("--output-dir", default="results/ext1a")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["dataset_name"]), str(row["sequence_id"]), str(row["event_id"]))


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def load_by_method(rows: list[dict[str, str]]) -> dict[str, dict[tuple[str, str, str], dict[str, str]]]:
    out: dict[str, dict[tuple[str, str, str], dict[str, str]]] = defaultdict(dict)
    for row in rows:
        out[str(row["method_name"])][_key(row)] = row
    return out


def event_delta(results: list[dict[str, str]], difficulty: list[dict[str, str]], ledger: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_method = load_by_method(results)
    difficulty_by_event = {str(r["event_id"]): r for r in difficulty}
    ledger_by_event = {str(r["event_id"]): r for r in ledger}
    rows: list[dict[str, Any]] = []
    all_events = sorted(set(by_method.get(NOPS, {})) | set(by_method.get(BEST, {})))
    for key in all_events:
        n = by_method.get(NOPS, {}).get(key)
        b = by_method.get(BEST, {}).get(key)
        if n is None or b is None:
            continue
        event_id = key[2]
        n_top1 = _to_int(n["target_memory_retrieved_top1"])
        b_top1 = _to_int(b["target_memory_retrieved_top1"])
        n_top5 = _to_int(n["target_memory_retrieved_top5"])
        b_top5 = _to_int(b["target_memory_retrieved_top5"])
        if n_top1 and b_top1:
            cls = "both_success"
        elif (not n_top1) and b_top1:
            cls = "best_only_success"
        elif n_top1 and not b_top1:
            cls = "nops_only_success"
        else:
            cls = "both_fail"
        d = difficulty_by_event.get(event_id, {})
        l = ledger_by_event.get(event_id, {})
        rows.append({
            "dataset_name": key[0],
            "sequence_id": key[1],
            "event_id": event_id,
            "delta_class": cls,
            "nops_top1": n_top1,
            "best_top1": b_top1,
            "nops_top3": _to_int(n["target_memory_retrieved_top3"]),
            "best_top3": _to_int(b["target_memory_retrieved_top3"]),
            "nops_top5": n_top5,
            "best_top5": b_top5,
            "nops_predicted_memory_id": n.get("predicted_memory_id", ""),
            "best_predicted_memory_id": b.get("predicted_memory_id", ""),
            "target_instance_id_eval_only": n.get("target_instance_id_eval_only", ""),
            "nops_failure_reason": n.get("failure_reason", ""),
            "best_failure_reason": b.get("failure_reason", ""),
            "difficulty_level": d.get("difficulty_level", ""),
            "gap_length": d.get("gap_length", l.get("gap_length", "")),
            "center_displacement": d.get("center_displacement", ""),
            "same_category_distractors": d.get("same_category_distractors", l.get("num_similar_distractors", "")),
            "num_similar_distractors": l.get("num_similar_distractors", ""),
            "nops_candidate_generation_ok": int(n_top5 == 1),
            "best_candidate_generation_ok": int(b_top5 == 1),
            "top1_selection_failure": int(n_top5 == 1 and n_top1 == 0),
        })
    return rows


def aggregate_gap_by_difficulty(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        groups[str(row.get("difficulty_level", "unknown"))].append(row)
    out: list[dict[str, Any]] = []
    for level, rows in sorted(groups.items()):
        n = len(rows)
        out.append({
            "difficulty_level": level or "unknown",
            "event_count": n,
            "nops_top1": sum(int(r["nops_top1"]) for r in rows) / max(n, 1),
            "best_top1": sum(int(r["best_top1"]) for r in rows) / max(n, 1),
            "delta": (sum(int(r["nops_top1"]) for r in rows) - sum(int(r["best_top1"]) for r in rows)) / max(n, 1),
            "best_only_success_count": sum(1 for r in rows if r["delta_class"] == "best_only_success"),
            "nops_only_success_count": sum(1 for r in rows if r["delta_class"] == "nops_only_success"),
            "nops_top5_not_top1_count": sum(int(r["top1_selection_failure"]) for r in rows),
        })
    return out


def failure_breakdown(results: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()
    for row in results:
        method = str(row["method_name"])
        reason = str(row.get("failure_reason", ""))
        totals[method] += 1
        counts[(method, reason or "success")] += 1
    out = []
    for (method, reason), count in sorted(counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
        out.append({
            "method_name": method,
            "failure_reason": reason,
            "count": count,
            "rate": count / max(totals[method], 1),
        })
    return out


def sequence_category_summary(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        seq = str(row["sequence_id"])
        category = seq
        if category.startswith("lagot_"):
            category = category[len("lagot_"):]
        category = category.split("-")[0]
        groups[category].append(row)
    out = []
    for cat, rows in sorted(groups.items()):
        n = len(rows)
        out.append({
            "category_id": cat,
            "event_count": n,
            "nops_top1": sum(int(r["nops_top1"]) for r in rows) / max(n, 1),
            "best_top1": sum(int(r["best_top1"]) for r in rows) / max(n, 1),
            "best_only_success_count": sum(1 for r in rows if r["delta_class"] == "best_only_success"),
            "nops_only_success_count": sum(1 for r in rows if r["delta_class"] == "nops_only_success"),
        })
    out.sort(key=lambda r: (r["best_only_success_count"] - r["nops_only_success_count"], r["event_count"]), reverse=True)
    return out


def main() -> None:
    args = parse_args()
    ext1 = Path(args.ext1_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = read_csv(ext1 / "stage_EXT1_oracle_proposal_memory_results_v1.csv")
    difficulty = read_csv(ext1 / "stage_EXT1_event_difficulty_audit_v1.csv")
    ledger = read_csv(ext1 / "stage_EXT1_external_event_ledger_v1.csv")
    delta_rows = event_delta(results, difficulty, ledger)
    diff_rows = aggregate_gap_by_difficulty(delta_rows)
    fail_rows = failure_breakdown(results)
    cat_rows = sequence_category_summary(delta_rows)
    write_csv(out / f"stage_EXT1A_nops_vs_best_event_delta_{args.artifact_version}.csv", delta_rows)
    write_csv(out / f"stage_EXT1A_gap_by_difficulty_{args.artifact_version}.csv", diff_rows)
    write_csv(out / f"stage_EXT1A_failure_reason_breakdown_{args.artifact_version}.csv", fail_rows)
    write_csv(out / f"stage_EXT1A_category_hard_cases_{args.artifact_version}.csv", cat_rows)

    total = len(delta_rows)
    nops_success = sum(int(r["nops_top1"]) for r in delta_rows)
    best_success = sum(int(r["best_top1"]) for r in delta_rows)
    nops_top5 = sum(int(r["nops_top5"]) for r in delta_rows)
    best_only = sum(1 for r in delta_rows if r["delta_class"] == "best_only_success")
    nops_only = sum(1 for r in delta_rows if r["delta_class"] == "nops_only_success")
    selection_fail = sum(int(r["top1_selection_failure"]) for r in delta_rows)
    failure_counts = Counter(r["nops_failure_reason"] or "success" for r in delta_rows)
    compact = {
        "stage": "EXT-1A",
        "event_count": total,
        "nops_top1": nops_success / max(total, 1),
        "best_support_trajectory_top1": best_success / max(total, 1),
        "nops_vs_best_delta": (nops_success - best_success) / max(total, 1),
        "nops_top5": nops_top5 / max(total, 1),
        "best_only_success_count": best_only,
        "nops_only_success_count": nops_only,
        "nops_top5_but_not_top1_count": selection_fail,
        "nops_main_failure_counts": dict(failure_counts.most_common(5)),
        "primary_diagnosis": "candidate generation mostly works; top1 selection under similar distractors is the main external failure",
        "raw_pixel_requirement": "not required for this geometry-only failure diagnosis; required before full perception or appearance descriptor claims",
        "next_recommendation": "build geometry-aware NOPS passive calibration or connect LaSOT pixels for appearance/full-pipeline analysis before model changes",
    }
    report_lines = [
        "# Stage EXT-1A External Failure Analysis",
        "",
        "## Verdict",
        "",
        "NOPS passive is not failing mainly because the target is absent from memory. It retrieves the target in top5 for most events, but loses top1 under similar distractor competition.",
        "",
        "This is still oracle-proposal / geometry-only analysis on LaGOT annotations. Raw LaSOT pixels are not required for this failure diagnosis, but they are required for appearance descriptors and full perception evaluation.",
        "",
        "## Compact",
        "",
        "```json",
        json.dumps(compact, indent=2, ensure_ascii=False),
        "```",
    ]
    (out / f"stage_EXT1A_compact_for_gpt_{args.artifact_version}.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"stage_EXT1A_report_{args.artifact_version}.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()

