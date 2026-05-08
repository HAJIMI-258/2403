from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ext1_utils import read_csv, write_csv


A0 = "A0_nops_current"
CAL = "A2_trajectory_heavy"
REF = "A3_support_trajectory_reference"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXT-3 geometry calibration robustness audit.")
    p.add_argument("--ext1-dir", default="results/ext1")
    p.add_argument("--ext2-dir", default="results/ext2")
    p.add_argument("--output-dir", default="results/ext3")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def as_int(v: Any) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0


def as_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def category_from_seq(seq: str) -> str:
    s = str(seq)
    if s.startswith("lagot_"):
        s = s[len("lagot_"):]
    return s.split("-")[0]


def gap_bin(gap: int) -> str:
    if gap < 10:
        return "gap_03_09"
    if gap < 30:
        return "gap_10_29"
    if gap < 100:
        return "gap_30_99"
    return "gap_100_plus"


def index_events(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        out[row["event_id"]][row["variant"]] = row
    return out


def event_delta(rows: list[dict[str, str]], difficulty_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    diff_by_event = {r["event_id"]: r for r in difficulty_rows}
    by_event = index_events(rows)
    out: list[dict[str, Any]] = []
    for event_id, variants in sorted(by_event.items()):
        if A0 not in variants or CAL not in variants or REF not in variants:
            continue
        base, cal, ref = variants[A0], variants[CAL], variants[REF]
        base_hit = as_int(base["top1"])
        cal_hit = as_int(cal["top1"])
        ref_hit = as_int(ref["top1"])
        if base_hit and cal_hit:
            cls = "unchanged_success"
        elif not base_hit and cal_hit:
            cls = "improved"
        elif base_hit and not cal_hit:
            cls = "regressed"
        else:
            cls = "unchanged_failure"
        diff = diff_by_event.get(event_id, {})
        gap = as_int(base.get("gap_length", diff.get("gap_length", 0)))
        out.append({
            "dataset_name": base["dataset_name"],
            "sequence_id": base["sequence_id"],
            "event_id": event_id,
            "split": base["split"],
            "category_id": category_from_seq(base["sequence_id"]),
            "gap_length": gap,
            "gap_bin": gap_bin(gap),
            "difficulty_level": diff.get("difficulty_level", ""),
            "candidate_count": as_int(base.get("candidate_count", 0)),
            "num_similar_distractors": as_int(base.get("num_similar_distractors", 0)),
            "a0_top1": base_hit,
            "cal_top1": cal_hit,
            "ref_top1": ref_hit,
            "a0_top5": as_int(base["top5"]),
            "cal_top5": as_int(cal["top5"]),
            "ref_top5": as_int(ref["top5"]),
            "a0_predicted_memory_id": base.get("predicted_memory_id", ""),
            "cal_predicted_memory_id": cal.get("predicted_memory_id", ""),
            "ref_predicted_memory_id": ref.get("predicted_memory_id", ""),
            "target_instance_id_eval_only": base.get("target_instance_id_eval_only", ""),
            "delta_class": cls,
            "cal_matches_reference": int(cal.get("predicted_memory_id", "") == ref.get("predicted_memory_id", "")),
            "cal_closes_reference_error": int((not base_hit) and cal_hit and ref_hit),
            "reference_succeeds_cal_fails": int(ref_hit and not cal_hit),
        })
    return out


def summarize_group(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key, ""))].append(row)
    out: list[dict[str, Any]] = []
    for key, rs in sorted(groups.items()):
        n = len(rs)
        a0 = sum(int(r["a0_top1"]) for r in rs)
        cal = sum(int(r["cal_top1"]) for r in rs)
        ref = sum(int(r["ref_top1"]) for r in rs)
        out.append({
            group_key: key or "unknown",
            "event_count": n,
            "a0_top1": a0 / max(n, 1),
            "cal_top1": cal / max(n, 1),
            "ref_top1": ref / max(n, 1),
            "cal_delta_vs_a0": (cal - a0) / max(n, 1),
            "remaining_gap_to_ref": (ref - cal) / max(n, 1),
            "improved_count": sum(1 for r in rs if r["delta_class"] == "improved"),
            "regressed_count": sum(1 for r in rs if r["delta_class"] == "regressed"),
            "unchanged_failure_count": sum(1 for r in rs if r["delta_class"] == "unchanged_failure"),
            "reference_succeeds_cal_fails_count": sum(int(r["reference_succeeds_cal_fails"]) for r in rs),
        })
    return out


def integration_gate(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(delta_rows)
    splits = ["all", "dev", "test"]
    for split in splits:
        rs = delta_rows if split == "all" else [r for r in delta_rows if r["split"] == split]
        n = len(rs)
        a0 = sum(int(r["a0_top1"]) for r in rs)
        cal = sum(int(r["cal_top1"]) for r in rs)
        ref = sum(int(r["ref_top1"]) for r in rs)
        improved = sum(1 for r in rs if r["delta_class"] == "improved")
        regressed = sum(1 for r in rs if r["delta_class"] == "regressed")
        rows.append({
            "split": split,
            "event_count": n,
            "a0_top1": a0 / max(n, 1),
            "cal_top1": cal / max(n, 1),
            "ref_top1": ref / max(n, 1),
            "cal_delta_vs_a0": (cal - a0) / max(n, 1),
            "remaining_gap_to_ref": (ref - cal) / max(n, 1),
            "improved_count": improved,
            "regressed_count": regressed,
            "regression_rate": regressed / max(n, 1),
            "passed": int((cal - a0) / max(n, 1) >= 0.05 and regressed / max(n, 1) <= 0.10),
        })
    all_row = rows[0]
    test_row = next(r for r in rows if r["split"] == "test")
    rows.append({
        "split": "integration_decision",
        "event_count": total,
        "a0_top1": "",
        "cal_top1": "",
        "ref_top1": "",
        "cal_delta_vs_a0": "",
        "remaining_gap_to_ref": "",
        "improved_count": "",
        "regressed_count": "",
        "regression_rate": "",
        "passed": int(int(all_row["passed"]) == 1 and int(test_row["passed"]) == 1),
    })
    return rows


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ext2_rows = read_csv(Path(args.ext2_dir) / "stage_EXT2_event_delta_v1.csv")
    difficulty = read_csv(Path(args.ext1_dir) / "stage_EXT1_event_difficulty_audit_v1.csv")
    delta_rows = event_delta(ext2_rows, difficulty)
    gap_rows = summarize_group(delta_rows, "gap_bin")
    difficulty_summary = summarize_group(delta_rows, "difficulty_level")
    category_summary = summarize_group(delta_rows, "category_id")
    category_summary.sort(key=lambda r: (r["regressed_count"], -r["cal_delta_vs_a0"]), reverse=True)
    regression_rows = [r for r in delta_rows if r["delta_class"] == "regressed"]
    gate_rows = integration_gate(delta_rows)
    write_csv(out / f"stage_EXT3_variant_pairwise_delta_{args.artifact_version}.csv", delta_rows)
    write_csv(out / f"stage_EXT3_gap_bin_summary_{args.artifact_version}.csv", gap_rows)
    write_csv(out / f"stage_EXT3_difficulty_summary_{args.artifact_version}.csv", difficulty_summary)
    write_csv(out / f"stage_EXT3_category_regression_summary_{args.artifact_version}.csv", category_summary)
    write_csv(out / f"stage_EXT3_regression_events_{args.artifact_version}.csv", regression_rows)
    write_csv(out / f"stage_EXT3_integration_gate_{args.artifact_version}.csv", gate_rows)
    total = len(delta_rows)
    a0 = sum(int(r["a0_top1"]) for r in delta_rows)
    cal = sum(int(r["cal_top1"]) for r in delta_rows)
    ref = sum(int(r["ref_top1"]) for r in delta_rows)
    improved = sum(1 for r in delta_rows if r["delta_class"] == "improved")
    regressed = len(regression_rows)
    unchanged_failure = sum(1 for r in delta_rows if r["delta_class"] == "unchanged_failure")
    gate_passed = int(next(r for r in gate_rows if r["split"] == "integration_decision")["passed"])
    reg_categories = Counter(r["category_id"] for r in regression_rows)
    compact = {
        "stage": "EXT-3",
        "event_count": total,
        "a0_top1": a0 / max(total, 1),
        "calibrated_variant": CAL,
        "calibrated_top1": cal / max(total, 1),
        "support_reference_top1": ref / max(total, 1),
        "calibrated_delta_vs_a0": (cal - a0) / max(total, 1),
        "remaining_gap_to_reference": (ref - cal) / max(total, 1),
        "improved_count": improved,
        "regressed_count": regressed,
        "unchanged_failure_count": unchanged_failure,
        "regression_rate": regressed / max(total, 1),
        "integration_gate_passed": gate_passed,
        "top_regression_categories": dict(reg_categories.most_common(10)),
        "manual_lasot_pixels_needed_now": 0,
        "next_recommendation": "calibration is robust enough for an isolated external geometry branch, but do not merge into main NOPS until synthetic regression and full-pixel validation are checked" if gate_passed else "do not integrate calibration; inspect regression categories and scoring conditions first",
    }
    report = "\n".join([
        "# Stage EXT-3 Geometry Calibration Robustness",
        "",
        "## Scope",
        "",
        "Robustness audit for EXT-2 trajectory-heavy geometry calibration. This does not modify main NOPS and does not use target identity in scoring.",
        "",
        "## Verdict",
        "",
        compact["next_recommendation"],
        "",
        "## Compact",
        "",
        "```json",
        json.dumps(compact, indent=2, ensure_ascii=False),
        "```",
    ]) + "\n"
    (out / f"stage_EXT3_compact_for_gpt_{args.artifact_version}.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"stage_EXT3_report_{args.artifact_version}.md").write_text(report, encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()

