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

from experiments.ext1_utils import write_csv


GEOM_BASE = "A0_nops_geometry_passive"
GEOM_APP = "A2_geometry_plus_appearance_w010"
EXT_BRANCH = "A4_external_trajectory_heavy"
EXT_APP = "A5_external_trajectory_plus_appearance_w010"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXT-4B appearance descriptor failure audit.")
    p.add_argument("--ext4a-dir", default="results/ext4a")
    p.add_argument("--output-dir", default="results/ext4b")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def delta_class(base_success: int, variant_success: int) -> str:
    if variant_success and not base_success:
        return "improved"
    if base_success and not variant_success:
        return "regressed"
    if base_success and variant_success:
        return "unchanged_success"
    return "unchanged_failure"


def compare_pair(
    by_event: dict[str, dict[str, dict[str, str]]],
    margin_by_event: dict[str, dict[str, str]],
    base: str,
    variant: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for event_id, variants in sorted(by_event.items()):
        if base not in variants or variant not in variants:
            continue
        b = variants[base]
        v = variants[variant]
        b_success = as_int(b.get("top1"))
        v_success = as_int(v.get("top1"))
        dc = delta_class(b_success, v_success)
        counts[dc] += 1
        margin = margin_by_event.get(event_id, {})
        rows.append({
            "event_id": event_id,
            "sequence_id": b.get("sequence_id", ""),
            "base_variant": base,
            "variant": variant,
            "base_top1": b.get("predicted_memory_id", ""),
            "variant_top1": v.get("predicted_memory_id", ""),
            "target_instance_id_eval_only": b.get("target_instance_id_eval_only", ""),
            "base_success": b_success,
            "variant_success": v_success,
            "delta_class": dc,
            "appearance_margin_target_minus_wrong": margin.get("appearance_margin_target_minus_wrong", ""),
            "appearance_margin_positive": margin.get("appearance_margin_positive", ""),
            "gap_length": b.get("gap_length", ""),
            "candidate_count": b.get("candidate_count", ""),
            "failure_interpretation": failure_interpretation(dc, margin),
        })
    return rows, counts


def failure_interpretation(delta: str, margin: dict[str, str]) -> str:
    m = as_float(margin.get("appearance_margin_target_minus_wrong", 0.0))
    if delta == "improved" and m > 0:
        return "appearance_rescued_with_positive_margin"
    if delta == "improved":
        return "appearance_rescued_despite_negative_pair_margin"
    if delta == "regressed" and m < 0:
        return "appearance_regressed_with_negative_margin"
    if delta == "regressed":
        return "appearance_regressed_despite_positive_margin"
    if delta == "unchanged_failure" and m < 0:
        return "appearance_not_discriminative_negative_margin"
    if delta == "unchanged_failure":
        return "appearance_margin_not_enough_to_change_top1"
    return "no_failure"


def main() -> None:
    args = parse_args()
    ext4a = Path(args.ext4a_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compact4a = read_json(ext4a / "stage_EXT4A_compact_for_gpt_v1.json")
    event_rows = read_csv(ext4a / "stage_EXT4A_event_results_v1.csv")
    margin_rows = read_csv(ext4a / "stage_EXT4A_appearance_margin_trace_v1.csv")
    summary_rows = read_csv(ext4a / "stage_EXT4A_ablation_summary_v1.csv")

    by_event: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in event_rows:
        by_event[row["event_id"]][row["variant"]] = row
    margin_by_event = {row["event_id"]: row for row in margin_rows}

    geom_rows, geom_counts = compare_pair(by_event, margin_by_event, GEOM_BASE, GEOM_APP)
    ext_rows, ext_counts = compare_pair(by_event, margin_by_event, EXT_BRANCH, EXT_APP)
    margin_values = [as_float(r.get("appearance_margin_target_minus_wrong")) for r in margin_rows]
    severe_negative = [r for r in margin_rows if as_float(r.get("appearance_margin_target_minus_wrong")) < -0.25]

    decision_rows = [
        {
            "decision_item": "appearance_as_main_branch",
            "allowed": 0,
            "reason": "appearance variants do not beat external trajectory-heavy geometry branch",
        },
        {
            "decision_item": "appearance_as_auxiliary_signal",
            "allowed": int(geom_counts["improved"] > geom_counts["regressed"]),
            "reason": "appearance improves current passive geometry on a small number of dog events",
        },
        {
            "decision_item": "main_nops_merge",
            "allowed": 0,
            "reason": "small dog subset only; mean appearance margin remains negative",
        },
        {
            "decision_item": "larger_multicategory_validation",
            "allowed": 1,
            "reason": "pixel path works and dog subset is insufficient for final external conclusion",
        },
    ]

    variant_top1 = {row["variant"]: as_float(row.get("global_top1")) for row in summary_rows}
    compact = {
        "stage": "EXT-4B",
        "source_stage": "EXT-4A",
        "num_events": len(by_event),
        "geometry_plus_appearance_improved": geom_counts["improved"],
        "geometry_plus_appearance_regressed": geom_counts["regressed"],
        "external_branch_plus_appearance_improved": ext_counts["improved"],
        "external_branch_plus_appearance_regressed": ext_counts["regressed"],
        "geometry_passive_top1": variant_top1.get(GEOM_BASE, 0.0),
        "geometry_plus_appearance_top1": variant_top1.get(GEOM_APP, 0.0),
        "external_branch_top1": variant_top1.get(EXT_BRANCH, 0.0),
        "external_branch_plus_appearance_top1": variant_top1.get(EXT_APP, 0.0),
        "appearance_margin_positive_rate": compact4a.get("appearance_margin_positive_rate", 0.0),
        "mean_appearance_margin": compact4a.get("mean_appearance_margin", 0.0),
        "severe_negative_margin_count": len(severe_negative),
        "appearance_safe_for_main_merge": 0,
        "next_recommendation": "run larger multi-category full-pixel validation; treat current crop appearance as auxiliary diagnostic only",
    }

    report = f"""# EXT-4B Appearance Failure Audit

## Result

Appearance is not ready as a main retrieval branch.

- Geometry passive top1: `{compact["geometry_passive_top1"]:.4f}`
- Geometry + appearance top1: `{compact["geometry_plus_appearance_top1"]:.4f}`
- External trajectory branch top1: `{compact["external_branch_top1"]:.4f}`
- External trajectory + appearance top1: `{compact["external_branch_plus_appearance_top1"]:.4f}`
- Geometry + appearance improved events: `{geom_counts["improved"]}`
- Geometry + appearance regressed events: `{geom_counts["regressed"]}`
- External branch + appearance improved events: `{ext_counts["improved"]}`
- External branch + appearance regressed events: `{ext_counts["regressed"]}`
- Mean appearance margin: `{compact["mean_appearance_margin"]:.6f}`
- Severe negative margin count: `{len(severe_negative)}`

## Interpretation

Raw crop appearance helps the weaker geometry-passive baseline in a few cases, but it weakens the stronger trajectory-heavy external branch. The mean target-vs-wrong appearance margin is still negative.

The correct next step is larger multi-category full-pixel validation and descriptor failure analysis, not main NOPS integration.
"""

    write_csv(out_dir / "stage_EXT4B_event_delta_vs_geometry_v1.csv", geom_rows)
    write_csv(out_dir / "stage_EXT4B_event_delta_vs_external_branch_v1.csv", ext_rows)
    write_csv(out_dir / "stage_EXT4B_appearance_margin_failure_audit_v1.csv", [
        {
            "event_id": r["event_id"],
            "sequence_id": r["sequence_id"],
            "appearance_margin_target_minus_wrong": r["appearance_margin_target_minus_wrong"],
            "appearance_margin_positive": r["appearance_margin_positive"],
            "severity": "severe_negative" if as_float(r["appearance_margin_target_minus_wrong"]) < -0.25 else "normal",
        }
        for r in margin_rows
    ])
    write_csv(out_dir / "stage_EXT4B_usage_decision_v1.csv", decision_rows)
    write_json(out_dir / "stage_EXT4B_compact_for_gpt_v1.json", compact)
    (out_dir / "stage_EXT4B_report_v1.md").write_text(report, encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
