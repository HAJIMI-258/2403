from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ext1_utils import write_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEO-BRANCH-1 isolated external geometry branch profile.")
    p.add_argument("--profile", default="configs/profiles/external_geometry_branch_v1.json")
    p.add_argument("--ext2", default="results/ext2/stage_EXT2_compact_for_gpt_v1.json")
    p.add_argument("--ext3", default="results/ext3/stage_EXT3_compact_for_gpt_v1.json")
    p.add_argument("--synreg1", default="results/synreg1/stage_SYNREG1_compact_for_gpt_v1.json")
    p.add_argument("--output-dir", default="results/geobranch1")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = load_json(args.profile)
    ext2 = load_json(args.ext2)
    ext3 = load_json(args.ext3)
    syn = load_json(args.synreg1)

    ext_delta = as_float(ext3.get("calibrated_delta_vs_a0", ext2.get("best_nops_calibrated_delta_vs_a0")))
    ext_top1 = as_float(ext3.get("calibrated_top1", ext2.get("best_nops_calibrated_global_top1")))
    ext_a0 = as_float(ext3.get("a0_top1", ext2.get("a0_nops_current_top1")))
    support_ref = as_float(ext3.get("support_reference_top1", ext2.get("support_trajectory_reference_top1")))
    syn_passed = as_int(syn.get("synthetic_regression_passed"))
    safe_external = as_int(syn.get("safe_external_geometry_branch"))
    safe_main = as_int(syn.get("safe_main_merge"))
    ext_gate = as_int(ext3.get("integration_gate_passed"))

    guard_rows = [
        {
            "guard_name": "profile_scope_is_external_geometry_only",
            "passed": int(profile.get("scope") == "external_annotation_oracle_proposal_geometry_only"),
            "evidence": profile.get("scope", ""),
            "required_for": "external_branch",
        },
        {
            "guard_name": "profile_forbids_main_merge",
            "passed": int("main NOPS merge" in profile.get("forbidden_use", [])),
            "evidence": "; ".join(profile.get("forbidden_use", [])),
            "required_for": "main_safety",
        },
        {
            "guard_name": "ext3_integration_gate_passed",
            "passed": ext_gate,
            "evidence": ext_gate,
            "required_for": "external_branch",
        },
        {
            "guard_name": "external_delta_positive",
            "passed": int(ext_delta > 0),
            "evidence": ext_delta,
            "required_for": "external_branch",
        },
        {
            "guard_name": "synreg_safe_external_branch",
            "passed": safe_external,
            "evidence": safe_external,
            "required_for": "external_branch",
        },
        {
            "guard_name": "synthetic_regression_passed",
            "passed": syn_passed,
            "evidence": syn_passed,
            "required_for": "main_merge",
        },
        {
            "guard_name": "synreg_safe_main_merge",
            "passed": safe_main,
            "evidence": safe_main,
            "required_for": "main_merge",
        },
        {
            "guard_name": "full_pixel_validation_available",
            "passed": 0,
            "evidence": "LaGOT annotations only; raw LaSOT pixels not validated",
            "required_for": "main_merge",
        },
    ]

    external_branch_valid = int(all(r["passed"] for r in guard_rows if r["required_for"] == "external_branch"))
    main_merge_valid = int(all(r["passed"] for r in guard_rows if r["required_for"] == "main_merge"))

    evidence_rows = [
        {
            "source_stage": "EXT-2",
            "metric_name": "a0_nops_current_top1",
            "metric_value": ext2.get("a0_nops_current_top1", ""),
            "interpretation": "current NOPS passive external geometry baseline",
        },
        {
            "source_stage": "EXT-2",
            "metric_name": "best_nops_calibrated_global_top1",
            "metric_value": ext2.get("best_nops_calibrated_global_top1", ""),
            "interpretation": "trajectory-heavy calibrated external geometry profile",
        },
        {
            "source_stage": "EXT-2",
            "metric_name": "recency_favors_wrong_when_trajectory_favors_target_count",
            "metric_value": ext2.get("recency_favors_wrong_when_trajectory_favors_target_count", ""),
            "interpretation": "external failure mechanism: recency over-selects similar distractors",
        },
        {
            "source_stage": "EXT-3",
            "metric_name": "calibrated_delta_vs_a0",
            "metric_value": ext_delta,
            "interpretation": "robust external geometry improvement",
        },
        {
            "source_stage": "EXT-3",
            "metric_name": "regression_rate",
            "metric_value": ext3.get("regression_rate", ""),
            "interpretation": "external regression cost under trajectory-heavy profile",
        },
        {
            "source_stage": "SYN-REG-1",
            "metric_name": "synthetic_regression_passed",
            "metric_value": syn_passed,
            "interpretation": "main NOPS merge blocked when false",
        },
        {
            "source_stage": "SYN-REG-1",
            "metric_name": "safe_external_geometry_branch",
            "metric_value": safe_external,
            "interpretation": "isolated external branch allowed when true",
        },
        {
            "source_stage": "SYN-REG-1",
            "metric_name": "safe_main_merge",
            "metric_value": safe_main,
            "interpretation": "main NOPS merge allowed only when true",
        },
    ]

    usage_rows = [
        {
            "use_case": "LaGOT annotation oracle-proposal geometry memory benchmark",
            "allowed": external_branch_valid,
            "reason": "external EXT-3 gate passed and SYN-REG allows isolated external branch",
        },
        {
            "use_case": "external geometry failure analysis",
            "allowed": external_branch_valid,
            "reason": "profile documents recency-vs-trajectory failure mechanism",
        },
        {
            "use_case": "main NOPS scoring merge",
            "allowed": main_merge_valid,
            "reason": "blocked by SYN-REG synthetic regression and missing full-pixel validation",
        },
        {
            "use_case": "attach/promotion decision",
            "allowed": 0,
            "reason": "profile is memory retrieval analysis only",
        },
        {
            "use_case": "full perception claim",
            "allowed": 0,
            "reason": "LaGOT annotations do not validate raw pixel perception or appearance descriptors",
        },
    ]

    profile_contract = {
        "stage": "GEO-BRANCH-1",
        "profile_name": profile.get("profile_name", "external_geometry_branch_v1"),
        "scope": profile.get("scope", "external_annotation_oracle_proposal_geometry_only"),
        "proposal_mode": profile.get("proposal_mode", "oracle_gt_box_memory_only"),
        "primary_variant": profile.get("primary_variant", "A2_trajectory_heavy"),
        "score_formula": profile.get("score_formula", {}),
        "allowed_use": profile.get("allowed_use", []),
        "forbidden_use": profile.get("forbidden_use", []),
        "guard_summary": {
            "external_branch_valid": external_branch_valid,
            "main_merge_valid": main_merge_valid,
            "synthetic_regression_passed": syn_passed,
            "safe_external_geometry_branch": safe_external,
            "safe_main_merge": safe_main,
            "full_pixel_validation_available": 0,
        },
    }

    compact = {
        "stage": "GEO-BRANCH-1",
        "profile_name": profile_contract["profile_name"],
        "branch_created": 1,
        "allowed_scope": profile_contract["scope"],
        "external_branch_valid": external_branch_valid,
        "external_a0_top1": ext_a0,
        "external_branch_top1": ext_top1,
        "external_support_reference_top1": support_ref,
        "external_top1_delta": ext_delta,
        "synthetic_regression_passed": syn_passed,
        "safe_external_geometry_branch": safe_external,
        "safe_main_merge": safe_main,
        "requires_lasot_pixels_for_full_validation": 1,
        "active_evidence_integrated": 0,
        "next_recommendation": (
            "use only as isolated external geometry profile; next do EXT-4 full-pixel appearance validation "
            "or keep profile quarantined"
        ),
    }

    report = f"""# GEO-BRANCH-1 External Geometry Branch

## Decision

The EXT-2 / EXT-3 trajectory-heavy calibration is packaged as an isolated external geometry branch.

It is not safe to merge into main NOPS.

## Evidence

- External A0 top1: `{ext_a0:.4f}`
- External trajectory-heavy top1: `{ext_top1:.4f}`
- External top1 delta: `{ext_delta:.4f}`
- Support-trajectory reference top1: `{support_ref:.4f}`
- EXT-3 integration gate: `{ext_gate}`
- SYN-REG synthetic regression passed: `{syn_passed}`
- SYN-REG safe external branch: `{safe_external}`
- SYN-REG safe main merge: `{safe_main}`

## Allowed Use

- LaGOT annotation oracle-proposal geometry-only memory benchmark.
- External geometry failure analysis.
- Isolated profile comparison.

## Forbidden Use

- Main NOPS scoring merge.
- Synthetic anchor/canonical/episodic path replacement.
- Active evidence integration.
- Attach / promotion decisions.
- Full perception claims.

## Next Step

Keep this branch quarantined. Use it for external annotation geometry analysis only.

Before any main merge, run full-pixel / appearance validation and pass synthetic regression.
"""

    write_json(out_dir / "stage_GEOBRANCH1_compact_for_gpt_v1.json", compact)
    write_json(out_dir / "stage_GEOBRANCH1_profile_contract_v1.json", profile_contract)
    write_csv(out_dir / "stage_GEOBRANCH1_guard_check_v1.csv", guard_rows)
    write_csv(out_dir / "stage_GEOBRANCH1_evidence_summary_v1.csv", evidence_rows)
    write_csv(out_dir / "stage_GEOBRANCH1_usage_decision_v1.csv", usage_rows)
    (out_dir / "stage_GEOBRANCH1_report_v1.md").write_text(report, encoding="utf-8")

    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
