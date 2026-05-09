from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ext1_utils import write_csv


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"missing": 1, "path": str(p)}
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def i(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def evidence_rows(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ext1 = data["ext1"]
    ext3 = data["ext3"]
    syn = data["synreg1"]
    geo = data["geobranch1"]
    ext5 = data["ext5"]
    ext5c = data["ext5c"]
    ext6 = data["ext6"]
    ext7 = data["ext7"]
    ext12 = data.get("ext12", {})
    return [
        {
            "stage": "EXT-1",
            "claim": "external annotation geometry benchmark is usable",
            "event_count": ext1.get("valid_event_count", ""),
            "primary_metric": "nops_passive_top1",
            "value": ext1.get("nops_passive_top1", ""),
            "supporting_value": ext1.get("best_baseline_top1", ""),
            "decision": "external_geometry_problem_confirmed",
        },
        {
            "stage": "EXT-3",
            "claim": "trajectory-heavy geometry calibration helps external annotations",
            "event_count": ext3.get("valid_event_count", ext1.get("valid_event_count", "")),
            "primary_metric": "calibrated_top1",
            "value": ext3.get("calibrated_top1", ""),
            "supporting_value": ext3.get("calibrated_delta_vs_a0", ""),
            "decision": "valid_as_external_geometry_calibration",
        },
        {
            "stage": "SYN-REG-1",
            "claim": "external geometry calibration cannot merge into synthetic NOPS",
            "event_count": "",
            "primary_metric": "synthetic_regression_passed",
            "value": syn.get("synthetic_regression_passed", ""),
            "supporting_value": syn.get("best_calibration_focus_success_count", ""),
            "decision": "keep_isolated_not_main_merge",
        },
        {
            "stage": "GEO-BRANCH-1",
            "claim": "external geometry branch is valid as isolated profile",
            "event_count": ext1.get("valid_event_count", ""),
            "primary_metric": "external_top1_delta",
            "value": geo.get("external_top1_delta", ""),
            "supporting_value": geo.get("safe_external_geometry_branch", ""),
            "decision": "keep_external_geometry_branch",
        },
        {
            "stage": "EXT-5",
            "claim": "multi-category raw appearance is not integration-ready",
            "event_count": ext5.get("pixel_ready_events", ""),
            "primary_metric": "appearance_controls_passed",
            "value": ext5.get("appearance_controls_passed", ""),
            "supporting_value": ext5.get("external_branch_plus_appearance_best_top1", ""),
            "decision": "diagnostic_only",
        },
        {
            "stage": "EXT-5C",
            "claim": "raw appearance gains fail real shuffle controls",
            "event_count": ext5c.get("num_events", ""),
            "primary_metric": "appearance_controls_passed",
            "value": ext5c.get("appearance_controls_passed", ""),
            "supporting_value": ext5c.get("category_shuffled_external_appearance_gain", ""),
            "decision": "do_not_integrate_raw_appearance",
        },
        {
            "stage": "EXT-6",
            "claim": "handcrafted strong descriptor status depends on controls after full-pixel expansion",
            "event_count": ext6.get("num_events", ""),
            "primary_metric": "strong_controls_passed",
            "value": ext6.get("strong_controls_passed", ""),
            "supporting_value": ext6.get("strong_external_gain", ""),
            "decision": (
                "split_gate_failed_do_not_integrate"
                if ext12 and not int(float(ext12.get("strong_auxiliary_split_gate_passed", 0) or 0))
                else "candidate_external_auxiliary_requires_split_and_synthetic_guard"
                if int(float(ext6.get("strong_descriptor_safe_for_external_branch", 0) or 0))
                else "do_not_integrate_handcrafted_descriptor"
            ),
        },
        {
            "stage": "EXT-12",
            "claim": "handcrafted strong descriptor external auxiliary must pass split gate",
            "event_count": ext12.get("test_num_events", ""),
            "primary_metric": "strong_auxiliary_split_gate_passed",
            "value": ext12.get("strong_auxiliary_split_gate_passed", ""),
            "supporting_value": ext12.get("test_selected_delta_vs_external_branch", ""),
            "decision": "do_not_integrate_strong_descriptor_auxiliary" if ext12 else "not_run",
        },
        {
            "stage": "EXT-7",
            "claim": "frozen embedding is useful as diagnostic baseline but not fusion signal",
            "event_count": ext7.get("num_events", ""),
            "primary_metric": "embedding_controls_passed",
            "value": ext7.get("embedding_controls_passed", ""),
            "supporting_value": ext7.get("significance_passed", ""),
            "decision": "external_baseline_only",
        },
    ]


def integration_matrix(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "method_or_branch": "main_nops_synthetic_mechanism_chain",
            "safe_for_main_merge": 1,
            "safe_as_external_branch": 0,
            "safe_as_diagnostic_baseline": 1,
            "reason": "baseline mechanism chain remains protected; no external calibration has passed synthetic merge gates",
        },
        {
            "method_or_branch": "external_geometry_branch_v1",
            "safe_for_main_merge": 0,
            "safe_as_external_branch": 1,
            "safe_as_diagnostic_baseline": 1,
            "reason": "improves external geometry annotations but fails SYN-REG-1 synthetic focus/anchor regression",
        },
        {
            "method_or_branch": "raw_crop_appearance",
            "safe_for_main_merge": 0,
            "safe_as_external_branch": 0,
            "safe_as_diagnostic_baseline": 1,
            "reason": "EXT-5C controls fail; category-shuffled gain matches or exceeds real gain",
        },
        {
            "method_or_branch": "handcrafted_strong_descriptor",
            "safe_for_main_merge": 0,
            "safe_as_external_branch": int(float(data["ext6"].get("strong_descriptor_safe_for_external_branch", 0) or 0)),
            "safe_as_diagnostic_baseline": 1,
            "reason": (
                "EXT-6 controls pass on expanded subset, but branch still needs split/significance and synthetic guard"
                if int(float(data["ext6"].get("strong_descriptor_safe_for_external_branch", 0) or 0))
                else "EXT-6 controls fail; shuffled gain can exceed real gain"
            ),
        },
        {
            "method_or_branch": "frozen_resnet18_embedding",
            "safe_for_main_merge": 0,
            "safe_as_external_branch": 0,
            "safe_as_diagnostic_baseline": 1,
            "reason": "EXT-7 extraction works, but controls/significance fail and external branch fusion regresses",
        },
    ]


def method_status(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ext5 = data["ext5"]
    ext6 = data["ext6"]
    ext7 = data["ext7"]
    return [
        {
            "method": "strong_descriptor_external_auxiliary_split_gate",
            "events": data.get("ext12", {}).get("test_num_events", ""),
            "top1": data.get("ext12", {}).get("test_selected_top1", ""),
            "controls_passed": data.get("ext12", {}).get("strong_auxiliary_split_gate_passed", ""),
            "status": "rejected_for_integration",
        },
        {
            "method": "geometry_passive_full_pixel",
            "events": ext5.get("pixel_ready_events", ""),
            "top1": ext5.get("geometry_passive_top1", ""),
            "controls_passed": "not_applicable",
            "status": "baseline",
        },
        {
            "method": "external_trajectory_branch_full_pixel",
            "events": ext5.get("pixel_ready_events", ""),
            "top1": ext5.get("external_branch_top1", ""),
            "controls_passed": "not_applicable",
            "status": "best_current_external_full_pixel_geometry",
        },
        {
            "method": "raw_appearance_fusion",
            "events": ext5.get("pixel_ready_events", ""),
            "top1": ext5.get("external_branch_plus_appearance_best_top1", ""),
            "controls_passed": data["ext5c"].get("appearance_controls_passed", ""),
            "status": "rejected_for_integration",
        },
        {
            "method": "handcrafted_strong_descriptor_fusion",
            "events": ext6.get("num_events", ""),
            "top1": ext6.get("external_branch_plus_strong_best_top1", ""),
            "controls_passed": ext6.get("strong_controls_passed", ""),
            "status": "rejected_for_integration",
        },
        {
            "method": "frozen_resnet18_embedding_nn",
            "events": ext7.get("num_events", ""),
            "top1": ext7.get("embedding_nn_top1", ""),
            "controls_passed": ext7.get("embedding_controls_passed", ""),
            "status": "diagnostic_external_baseline",
        },
        {
            "method": "frozen_resnet18_embedding_fusion",
            "events": ext7.get("num_events", ""),
            "top1": ext7.get("external_branch_plus_embedding_best_top1", ""),
            "controls_passed": ext7.get("embedding_controls_passed", ""),
            "status": "rejected_for_integration",
        },
    ]


def next_action_gate(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ext5 = data["ext5"]
    ext7 = data["ext7"]
    pixel_events = i(ext5.get("pixel_ready_events"))
    embedding_controls = i(ext7.get("embedding_controls_passed"))
    return [
        {
            "candidate_next_action": "merge_external_geometry_into_main_nops",
            "allowed": 0,
            "reason": "SYN-REG-1 failed; would break internal focus/anchor path",
        },
        {
            "candidate_next_action": "integrate_raw_or_handcrafted_appearance",
            "allowed": 0,
            "reason": "EXT-5C/EXT-6 controls failed",
        },
        {
            "candidate_next_action": "integrate_frozen_embedding",
            "allowed": 0,
            "reason": "EXT-7 controls/significance failed and pretrained embedding cannot be main no-pretrain method",
        },
        {
            "candidate_next_action": "expand_full_pixel_categories",
            "allowed": int(pixel_events < 500),
            "reason": "current full-pixel subset has 234 events; larger pixel subset needed before paper-level conclusion",
        },
        {
            "candidate_next_action": "event_conditioned_geometry_analysis",
            "allowed": 1,
            "reason": "external failures are geometry/trajectory dominated; analyze gap/category/distractor regimes without adding appearance",
        },
        {
            "candidate_next_action": "try_more_embedding_fusion_weights",
            "allowed": 0 if embedding_controls == 0 else 1,
            "reason": "not useful while controls/significance fail",
        },
    ]


def main() -> None:
    out_dir = ROOT / "results" / "ext8"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "ext1": read_json(ROOT / "results" / "ext1" / "stage_EXT1_compact_for_gpt_v1.json"),
        "ext1a": read_json(ROOT / "results" / "ext1a" / "stage_EXT1A_compact_for_gpt_v1.json"),
        "ext2": read_json(ROOT / "results" / "ext2" / "stage_EXT2_compact_for_gpt_v1.json"),
        "ext3": read_json(ROOT / "results" / "ext3" / "stage_EXT3_compact_for_gpt_v1.json"),
        "synreg1": read_json(ROOT / "results" / "synreg1" / "stage_SYNREG1_compact_for_gpt_v1.json"),
        "geobranch1": read_json(ROOT / "results" / "geobranch1" / "stage_GEOBRANCH1_compact_for_gpt_v1.json"),
        "ext5": read_json(ROOT / "results" / "ext5" / "stage_EXT5_compact_for_gpt_v1.json"),
        "ext5c": read_json(ROOT / "results" / "ext5c" / "stage_EXT5C_compact_for_gpt_v1.json"),
        "ext6": read_json(ROOT / "results" / "ext6" / "stage_EXT6_compact_for_gpt_v1.json"),
        "ext7": read_json(ROOT / "results" / "ext7" / "stage_EXT7_compact_for_gpt_v1.json"),
        "ext12": read_json(ROOT / "results" / "ext12" / "stage_EXT12_compact_for_gpt_v1.json"),
    }
    evidence = evidence_rows(data)
    matrix = integration_matrix(data)
    status = method_status(data)
    gates = next_action_gate(data)
    compact = {
        "stage": "EXT-8",
        "external_geometry_branch_status": "valid_isolated_not_main_merge",
        "main_nops_merge_allowed": 0,
        "appearance_integration_allowed": 0,
        "embedding_integration_allowed": 0,
        "best_external_annotation_top1": data["geobranch1"].get("external_branch_top1"),
        "best_full_pixel_top1": data["ext5"].get("external_branch_top1"),
        "full_pixel_event_count": data["ext5"].get("pixel_ready_events"),
        "full_pixel_category_count": data["ext5"].get("num_categories"),
        "raw_appearance_controls_passed": data["ext5c"].get("appearance_controls_passed"),
        "strong_descriptor_controls_passed": data["ext6"].get("strong_controls_passed"),
        "strong_descriptor_external_auxiliary_allowed": data["ext12"].get("safe_for_external_auxiliary", 0),
        "strong_descriptor_split_gate_passed": data["ext12"].get("strong_auxiliary_split_gate_passed", 0),
        "frozen_embedding_controls_passed": data["ext7"].get("embedding_controls_passed"),
        "frozen_embedding_significance_passed": data["ext7"].get("significance_passed"),
        "recommended_next_stage": (
            "download target-500 categories or keep isolated all-external geometry branch; do not run more appearance/embedding fusion"
        ),
    }
    write_csv(out_dir / "stage_EXT8_evidence_table_v1.csv", evidence)
    write_csv(out_dir / "stage_EXT8_integration_matrix_v1.csv", matrix)
    write_csv(out_dir / "stage_EXT8_method_status_v1.csv", status)
    write_csv(out_dir / "stage_EXT8_next_action_gate_v1.csv", gates)
    write_json(out_dir / "stage_EXT8_compact_for_gpt_v1.json", compact)
    report = [
        "# EXT-8 External Evidence Synthesis",
        "",
        "## Decision",
        "",
        "- External geometry branch is valid only as an isolated external profile.",
        "- Main NOPS merge is not allowed because SYN-REG-1 failed.",
        "- Raw appearance and frozen embedding fusion are rejected for integration because controls/significance failed.",
        "- Handcrafted strong descriptor also failed the split gate, so it is not allowed as external auxiliary.",
        "- Frozen ResNet18 remains useful only as an external pretrained diagnostic baseline.",
        "",
        "## Current Best Numbers",
        "",
        f"- Annotation external geometry branch top1: `{compact['best_external_annotation_top1']}`",
        f"- Full-pixel external geometry branch top1: `{compact['best_full_pixel_top1']}`",
        f"- Full-pixel events/categories: `{compact['full_pixel_event_count']}` / `{compact['full_pixel_category_count']}`",
        "",
        "## Next",
        "",
        compact["recommended_next_stage"],
    ]
    (out_dir / "stage_EXT8_report_v1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
