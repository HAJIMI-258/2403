from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ext1_utils import read_csv, write_csv


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


def target500_rows(ext11_plan: dict[str, Any], current_events: int) -> list[dict[str, Any]]:
    plan = next((p for p in ext11_plan.get("plans", []) if int(p.get("target_total_events", 0)) == 500), {})
    selected = plan.get("selected_categories", [])
    candidates = {r["category"]: r for r in read_csv(ROOT / "results" / "ext11" / "stage_EXT11_missing_category_candidates_v1.csv")}
    rows = []
    running_total = current_events
    for cat in selected:
        c = candidates.get(cat, {})
        running_total += int(float(c.get("missing_event_count", 0) or 0))
        rows.append({
            "category": cat,
            "estimated_download_gb": c.get("hf_zip_size_gb", ""),
            "projected_new_events": c.get("missing_event_count", ""),
            "projected_total_events_after_download": running_total,
            "download_command": c.get("download_command", ""),
            "requires_user_confirmation": 1,
        })
    return rows


def current_evidence(ext5: dict[str, Any], ext5c: dict[str, Any], ext6: dict[str, Any], ext7: dict[str, Any], ext8: dict[str, Any], ext10: dict[str, Any], ext12: dict[str, Any]) -> list[dict[str, Any]]:
    raw_controls_passed = int(ext5c.get("appearance_controls_passed", ext5.get("appearance_controls_passed", 0)) or 0)
    raw_decision = "diagnostic_only" if raw_controls_passed else "rejected_controls_failed"
    raw_reason = (
        "raw appearance controls pass after EXT-5C, but EXT-8 still blocks integration; keep diagnostic only"
        if raw_controls_passed
        else "raw appearance controls fail"
    )
    return [
        {
            "stage": "EXT-5",
            "event_count": ext5.get("pixel_ready_events"),
            "category_count": ext5.get("num_categories"),
            "method_or_branch": "main_nops_passive_geometry",
            "top1": ext5.get("geometry_passive_top1"),
            "top3": "",
            "top5": "",
            "control_passed": "not_applicable",
            "split_gate_passed": "not_applicable",
            "safe_for_external_branch": 0,
            "safe_for_main_merge": 1,
            "decision": "baseline_only",
            "reason": "main NOPS is protected; no external branch merged",
        },
        {
            "stage": "EXT-5/EXT-8",
            "event_count": ext5.get("pixel_ready_events"),
            "category_count": ext5.get("num_categories"),
            "method_or_branch": "external_geometry_branch",
            "top1": ext5.get("external_branch_top1"),
            "top3": "",
            "top5": "",
            "control_passed": "not_applicable",
            "split_gate_passed": "not_applicable",
            "safe_for_external_branch": 1,
            "safe_for_main_merge": 0,
            "decision": "isolated_external_branch_only",
            "reason": "external branch improves full-pixel geometry but SYN-REG-1 blocks main merge",
        },
        {
            "stage": "EXT-5C",
            "event_count": ext5.get("pixel_ready_events"),
            "category_count": ext5.get("num_categories"),
            "method_or_branch": "raw_appearance",
            "top1": ext5.get("external_branch_plus_appearance_best_top1"),
            "top3": "",
            "top5": "",
            "control_passed": raw_controls_passed,
            "split_gate_passed": "not_run",
            "safe_for_external_branch": 0,
            "safe_for_main_merge": 0,
            "decision": raw_decision,
            "reason": raw_reason,
        },
        {
            "stage": "EXT-6/EXT-12",
            "event_count": ext6.get("num_events"),
            "category_count": ext5.get("num_categories"),
            "method_or_branch": "handcrafted_strong_descriptor",
            "top1": ext6.get("external_branch_plus_strong_best_top1"),
            "top3": "",
            "top5": "",
            "control_passed": ext6.get("strong_controls_passed"),
            "split_gate_passed": ext12.get("strong_auxiliary_split_gate_passed"),
            "safe_for_external_branch": 0,
            "safe_for_main_merge": 0,
            "decision": "diagnostic_only",
            "reason": "aggregate controls pass but EXT-12 sequence-level split gate fails",
        },
        {
            "stage": "EXT-7",
            "event_count": ext7.get("num_events"),
            "category_count": ext5.get("num_categories"),
            "method_or_branch": "frozen_resnet18_embedding",
            "top1": ext7.get("external_branch_plus_embedding_best_top1"),
            "top3": "",
            "top5": "",
            "control_passed": ext7.get("embedding_controls_passed"),
            "split_gate_passed": "not_run",
            "safe_for_external_branch": 0,
            "safe_for_main_merge": 0,
            "decision": "external_baseline_only",
            "reason": "controls and significance fail; pretrained embedding cannot be no-pretrain main method",
        },
        {
            "stage": "EXT-10",
            "event_count": ext10.get("num_events"),
            "category_count": ext5.get("num_categories"),
            "method_or_branch": "event_conditioned_routing",
            "top1": ext10.get("test_selected_gate_top1"),
            "top3": "",
            "top5": "",
            "control_passed": "not_applicable",
            "split_gate_passed": ext10.get("routing_integration_ready"),
            "safe_for_external_branch": 0,
            "safe_for_main_merge": 0,
            "decision": "rejected_for_now",
            "reason": "split gate fails; selected gate does not beat all-external on test",
        },
    ]


def decision_matrix() -> list[dict[str, Any]]:
    return [
        {
            "component": "external_geometry_branch",
            "can_use_as_main_nops": 0,
            "can_use_as_isolated_external_branch": 1,
            "can_use_as_diagnostic_baseline": 1,
            "requires_more_data": 1,
            "requires_synthetic_regression": 1,
            "requires_controls": 0,
            "decision": "keep isolated profile; do not main-merge",
        },
        {
            "component": "raw_appearance",
            "can_use_as_main_nops": 0,
            "can_use_as_isolated_external_branch": 0,
            "can_use_as_diagnostic_baseline": 1,
            "requires_more_data": 1,
            "requires_synthetic_regression": 0,
            "requires_controls": 1,
            "decision": "diagnostic only; even when controls pass, EXT-8 blocks integration",
        },
        {
            "component": "handcrafted_strong_descriptor",
            "can_use_as_main_nops": 0,
            "can_use_as_isolated_external_branch": 0,
            "can_use_as_diagnostic_baseline": 1,
            "requires_more_data": 1,
            "requires_synthetic_regression": 0,
            "requires_controls": 1,
            "decision": "diagnostic only; split gate failed on EXT-12",
        },
        {
            "component": "frozen_resnet18_embedding",
            "can_use_as_main_nops": 0,
            "can_use_as_isolated_external_branch": 0,
            "can_use_as_diagnostic_baseline": 1,
            "requires_more_data": 1,
            "requires_synthetic_regression": 0,
            "requires_controls": 1,
            "decision": "external pretrained baseline only",
        },
        {
            "component": "event_conditioned_routing",
            "can_use_as_main_nops": 0,
            "can_use_as_isolated_external_branch": 0,
            "can_use_as_diagnostic_baseline": 1,
            "requires_more_data": 1,
            "requires_synthetic_regression": 1,
            "requires_controls": 0,
            "decision": "rejected for now; split gate failed / no routing generalization",
        },
    ]


def frozen_protocol_md() -> str:
    return """# EXT-13 Frozen External Evaluation Protocol

This protocol freezes the current full-pixel evaluation state.

Allowed:
- Rerun existing EXT-4, EXT-5, EXT-5C, EXT-6, EXT-7, EXT-9, EXT-10, EXT-12, EXT-8 scripts unchanged.
- Add more LaSOT categories only through the EXT-11/EXT-13 download plan.
- Report oracle-proposal memory-only results as oracle-proposal memory-only results.

Forbidden:
- Do not add new appearance / descriptor / embedding fusion variants inside the frozen rerun.
- Do not tune event-conditioned routing rules on the test split.
- Do not merge external geometry calibration into main NOPS.
- Do not claim full-perception results.
- Do not use target identity, GT instance id, or future frames in online scoring.

If a new method is needed, create a new stage after EXT-13 and keep it separate from the frozen protocol.
"""


def rerun_protocol_md() -> str:
    return """# EXT-13 Rerun Protocol After Target-500 Download

After downloading `guitar, car, drone`, run exactly:

```powershell
python experiments\\run_ext4_full_pixel_readiness.py
python experiments\\run_ext5_multicategory_full_pixel_validation.py
python experiments\\run_ext5c_appearance_control_audit.py
python experiments\\run_ext6_stronger_local_descriptor_validation.py
python experiments\\run_ext7_frozen_embedding_baseline.py --embedding-model resnet18 --bootstrap-samples 300
python experiments\\run_ext9_event_conditioned_geometry_analysis.py
python experiments\\run_ext10_geometry_routing_split_gate.py
python experiments\\run_ext12_strong_descriptor_split_gate.py
python experiments\\run_ext8_external_evidence_synthesis.py
python experiments\\run_ext13_freeze_external_protocol.py
```

No variants or scoring rules may be added during this rerun.
If a new method is required, open EXT-14.
"""


def main() -> None:
    out_dir = ROOT / "results" / "ext13"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext5 = read_json(ROOT / "results" / "ext5" / "stage_EXT5_compact_for_gpt_v1.json")
    ext5c = read_json(ROOT / "results" / "ext5c" / "stage_EXT5C_compact_for_gpt_v1.json")
    ext6 = read_json(ROOT / "results" / "ext6" / "stage_EXT6_compact_for_gpt_v1.json")
    ext7 = read_json(ROOT / "results" / "ext7" / "stage_EXT7_compact_for_gpt_v1.json")
    ext8 = read_json(ROOT / "results" / "ext8" / "stage_EXT8_compact_for_gpt_v1.json")
    ext10 = read_json(ROOT / "results" / "ext10" / "stage_EXT10_compact_for_gpt_v1.json")
    ext11 = read_json(ROOT / "results" / "ext11" / "stage_EXT11_plan_summary_v1.json")
    ext12 = read_json(ROOT / "results" / "ext12" / "stage_EXT12_compact_for_gpt_v1.json")
    current_events = int(f(ext5.get("pixel_ready_events"), 0))
    download_rows = target500_rows(ext11, current_events)
    target500_cats = [r["category"] for r in download_rows]
    raw_controls_passed = int(ext5c.get("appearance_controls_passed", ext5.get("appearance_controls_passed", 0)) or 0)
    target500_reached = current_events >= 500
    raw_status = "diagnostic_only_controls_passed_no_integration" if raw_controls_passed else "rejected_controls_failed"
    recommended_next_action = (
        "keep_current_504_event_evidence_and_write_report"
        if target500_reached
        else "await_user_download_confirmation"
    )
    compact = {
        "stage": "EXT-13",
        "current_pixel_ready_events": current_events,
        "current_categories": ext5.get("num_categories"),
        "external_geometry_branch_status": "isolated_external_branch_only",
        "raw_appearance_status": raw_status,
        "strong_descriptor_status": "rejected_split_gate_failed",
        "embedding_status": "external_baseline_only_controls_failed",
        "routing_status": "rejected_split_gate_failed",
        "target500_categories": target500_cats,
        "target500_estimated_download_gb": sum(f(r["estimated_download_gb"]) for r in download_rows),
        "target500_projected_events": download_rows[-1]["projected_total_events_after_download"] if download_rows else ext5.get("pixel_ready_events"),
        "download_requires_user_confirmation": 0 if target500_reached else 1,
        "recommended_next_action": recommended_next_action,
    }
    write_csv(out_dir / "stage_EXT13_current_evidence_table_v1.csv", current_evidence(ext5, ext5c, ext6, ext7, ext8, ext10, ext12))
    write_csv(out_dir / "stage_EXT13_integration_decision_matrix_v1.csv", decision_matrix())
    write_csv(out_dir / "stage_EXT13_target500_download_plan_v1.csv", download_rows)
    (out_dir / "stage_EXT13_frozen_protocol_v1.md").write_text(frozen_protocol_md(), encoding="utf-8")
    (out_dir / "stage_EXT13_rerun_protocol_after_download_v1.md").write_text(rerun_protocol_md(), encoding="utf-8")
    write_json(out_dir / "stage_EXT13_compact_for_gpt_v1.json", compact)
    report = [
        "# EXT-13 Full-pixel Expansion Decision and Frozen Evaluation Protocol",
        "",
        "## Current State",
        "",
        f"- Current events/categories: `{compact['current_pixel_ready_events']}` / `{compact['current_categories']}`",
        f"- External geometry branch: `{compact['external_geometry_branch_status']}`",
        f"- Raw appearance: `{compact['raw_appearance_status']}`",
        f"- Strong descriptor: `{compact['strong_descriptor_status']}`",
        f"- Frozen embedding: `{compact['embedding_status']}`",
        f"- Routing: `{compact['routing_status']}`",
        "",
        "## Target-500 Plan",
        "",
        f"- Categories: `{','.join(target500_cats)}`",
        f"- Estimated size: `{compact['target500_estimated_download_gb']:.2f} GB`",
        f"- Projected events: `{compact['target500_projected_events']}`",
        "",
        "## Decision",
        "",
        (
            "Target-500 is already reached. Keep this frozen 504-event evidence and write the stage report; do not run new model experiments under the frozen protocol."
            if target500_reached
            else "Await user confirmation before downloading target-500 categories. Do not run new model experiments under the frozen protocol."
        ),
    ]
    (out_dir / "stage_EXT13_report_v1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
