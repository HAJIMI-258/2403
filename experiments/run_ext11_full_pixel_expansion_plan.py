from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ext1_utils import read_csv, write_csv


def as_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def category_rows() -> list[dict[str, Any]]:
    rows = read_csv(ROOT / "results" / "ext4" / "stage_EXT4_lasot_download_manifest_v1.csv")
    out = []
    for r in rows:
        events = as_int(r.get("lagot_event_count"))
        ready_events = events if as_int(r.get("local_pixel_ready_sequences")) > 0 else 0
        size_gb = as_float(r.get("hf_zip_size_gb"))
        missing_events = max(0, events - ready_events)
        out.append({
            "category": r.get("category", ""),
            "lagot_event_count": events,
            "current_pixel_ready_events": ready_events,
            "missing_event_count": missing_events,
            "hf_zip_name": r.get("hf_zip_name", ""),
            "hf_zip_size_gb": size_gb,
            "events_per_gb": (missing_events / size_gb) if size_gb > 0 and missing_events > 0 else 0.0,
            "already_downloaded": int(ready_events > 0),
            "download_command": r.get("download_command_execute", ""),
        })
    return out


def greedy_plan(rows: list[dict[str, Any]], target_total_events: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current = sum(as_int(r["current_pixel_ready_events"]) for r in rows)
    selected: list[dict[str, Any]] = []
    total_size = 0.0
    total_new_events = 0
    candidates = [r for r in rows if not as_int(r["already_downloaded"]) and as_int(r["missing_event_count"]) > 0 and as_float(r["hf_zip_size_gb"]) > 0]
    candidates.sort(key=lambda r: (as_float(r["events_per_gb"]), as_int(r["missing_event_count"])), reverse=True)
    for r in candidates:
        if current + total_new_events >= target_total_events:
            break
        selected.append(r)
        total_new_events += as_int(r["missing_event_count"])
        total_size += as_float(r["hf_zip_size_gb"])
    summary = {
        "target_total_events": target_total_events,
        "current_pixel_ready_events": current,
        "selected_category_count": len(selected),
        "new_event_count": total_new_events,
        "projected_total_events": current + total_new_events,
        "estimated_download_gb": total_size,
        "selected_categories": [r["category"] for r in selected],
        "combined_download_command": (
            "python scripts/download_lasot_hf_categories.py --categories "
            + ",".join(r["category"] for r in selected)
            + " --execute"
            if selected else ""
        ),
    }
    return selected, summary


def manual_instructions(plans: list[dict[str, Any]]) -> str:
    lines = [
        "# EXT-11 Full-pixel Expansion Instructions",
        "",
        "Current full-pixel subset has 234 events across 5 categories.",
        "Do not download the full LaSOT dataset yet. Use one of the staged plans below.",
        "",
    ]
    for p in plans:
        lines.extend([
            f"## Target {p['target_total_events']} events",
            "",
            f"- Categories: `{','.join(p['selected_categories'])}`",
            f"- Estimated new events: `{p['new_event_count']}`",
            f"- Estimated download size: `{p['estimated_download_gb']:.2f} GB`",
            f"- Projected total events: `{p['projected_total_events']}`",
            "",
            "```powershell",
            p["combined_download_command"],
            "```",
            "",
        ])
    lines.extend([
        "After download/extraction, rerun:",
        "",
        "```powershell",
        "python experiments\\run_ext4_full_pixel_readiness.py",
        "python experiments\\run_ext5_multicategory_full_pixel_validation.py",
        "python experiments\\run_ext5c_appearance_control_audit.py",
        "python experiments\\run_ext6_stronger_local_descriptor_validation.py",
        "python experiments\\run_ext7_frozen_embedding_baseline.py --embedding-model resnet18 --bootstrap-samples 300",
        "python experiments\\run_ext8_external_evidence_synthesis.py",
        "python experiments\\run_ext9_event_conditioned_geometry_analysis.py",
        "python experiments\\run_ext10_geometry_routing_split_gate.py",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    out_dir = ROOT / "results" / "ext11"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = category_rows()
    ranking = sorted(rows, key=lambda r: (as_int(r["already_downloaded"]), as_float(r["events_per_gb"])), reverse=True)
    candidate_ranking = [r for r in sorted(rows, key=lambda r: as_float(r["events_per_gb"]), reverse=True) if not as_int(r["already_downloaded"]) and as_int(r["missing_event_count"]) > 0]
    selected_rows = []
    plan_summaries = []
    for target in (400, 500, 750):
        selected, summary = greedy_plan(rows, target)
        plan_summaries.append(summary)
        for idx, r in enumerate(selected, start=1):
            rr = dict(r)
            rr.update({
                "target_total_events": target,
                "selection_rank": idx,
                "projected_total_after_this": sum(as_int(x["current_pixel_ready_events"]) for x in rows) + sum(as_int(x["missing_event_count"]) for x in selected[:idx]),
            })
            selected_rows.append(rr)
    current_events = sum(as_int(r["current_pixel_ready_events"]) for r in rows)
    compact = {
        "stage": "EXT-11",
        "current_pixel_ready_events": current_events,
        "current_pixel_ready_categories": sum(1 for r in rows if as_int(r["already_downloaded"])),
        "target_500_plan_categories": next(p["selected_categories"] for p in plan_summaries if p["target_total_events"] == 500),
        "target_500_estimated_download_gb": next(p["estimated_download_gb"] for p in plan_summaries if p["target_total_events"] == 500),
        "target_500_projected_total_events": next(p["projected_total_events"] for p in plan_summaries if p["target_total_events"] == 500),
        "download_should_be_manual_confirmed": 1,
        "next_recommendation": "download target_500_plan categories if storage/network budget allows; otherwise target_400 plan is the smaller expansion",
    }
    write_csv(out_dir / "stage_EXT11_category_download_ranking_v1.csv", ranking)
    write_csv(out_dir / "stage_EXT11_missing_category_candidates_v1.csv", candidate_ranking)
    write_csv(out_dir / "stage_EXT11_selected_download_plans_v1.csv", selected_rows)
    write_json(out_dir / "stage_EXT11_plan_summary_v1.json", {"plans": plan_summaries})
    write_json(out_dir / "stage_EXT11_compact_for_gpt_v1.json", compact)
    (out_dir / "stage_EXT11_manual_download_instructions_v1.md").write_text(manual_instructions(plan_summaries), encoding="utf-8")
    report = [
        "# EXT-11 Full-pixel Expansion Plan",
        "",
        f"- Current pixel-ready events: `{current_events}`",
        f"- Target 500 categories: `{','.join(compact['target_500_plan_categories'])}`",
        f"- Target 500 estimated size: `{compact['target_500_estimated_download_gb']:.2f} GB`",
        f"- Target 500 projected events: `{compact['target_500_projected_total_events']}`",
        "",
        "No model changes are made in this stage.",
    ]
    (out_dir / "stage_EXT11_report_v1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
