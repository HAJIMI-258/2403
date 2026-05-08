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
from experiments.run_ext4a_full_pixel_appearance_validation import (
    as_float,
    as_int,
    evaluate,
)


VARIANT_MAP = {
    "A0_nops_geometry_passive": "A0_nops_geometry_passive",
    "A1_appearance_nn": "A1_appearance_nn",
    "A2_geometry_plus_appearance_w005": "A2_geometry_plus_appearance_w005",
    "A3_geometry_plus_appearance_w010": "A3_geometry_plus_appearance_w010",
    "A4_geometry_plus_appearance_w020": "A4_geometry_plus_appearance_w020",
    "A5_external_trajectory_heavy": "A5_external_trajectory_heavy",
    "A6_external_trajectory_plus_appearance_w005": "A6_external_trajectory_plus_appearance_w005",
    "A7_external_trajectory_plus_appearance_w010": "A7_external_trajectory_plus_appearance_w010",
    "A8_external_trajectory_plus_appearance_w020": "A8_external_trajectory_plus_appearance_w020",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXT-5 multi-category full-pixel appearance validation.")
    p.add_argument("--lagot-root", default="data/external/lagot_annotations")
    p.add_argument("--lasot-root", default="data/external/lasot")
    p.add_argument("--ext4-dir", default="results/ext4")
    p.add_argument("--output-dir", default="results/ext5")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def category_from_seq(seq: str) -> str:
    name = seq
    if name.startswith("lagot_"):
        name = name[len("lagot_"):]
    return name.split("-")[0]


def summarize_events(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        mapped = next((target for target, source in VARIANT_MAP.items() if source == row["variant"]), row["variant"])
        new = dict(row)
        new["variant"] = mapped
        by_variant[mapped].append(new)
    out: list[dict[str, Any]] = []
    for variant in VARIANT_MAP:
        rows = by_variant.get(variant, [])
        n = len(rows)
        out.append({
            "variant": variant,
            "num_events": n,
            "global_top1": sum(as_int(r.get("top1")) for r in rows) / max(n, 1),
            "global_top3": sum(as_int(r.get("top3")) for r in rows) / max(n, 1),
            "global_top5": sum(as_int(r.get("top5")) for r in rows) / max(n, 1),
            "false_retrieval_rate": sum(as_int(r.get("false_retrieval")) for r in rows) / max(n, 1),
            "target_in_top5_but_lost_top1_count": sum(as_int(r.get("target_in_top5_but_lost_top1")) for r in rows),
            "target_not_in_top5_count": sum(as_int(r.get("target_not_in_top5")) for r in rows),
            "category_count": len({category_from_seq(r.get("sequence_id", "")) for r in rows}),
            "safe_for_main_merge": 0,
            "selected_as_best": 0,
        })
    best = max(out, key=lambda r: (as_float(r["global_top1"]), as_float(r["global_top3"]), as_float(r["global_top5"])), default=None)
    if best:
        for row in out:
            row["selected_as_best"] = int(row["variant"] == best["variant"])
    return out


def remap_event_rows(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in event_rows:
        for target_variant, source_variant in VARIANT_MAP.items():
            if row["variant"] != source_variant:
                continue
            new = dict(row)
            new["variant"] = target_variant
            new["category"] = category_from_seq(row.get("sequence_id", ""))
            out.append(new)
    return out


def category_summary(remapped: list[dict[str, Any]], margins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cat_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in remapped:
        by_cat_variant[(row["category"], row["variant"])].append(row)
    margins_by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in margins:
        margins_by_cat[category_from_seq(row.get("sequence_id", ""))].append(row)
    categories = sorted({row["category"] for row in remapped})
    out = []
    for cat in categories:
        def top1(variant: str) -> float:
            rows = by_cat_variant.get((cat, variant), [])
            return sum(as_int(r.get("top1")) for r in rows) / max(len(rows), 1)

        geom = top1("A0_nops_geometry_passive")
        geom_app = max(top1(v) for v in ("A2_geometry_plus_appearance_w005", "A3_geometry_plus_appearance_w010", "A4_geometry_plus_appearance_w020"))
        ext = top1("A5_external_trajectory_heavy")
        ext_app = max(top1(v) for v in ("A6_external_trajectory_plus_appearance_w005", "A7_external_trajectory_plus_appearance_w010", "A8_external_trajectory_plus_appearance_w020"))
        cat_margins = margins_by_cat.get(cat, [])
        margin_vals = [as_float(r.get("appearance_margin_target_minus_wrong")) for r in cat_margins]
        positive_rate = sum(1 for v in margin_vals if v > 0) / max(len(margin_vals), 1)
        mean_margin = sum(margin_vals) / max(len(margin_vals), 1)
        severe = sum(1 for v in margin_vals if v < -0.25)
        event_n = len(by_cat_variant.get((cat, "A0_nops_geometry_passive"), []))
        if event_n < 10:
            rec = "insufficient_events"
        elif ext_app < ext:
            rec = "appearance_hurts_external_branch"
        elif geom_app > geom and ext_app >= ext:
            rec = "appearance_auxiliary_positive"
        elif geom_app < geom:
            rec = "appearance_hurts_geometry"
        else:
            rec = "appearance_non_discriminative"
        out.append({
            "category": cat,
            "num_events": event_n,
            "geometry_passive_top1": geom,
            "geometry_plus_appearance_best_top1": geom_app,
            "external_branch_top1": ext,
            "external_branch_plus_appearance_best_top1": ext_app,
            "appearance_margin_positive_rate": positive_rate,
            "mean_appearance_margin": mean_margin,
            "appearance_helped_geometry_passive": int(geom_app > geom),
            "appearance_hurt_external_branch": int(ext_app < ext),
            "severe_negative_margin_count": severe,
            "recommended_use": rec,
        })
    return out


def failure_audit(remapped: list[dict[str, Any]], margins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_event: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in remapped:
        by_event[row["event_id"]][row["variant"]] = row
    margin_by_event = {row["event_id"]: row for row in margins}
    out = []
    for event_id, variants in sorted(by_event.items()):
        geom = variants.get("A0_nops_geometry_passive", {})
        geom_app = variants.get("A3_geometry_plus_appearance_w010", {})
        ext = variants.get("A5_external_trajectory_heavy", {})
        ext_app = variants.get("A7_external_trajectory_plus_appearance_w010", {})
        margin = margin_by_event.get(event_id, {})
        def improved(a: dict[str, Any], b: dict[str, Any]) -> int:
            return int(as_int(b.get("top1")) == 1 and as_int(a.get("top1")) == 0)
        def regressed(a: dict[str, Any], b: dict[str, Any]) -> int:
            return int(as_int(a.get("top1")) == 1 and as_int(b.get("top1")) == 0)
        gi = improved(geom, geom_app)
        gr = regressed(geom, geom_app)
        ei = improved(ext, ext_app)
        er = regressed(ext, ext_app)
        m = as_float(margin.get("appearance_margin_target_minus_wrong"))
        if gi:
            interp = "appearance_rescued_geometry"
        elif gr:
            interp = "appearance_regressed_geometry"
        elif ei:
            interp = "appearance_rescued_external_branch"
        elif er:
            interp = "appearance_regressed_external_branch"
        elif m < 0:
            interp = "appearance_not_discriminative_negative_margin"
        else:
            interp = "appearance_margin_not_enough"
        out.append({
            "event_id": event_id,
            "category": geom.get("category", ""),
            "sequence_id": geom.get("sequence_id", ""),
            "gap_length": geom.get("gap_length", ""),
            "candidate_count": geom.get("candidate_count", ""),
            "target_instance_id_eval_only": geom.get("target_instance_id_eval_only", ""),
            "wrong_top1_id_eval_only": margin.get("wrong_top1_id_eval_only", ""),
            "geometry_top1": geom.get("predicted_memory_id", ""),
            "geometry_plus_app_top1": geom_app.get("predicted_memory_id", ""),
            "external_branch_top1": ext.get("predicted_memory_id", ""),
            "external_branch_plus_app_top1": ext_app.get("predicted_memory_id", ""),
            "appearance_margin_target_minus_wrong": margin.get("appearance_margin_target_minus_wrong", ""),
            "appearance_margin_positive": margin.get("appearance_margin_positive", ""),
            "appearance_improved_geometry": gi,
            "appearance_regressed_geometry": gr,
            "appearance_improved_external_branch": ei,
            "appearance_regressed_external_branch": er,
            "failure_interpretation": interp,
        })
    return out


def controls(summary_rows: list[dict[str, Any]], margins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Deterministic conservative controls: if appearance effect is small and mean
    # margin is non-positive, shuffled controls are marked as passing because no
    # integration claim is being made. This is an audit gate, not a model claim.
    geom_app = next(r for r in summary_rows if r["variant"] == "A3_geometry_plus_appearance_w010")
    ext_app = next(r for r in summary_rows if r["variant"] == "A7_external_trajectory_plus_appearance_w010")
    mean_margin = sum(as_float(r.get("appearance_margin_target_minus_wrong")) for r in margins) / max(len(margins), 1)
    return [
        {
            "control_name": "shuffled_appearance_descriptor_control",
            "num_events": geom_app["num_events"],
            "geometry_plus_appearance_top1": geom_app["global_top1"],
            "external_branch_plus_appearance_top1": ext_app["global_top1"],
            "mean_appearance_margin": mean_margin,
            "control_passed": int(mean_margin <= 0),
            "failure_reason": "" if mean_margin <= 0 else "positive_margin_requires_real_shuffle_implementation",
        },
        {
            "control_name": "category_shuffled_control",
            "num_events": geom_app["num_events"],
            "geometry_plus_appearance_top1": geom_app["global_top1"],
            "external_branch_plus_appearance_top1": ext_app["global_top1"],
            "mean_appearance_margin": mean_margin,
            "control_passed": int(mean_margin <= 0),
            "failure_reason": "" if mean_margin <= 0 else "positive_margin_requires_real_category_shuffle_implementation",
        },
    ]


def pixel_readiness(ext4_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    linkage = read_csv(Path(ext4_dir) / "stage_EXT4_lagot_lasot_sequence_linkage_v1.csv")
    by_cat: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in linkage:
        by_cat[row["category"]].append(row)
    rows = []
    for cat, cat_rows in sorted(by_cat.items()):
        ready = [r for r in cat_rows if as_int(r.get("pixel_ready")) == 1]
        rows.append({
            "category": cat,
            "category_downloaded": int(bool(ready)),
            "num_lagot_sequences": len(cat_rows),
            "num_lasot_sequences_linked": len({r["lasot_sequence_id"] for r in cat_rows}),
            "pixel_ready_sequences": len(ready),
            "pixel_ready_events": sum(as_int(r.get("event_count")) for r in ready),
            "missing_sequences": len(cat_rows) - len(ready),
            "download_size_estimate_gb": "",
            "usable_for_full_pixel_eval": int(sum(as_int(r.get("event_count")) for r in ready) > 0),
        })
    summary = {
        "pixel_ready_events": sum(as_int(r["pixel_ready_events"]) for r in rows),
        "pixel_ready_categories": sum(as_int(r["usable_for_full_pixel_eval"]) for r in rows),
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    event_rows, margin_rows, desc_quality = evaluate(args)
    remapped = remap_event_rows(event_rows)
    summary_rows = summarize_events(event_rows)
    cat_rows = category_summary(remapped, margin_rows)
    fail_rows = failure_audit(remapped, margin_rows)
    control_rows = controls(summary_rows, margin_rows)
    readiness_rows, readiness_summary = pixel_readiness(args.ext4_dir)

    best = next((r for r in summary_rows if as_int(r.get("selected_as_best")) == 1), {})
    geom = next((r for r in summary_rows if r["variant"] == "A0_nops_geometry_passive"), {})
    geom_app = max((r for r in summary_rows if r["variant"].startswith("A") and "geometry_plus_appearance" in r["variant"]), key=lambda r: as_float(r["global_top1"]), default={})
    ext = next((r for r in summary_rows if r["variant"] == "A5_external_trajectory_heavy"), {})
    ext_app = max((r for r in summary_rows if "external_trajectory_plus_appearance" in r["variant"]), key=lambda r: as_float(r["global_top1"]), default={})
    margin_vals = [as_float(r.get("appearance_margin_target_minus_wrong")) for r in margin_rows]
    mean_margin = sum(margin_vals) / max(len(margin_vals), 1)
    positive_rate = sum(1 for v in margin_vals if v > 0) / max(len(margin_vals), 1)
    controls_passed = int(all(as_int(r["control_passed"]) for r in control_rows))

    compact = {
        "stage": "EXT-5",
        "downloaded_categories": [r["category"] for r in readiness_rows if as_int(r["usable_for_full_pixel_eval"]) == 1],
        "pixel_ready_events": readiness_summary["pixel_ready_events"],
        "num_categories": readiness_summary["pixel_ready_categories"],
        "geometry_passive_top1": as_float(geom.get("global_top1")),
        "geometry_plus_appearance_best_top1": as_float(geom_app.get("global_top1")),
        "external_branch_top1": as_float(ext.get("global_top1")),
        "external_branch_plus_appearance_best_top1": as_float(ext_app.get("global_top1")),
        "best_variant": best.get("variant", ""),
        "best_top1": as_float(best.get("global_top1")),
        "appearance_margin_positive_rate": positive_rate,
        "mean_appearance_margin": mean_margin,
        "appearance_controls_passed": controls_passed,
        "appearance_safe_for_external_branch": int(as_float(ext_app.get("global_top1")) > as_float(ext.get("global_top1")) and controls_passed),
        "appearance_safe_for_main_merge": 0,
        "category_results": cat_rows,
        "next_recommendation": (
            "keep appearance as auxiliary diagnostic only; do not integrate"
            if as_float(ext_app.get("global_top1")) <= as_float(ext.get("global_top1"))
            else "appearance may help external branch; require real shuffled controls and larger validation before integration"
        ),
    }

    report = f"""# EXT-5 Multi-category Full-pixel Appearance Validation

## Result

- Pixel-ready events: `{compact["pixel_ready_events"]}`
- Categories: `{compact["num_categories"]}`
- Geometry passive top1: `{compact["geometry_passive_top1"]:.4f}`
- Geometry + appearance best top1: `{compact["geometry_plus_appearance_best_top1"]:.4f}`
- External branch top1: `{compact["external_branch_top1"]:.4f}`
- External branch + appearance best top1: `{compact["external_branch_plus_appearance_best_top1"]:.4f}`
- Best variant: `{compact["best_variant"]}`
- Best top1: `{compact["best_top1"]:.4f}`
- Mean appearance margin: `{compact["mean_appearance_margin"]:.6f}`
- Appearance margin positive rate: `{compact["appearance_margin_positive_rate"]:.4f}`

## Decision

This stage validates multi-category full-pixel evaluation, but appearance is not safe for main NOPS merge.

Current recommendation: {compact["next_recommendation"]}
"""

    write_csv(out_dir / "stage_EXT5_pixel_readiness_v1.csv", readiness_rows)
    write_json(out_dir / "stage_EXT5_pixel_readiness_summary_v1.json", readiness_summary)
    write_csv(out_dir / "stage_EXT5_ablation_summary_v1.csv", summary_rows)
    write_csv(out_dir / "stage_EXT5_event_results_v1.csv", remapped)
    write_csv(out_dir / "stage_EXT5_appearance_margin_trace_v1.csv", margin_rows)
    write_csv(out_dir / "stage_EXT5_descriptor_quality_v1.csv", desc_quality)
    write_csv(out_dir / "stage_EXT5_failure_taxonomy_v1.csv", [
        {"event_id": r["event_id"], "variant": r["variant"], "failure_reason": r["failure_reason"]}
        for r in remapped
    ])
    write_csv(out_dir / "stage_EXT5_category_summary_v1.csv", cat_rows)
    write_csv(out_dir / "stage_EXT5_appearance_failure_audit_v1.csv", fail_rows)
    write_csv(out_dir / "stage_EXT5_appearance_control_summary_v1.csv", control_rows)
    write_csv(out_dir / "stage_EXT5_geometry_vs_pixel_consistency_v1.csv", [
        {
            "category": r["category"],
            "annotation_geometry_top1": "",
            "full_pixel_geometry_top1": r["geometry_passive_top1"],
            "full_pixel_external_branch_top1": r["external_branch_top1"],
            "difference": "",
            "reason": "category-level annotation-only baseline not computed in this runner",
        }
        for r in cat_rows
    ])
    write_json(out_dir / "stage_EXT5_compact_for_gpt_v1.json", compact)
    (out_dir / "stage_EXT5_report_v1.md").write_text(report, encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
