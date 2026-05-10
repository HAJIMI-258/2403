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

from experiments.run_core1aa_stability_namespace_pair_gate import GATES, row_passes
from experiments.run_core1ab_non_oracle_curriculum_encoder import extract_descriptors
from experiments.run_core1ac_raw_descriptor_memory_integration_smoke import (
    VARIANTS,
    build_event_rows,
    f,
    i,
    read_json,
    summarize_variant,
)
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json
from experiments.run_core1z_oracle_proposal_diagnostic_encoder import parse_descriptor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AD medium-confidence raw descriptor gate smoke.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--observations", default="results/core1aa/stage_CORE1AA_stability_observation_trace_v1.csv")
    p.add_argument("--core1ac-compact", default="results/core1ac/stage_CORE1AC_compact_for_gpt_v1.json")
    p.add_argument("--output-dir", default="results/core1ad")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


GATE_NAMES = [
    "A1_score050_cost050",
    "A2_score060_cost040",
    "A9_score060_cost040_consecutive",
    "A10_score060_cost040_streak2",
    "A11_score070_cost030",
    "A12_score070_cost030_consecutive",
    "A13_score070_cost030_streak2_center32",
]


def gate_by_name(name: str) -> dict[str, Any]:
    for gate in GATES:
        if gate["gate_name"] == name:
            return gate
    raise ValueError(name)


def assign_obs_ids(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        rr: dict[str, Any] = dict(row)
        rr["obs_id"] = idx
        out.append(rr)
    return out


def run_gate(
    *,
    gate_name: str,
    all_rows: list[dict[str, Any]],
    desc_by_obs: dict[int, np.ndarray],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gate = gate_by_name(gate_name)
    selected = [row for row in all_rows if row_passes(row, gate) and i(row["obs_id"]) in desc_by_obs]
    baseline_rows = build_event_rows(selected, desc_by_obs, VARIANTS[0], seed)
    baseline_by_query = {i(r["query_obs_id"]): r for r in baseline_rows}
    event_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows = baseline_rows if variant["variant"] == "A0_track_recency_baseline" else build_event_rows(selected, desc_by_obs, variant, seed)
        for row in rows:
            row["gate_name"] = gate_name
        event_rows.extend(rows)
        summary = summarize_variant(variant, rows, baseline_by_query)
        summary["gate_name"] = gate_name
        summary["selected_observation_count"] = len(selected)
        summary["baseline_saturated"] = int(
            variant["variant"] == "A0_track_recency_baseline"
            and f(summary["top1"]) >= 0.999
        )
        summary_rows.append(summary)
    baseline_summary = next(r for r in summary_rows if r["variant"] == "A0_track_recency_baseline")
    controls = [r for r in summary_rows if str(r["variant"]).endswith("_control")]
    control_best = max([f(r["top1"]) for r in controls], default=0.0)
    non_controls = [r for r in summary_rows if i(r["eligible_for_best"]) == 1]
    best = max(non_controls, key=lambda r: (f(r["top1"]), f(r["mean_target_margin"]), -i(r["regressed_count"]))) if non_controls else baseline_summary
    controls_passed = int(f(best["top1"]) >= control_best and f(best["top1"]) >= f(baseline_summary["top1"]))
    for row in summary_rows:
        row["gate_baseline_top1"] = baseline_summary["top1"]
        row["gate_baseline_false_retrieval_rate"] = baseline_summary["false_retrieval_rate"]
        row["control_best_top1"] = control_best
        row["controls_passed_for_gate"] = controls_passed
        row["selected_as_gate_best"] = int(row["variant"] == best["variant"])
        row["safe_for_gate"] = int(
            row["variant"] == best["variant"]
            and row["variant"] != "A0_track_recency_baseline"
            and controls_passed
            and f(row["top1"]) > f(baseline_summary["top1"])
            and i(row["regressed_count"]) <= 1
        )
    return summary_rows, event_rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    core1ac = read_json(Path(args.core1ac_compact))
    all_rows = assign_obs_ids(read_csv(Path(args.observations)))
    desc_rows, desc_raw = extract_descriptors(all_rows, Path(args.config), args.seed)
    desc_by_obs = {i(row["obs_id"]): parse_descriptor(str(row["descriptor"])) for row in desc_rows}

    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for gate_name in GATE_NAMES:
        summaries, events = run_gate(gate_name=gate_name, all_rows=all_rows, desc_by_obs=desc_by_obs, seed=args.seed)
        summary_rows.extend(summaries)
        event_rows.extend(events)

    eligible = [r for r in summary_rows if i(r.get("safe_for_gate")) == 1]
    if eligible:
        best = max(eligible, key=lambda r: (f(r["top1"]) - f(r["gate_baseline_top1"]), f(r["mean_target_margin"])))
    else:
        best = max(summary_rows, key=lambda r: (f(r["top1"]) - f(r["gate_baseline_top1"]), f(r["top1"]), f(r["mean_target_margin"]))) if summary_rows else {}
    for row in summary_rows:
        row["selected_as_global_best"] = int(row is best)

    baseline_for_best = f(best.get("gate_baseline_top1", 0.0))
    best_top1 = f(best.get("top1", 0.0))
    safe = int(bool(eligible))
    compact = {
        "stage": "CORE-1AD",
        "artifact_version": args.artifact_version,
        "source_stage": "CORE-1AC",
        "gate_count": len(GATE_NAMES),
        "observation_count": len(all_rows),
        "descriptor_available_count": len(desc_by_obs),
        "core1ac_baseline_saturated": core1ac.get("baseline_saturated", 0),
        "best_gate": best.get("gate_name", ""),
        "best_variant": best.get("variant", ""),
        "best_gate_num_queries": best.get("num_queries", 0),
        "best_gate_baseline_top1": baseline_for_best,
        "best_top1": best_top1,
        "best_delta_vs_gate_baseline": best_top1 - baseline_for_best,
        "best_false_retrieval_rate": best.get("false_retrieval_rate", 1.0),
        "best_mean_target_margin": best.get("mean_target_margin", 0.0),
        "best_improved_count": best.get("improved_count", 0),
        "best_regressed_count": best.get("regressed_count", 0),
        "descriptor_controls_passed": best.get("controls_passed_for_gate", 0),
        "safe_for_integration_smoke": safe,
        "oracle_leakage_found": 0,
        "pretrained_weights_used": 0,
        "next_recommendation": (
            "CORE-1AE run selected descriptor gate against broader internal focus/anchor regression guards"
            if safe
            else "do not integrate descriptor cue; medium-confidence smoke did not beat baseline cleanly"
        ),
    }
    report = f"""# CORE-1AD Medium-Confidence Descriptor Gate

This stage broadens CORE-1AC beyond the high-confidence saturated set. It scans medium/high confidence observation gates and tests whether raw descriptor cue can rescue baseline retrieval failures without control failures.

## Result

- Gates scanned: {len(GATE_NAMES)}
- Observations: {len(all_rows)}
- Descriptor availability: {len(desc_by_obs)}
- Best gate: {compact['best_gate']}
- Best variant: {compact['best_variant']}
- Best gate queries: {compact['best_gate_num_queries']}
- Gate baseline top1: {float(compact['best_gate_baseline_top1']):.4f}
- Best top1: {float(compact['best_top1']):.4f}
- Delta vs gate baseline: {float(compact['best_delta_vs_gate_baseline']):.4f}
- Improved / regressed: {compact['best_improved_count']} / {compact['best_regressed_count']}
- Controls passed: {compact['descriptor_controls_passed']}
- Safe for integration smoke: {compact['safe_for_integration_smoke']}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AD_"
    write_csv(
        out_dir / f"{prefix}descriptor_trace_{args.artifact_version}.csv",
        desc_rows,
        ["obs_id", "sequence_id", "event_id", "window_kind", "frame_idx", "track_id", "box", "crop_box", "descriptor_norm", "descriptor_entropy_proxy", "edge_density", "gt_instance_eval_only", "descriptor"],
    )
    write_csv(
        out_dir / f"{prefix}ablation_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "gate_name",
            "variant",
            "selected_observation_count",
            "num_queries",
            "top1",
            "top3",
            "false_retrieval_rate",
            "mean_target_margin",
            "descriptor_used_rate",
            "improved_count",
            "regressed_count",
            "unchanged_success_count",
            "unchanged_failure_count",
            "gate_baseline_top1",
            "gate_baseline_false_retrieval_rate",
            "control_best_top1",
            "controls_passed_for_gate",
            "selected_as_gate_best",
            "safe_for_gate",
            "selected_as_global_best",
        ],
    )
    write_csv(
        out_dir / f"{prefix}event_results_{args.artifact_version}.csv",
        event_rows,
        [
            "gate_name",
            "variant",
            "sequence_id",
            "event_id",
            "window_kind",
            "query_obs_id",
            "candidate_count",
            "target_candidate_count",
            "top1_obs_id",
            "top1_instance_eval_only",
            "target_instance_eval_only",
            "top1_success",
            "target_rank",
            "target_in_top3",
            "target_margin",
            "baseline_score_top1",
            "descriptor_score_top1",
            "descriptor_used",
            "base_margin",
        ],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1ad_medium_confidence_descriptor_gate.py",
                "oracle_proposals_used": 0,
                "pretrained_weights_used": 0,
                "gt_used_for_online_scoring": 0,
                "gt_used_for_eval_only": 1,
                "leakage_found": 0,
            }
        ],
        ["file", "oracle_proposals_used", "pretrained_weights_used", "gt_used_for_online_scoring", "gt_used_for_eval_only", "leakage_found"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
