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
from experiments.run_core1ac_raw_descriptor_memory_integration_smoke import (
    VARIANTS,
    build_event_rows,
    f,
    i,
    summarize_variant,
)
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json
from experiments.run_core1z_oracle_proposal_diagnostic_encoder import parse_descriptor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AH broad hard-pool tradeoff audit.")
    p.add_argument("--observations", default="results/core1aa/stage_CORE1AA_stability_observation_trace_v1.csv")
    p.add_argument("--descriptor-trace", default="results/core1ad/stage_CORE1AD_descriptor_trace_v1.csv")
    p.add_argument("--pair-quality", default="results/core1aa/stage_CORE1AA_gate_summary_v1.csv")
    p.add_argument("--output-dir", default="results/core1ah")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def assign_obs_ids(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        rr: dict[str, Any] = dict(row)
        rr["obs_id"] = idx
        out.append(rr)
    return out


def load_desc(path: Path) -> dict[int, np.ndarray]:
    rows = read_csv(path)
    return {i(row["obs_id"]): parse_descriptor(str(row["descriptor"])) for row in rows}


def pair_quality_by_gate(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for row in read_csv(path):
        if row.get("negative_mode") != "cross_sequence":
            continue
        out[row["gate_name"]] = row
    return out


def evaluate_gate(
    gate: dict[str, Any],
    rows: list[dict[str, Any]],
    desc_by_obs: dict[int, np.ndarray],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [row for row in rows if row_passes(row, gate) and i(row["obs_id"]) in desc_by_obs]
    baseline_rows = build_event_rows(selected, desc_by_obs, VARIANTS[0], seed)
    baseline_by_query = {i(r["query_obs_id"]): r for r in baseline_rows}
    summaries: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_rows = baseline_rows if variant["variant"] == "A0_track_recency_baseline" else build_event_rows(selected, desc_by_obs, variant, seed)
        for row in variant_rows:
            row["gate_name"] = gate["gate_name"]
        events.extend(variant_rows)
        summary = summarize_variant(variant, variant_rows, baseline_by_query)
        summary["gate_name"] = gate["gate_name"]
        summary["selected_observation_count"] = len(selected)
        summaries.append(summary)
    return summaries, events


def clean_rescue_count(events: list[dict[str, Any]], gate: str, variant: str) -> tuple[int, int, int, int]:
    def by_variant(v: str) -> dict[int, dict[str, Any]]:
        return {i(r["query_obs_id"]): r for r in events if r["gate_name"] == gate and r["variant"] == v}

    base = by_variant("A0_track_recency_baseline")
    selected = by_variant(variant)
    controls = [
        by_variant("A7_shuffled_descriptor_w010_control"),
        by_variant("A8_wrong_binding_descriptor_w010_control"),
        by_variant("A9_random_descriptor_w010_control"),
    ]
    clean = control_confounded = regressed = baseline_failures = 0
    for qid, b in base.items():
        bs = i(b["top1_success"])
        if bs == 0:
            baseline_failures += 1
        s = selected.get(qid)
        if s is None:
            continue
        ss = i(s["top1_success"])
        if bs == 1 and ss == 0:
            regressed += 1
        if bs == 0 and ss == 1:
            if any(i(c.get(qid, {}).get("top1_success")) == 1 for c in controls):
                control_confounded += 1
            else:
                clean += 1
    return baseline_failures, clean, control_confounded, regressed


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = assign_obs_ids(read_csv(Path(args.observations)))
    desc_by_obs = load_desc(Path(args.descriptor_trace))
    quality = pair_quality_by_gate(Path(args.pair_quality))

    all_summary: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    for gate in GATES:
        summaries, events = evaluate_gate(gate, rows, desc_by_obs, args.seed)
        q = quality.get(gate["gate_name"], {})
        baseline = next((s for s in summaries if s["variant"] == "A0_track_recency_baseline"), {})
        controls = [s for s in summaries if str(s["variant"]).endswith("_control")]
        control_best = max([f(s["top1"]) for s in controls], default=0.0)
        for summary in summaries:
            bf, clean, confounded, regressed = clean_rescue_count(events, gate["gate_name"], str(summary["variant"]))
            summary["pair_positive_precision"] = q.get("positive_pair_precision_eval_only", "")
            summary["pair_negative_precision_namespace"] = q.get("negative_pair_precision_namespace_eval_only", "")
            summary["pair_negative_precision_local"] = q.get("negative_pair_precision_local_id_eval_only", "")
            summary["baseline_top1_for_gate"] = baseline.get("top1", 0.0)
            summary["baseline_failure_count"] = bf
            summary["clean_rescue_count"] = clean
            summary["control_confounded_rescue_count"] = confounded
            summary["control_best_top1"] = control_best
            summary["delta_vs_gate_baseline"] = f(summary["top1"]) - f(baseline.get("top1", 0.0))
            pos_ok = f(q.get("positive_pair_precision_eval_only", 0.0)) >= 0.85
            neg_ok = f(q.get("negative_pair_precision_namespace_eval_only", 0.0)) >= 0.85
            hard_ok = bf >= 20 and clean >= 3 and regressed <= clean
            summary["pair_quality_passed"] = int(pos_ok and neg_ok)
            summary["hard_pool_candidate"] = int(hard_ok and not str(summary["variant"]).endswith("_control"))
        all_summary.extend(summaries)
        all_events.extend(events)

    candidates = [s for s in all_summary if i(s["hard_pool_candidate"]) == 1]
    if candidates:
        best = max(candidates, key=lambda s: (i(s["clean_rescue_count"]), f(s["delta_vs_gate_baseline"]), -i(s["regressed_count"])))
    else:
        best = max(all_summary, key=lambda s: (i(s["clean_rescue_count"]), i(s["baseline_failure_count"]), f(s["delta_vs_gate_baseline"]))) if all_summary else {}

    hard_events = [
        r
        for r in all_events
        if r.get("gate_name") == best.get("gate_name")
        and r.get("variant") == best.get("variant")
        and i(r.get("top1_success")) == 1
    ]
    compact = {
        "stage": "CORE-1AH",
        "artifact_version": args.artifact_version,
        "gate_count": len(GATES),
        "gate_variant_count": len(all_summary),
        "best_gate": best.get("gate_name", ""),
        "best_variant": best.get("variant", ""),
        "best_num_queries": best.get("num_queries", 0),
        "best_selected_observation_count": best.get("selected_observation_count", 0),
        "best_baseline_top1": best.get("baseline_top1_for_gate", 0.0),
        "best_top1": best.get("top1", 0.0),
        "best_delta_vs_gate_baseline": best.get("delta_vs_gate_baseline", 0.0),
        "best_baseline_failure_count": best.get("baseline_failure_count", 0),
        "best_clean_rescue_count": best.get("clean_rescue_count", 0),
        "best_regressed_count": best.get("regressed_count", 0),
        "best_pair_quality_passed": best.get("pair_quality_passed", 0),
        "hard_pool_found": int(bool(candidates)),
        "oracle_leakage_found": 0,
        "next_recommendation": (
            "CORE-1AI build hard-pool training/eval split from selected gate"
            if candidates
            else "no reliable hard descriptor pool; prioritize proposal/observation quality repair over descriptor integration"
        ),
    }
    report = f"""# CORE-1AH Broad Hard-Pool Tradeoff

This stage scans all CORE-1AA gates, including noisy low-confidence gates, to measure the tradeoff between hard retrieval opportunities, pair quality, descriptor rescues, and controls.

## Result

- Gate/variant combinations: {len(all_summary)}
- Best gate: {compact['best_gate']}
- Best variant: {compact['best_variant']}
- Best baseline top1: {float(compact['best_baseline_top1']):.4f}
- Best top1: {float(compact['best_top1']):.4f}
- Baseline failures: {compact['best_baseline_failure_count']}
- Clean rescues: {compact['best_clean_rescue_count']}
- Regressions: {compact['best_regressed_count']}
- Pair quality passed: {compact['best_pair_quality_passed']}
- Hard pool found: {compact['hard_pool_found']}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AH_"
    write_csv(
        out_dir / f"{prefix}tradeoff_summary_{args.artifact_version}.csv",
        all_summary,
        [
            "gate_name",
            "variant",
            "selected_observation_count",
            "num_queries",
            "top1",
            "top3",
            "false_retrieval_rate",
            "mean_target_margin",
            "delta_vs_gate_baseline",
            "baseline_top1_for_gate",
            "baseline_failure_count",
            "clean_rescue_count",
            "control_confounded_rescue_count",
            "regressed_count",
            "pair_positive_precision",
            "pair_negative_precision_namespace",
            "pair_negative_precision_local",
            "pair_quality_passed",
            "hard_pool_candidate",
        ],
    )
    write_csv(
        out_dir / f"{prefix}best_variant_success_events_{args.artifact_version}.csv",
        hard_events,
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
            "target_margin",
            "descriptor_used",
        ],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
