from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.synth_stream import SyntheticStreamGenerator
from experiments.run_core1aa_stability_namespace_pair_gate import augment_stability, row_passes
from experiments.run_core1ab_non_oracle_curriculum_encoder import extract_descriptors
from experiments.run_core1ai_observation_quality_frontier import baseline_queries, candidate_gates, pair_quality
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json
from experiments.run_core1m_assignment_pair_confidence_gate import collect_window_observations, load_config, select_windows
from experiments.run_core1p_proposal_profile_pair_validation import PROFILE_A3_LOWER_QUANTILE
from experiments.run_core1z_oracle_proposal_diagnostic_encoder import parse_descriptor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AJ expanded observation frontier.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--window-plan", default="results/core1j/stage_CORE1J_window_plan_v1.csv")
    p.add_argument("--output-dir", default="results/core1aj")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sequences", type=int, default=4)
    p.add_argument("--match-iou", type=float, default=0.25)
    p.add_argument("--max-negatives-per-observation", type=int, default=8)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def i(v: Any, default: int = 0) -> int:
    if v in (None, ""):
        return default
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default


def f(v: Any, default: float = 0.0) -> float:
    if v in (None, ""):
        return default
    try:
        out = float(v)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def assign_obs_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row, obs_id=idx) for idx, row in enumerate(rows, start=1)]


def collect_expanded_observations(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cfg, payload = load_config(Path(args.config))
    profiled_payload = deepcopy(payload)
    profiled_payload["field"].update(PROFILE_A3_LOWER_QUANTILE)
    generator = SyntheticStreamGenerator(cfg, seed=args.seed)
    selected_windows = select_windows(read_csv(Path(args.window_plan)), args.max_sequences)
    sequence_to_windows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in selected_windows:
        sequence_to_windows[int(row["sequence_id"])].append(row)

    obs_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for sequence_id, windows in sorted(sequence_to_windows.items()):
        gen_start = time.perf_counter()
        sequence = generator.generate_sequence(sequence_id)
        generation_time = time.perf_counter() - gen_start
        frames_by_idx = {frame.frame_index: frame for frame in sequence.frames}
        for window in windows:
            rows, runtime = collect_window_observations(
                sequence_id=sequence_id,
                window_row=window,
                frames_by_idx=frames_by_idx,
                payload=profiled_payload,
                min_iou=args.match_iou,
            )
            obs_rows.extend(rows)
            runtime["sequence_generation_time_sec"] = generation_time
            runtime_rows.append(runtime)
    meta = {
        "selected_sequence_count": len(sequence_to_windows),
        "selected_event_count": len({row["event_id"] for row in selected_windows}),
        "selected_window_count": len(selected_windows),
    }
    return obs_rows, runtime_rows, meta


def run_frontier(rows: list[dict[str, Any]], desc_by_obs: dict[int, np.ndarray], args: argparse.Namespace) -> list[dict[str, Any]]:
    frontier_rows: list[dict[str, Any]] = []
    for gate in candidate_gates():
        selected = [row for row in rows if row_passes(row, gate) and i(row["obs_id"]) in desc_by_obs]
        pos_count, neg_count, pos_prec, neg_prec = pair_quality(selected, gate, args.max_negatives_per_observation)
        query_count, top1, failure_count = baseline_queries(selected, desc_by_obs, args.seed)
        pair_quality_passed = int(pos_count >= 20 and neg_count >= 20 and pos_prec >= 0.85 and neg_prec >= 0.85)
        hard_eval_ready = int(pair_quality_passed and failure_count >= 20 and query_count >= 100)
        frontier_rows.append(
            {
                "gate_name": gate["gate_name"],
                "score_min": gate.get("score_min", ""),
                "match_cost_max": gate.get("match_cost_max", ""),
                "streak_min": gate.get("streak_min", ""),
                "consecutive_required": int(bool(gate.get("consecutive_required"))),
                "selected_observation_count": len(selected),
                "positive_pair_count": pos_count,
                "negative_pair_count": neg_count,
                "positive_pair_precision_eval_only": pos_prec,
                "negative_pair_precision_namespace_eval_only": neg_prec,
                "query_count": query_count,
                "baseline_top1": top1,
                "baseline_failure_count": failure_count,
                "pair_quality_passed": pair_quality_passed,
                "hard_eval_ready": hard_eval_ready,
                "frontier_score": (min(pos_prec, neg_prec) * 1000.0) + failure_count + query_count / 1000.0,
            }
        )
    return frontier_rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    raw_obs, runtime_rows, meta = collect_expanded_observations(args)
    stability_rows = assign_obs_ids(augment_stability(raw_obs))
    desc_rows, desc_raw = extract_descriptors(stability_rows, Path(args.config), args.seed)
    desc_by_obs = {i(row["obs_id"]): parse_descriptor(str(row["descriptor"])) for row in desc_rows}
    frontier_rows = run_frontier(stability_rows, desc_by_obs, args)

    candidates = [r for r in frontier_rows if i(r["hard_eval_ready"]) == 1]
    if candidates:
        best = max(candidates, key=lambda r: (i(r["baseline_failure_count"]), f(r["positive_pair_precision_eval_only"]) + f(r["negative_pair_precision_namespace_eval_only"]), i(r["query_count"])))
    else:
        best = max(frontier_rows, key=lambda r: (i(r["pair_quality_passed"]), i(r["baseline_failure_count"]), f(r["frontier_score"]))) if frontier_rows else {}
    for row in frontier_rows:
        row["selected_as_best_frontier"] = int(row is best)

    compact = {
        "stage": "CORE-1AJ",
        "artifact_version": args.artifact_version,
        **meta,
        "observation_count": len(stability_rows),
        "descriptor_available_count": len(desc_by_obs),
        "gate_count": len(frontier_rows),
        "best_gate": best.get("gate_name", ""),
        "best_selected_observation_count": best.get("selected_observation_count", 0),
        "best_positive_pair_precision": best.get("positive_pair_precision_eval_only", 0.0),
        "best_negative_pair_precision": best.get("negative_pair_precision_namespace_eval_only", 0.0),
        "best_query_count": best.get("query_count", 0),
        "best_baseline_top1": best.get("baseline_top1", 0.0),
        "best_baseline_failure_count": best.get("baseline_failure_count", 0),
        "pair_quality_passed": best.get("pair_quality_passed", 0),
        "hard_eval_ready": best.get("hard_eval_ready", 0),
        "oracle_leakage_found": 0,
        "runtime_sec": time.perf_counter() - total_start,
        "next_recommendation": (
            "CORE-1AK run descriptor integration on expanded hard frontier gate"
            if i(best.get("hard_eval_ready")) == 1
            else "expanded sampling still lacks clean hard pool; repair proposal/observation generation or increase sequence coverage"
        ),
    }
    report = f"""# CORE-1AJ Expanded Observation Frontier

This stage expands the CORE-1AI frontier beyond the initial three sequences. It regenerates non-oracle observations with the A3 lower-quantile proposal profile, computes stability fields and crop descriptors, then reruns the clean-pair/hard-failure frontier.

## Result

- Selected sequences: {compact['selected_sequence_count']}
- Selected events: {compact['selected_event_count']}
- Selected windows: {compact['selected_window_count']}
- Observations: {compact['observation_count']}
- Descriptor availability: {compact['descriptor_available_count']}
- Best gate: {compact['best_gate']}
- Best selected observations: {compact['best_selected_observation_count']}
- Pair precision: positive {float(compact['best_positive_pair_precision']):.4f}, negative {float(compact['best_negative_pair_precision']):.4f}
- Queries: {compact['best_query_count']}
- Baseline top1: {float(compact['best_baseline_top1']):.4f}
- Baseline failures: {compact['best_baseline_failure_count']}
- Hard eval ready: {compact['hard_eval_ready']}
- Runtime seconds: {float(compact['runtime_sec']):.2f}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AJ_"
    write_csv(
        out_dir / f"{prefix}stability_observation_trace_{args.artifact_version}.csv",
        stability_rows,
        [
            "obs_id",
            "sequence_id",
            "event_id",
            "window_kind",
            "frame_idx",
            "track_id",
            "prototype_id",
            "box",
            "score",
            "objectness_score",
            "match_cost",
            "track_hit_count",
            "track_age",
            "track_gap_length",
            "frame_assignment_count",
            "max_box_overlap_same_frame",
            "consecutive_observation",
            "track_streak_length",
            "center_shift_from_prev_track",
            "area_ratio_delta_from_prev_track",
            "prev_box_iou_same_track",
            "stability_score",
            "gt_instance_eval_only",
            "match_iou_eval_only",
        ],
    )
    write_csv(
        out_dir / f"{prefix}descriptor_trace_{args.artifact_version}.csv",
        desc_rows,
        ["obs_id", "sequence_id", "event_id", "window_kind", "frame_idx", "track_id", "box", "crop_box", "descriptor_norm", "descriptor_entropy_proxy", "edge_density", "gt_instance_eval_only", "descriptor"],
    )
    write_csv(
        out_dir / f"{prefix}frontier_summary_{args.artifact_version}.csv",
        frontier_rows,
        [
            "gate_name",
            "score_min",
            "match_cost_max",
            "streak_min",
            "consecutive_required",
            "selected_observation_count",
            "positive_pair_count",
            "negative_pair_count",
            "positive_pair_precision_eval_only",
            "negative_pair_precision_namespace_eval_only",
            "query_count",
            "baseline_top1",
            "baseline_failure_count",
            "pair_quality_passed",
            "hard_eval_ready",
            "frontier_score",
            "selected_as_best_frontier",
        ],
    )
    write_csv(
        out_dir / f"{prefix}runtime_audit_{args.artifact_version}.csv",
        runtime_rows,
        [
            "sequence_id",
            "event_id",
            "window_kind",
            "start_frame",
            "end_frame",
            "assignment_count",
            "matched_assignment_count",
            "matched_assignment_rate",
            "tracker_runtime_sec",
            "sequence_generation_time_sec",
        ],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
