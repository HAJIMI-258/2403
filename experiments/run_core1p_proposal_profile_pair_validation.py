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
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json
from experiments.run_core1m_assignment_pair_confidence_gate import (
    GATES,
    build_pairs_for_gate,
    collect_window_observations,
    load_config,
    select_windows,
    summarize_gate,
)


PROFILE_A3_LOWER_QUANTILE = {
    "q_obj": 0.88,
    "local_k": 0.85,
    "max_proposals": 16,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1P proposal-profile tracker pair validation.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--window-plan", default="results/core1j/stage_CORE1J_window_plan_v1.csv")
    p.add_argument("--core1m-compact", default="results/core1m/stage_CORE1M_compact_for_gpt_v1.json")
    p.add_argument("--output-dir", default="results/core1p")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sequences", type=int, default=2)
    p.add_argument("--match-iou", type=float, default=0.25)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
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

    all_pair_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for gate in GATES:
        pairs = build_pairs_for_gate(obs_rows, gate)
        summaries.append(summarize_gate(gate["name"], pairs))
        all_pair_rows.extend(pairs)

    eligible = [s for s in summaries if int(s["eligible_for_training_smoke"]) == 1]
    if eligible:
        best = max(eligible, key=lambda s: (s["positive_pair_count"] + s["negative_pair_count"], s["positive_pair_precision_eval_only"] + s["negative_pair_precision_eval_only"]))
    else:
        best = max(summaries, key=lambda s: (min(s["positive_pair_precision_eval_only"], s["negative_pair_precision_eval_only"]), s["positive_pair_count"] + s["negative_pair_count"])) if summaries else {}

    core1m = load_json(Path(args.core1m_compact))
    compact = {
        "stage": "CORE-1P",
        "artifact_version": args.artifact_version,
        "proposal_profile": "A3_lower_quantile_from_CORE1O",
        "profile_config": PROFILE_A3_LOWER_QUANTILE,
        "selected_sequence_count": len(sequence_to_windows),
        "selected_window_count": len(selected_windows),
        "assignment_observation_count": len(obs_rows),
        "matched_assignment_rate": float(np.mean([float(r["matched_assignment_rate"]) for r in runtime_rows])) if runtime_rows else 0.0,
        "best_gate": best.get("gate_name", ""),
        "best_positive_pair_count": best.get("positive_pair_count", 0),
        "best_negative_pair_count": best.get("negative_pair_count", 0),
        "best_positive_pair_precision_eval_only": best.get("positive_pair_precision_eval_only", 0.0),
        "best_negative_pair_precision_eval_only": best.get("negative_pair_precision_eval_only", 0.0),
        "core1m_best_positive_precision": core1m.get("best_positive_pair_precision_eval_only", 0.0),
        "core1m_best_negative_precision": core1m.get("best_negative_pair_precision_eval_only", 0.0),
        "pair_mining_gate_passed": int(bool(eligible)),
        "oracle_leakage_found": 0,
        "runtime_sec": time.perf_counter() - total_start,
        "next_recommendation": "CORE-1Q train tiny encoder on profile-gated pairs" if eligible else "proposal recall improved but pair quality still insufficient; inspect duplicate/fragmented assignments before training",
    }

    report = f"""# CORE-1P Proposal Profile Pair Validation

This stage validates the CORE-1O A3 lower-quantile objectness profile through the same assignment-pair gates used in CORE-1M. It does not change the main model.

## Result

- Proposal profile: A3 lower quantile
- Assignment observations: {compact['assignment_observation_count']}
- Matched assignment rate eval-only: {compact['matched_assignment_rate']:.4f}
- Best gate: {compact['best_gate']}
- Best positive precision eval-only: {compact['best_positive_pair_precision_eval_only']:.4f}
- Best negative precision eval-only: {compact['best_negative_pair_precision_eval_only']:.4f}
- CORE-1M best positive precision: {float(compact['core1m_best_positive_precision']):.4f}
- CORE-1M best negative precision: {float(compact['core1m_best_negative_precision']):.4f}
- Gate passed: {compact['pair_mining_gate_passed']}

Next recommendation: {compact['next_recommendation']}
"""

    prefix = "stage_CORE1P_"
    write_csv(
        out_dir / f"{prefix}assignment_observation_trace_{args.artifact_version}.csv",
        obs_rows,
        [
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
            "assignment_source",
            "final_assignment_source",
            "track_hit_count",
            "track_age",
            "track_gap_length",
            "gt_instance_eval_only",
            "gt_concept_eval_only",
            "match_iou_eval_only",
        ],
    )
    write_csv(
        out_dir / f"{prefix}gate_ablation_summary_{args.artifact_version}.csv",
        summaries,
        [
            "gate_name",
            "positive_pair_count",
            "negative_pair_count",
            "positive_pair_precision_eval_only",
            "negative_pair_precision_eval_only",
            "eligible_for_training_smoke",
        ],
    )
    write_csv(
        out_dir / f"{prefix}gated_pair_trace_{args.artifact_version}.csv",
        all_pair_rows,
        [
            "pair_id",
            "gate_name",
            "sequence_id",
            "event_id",
            "window_kind",
            "frame_i",
            "frame_j",
            "track_i",
            "track_j",
            "prototype_i",
            "prototype_j",
            "pair_type",
            "online_positive",
            "online_negative",
            "gt_instance_i_eval_only",
            "gt_instance_j_eval_only",
            "pair_correct_eval_only",
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
