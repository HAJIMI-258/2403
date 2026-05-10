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
from experiments.run_core1m_assignment_pair_confidence_gate import collect_window_observations, load_config, select_windows
from experiments.run_core1p_proposal_profile_pair_validation import PROFILE_A3_LOWER_QUANTILE
from experiments.run_core1x_cross_event_negative_mining import NEGATIVE_MODES, build_cross_negatives, prepare_rows, summarize
from experiments.run_core1w_negative_curriculum_audit import build_positive_pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1Y expanded cross-event negative mining.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--window-plan", default="results/core1j/stage_CORE1J_window_plan_v1.csv")
    p.add_argument("--output-dir", default="results/core1y")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sequences", type=int, default=3)
    p.add_argument("--match-iou", type=float, default=0.25)
    p.add_argument("--max-negatives-per-observation", type=int, default=6)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


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

    prepared = prepare_rows(obs_rows)
    positives = build_positive_pairs(prepared)
    summary_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for mode in NEGATIVE_MODES:
        negatives = build_cross_negatives(prepared, mode, args.max_negatives_per_observation)
        summary_rows.append(summarize(mode, positives, negatives))
        for p in positives:
            pp = dict(p)
            pp["negative_mode"] = mode
            pair_rows.append(pp)
        pair_rows.extend(negatives)
    eligible = [r for r in summary_rows if int(r["eligible_for_training_smoke"]) == 1]
    if eligible:
        best = max(eligible, key=lambda r: r["negative_pair_count"])
    else:
        best = max(summary_rows, key=lambda r: (min(r["positive_pair_precision_eval_only"], r["negative_pair_precision_eval_only"]), r["negative_pair_count"])) if summary_rows else {}

    sequence_count = len(sequence_to_windows)
    event_count = len({row["event_id"] for row in selected_windows})
    compact = {
        "stage": "CORE-1Y",
        "artifact_version": args.artifact_version,
        "selected_sequence_count": sequence_count,
        "selected_event_count": event_count,
        "selected_window_count": len(selected_windows),
        "assignment_observation_count": len(obs_rows),
        "positive_pair_count": len(positives),
        "best_negative_mode": best.get("negative_mode", ""),
        "best_negative_pair_count": best.get("negative_pair_count", 0),
        "best_positive_pair_precision_eval_only": best.get("positive_pair_precision_eval_only", 0.0),
        "best_negative_pair_precision_eval_only": best.get("negative_pair_precision_eval_only", 0.0),
        "expanded_cross_event_negative_mining_passed": int(bool(eligible)),
        "oracle_leakage_found": 0,
        "ready_for_encoder_training": int(bool(eligible)),
        "runtime_sec": time.perf_counter() - total_start,
        "next_recommendation": (
            "CORE-1Z train tiny encoder on expanded cross-event negative curriculum"
            if eligible
            else "expand beyond 3 rendered sequences or switch to oracle-proposal diagnostic encoder; current objectness-derived negatives remain too noisy"
        ),
    }
    report = f"""# CORE-1Y Expanded Cross-Event Negative Mining

This stage regenerates assignment observations for a larger selected-window set and re-runs cross-context negative mining. It still does not train an encoder.

## Result

- Selected sequences: {sequence_count}
- Selected events: {event_count}
- Selected windows: {len(selected_windows)}
- Assignment observations: {len(obs_rows)}
- Positive pairs: {len(positives)}
- Best negative mode: {compact['best_negative_mode']}
- Best negative pair count: {compact['best_negative_pair_count']}
- Best positive precision eval-only: {float(compact['best_positive_pair_precision_eval_only']):.4f}
- Best negative precision eval-only: {float(compact['best_negative_pair_precision_eval_only']):.4f}
- Passed: {compact['expanded_cross_event_negative_mining_passed']}
- Runtime seconds: {compact['runtime_sec']:.2f}

Next recommendation: {compact['next_recommendation']}
"""

    prefix = "stage_CORE1Y_"
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
        out_dir / f"{prefix}negative_mode_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "negative_mode",
            "positive_pair_count",
            "negative_pair_count",
            "positive_pair_precision_eval_only",
            "negative_pair_precision_eval_only",
            "eligible_for_training_smoke",
        ],
    )
    write_csv(
        out_dir / f"{prefix}pair_trace_{args.artifact_version}.csv",
        pair_rows,
        [
            "negative_mode",
            "pair_id",
            "pair_type",
            "sequence_id",
            "event_id",
            "window_kind",
            "frame_i",
            "frame_j",
            "track_i",
            "track_j",
            "prototype_i",
            "prototype_j",
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
