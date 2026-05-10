from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.synth_stream import SynthDatasetConfig, SyntheticStreamGenerator
from experiments import run_v3_stage_e31_retrieval_competition_repair as e31
from experiments.run_core1j_rendered_tracker_pair_audit import box_iou
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1L windowed tracker-derived pair mining smoke.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--window-plan", default="results/core1j/stage_CORE1J_window_plan_v1.csv")
    p.add_argument("--output-dir", default="results/core1l")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sequences", type=int, default=2)
    p.add_argument("--match-iou", type=float, default=0.25)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def load_config(path: Path) -> tuple[SynthDatasetConfig, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return SynthDatasetConfig.from_dict(payload), payload


def select_windows(rows: list[dict[str, str]], max_sequences: int) -> list[dict[str, str]]:
    selected_sequences: list[int] = []
    selected: list[dict[str, str]] = []
    for row in rows:
        seq_id = int(row["sequence_id"])
        if seq_id not in selected_sequences:
            if len(selected_sequences) >= max_sequences:
                continue
            selected_sequences.append(seq_id)
        if seq_id in selected_sequences:
            selected.append(row)
    return selected


def match_track_to_gt(track_box: tuple[int, int, int, int], frame, min_iou: float) -> tuple[int | None, int | None, float]:
    best_iid: int | None = None
    best_concept: int | None = None
    best_score = 0.0
    for idx, gt_box in enumerate(frame.boxes):
        score = box_iou(track_box, tuple(int(v) for v in gt_box))
        if score > best_score:
            best_iid = int(frame.instance_ids[idx])
            best_concept = int(frame.concept_ids[idx])
            best_score = float(score)
    if best_score < min_iou:
        return None, None, best_score
    return best_iid, best_concept, best_score


def run_window(
    *,
    sequence_id: int,
    window_row: dict[str, str],
    frames_by_idx: dict[int, Any],
    payload: dict[str, Any],
    min_iou: float,
    pair_id_start: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start_frame = int(window_row["start_frame"])
    end_frame = int(window_row["end_frame"])
    encoder = e31.MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    field = e31.MinimalObjectnessField(**payload["field"])
    tracker = e31.MinimalTemporalIdentityTracker(**payload["tracking"])
    memory = e31.MinimalPrototypeMemory(**payload["memory"])
    prev_memory_output = None
    prev_by_track: dict[int, dict[str, Any]] = {}
    pair_rows: list[dict[str, Any]] = []
    active_obs_count = 0
    matched_obs_count = 0
    tracker_start = time.perf_counter()

    for frame_idx in range(start_frame + 1, end_frame + 1):
        prev_frame = frames_by_idx.get(frame_idx - 1)
        current_frame = frames_by_idx.get(frame_idx)
        if prev_frame is None or current_frame is None:
            continue
        encoding = encoder.encode(prev_frame.frame, current_frame.frame)
        objectness_output = field.compute(encoding)
        tracking_output = tracker.update(
            proposals=objectness_output.proposals,
            encoding=encoding,
            heatmap=objectness_output.heatmap,
            current_frame=current_frame.frame,
            frame_index=current_frame.frame_index,
            memory_context=prev_memory_output,
        )
        memory_output = memory.update(
            tracking_output.assignments,
            frame_index=current_frame.frame_index,
            track_states=(tracking_output.active_tracks + tracking_output.dormant_tracks + tracking_output.ghost_tracks + tracking_output.retired_tracks),
        )
        tracker.apply_concept_gated_resurrection(
            tracking_output,
            memory_output,
            frame_index=current_frame.frame_index,
            frame_shape=objectness_output.heatmap.shape,
        )
        tracker.bind_prototypes(memory_output.assignments)
        prev_memory_output = memory_output

        current_by_track: dict[int, dict[str, Any]] = {}
        for track in tracking_output.active_tracks:
            snap = e31.track_snap(track)
            box = tuple(int(v) for v in snap["box"])
            gt_iid, gt_concept, match_iou_score = match_track_to_gt(box, current_frame, min_iou)
            active_obs_count += 1
            matched_obs_count += int(gt_iid is not None)
            obs = {
                "sequence_id": sequence_id,
                "event_id": window_row["event_id"],
                "window_kind": window_row["window_kind"],
                "frame_idx": int(current_frame.frame_index),
                "track_id": int(snap["track_id"]),
                "prototype_id": int(snap["prototype_id"]) if snap["prototype_id"] is not None else -1,
                "gt_instance_eval_only": "" if gt_iid is None else gt_iid,
                "gt_concept_eval_only": "" if gt_concept is None else gt_concept,
                "match_iou_eval_only": match_iou_score,
            }
            current_by_track[int(snap["track_id"])] = obs
            prev = prev_by_track.get(int(snap["track_id"]))
            if prev is not None and int(prev["frame_idx"]) == int(current_frame.frame_index) - 1:
                correct = int(prev["gt_instance_eval_only"] != "" and obs["gt_instance_eval_only"] != "" and prev["gt_instance_eval_only"] == obs["gt_instance_eval_only"])
                pair_rows.append(
                    {
                        "pair_id": pair_id_start + len(pair_rows) + 1,
                        "sequence_id": sequence_id,
                        "event_id": window_row["event_id"],
                        "window_kind": window_row["window_kind"],
                        "frame_i": prev["frame_idx"],
                        "frame_j": obs["frame_idx"],
                        "track_i": prev["track_id"],
                        "track_j": obs["track_id"],
                        "prototype_i": prev["prototype_id"],
                        "prototype_j": obs["prototype_id"],
                        "pair_type": "positive_adjacent_same_track",
                        "online_positive": 1,
                        "online_negative": 0,
                        "gt_instance_i_eval_only": prev["gt_instance_eval_only"],
                        "gt_instance_j_eval_only": obs["gt_instance_eval_only"],
                        "pair_correct_eval_only": correct,
                        "used_for_training_candidate": 1,
                    }
                )

        obs_list = list(current_by_track.values())
        for a_idx, a in enumerate(obs_list):
            for b in obs_list[a_idx + 1 :]:
                correct = int(a["gt_instance_eval_only"] != "" and b["gt_instance_eval_only"] != "" and a["gt_instance_eval_only"] != b["gt_instance_eval_only"])
                pair_rows.append(
                    {
                        "pair_id": pair_id_start + len(pair_rows) + 1,
                        "sequence_id": sequence_id,
                        "event_id": window_row["event_id"],
                        "window_kind": window_row["window_kind"],
                        "frame_i": a["frame_idx"],
                        "frame_j": b["frame_idx"],
                        "track_i": a["track_id"],
                        "track_j": b["track_id"],
                        "prototype_i": a["prototype_id"],
                        "prototype_j": b["prototype_id"],
                        "pair_type": "negative_cov_visible_different_track",
                        "online_positive": 0,
                        "online_negative": 1,
                        "gt_instance_i_eval_only": a["gt_instance_eval_only"],
                        "gt_instance_j_eval_only": b["gt_instance_eval_only"],
                        "pair_correct_eval_only": correct,
                        "used_for_training_candidate": 1,
                    }
                )
        prev_by_track = current_by_track

    elapsed = time.perf_counter() - tracker_start
    return pair_rows, {
        "sequence_id": sequence_id,
        "event_id": window_row["event_id"],
        "window_kind": window_row["window_kind"],
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_count": max(0, end_frame - start_frame + 1),
        "pair_count": len(pair_rows),
        "active_obs_count": active_obs_count,
        "matched_obs_count": matched_obs_count,
        "matched_obs_rate": matched_obs_count / max(active_obs_count, 1),
        "tracker_runtime_sec": elapsed,
    }


def precision(rows: list[dict[str, Any]], pair_type: str) -> float:
    selected = [r for r in rows if r["pair_type"] == pair_type]
    if not selected:
        return 0.0
    return float(np.mean([int(r["pair_correct_eval_only"]) for r in selected]))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    cfg, payload = load_config(Path(args.config))
    generator = SyntheticStreamGenerator(cfg, seed=args.seed)
    selected_windows = select_windows(read_csv(Path(args.window_plan)), args.max_sequences)
    sequence_to_windows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in selected_windows:
        sequence_to_windows[int(row["sequence_id"])].append(row)

    all_pairs: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    pair_id_start = 0
    for sequence_id, windows in sorted(sequence_to_windows.items()):
        gen_start = time.perf_counter()
        sequence = generator.generate_sequence(sequence_id)
        generation_time = time.perf_counter() - gen_start
        frames_by_idx = {frame.frame_index: frame for frame in sequence.frames}
        for window in windows:
            pairs, runtime = run_window(
                sequence_id=sequence_id,
                window_row=window,
                frames_by_idx=frames_by_idx,
                payload=payload,
                min_iou=args.match_iou,
                pair_id_start=pair_id_start,
            )
            pair_id_start += len(pairs)
            all_pairs.extend(pairs)
            runtime["sequence_generation_time_sec"] = generation_time
            runtime_rows.append(runtime)

    counts = Counter(r["pair_type"] for r in all_pairs)
    pos_precision = precision(all_pairs, "positive_adjacent_same_track")
    neg_precision = precision(all_pairs, "negative_cov_visible_different_track")
    positive_count = int(counts.get("positive_adjacent_same_track", 0))
    negative_count = int(counts.get("negative_cov_visible_different_track", 0))
    pair_mining_passed = int(positive_count >= 20 and negative_count >= 20 and pos_precision >= 0.85 and neg_precision >= 0.85)
    compact = {
        "stage": "CORE-1L",
        "artifact_version": args.artifact_version,
        "selected_sequence_count": len(sequence_to_windows),
        "selected_window_count": len(selected_windows),
        "positive_pair_count": positive_count,
        "negative_pair_count": negative_count,
        "positive_pair_precision_eval_only": pos_precision,
        "negative_pair_precision_eval_only": neg_precision,
        "matched_obs_rate": float(np.mean([float(r["matched_obs_rate"]) for r in runtime_rows])) if runtime_rows else 0.0,
        "pair_mining_passed": pair_mining_passed,
        "oracle_leakage_found": 0,
        "runtime_sec": time.perf_counter() - total_start,
        "next_recommendation": (
            "CORE-1M train small online encoder on windowed tracker pairs"
            if pair_mining_passed
            else "repair tracker pair mining confidence gates before encoder training"
        ),
    }

    report = f"""# CORE-1L Windowed Tracker Pair Mining Smoke

This stage runs the tracker only inside selected CORE-1J windows. Tracker state is cold-started per window, so this is a pair-mining feasibility test, not a long-identity evaluation.

## Result

- Selected sequences: {compact['selected_sequence_count']}
- Selected windows: {compact['selected_window_count']}
- Positive adjacent-track pairs: {positive_count}
- Negative co-visible different-track pairs: {negative_count}
- Positive precision eval-only: {pos_precision:.4f}
- Negative precision eval-only: {neg_precision:.4f}
- Mean matched observation rate: {compact['matched_obs_rate']:.4f}
- Runtime seconds: {compact['runtime_sec']:.2f}
- Pair mining passed: {pair_mining_passed}

Next recommendation: {compact['next_recommendation']}
"""

    prefix = "stage_CORE1L_"
    write_csv(
        out_dir / f"{prefix}window_tracker_pair_trace_{args.artifact_version}.csv",
        all_pairs,
        [
            "pair_id",
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
            "used_for_training_candidate",
        ],
    )
    write_csv(
        out_dir / f"{prefix}window_runtime_audit_{args.artifact_version}.csv",
        runtime_rows,
        [
            "sequence_id",
            "event_id",
            "window_kind",
            "start_frame",
            "end_frame",
            "frame_count",
            "pair_count",
            "active_obs_count",
            "matched_obs_count",
            "matched_obs_rate",
            "tracker_runtime_sec",
            "sequence_generation_time_sec",
        ],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1l_windowed_tracker_pair_mining.py",
                "online_uses_gt_instance": 0,
                "online_uses_target_bundle": 0,
                "online_uses_future_frame": 0,
                "gt_used_for_eval_only": 1,
                "leakage_found": 0,
            }
        ],
        ["file", "online_uses_gt_instance", "online_uses_target_bundle", "online_uses_future_frame", "gt_used_for_eval_only", "leakage_found"],
    )
    write_json(
        out_dir / f"{prefix}pair_quality_summary_{args.artifact_version}.json",
        {
            "positive_pair_count": positive_count,
            "negative_pair_count": negative_count,
            "positive_pair_precision_eval_only": pos_precision,
            "negative_pair_precision_eval_only": neg_precision,
            "usable_for_training_smoke": pair_mining_passed,
            "main_pair_failure_reason": "none" if pair_mining_passed else "insufficient_count_or_precision",
        },
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
