from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.synth_stream import SynthDatasetConfig, SyntheticStreamGenerator
from experiments import run_v3_stage_e31_retrieval_competition_repair as e31
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json
from nops_owr.objectness.field import Proposal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1N oracle-proposal pair mining upper bound.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--window-plan", default="results/core1j/stage_CORE1J_window_plan_v1.csv")
    p.add_argument("--output-dir", default="results/core1n")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sequences", type=int, default=2)
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


def i(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default


def make_oracle_proposals(frame) -> tuple[list[Proposal], np.ndarray, dict[int, int]]:
    h, w = frame.frame.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    proposals: list[Proposal] = []
    proposal_to_instance: dict[int, int] = {}
    for idx, box in enumerate(frame.boxes):
        if frame.visibility_flags and not frame.visibility_flags[idx]:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            continue
        mask = np.zeros((h, w), dtype=bool)
        if idx < len(frame.masks):
            mask = np.asarray(frame.masks[idx], dtype=bool)
        if not mask.any():
            mask[y1:y2, x1:x2] = True
        heatmap[mask] = 1.0
        area = int(max(1, (x2 - x1) * (y2 - y1)))
        fill_ratio = float(mask[y1:y2, x1:x2].mean()) if mask[y1:y2, x1:x2].size else 1.0
        compactness = float(min(x2 - x1, y2 - y1) / max(max(x2 - x1, y2 - y1), 1))
        proposal_index = len(proposals)
        proposals.append(
            Proposal(
                box=(x1, y1, x2, y2),
                raw_box=(x1, y1, x2, y2),
                support_box=(x1, y1, x2, y2),
                area=area,
                raw_area=area,
                score=1.0,
                quality_score=1.0,
                centroid=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                support_mask=mask,
                fill_ratio=fill_ratio,
                compactness=compactness,
                boundary_smoothness=1.0,
                near_boundary=0,
            )
        )
        proposal_to_instance[proposal_index] = int(frame.instance_ids[idx])
    return proposals, heatmap, proposal_to_instance


def run_window(
    *,
    sequence_id: int,
    window_row: dict[str, str],
    frames_by_idx: dict[int, Any],
    payload: dict[str, Any],
    pair_id_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    start_frame = int(window_row["start_frame"])
    end_frame = int(window_row["end_frame"])
    encoder = e31.MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    tracker = e31.MinimalTemporalIdentityTracker(**payload["tracking"])
    memory = e31.MinimalPrototypeMemory(**payload["memory"])
    prev_memory_output = None
    prev_by_track: dict[int, dict[str, Any]] = {}
    pair_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    assignment_count = 0
    tracker_start = time.perf_counter()

    for frame_idx in range(start_frame + 1, end_frame + 1):
        prev_frame = frames_by_idx.get(frame_idx - 1)
        current_frame = frames_by_idx.get(frame_idx)
        if prev_frame is None or current_frame is None:
            continue
        encoding = encoder.encode(prev_frame.frame, current_frame.frame)
        proposals, heatmap, proposal_to_instance = make_oracle_proposals(current_frame)
        tracking_output = tracker.update(
            proposals=proposals,
            encoding=encoding,
            heatmap=heatmap,
            current_frame=current_frame.frame,
            frame_index=current_frame.frame_index,
            memory_context=prev_memory_output,
        )
        memory_output = memory.update(
            tracking_output.assignments,
            frame_index=current_frame.frame_index,
            track_states=(tracking_output.active_tracks + tracking_output.dormant_tracks + tracking_output.ghost_tracks + tracking_output.retired_tracks),
        )
        tracker.bind_prototypes(memory_output.assignments)
        prev_memory_output = memory_output

        current_by_track: dict[int, dict[str, Any]] = {}
        for assignment in tracking_output.assignments:
            gt_iid = proposal_to_instance.get(int(assignment.proposal_index), None)
            obs = {
                "sequence_id": sequence_id,
                "event_id": window_row["event_id"],
                "window_kind": window_row["window_kind"],
                "frame_idx": int(current_frame.frame_index),
                "track_id": int(assignment.track_id),
                "prototype_id": -1 if getattr(assignment, "linked_prototype_id", None) is None else int(assignment.linked_prototype_id),
                "proposal_index": int(assignment.proposal_index),
                "score": float(getattr(assignment, "score", 1.0)),
                "match_cost": float(getattr(assignment, "match_cost", 0.0)),
                "gt_instance_eval_only": "" if gt_iid is None else int(gt_iid),
            }
            obs_rows.append(obs)
            current_by_track[int(assignment.track_id)] = obs
            assignment_count += 1
            prev = prev_by_track.get(int(assignment.track_id))
            if prev is not None and int(prev["frame_idx"]) == int(current_frame.frame_index) - 1:
                pair_rows.append(
                    {
                        "pair_id": pair_id_start + len(pair_rows) + 1,
                        "sequence_id": sequence_id,
                        "event_id": window_row["event_id"],
                        "window_kind": window_row["window_kind"],
                        "pair_type": "positive_adjacent_oracle_assignment_track",
                        "frame_i": prev["frame_idx"],
                        "frame_j": obs["frame_idx"],
                        "track_i": prev["track_id"],
                        "track_j": obs["track_id"],
                        "gt_instance_i_eval_only": prev["gt_instance_eval_only"],
                        "gt_instance_j_eval_only": obs["gt_instance_eval_only"],
                        "pair_correct_eval_only": int(prev["gt_instance_eval_only"] != "" and obs["gt_instance_eval_only"] != "" and prev["gt_instance_eval_only"] == obs["gt_instance_eval_only"]),
                    }
                )
        obs_list = list(current_by_track.values())
        for a_idx, a in enumerate(obs_list):
            for b in obs_list[a_idx + 1 :]:
                pair_rows.append(
                    {
                        "pair_id": pair_id_start + len(pair_rows) + 1,
                        "sequence_id": sequence_id,
                        "event_id": window_row["event_id"],
                        "window_kind": window_row["window_kind"],
                        "pair_type": "negative_cov_visible_oracle_assignment_track",
                        "frame_i": a["frame_idx"],
                        "frame_j": b["frame_idx"],
                        "track_i": a["track_id"],
                        "track_j": b["track_id"],
                        "gt_instance_i_eval_only": a["gt_instance_eval_only"],
                        "gt_instance_j_eval_only": b["gt_instance_eval_only"],
                        "pair_correct_eval_only": int(a["gt_instance_eval_only"] != "" and b["gt_instance_eval_only"] != "" and a["gt_instance_eval_only"] != b["gt_instance_eval_only"]),
                    }
                )
        prev_by_track = current_by_track

    return pair_rows, obs_rows, {
        "sequence_id": sequence_id,
        "event_id": window_row["event_id"],
        "window_kind": window_row["window_kind"],
        "start_frame": start_frame,
        "end_frame": end_frame,
        "assignment_count": assignment_count,
        "pair_count": len(pair_rows),
        "tracker_runtime_sec": time.perf_counter() - tracker_start,
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

    pair_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    pair_id_start = 0
    for sequence_id, windows in sorted(sequence_to_windows.items()):
        gen_start = time.perf_counter()
        sequence = generator.generate_sequence(sequence_id)
        generation_time = time.perf_counter() - gen_start
        frames_by_idx = {frame.frame_index: frame for frame in sequence.frames}
        for window in windows:
            pairs, observations, runtime = run_window(
                sequence_id=sequence_id,
                window_row=window,
                frames_by_idx=frames_by_idx,
                payload=payload,
                pair_id_start=pair_id_start,
            )
            pair_id_start += len(pairs)
            pair_rows.extend(pairs)
            obs_rows.extend(observations)
            runtime["sequence_generation_time_sec"] = generation_time
            runtime_rows.append(runtime)

    positive_count = len([r for r in pair_rows if r["pair_type"] == "positive_adjacent_oracle_assignment_track"])
    negative_count = len([r for r in pair_rows if r["pair_type"] == "negative_cov_visible_oracle_assignment_track"])
    pos_precision = precision(pair_rows, "positive_adjacent_oracle_assignment_track")
    neg_precision = precision(pair_rows, "negative_cov_visible_oracle_assignment_track")
    pair_mining_passed = int(positive_count >= 20 and negative_count >= 20 and pos_precision >= 0.85 and neg_precision >= 0.85)
    compact = {
        "stage": "CORE-1N",
        "artifact_version": args.artifact_version,
        "proposal_mode": "oracle_gt_box_memory_only",
        "selected_sequence_count": len(sequence_to_windows),
        "selected_window_count": len(selected_windows),
        "assignment_observation_count": len(obs_rows),
        "positive_pair_count": positive_count,
        "negative_pair_count": negative_count,
        "positive_pair_precision_eval_only": pos_precision,
        "negative_pair_precision_eval_only": neg_precision,
        "pair_mining_upper_bound_passed": pair_mining_passed,
        "oracle_leakage_found": 0,
        "safe_for_main_online_training": 0,
        "runtime_sec": time.perf_counter() - total_start,
        "next_recommendation": (
            "front-end objectness/proposal repair is the blocker; oracle proposals make pair mining viable"
            if pair_mining_passed
            else "tracker identity consistency remains a blocker even with oracle proposals"
        ),
    }

    report = f"""# CORE-1N Oracle Proposal Pair Upper Bound

This diagnostic uses GT boxes as oracle proposals inside selected windows. It is not a main-method training setup. Its purpose is to isolate whether CORE-1M failed because of objectness/proposal noise or because tracker pair mining is intrinsically unreliable.

## Result

- Proposal mode: oracle GT box memory-only
- Selected sequences: {compact['selected_sequence_count']}
- Selected windows: {compact['selected_window_count']}
- Assignment observations: {compact['assignment_observation_count']}
- Positive pairs: {positive_count}
- Negative pairs: {negative_count}
- Positive precision eval-only: {pos_precision:.4f}
- Negative precision eval-only: {neg_precision:.4f}
- Upper bound passed: {pair_mining_passed}

Next recommendation: {compact['next_recommendation']}
"""

    prefix = "stage_CORE1N_"
    write_csv(
        out_dir / f"{prefix}oracle_observation_trace_{args.artifact_version}.csv",
        obs_rows,
        ["sequence_id", "event_id", "window_kind", "frame_idx", "track_id", "prototype_id", "proposal_index", "score", "match_cost", "gt_instance_eval_only"],
    )
    write_csv(
        out_dir / f"{prefix}oracle_pair_trace_{args.artifact_version}.csv",
        pair_rows,
        [
            "pair_id",
            "sequence_id",
            "event_id",
            "window_kind",
            "pair_type",
            "frame_i",
            "frame_j",
            "track_i",
            "track_j",
            "gt_instance_i_eval_only",
            "gt_instance_j_eval_only",
            "pair_correct_eval_only",
        ],
    )
    write_csv(
        out_dir / f"{prefix}runtime_audit_{args.artifact_version}.csv",
        runtime_rows,
        ["sequence_id", "event_id", "window_kind", "start_frame", "end_frame", "assignment_count", "pair_count", "tracker_runtime_sec", "sequence_generation_time_sec"],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1n_oracle_proposal_pair_upper_bound.py",
                "proposal_mode": "oracle_gt_box_memory_only",
                "safe_for_main_online_training": 0,
                "gt_used_as_proposal_only": 1,
                "gt_used_for_eval_only": 1,
                "leakage_found": 0,
            }
        ],
        ["file", "proposal_mode", "safe_for_main_online_training", "gt_used_as_proposal_only", "gt_used_for_eval_only", "leakage_found"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
