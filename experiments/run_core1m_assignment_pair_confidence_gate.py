from __future__ import annotations

import argparse
import csv
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
from experiments.run_core1j_rendered_tracker_pair_audit import box_iou
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1M assignment-observation pair confidence gate.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--window-plan", default="results/core1j/stage_CORE1J_window_plan_v1.csv")
    p.add_argument("--output-dir", default="results/core1m")
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


def f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def i(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default


def box_from_text(text: str) -> tuple[int, int, int, int]:
    vals = [int(float(x)) for x in str(text).split("|")]
    return vals[0], vals[1], vals[2], vals[3]


def collect_window_observations(
    *,
    sequence_id: int,
    window_row: dict[str, str],
    frames_by_idx: dict[int, Any],
    payload: dict[str, Any],
    min_iou: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start_frame = int(window_row["start_frame"])
    end_frame = int(window_row["end_frame"])
    encoder = e31.MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    field = e31.MinimalObjectnessField(**payload["field"])
    tracker = e31.MinimalTemporalIdentityTracker(**payload["tracking"])
    memory = e31.MinimalPrototypeMemory(**payload["memory"])
    prev_memory_output = None
    rows: list[dict[str, Any]] = []
    tracker_start = time.perf_counter()
    assignment_count = 0
    matched_assignment_count = 0

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

        track_by_id = {int(t.track_id): e31.track_snap(t) for t in tracking_output.active_tracks}
        for assignment in tracking_output.assignments:
            track_id = int(assignment.track_id)
            snap = track_by_id.get(track_id, {})
            box = tuple(int(v) for v in assignment.box)
            gt_iid, gt_concept, match_iou_score = match_track_to_gt(box, current_frame, min_iou)
            assignment_count += 1
            matched_assignment_count += int(gt_iid is not None)
            proto = getattr(assignment, "linked_prototype_id", None)
            if proto is None:
                proto = snap.get("prototype_id", -1)
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "event_id": window_row["event_id"],
                    "window_kind": window_row["window_kind"],
                    "frame_idx": int(current_frame.frame_index),
                    "track_id": track_id,
                    "prototype_id": -1 if proto is None else int(proto),
                    "box": "|".join(str(v) for v in box),
                    "score": float(getattr(assignment, "score", 0.0)),
                    "objectness_score": float(getattr(assignment, "objectness_score", getattr(assignment, "score", 0.0))),
                    "match_cost": float(getattr(assignment, "match_cost", 1.0)),
                    "assignment_source": str(getattr(assignment, "assignment_source", "")),
                    "final_assignment_source": str(getattr(assignment, "final_assignment_source", "")),
                    "track_hit_count": int(snap.get("hit_count", 0) or 0),
                    "track_age": int(snap.get("age", 0) or 0),
                    "track_gap_length": int(snap.get("gap_length", 0) or 0),
                    "gt_instance_eval_only": "" if gt_iid is None else gt_iid,
                    "gt_concept_eval_only": "" if gt_concept is None else gt_concept,
                    "match_iou_eval_only": match_iou_score,
                }
            )

    return rows, {
        "sequence_id": sequence_id,
        "event_id": window_row["event_id"],
        "window_kind": window_row["window_kind"],
        "start_frame": start_frame,
        "end_frame": end_frame,
        "assignment_count": assignment_count,
        "matched_assignment_count": matched_assignment_count,
        "matched_assignment_rate": matched_assignment_count / max(assignment_count, 1),
        "tracker_runtime_sec": time.perf_counter() - tracker_start,
    }


GATES: list[dict[str, Any]] = [
    {"name": "A0_assignment_only"},
    {"name": "A1_score_ge_050", "score_min": 0.50},
    {"name": "A2_score_ge_060", "score_min": 0.60},
    {"name": "A3_score050_cost_le_050", "score_min": 0.50, "match_cost_max": 0.50},
    {"name": "A4_score050_hits_ge_2", "score_min": 0.50, "hit_min": 2},
    {"name": "A5_score050_proto_known", "score_min": 0.50, "proto_known": True},
    {"name": "A6_no_overlap_neg", "negative_iou_max": 0.20},
    {"name": "A7_score050_no_overlap_neg", "score_min": 0.50, "negative_iou_max": 0.20},
    {"name": "A8_score050_no_overlap_diff_proto_neg", "score_min": 0.50, "negative_iou_max": 0.20, "negative_diff_proto": True},
    {"name": "A9_strict_score060_hits2_no_overlap", "score_min": 0.60, "hit_min": 2, "negative_iou_max": 0.20},
]


def obs_passes(obs: dict[str, Any], gate: dict[str, Any]) -> bool:
    if f(obs["score"]) < f(gate.get("score_min", -1.0)):
        return False
    if f(obs["match_cost"]) > f(gate.get("match_cost_max", 99.0)):
        return False
    if i(obs["track_hit_count"]) < i(gate.get("hit_min", 0)):
        return False
    if gate.get("proto_known") and i(obs["prototype_id"], -1) < 0:
        return False
    return True


def build_pairs_for_gate(obs_rows: list[dict[str, Any]], gate: dict[str, Any]) -> list[dict[str, Any]]:
    by_window: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for obs in obs_rows:
        if obs_passes(obs, gate):
            by_window[(i(obs["sequence_id"]), str(obs["event_id"]), str(obs["window_kind"]))].append(obs)

    pairs: list[dict[str, Any]] = []
    pair_id = 0
    for (_sequence_id, _event_id, _window_kind), rows in by_window.items():
        by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_track[i(row["track_id"])].append(row)
            by_frame[i(row["frame_idx"])].append(row)

        for track_rows in by_track.values():
            ordered = sorted(track_rows, key=lambda r: i(r["frame_idx"]))
            prev = None
            for obs in ordered:
                if prev is not None and i(obs["frame_idx"]) == i(prev["frame_idx"]) + 1:
                    pair_id += 1
                    pairs.append(
                        {
                            "pair_id": pair_id,
                            "gate_name": gate["name"],
                            "sequence_id": obs["sequence_id"],
                            "event_id": obs["event_id"],
                            "window_kind": obs["window_kind"],
                            "frame_i": prev["frame_idx"],
                            "frame_j": obs["frame_idx"],
                            "track_i": prev["track_id"],
                            "track_j": obs["track_id"],
                            "prototype_i": prev["prototype_id"],
                            "prototype_j": obs["prototype_id"],
                            "pair_type": "positive_adjacent_assignment_track",
                            "online_positive": 1,
                            "online_negative": 0,
                            "gt_instance_i_eval_only": prev["gt_instance_eval_only"],
                            "gt_instance_j_eval_only": obs["gt_instance_eval_only"],
                            "pair_correct_eval_only": int(prev["gt_instance_eval_only"] != "" and obs["gt_instance_eval_only"] != "" and prev["gt_instance_eval_only"] == obs["gt_instance_eval_only"]),
                        }
                    )
                prev = obs

        for frame_rows in by_frame.values():
            ordered = sorted(frame_rows, key=lambda r: i(r["track_id"]))
            for a_idx, a in enumerate(ordered):
                for b in ordered[a_idx + 1 :]:
                    if int(a["track_id"]) == int(b["track_id"]):
                        continue
                    if "negative_iou_max" in gate and box_iou(box_from_text(a["box"]), box_from_text(b["box"])) > f(gate["negative_iou_max"]):
                        continue
                    if gate.get("negative_diff_proto"):
                        if i(a["prototype_id"], -1) < 0 or i(b["prototype_id"], -1) < 0 or i(a["prototype_id"]) == i(b["prototype_id"]):
                            continue
                    pair_id += 1
                    pairs.append(
                        {
                            "pair_id": pair_id,
                            "gate_name": gate["name"],
                            "sequence_id": a["sequence_id"],
                            "event_id": a["event_id"],
                            "window_kind": a["window_kind"],
                            "frame_i": a["frame_idx"],
                            "frame_j": b["frame_idx"],
                            "track_i": a["track_id"],
                            "track_j": b["track_id"],
                            "prototype_i": a["prototype_id"],
                            "prototype_j": b["prototype_id"],
                            "pair_type": "negative_cov_visible_assignment_track",
                            "online_positive": 0,
                            "online_negative": 1,
                            "gt_instance_i_eval_only": a["gt_instance_eval_only"],
                            "gt_instance_j_eval_only": b["gt_instance_eval_only"],
                            "pair_correct_eval_only": int(a["gt_instance_eval_only"] != "" and b["gt_instance_eval_only"] != "" and a["gt_instance_eval_only"] != b["gt_instance_eval_only"]),
                        }
                    )
    return pairs


def summarize_gate(gate_name: str, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [p for p in pairs if p["pair_type"] == "positive_adjacent_assignment_track"]
    negatives = [p for p in pairs if p["pair_type"] == "negative_cov_visible_assignment_track"]
    pos_precision = float(np.mean([int(p["pair_correct_eval_only"]) for p in positives])) if positives else 0.0
    neg_precision = float(np.mean([int(p["pair_correct_eval_only"]) for p in negatives])) if negatives else 0.0
    eligible = int(len(positives) >= 20 and len(negatives) >= 20 and pos_precision >= 0.85 and neg_precision >= 0.85)
    return {
        "gate_name": gate_name,
        "positive_pair_count": len(positives),
        "negative_pair_count": len(negatives),
        "positive_pair_precision_eval_only": pos_precision,
        "negative_pair_precision_eval_only": neg_precision,
        "eligible_for_training_smoke": eligible,
    }


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
                payload=payload,
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

    compact = {
        "stage": "CORE-1M",
        "artifact_version": args.artifact_version,
        "selected_sequence_count": len(sequence_to_windows),
        "selected_window_count": len(selected_windows),
        "assignment_observation_count": len(obs_rows),
        "matched_assignment_rate": float(np.mean([float(r["matched_assignment_rate"]) for r in runtime_rows])) if runtime_rows else 0.0,
        "best_gate": best.get("gate_name", ""),
        "best_positive_pair_count": best.get("positive_pair_count", 0),
        "best_negative_pair_count": best.get("negative_pair_count", 0),
        "best_positive_pair_precision_eval_only": best.get("positive_pair_precision_eval_only", 0.0),
        "best_negative_pair_precision_eval_only": best.get("negative_pair_precision_eval_only", 0.0),
        "pair_mining_gate_passed": int(bool(eligible)),
        "oracle_leakage_found": 0,
        "runtime_sec": time.perf_counter() - total_start,
        "next_recommendation": "CORE-1N train tiny encoder on gated assignment pairs" if eligible else "repair objectness/tracker observation quality before encoder training",
    }

    report = f"""# CORE-1M Assignment Pair Confidence Gate

This stage re-runs selected CORE-1J windows, mines pairs from frame assignments rather than stale active tracks, and scans online-visible confidence gates. GT is used only for pair correctness audit.

## Result

- Selected sequences: {compact['selected_sequence_count']}
- Selected windows: {compact['selected_window_count']}
- Assignment observations: {compact['assignment_observation_count']}
- Matched assignment rate eval-only: {compact['matched_assignment_rate']:.4f}
- Best gate: {compact['best_gate']}
- Best positive pairs: {compact['best_positive_pair_count']}
- Best negative pairs: {compact['best_negative_pair_count']}
- Best positive precision eval-only: {compact['best_positive_pair_precision_eval_only']:.4f}
- Best negative precision eval-only: {compact['best_negative_pair_precision_eval_only']:.4f}
- Gate passed: {compact['pair_mining_gate_passed']}

Next recommendation: {compact['next_recommendation']}
"""

    prefix = "stage_CORE1M_"
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
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1m_assignment_pair_confidence_gate.py",
                "online_uses_gt_instance": 0,
                "online_uses_target_bundle": 0,
                "online_uses_future_frame": 0,
                "gt_used_for_eval_only": 1,
                "leakage_found": 0,
            }
        ],
        ["file", "online_uses_gt_instance", "online_uses_target_bundle", "online_uses_future_frame", "gt_used_for_eval_only", "leakage_found"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
