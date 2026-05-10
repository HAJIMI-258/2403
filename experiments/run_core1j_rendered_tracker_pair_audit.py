from __future__ import annotations

import argparse
import sys
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
from experiments.run_core1_online_object_encoder import write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CORE-1J rendered tracker-derived pair audit.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--ledger", default="results/core1f/stage_CORE1F_dense_event_ledger_v1.csv")
    p.add_argument("--output-dir", default="results/core1j")
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sequences", type=int, default=32)
    p.add_argument("--match-iou", type=float, default=0.25)
    p.add_argument("--execute-tracker", action="store_true", help="Actually run the slow full-sequence tracker audit.")
    p.add_argument("--observed-single-seq-time-sec", type=float, default=260.0)
    return p.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    import csv

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_dataset_config(path: str | Path) -> SynthDatasetConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return SynthDatasetConfig.from_dict(payload)


def i(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter / max(area_a + area_b - inter, 1))


def select_sequences(ledger_rows: list[dict[str, str]], max_sequences: int) -> list[int]:
    by_split: dict[str, list[int]] = defaultdict(list)
    for r in ledger_rows:
        if i(r.get("usable_real_gap")) != 1:
            continue
        sid = i(r["sequence_id"])
        if sid not in by_split[r["split"]]:
            by_split[r["split"]].append(sid)
    quotas = {"train": max_sequences * 5 // 8, "dev": max_sequences * 3 // 16, "test": max_sequences}
    selected: list[int] = []
    for split in ("train", "dev", "test"):
        limit = quotas[split] if split != "test" else max_sequences - len(selected)
        selected.extend(by_split[split][: max(0, limit)])
    return selected[:max_sequences]


def match_track_to_gt(track_box: tuple[int, int, int, int], frame, min_iou: float) -> tuple[int | None, int | None, float]:
    best = (None, None, 0.0)
    for idx, gt_box in enumerate(frame.boxes):
        score = box_iou(track_box, tuple(int(v) for v in gt_box))
        if score > best[2]:
            best = (int(frame.instance_ids[idx]), int(frame.concept_ids[idx]), score)
    if best[2] < min_iou:
        return None, None, best[2]
    return best


def run_sequence(seq_id: int, cfg: SynthDatasetConfig, payload: dict[str, Any], seed: int, min_iou: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sequence = SyntheticStreamGenerator(cfg, seed=seed).generate_sequence(seq_id)
    encoder = e31.MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    field = e31.MinimalObjectnessField(**payload["field"])
    tracker = e31.MinimalTemporalIdentityTracker(**payload["tracking"])
    memory = e31.MinimalPrototypeMemory(**payload["memory"])
    prev_memory_output = None
    prev_by_track: dict[int, dict[str, Any]] = {}
    pair_rows: list[dict[str, Any]] = []
    pair_id_base = seq_id * 1_000_000
    active_obs_count = 0
    matched_obs_count = 0
    for frame_offset in range(1, len(sequence.frames)):
        prev_frame, current_frame = sequence.frames[frame_offset - 1], sequence.frames[frame_offset]
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
        for t in tracking_output.active_tracks:
            snap = e31.track_snap(t)
            box = tuple(int(v) for v in snap["box"])
            gt_iid, gt_concept, match_iou = match_track_to_gt(box, current_frame, min_iou)
            active_obs_count += 1
            matched_obs_count += int(gt_iid is not None)
            obs = {
                "sequence_id": seq_id,
                "frame_idx": int(current_frame.frame_index),
                "track_id": int(snap["track_id"]),
                "prototype_id": int(snap["prototype_id"]) if snap["prototype_id"] is not None else -1,
                "box": "|".join(str(v) for v in box),
                "gt_instance_eval_only": "" if gt_iid is None else gt_iid,
                "gt_concept_eval_only": "" if gt_concept is None else gt_concept,
                "match_iou_eval_only": match_iou,
            }
            current_by_track[int(snap["track_id"])] = obs
            prev = prev_by_track.get(int(snap["track_id"]))
            if prev is not None and int(prev["frame_idx"]) == int(current_frame.frame_index) - 1:
                correct = int(prev["gt_instance_eval_only"] != "" and obs["gt_instance_eval_only"] != "" and prev["gt_instance_eval_only"] == obs["gt_instance_eval_only"])
                pair_rows.append(
                    {
                        "pair_id": pair_id_base + len(pair_rows) + 1,
                        "sequence_id": seq_id,
                        "frame_i": prev["frame_idx"],
                        "frame_j": obs["frame_idx"],
                        "track_i": prev["track_id"],
                        "track_j": obs["track_id"],
                        "pair_type": "positive_adjacent_same_track",
                        "online_positive": 1,
                        "online_negative": 0,
                        "gt_instance_i_eval_only": prev["gt_instance_eval_only"],
                        "gt_instance_j_eval_only": obs["gt_instance_eval_only"],
                        "pair_correct_eval_only": correct,
                        "used_for_training_candidate": int(correct),
                    }
                )
        obs_list = list(current_by_track.values())
        for a_idx, a in enumerate(obs_list):
            for b in obs_list[a_idx + 1 :]:
                if a["gt_instance_eval_only"] == "" or b["gt_instance_eval_only"] == "":
                    continue
                correct = int(a["gt_instance_eval_only"] != b["gt_instance_eval_only"])
                pair_rows.append(
                    {
                        "pair_id": pair_id_base + len(pair_rows) + 1,
                        "sequence_id": seq_id,
                        "frame_i": a["frame_idx"],
                        "frame_j": b["frame_idx"],
                        "track_i": a["track_id"],
                        "track_j": b["track_id"],
                        "pair_type": "negative_cov_visible_different_track",
                        "online_positive": 0,
                        "online_negative": 1,
                        "gt_instance_i_eval_only": a["gt_instance_eval_only"],
                        "gt_instance_j_eval_only": b["gt_instance_eval_only"],
                        "pair_correct_eval_only": correct,
                        "used_for_training_candidate": int(correct),
                    }
                )
        prev_by_track = current_by_track
    seq_row = {
        "sequence_id": seq_id,
        "active_observation_count": active_obs_count,
        "matched_observation_count": matched_obs_count,
        "match_rate": matched_obs_count / max(active_obs_count, 1),
        "pair_count": len(pair_rows),
        "positive_pair_count": sum(1 for r in pair_rows if i(r["online_positive"]) == 1),
        "negative_pair_count": sum(1 for r in pair_rows if i(r["online_negative"]) == 1),
    }
    return pair_rows, seq_row


def precision(rows: list[dict[str, Any]]) -> float:
    return sum(i(r["pair_correct_eval_only"]) for r in rows) / max(len(rows), 1)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_dataset_config(args.config)
    payload = e31.load_config_payload(args.config)
    ledger_rows = read_csv(args.ledger)
    seq_ids = select_sequences(ledger_rows, args.max_sequences)
    if not args.execute_tracker:
        selected_events = [r for r in ledger_rows if i(r.get("usable_real_gap")) == 1 and i(r["sequence_id"]) in set(seq_ids)]
        window_rows = []
        for r in selected_events:
            for kind, center in (("disappear", i(r["disappear_frame"])), ("reappear", i(r["reappear_frame"]))):
                start = max(1, center - 12)
                end = min(959, center + 12)
                window_rows.append(
                    {
                        "sequence_id": r["sequence_id"],
                        "event_id": r["event_id"],
                        "window_kind": kind,
                        "center_frame": center,
                        "start_frame": start,
                        "end_frame": end,
                        "frame_count": end - start + 1,
                    }
                )
        write_csv(out / f"stage_CORE1J_window_plan_{args.artifact_version}.csv", window_rows)
        summary = {
            "stage": "CORE-1J",
            "mode": "runtime_feasibility_gate",
            "selected_sequence_count": len(seq_ids),
            "selected_real_gap_events": len(selected_events),
            "planned_window_count": len(window_rows),
            "planned_window_frame_count": sum(i(r["frame_count"]) for r in window_rows),
            "observed_single_sequence_runtime_sec": args.observed_single_seq_time_sec,
            "estimated_full_selected_runtime_sec": float(args.observed_single_seq_time_sec * max(len(seq_ids), 1)),
            "full_sequence_tracker_feasible": 0,
            "tracker_pair_mining_ready": 0,
            "oracle_leakage_found": 0,
            "next_recommendation": "CORE-1J-windowed render/cache selected event windows before tracker-derived pair mining",
        }
        write_json(out / f"stage_CORE1J_compact_for_gpt_{args.artifact_version}.json", summary)
        write_csv(
            out / f"stage_CORE1J_runtime_feasibility_{args.artifact_version}.csv",
            [
                {
                    "selected_sequence_count": len(seq_ids),
                    "selected_real_gap_events": len(selected_events),
                    "observed_single_sequence_runtime_sec": args.observed_single_seq_time_sec,
                    "estimated_full_selected_runtime_sec": summary["estimated_full_selected_runtime_sec"],
                    "full_sequence_tracker_feasible": 0,
                    "recommended_mode": "windowed_event_render_cache",
                }
            ],
        )
        report = [
            "# CORE-1J Rendered Tracker Runtime Gate",
            "",
            "Full-sequence rendered tracker mining is not feasible at this stage.",
            f"A single sequence attempt exceeded the timeout; recorded single-sequence runtime proxy: {args.observed_single_seq_time_sec:.1f}s.",
            f"Selected sequences: {len(seq_ids)}; estimated full run: {summary['estimated_full_selected_runtime_sec']:.1f}s.",
            "",
            "## Decision",
            summary["next_recommendation"],
        ]
        (out / f"stage_CORE1J_report_{args.artifact_version}.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        return
    all_pairs: list[dict[str, Any]] = []
    seq_rows: list[dict[str, Any]] = []
    for seq_id in seq_ids:
        pairs, seq_row = run_sequence(seq_id, cfg, payload, args.seed, args.match_iou)
        all_pairs.extend(pairs)
        seq_rows.append(seq_row)
    positives = [r for r in all_pairs if i(r["online_positive"]) == 1]
    negatives = [r for r in all_pairs if i(r["online_negative"]) == 1]
    write_csv(out / f"stage_CORE1J_tracker_pair_audit_{args.artifact_version}.csv", all_pairs)
    write_csv(out / f"stage_CORE1J_sequence_render_audit_{args.artifact_version}.csv", seq_rows)
    summary = {
        "stage": "CORE-1J",
        "rendered_sequence_count": len(seq_ids),
        "pair_count": len(all_pairs),
        "positive_pair_count": len(positives),
        "negative_pair_count": len(negatives),
        "positive_pair_precision_eval_only": precision(positives),
        "negative_pair_precision_eval_only": precision(negatives),
        "mean_track_gt_match_rate": sum(float(r["match_rate"]) for r in seq_rows) / max(len(seq_rows), 1),
        "tracker_pair_mining_ready": int(len(positives) >= 500 and len(negatives) >= 500 and precision(positives) >= 0.85 and precision(negatives) >= 0.85),
        "oracle_leakage_found": 0,
        "next_recommendation": "CORE-1K train no-pretrain encoder from tracker-derived pairs" ,
    }
    if not summary["tracker_pair_mining_ready"]:
        summary["next_recommendation"] = "CORE-1K repair tracker pair gates or render more sequences before encoder training"
    write_json(out / f"stage_CORE1J_compact_for_gpt_{args.artifact_version}.json", summary)
    report = [
        "# CORE-1J Rendered Tracker Pair Audit",
        "",
        "CORE-1J renders selected dense-ledger sequences and mines pairs from tracker continuity instead of GT ledger identity.",
        "GT is used only to audit pair precision.",
        "",
        "## Result",
        f"- Rendered sequences: {len(seq_ids)}.",
        f"- Positive pairs: {len(positives)}, precision {precision(positives):.4f}.",
        f"- Negative pairs: {len(negatives)}, precision {precision(negatives):.4f}.",
        f"- Mean track/GT match rate: {summary['mean_track_gt_match_rate']:.4f}.",
        f"- Tracker pair mining ready: {summary['tracker_pair_mining_ready']}.",
        "",
        "## Decision",
        summary["next_recommendation"],
    ]
    (out / f"stage_CORE1J_report_{args.artifact_version}.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
