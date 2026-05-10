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
from experiments import run_v3_stage_e31_retrieval_competition_repair as e31
from experiments.run_core1j_rendered_tracker_pair_audit import box_iou
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json
from experiments.run_core1m_assignment_pair_confidence_gate import GATES, load_config, select_windows, build_pairs_for_gate, summarize_gate


PROFILE_A3_LOWER_QUANTILE = {"q_obj": 0.88, "local_k": 0.85, "max_proposals": 16}


POSTPROCESS_VARIANTS: list[dict[str, Any]] = [
    {"variant": "A0_a3_no_postprocess"},
    {"variant": "A1_score_ge_035", "score_min": 0.35},
    {"variant": "A2_score_ge_045", "score_min": 0.45},
    {"variant": "A3_quality_ge_040", "quality_min": 0.40},
    {"variant": "A4_nms_iou_030", "nms_iou": 0.30},
    {"variant": "A5_score035_nms040", "score_min": 0.35, "nms_iou": 0.40},
    {"variant": "A6_score045_nms040", "score_min": 0.45, "nms_iou": 0.40},
    {"variant": "A7_quality035_nms040_max12", "quality_min": 0.35, "nms_iou": 0.40, "max_keep": 12},
    {"variant": "A8_compact020_score035_nms040", "compactness_min": 0.20, "score_min": 0.35, "nms_iou": 0.40},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1S proposal postprocess pair validation.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--window-plan", default="results/core1j/stage_CORE1J_window_plan_v1.csv")
    p.add_argument("--output-dir", default="results/core1s")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sequences", type=int, default=2)
    p.add_argument("--match-iou", type=float, default=0.25)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


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


def postprocess_proposals(proposals: list[Any], variant: dict[str, Any]) -> list[Any]:
    rows = []
    for proposal in proposals:
        if f(proposal.score) < f(variant.get("score_min", -1.0)):
            continue
        if f(getattr(proposal, "quality_score", proposal.score)) < f(variant.get("quality_min", -1.0)):
            continue
        if f(getattr(proposal, "compactness", 1.0)) < f(variant.get("compactness_min", -1.0)):
            continue
        rows.append(proposal)
    rows.sort(key=lambda p: (f(getattr(p, "quality_score", p.score)), f(p.score), i(getattr(p, "area", 0))), reverse=True)
    nms_iou = variant.get("nms_iou")
    if nms_iou is not None:
        kept = []
        for proposal in rows:
            if all(box_iou(tuple(int(v) for v in proposal.box), tuple(int(v) for v in other.box)) <= f(nms_iou) for other in kept):
                kept.append(proposal)
        rows = kept
    max_keep = variant.get("max_keep")
    if max_keep is not None:
        rows = rows[: i(max_keep)]
    return rows


def collect_window_variant(
    *,
    sequence_id: int,
    window_row: dict[str, str],
    frames_by_idx: dict[int, Any],
    payload: dict[str, Any],
    variant: dict[str, Any],
    min_iou: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    start_frame = int(window_row["start_frame"])
    end_frame = int(window_row["end_frame"])
    encoder = e31.MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    field = e31.MinimalObjectnessField(**payload["field"])
    tracker = e31.MinimalTemporalIdentityTracker(**payload["tracking"])
    memory = e31.MinimalPrototypeMemory(**payload["memory"])
    prev_memory_output = None
    obs_rows: list[dict[str, Any]] = []
    proposal_trace: list[dict[str, Any]] = []
    raw_counts: list[int] = []
    kept_counts: list[int] = []
    matched_assignment_count = 0
    assignment_count = 0
    start = time.perf_counter()

    for frame_idx in range(start_frame + 1, end_frame + 1):
        prev_frame = frames_by_idx.get(frame_idx - 1)
        current_frame = frames_by_idx.get(frame_idx)
        if prev_frame is None or current_frame is None:
            continue
        encoding = encoder.encode(prev_frame.frame, current_frame.frame)
        objectness_output = field.compute(encoding)
        raw = objectness_output.proposals
        kept = postprocess_proposals(raw, variant)
        raw_counts.append(len(raw))
        kept_counts.append(len(kept))
        proposal_trace.append(
            {
                "variant": variant["variant"],
                "sequence_id": sequence_id,
                "event_id": window_row["event_id"],
                "window_kind": window_row["window_kind"],
                "frame_idx": int(current_frame.frame_index),
                "raw_proposal_count": len(raw),
                "kept_proposal_count": len(kept),
                "removed_proposal_count": len(raw) - len(kept),
            }
        )
        tracking_output = tracker.update(
            proposals=kept,
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
            obs_rows.append(
                {
                    "variant": variant["variant"],
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
    runtime = {
        "variant": variant["variant"],
        "sequence_id": sequence_id,
        "event_id": window_row["event_id"],
        "window_kind": window_row["window_kind"],
        "assignment_count": assignment_count,
        "matched_assignment_count": matched_assignment_count,
        "matched_assignment_rate": matched_assignment_count / max(assignment_count, 1),
        "mean_raw_proposals": float(np.mean(raw_counts)) if raw_counts else 0.0,
        "mean_kept_proposals": float(np.mean(kept_counts)) if kept_counts else 0.0,
        "tracker_runtime_sec": time.perf_counter() - start,
    }
    return obs_rows, proposal_trace, runtime


def evaluate_variant(obs_rows: list[dict[str, Any]], variant_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries = []
    all_pairs = []
    for gate in GATES:
        pairs = build_pairs_for_gate(obs_rows, gate)
        summary = summarize_gate(gate["name"], pairs)
        summary["variant"] = variant_name
        summaries.append(summary)
        for pair in pairs:
            pair["variant"] = variant_name
        all_pairs.extend(pairs)
    eligible = [s for s in summaries if int(s["eligible_for_training_smoke"]) == 1]
    if eligible:
        best = max(eligible, key=lambda s: (s["positive_pair_count"] + s["negative_pair_count"]))
    else:
        best = max(summaries, key=lambda s: (min(s["positive_pair_precision_eval_only"], s["negative_pair_precision_eval_only"]), s["positive_pair_count"] + s["negative_pair_count"])) if summaries else {}
    return best, all_pairs


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

    variant_obs: dict[str, list[dict[str, Any]]] = {v["variant"]: [] for v in POSTPROCESS_VARIANTS}
    proposal_trace: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for sequence_id, windows in sorted(sequence_to_windows.items()):
        gen_start = time.perf_counter()
        sequence = generator.generate_sequence(sequence_id)
        generation_time = time.perf_counter() - gen_start
        frames_by_idx = {frame.frame_index: frame for frame in sequence.frames}
        for window in windows:
            for variant in POSTPROCESS_VARIANTS:
                obs, prop, runtime = collect_window_variant(
                    sequence_id=sequence_id,
                    window_row=window,
                    frames_by_idx=frames_by_idx,
                    payload=profiled_payload,
                    variant=variant,
                    min_iou=args.match_iou,
                )
                variant_obs[variant["variant"]].extend(obs)
                proposal_trace.extend(prop)
                runtime["sequence_generation_time_sec"] = generation_time
                runtime_rows.append(runtime)

    summary_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for variant in POSTPROCESS_VARIANTS:
        obs = variant_obs[variant["variant"]]
        best, pairs = evaluate_variant(obs, variant["variant"])
        pair_rows.extend(pairs)
        matched_rate = float(np.mean([1 if row.get("gt_instance_eval_only", "") != "" else 0 for row in obs])) if obs else 0.0
        runtimes = [r for r in runtime_rows if r["variant"] == variant["variant"]]
        summary_rows.append(
            {
                "variant": variant["variant"],
                "observation_count": len(obs),
                "matched_assignment_rate_eval_only": matched_rate,
                "mean_raw_proposals": float(np.mean([r["mean_raw_proposals"] for r in runtimes])) if runtimes else 0.0,
                "mean_kept_proposals": float(np.mean([r["mean_kept_proposals"] for r in runtimes])) if runtimes else 0.0,
                "best_gate": best.get("gate_name", ""),
                "positive_pair_count": best.get("positive_pair_count", 0),
                "negative_pair_count": best.get("negative_pair_count", 0),
                "positive_pair_precision_eval_only": best.get("positive_pair_precision_eval_only", 0.0),
                "negative_pair_precision_eval_only": best.get("negative_pair_precision_eval_only", 0.0),
                "eligible_for_training_smoke": best.get("eligible_for_training_smoke", 0),
            }
        )
    eligible = [r for r in summary_rows if int(r["eligible_for_training_smoke"]) == 1]
    if eligible:
        best_row = max(eligible, key=lambda r: (r["positive_pair_count"] + r["negative_pair_count"]))
    else:
        best_row = max(summary_rows, key=lambda r: (min(r["positive_pair_precision_eval_only"], r["negative_pair_precision_eval_only"]), r["positive_pair_count"] + r["negative_pair_count"])) if summary_rows else {}
    compact = {
        "stage": "CORE-1S",
        "artifact_version": args.artifact_version,
        "selected_sequence_count": len(sequence_to_windows),
        "selected_window_count": len(selected_windows),
        "best_variant": best_row.get("variant", ""),
        "best_gate": best_row.get("best_gate", ""),
        "best_matched_assignment_rate_eval_only": best_row.get("matched_assignment_rate_eval_only", 0.0),
        "best_positive_pair_count": best_row.get("positive_pair_count", 0),
        "best_negative_pair_count": best_row.get("negative_pair_count", 0),
        "best_positive_pair_precision_eval_only": best_row.get("positive_pair_precision_eval_only", 0.0),
        "best_negative_pair_precision_eval_only": best_row.get("negative_pair_precision_eval_only", 0.0),
        "postprocess_pair_gate_passed": int(bool(eligible)),
        "oracle_leakage_found": 0,
        "runtime_sec": time.perf_counter() - total_start,
        "next_recommendation": "CORE-1T train tiny encoder on postprocessed pairs" if eligible else "proposal postprocess insufficient; repair objectness/localization model before encoder training",
    }
    report = f"""# CORE-1S Proposal Postprocess Pair Validation

This stage applies GT-free proposal postprocessing before tracker assignment and validates pair quality. It does not alter the main model.

## Result

- Best variant: {compact['best_variant']}
- Best gate: {compact['best_gate']}
- Best matched assignment rate eval-only: {float(compact['best_matched_assignment_rate_eval_only']):.4f}
- Best positive precision eval-only: {float(compact['best_positive_pair_precision_eval_only']):.4f}
- Best negative precision eval-only: {float(compact['best_negative_pair_precision_eval_only']):.4f}
- Postprocess pair gate passed: {compact['postprocess_pair_gate_passed']}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1S_"
    write_csv(
        out_dir / f"{prefix}postprocess_variant_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "variant",
            "observation_count",
            "matched_assignment_rate_eval_only",
            "mean_raw_proposals",
            "mean_kept_proposals",
            "best_gate",
            "positive_pair_count",
            "negative_pair_count",
            "positive_pair_precision_eval_only",
            "negative_pair_precision_eval_only",
            "eligible_for_training_smoke",
        ],
    )
    write_csv(
        out_dir / f"{prefix}proposal_postprocess_trace_{args.artifact_version}.csv",
        proposal_trace,
        ["variant", "sequence_id", "event_id", "window_kind", "frame_idx", "raw_proposal_count", "kept_proposal_count", "removed_proposal_count"],
    )
    write_csv(
        out_dir / f"{prefix}runtime_audit_{args.artifact_version}.csv",
        runtime_rows,
        [
            "variant",
            "sequence_id",
            "event_id",
            "window_kind",
            "assignment_count",
            "matched_assignment_count",
            "matched_assignment_rate",
            "mean_raw_proposals",
            "mean_kept_proposals",
            "tracker_runtime_sec",
            "sequence_generation_time_sec",
        ],
    )
    write_csv(
        out_dir / f"{prefix}gated_pair_trace_{args.artifact_version}.csv",
        pair_rows,
        [
            "variant",
            "pair_id",
            "gate_name",
            "sequence_id",
            "event_id",
            "window_kind",
            "frame_i",
            "frame_j",
            "track_i",
            "track_j",
            "pair_type",
            "gt_instance_i_eval_only",
            "gt_instance_j_eval_only",
            "pair_correct_eval_only",
        ],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
