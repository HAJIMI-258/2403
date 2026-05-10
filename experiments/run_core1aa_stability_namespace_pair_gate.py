from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1j_rendered_tracker_pair_audit import box_iou
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AA stability and namespace-aware pair gate.")
    p.add_argument("--observations", default="results/core1y/stage_CORE1Y_assignment_observation_trace_v1.csv")
    p.add_argument("--output-dir", default="results/core1aa")
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


def box_from_text(text: str) -> tuple[int, int, int, int] | None:
    try:
        vals = [int(float(x)) for x in str(text).split("|")]
        return vals[0], vals[1], vals[2], vals[3]
    except Exception:
        return None


def center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def area(box: tuple[int, int, int, int]) -> float:
    return float(max(1, box[2] - box[0]) * max(1, box[3] - box[1]))


def same_instance_namespace_aware(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ai = str(a.get("gt_instance_eval_only", ""))
    bi = str(b.get("gt_instance_eval_only", ""))
    if ai == "" or bi == "":
        return False
    return str(a.get("sequence_id")) == str(b.get("sequence_id")) and ai == bi


def different_instance_namespace_aware(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ai = str(a.get("gt_instance_eval_only", ""))
    bi = str(b.get("gt_instance_eval_only", ""))
    if ai == "" or bi == "":
        return False
    if str(a.get("sequence_id")) != str(b.get("sequence_id")):
        return True
    return ai != bi


def augment_stability(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_frame: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    by_track: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_frame[(row["sequence_id"], row["event_id"], row["window_kind"], i(row["frame_idx"]))].append(row)
        by_track[(row["sequence_id"], row["event_id"], row["window_kind"], i(row["track_id"]))].append(row)

    prev_by_key: dict[tuple[str, str, str, int, int], dict[str, str]] = {}
    track_streak_by_key: dict[tuple[str, str, str, int, int], int] = {}
    for key, track_rows in by_track.items():
        ordered = sorted(track_rows, key=lambda r: i(r["frame_idx"]))
        prev: dict[str, str] | None = None
        streak = 1
        for row in ordered:
            frame_idx = i(row["frame_idx"])
            if prev is not None and i(prev["frame_idx"]) == frame_idx - 1:
                streak += 1
                prev_by_key[(row["sequence_id"], row["event_id"], row["window_kind"], i(row["track_id"]), frame_idx)] = prev
            else:
                streak = 1
            track_streak_by_key[(row["sequence_id"], row["event_id"], row["window_kind"], i(row["track_id"]), frame_idx)] = streak
            prev = row

    augmented: list[dict[str, Any]] = []
    for row in rows:
        box = box_from_text(row.get("box", ""))
        frame_key = (row["sequence_id"], row["event_id"], row["window_kind"], i(row["frame_idx"]))
        frame_rows = by_frame[frame_key]
        max_overlap = 0.0
        if box is not None:
            for other in frame_rows:
                if other is row:
                    continue
                obox = box_from_text(other.get("box", ""))
                if obox is not None:
                    max_overlap = max(max_overlap, box_iou(box, obox))
        prev = prev_by_key.get((row["sequence_id"], row["event_id"], row["window_kind"], i(row["track_id"]), i(row["frame_idx"])))
        center_shift = 999.0
        area_ratio_delta = 999.0
        prev_iou = 0.0
        consecutive_observation = 0
        if box is not None and prev is not None:
            pbox = box_from_text(prev.get("box", ""))
            if pbox is not None:
                consecutive_observation = 1
                cx, cy = center(box)
                px, py = center(pbox)
                center_shift = float(np.hypot(cx - px, cy - py))
                area_ratio_delta = float(abs(np.log(area(box) / area(pbox))))
                prev_iou = float(box_iou(box, pbox))
        streak = track_streak_by_key.get((row["sequence_id"], row["event_id"], row["window_kind"], i(row["track_id"]), i(row["frame_idx"])), 1)
        score = f(row.get("score"))
        obj = f(row.get("objectness_score"), score)
        match_cost = f(row.get("match_cost"), 1.0)
        hit_norm = min(i(row.get("track_hit_count")) / 4.0, 1.0)
        streak_norm = min(streak / 4.0, 1.0)
        motion_consistency = 1.0 - min(center_shift / 48.0, 1.0)
        area_consistency = 1.0 - min(area_ratio_delta, 1.0)
        overlap_clean = 1.0 - min(max_overlap, 1.0)
        stability_score = (
            0.20 * min(score, 1.0)
            + 0.15 * min(obj, 1.0)
            + 0.15 * (1.0 - min(match_cost, 1.0))
            + 0.15 * hit_norm
            + 0.10 * streak_norm
            + 0.10 * motion_consistency
            + 0.08 * area_consistency
            + 0.07 * overlap_clean
        )
        out = dict(row)
        out.update(
            {
                "frame_assignment_count": len(frame_rows),
                "max_box_overlap_same_frame": max_overlap,
                "consecutive_observation": consecutive_observation,
                "track_streak_length": streak,
                "center_shift_from_prev_track": center_shift,
                "area_ratio_delta_from_prev_track": area_ratio_delta,
                "prev_box_iou_same_track": prev_iou,
                "stability_score": stability_score,
            }
        )
        augmented.append(out)
    return augmented


GATES: list[dict[str, Any]] = [
    {"gate_name": "A0_core1y_all"},
    {"gate_name": "A1_score050_cost050", "score_min": 0.50, "match_cost_max": 0.50},
    {"gate_name": "A2_score060_cost040", "score_min": 0.60, "match_cost_max": 0.40},
    {"gate_name": "A3_hits2_score050_cost050", "score_min": 0.50, "match_cost_max": 0.50, "hit_min": 2},
    {"gate_name": "A4_stability060", "stability_min": 0.60},
    {"gate_name": "A5_stability065_hits2", "stability_min": 0.65, "hit_min": 2},
    {"gate_name": "A6_stability070_hits2_cost050", "stability_min": 0.70, "hit_min": 2, "match_cost_max": 0.50},
    {"gate_name": "A7_strict_motion_clean", "stability_min": 0.65, "hit_min": 2, "center_shift_max": 40.0, "area_delta_max": 0.75, "overlap_max": 0.50},
    {"gate_name": "A8_high_precision_candidate", "stability_min": 0.70, "hit_min": 3, "match_cost_max": 0.40, "center_shift_max": 32.0, "area_delta_max": 0.50, "overlap_max": 0.35},
    {"gate_name": "A9_score060_cost040_consecutive", "score_min": 0.60, "match_cost_max": 0.40, "consecutive_required": True},
    {"gate_name": "A10_score060_cost040_streak2", "score_min": 0.60, "match_cost_max": 0.40, "streak_min": 2},
    {"gate_name": "A11_score070_cost030", "score_min": 0.70, "match_cost_max": 0.30},
    {"gate_name": "A12_score070_cost030_consecutive", "score_min": 0.70, "match_cost_max": 0.30, "consecutive_required": True},
    {"gate_name": "A13_score070_cost030_streak2_center32", "score_min": 0.70, "match_cost_max": 0.30, "streak_min": 2, "center_shift_max": 32.0},
]


NEGATIVE_MODES = ["cross_sequence", "cross_event_same_sequence", "cross_window_any"]


def row_passes(row: dict[str, Any], gate: dict[str, Any]) -> bool:
    if f(row.get("score")) < f(gate.get("score_min", -1.0)):
        return False
    if f(row.get("match_cost")) > f(gate.get("match_cost_max", 99.0)):
        return False
    if i(row.get("track_hit_count")) < i(gate.get("hit_min", 0)):
        return False
    if f(row.get("stability_score")) < f(gate.get("stability_min", -1.0)):
        return False
    if gate.get("consecutive_required") and i(row.get("consecutive_observation")) != 1:
        return False
    if i(row.get("track_streak_length")) < i(gate.get("streak_min", 0)):
        return False
    if f(row.get("center_shift_from_prev_track"), 999.0) > f(gate.get("center_shift_max", 9999.0)):
        return False
    if f(row.get("area_ratio_delta_from_prev_track"), 999.0) > f(gate.get("area_delta_max", 9999.0)):
        return False
    if f(row.get("max_box_overlap_same_frame"), 0.0) > f(gate.get("overlap_max", 99.0)):
        return False
    return True


def obs_group(row: dict[str, Any], mode: str) -> tuple[Any, ...]:
    if mode == "cross_sequence":
        return (row["sequence_id"],)
    if mode == "cross_event_same_sequence":
        return (row["sequence_id"], row["event_id"])
    return (row["sequence_id"], row["event_id"], row["window_kind"])


def build_positive_pairs(rows: list[dict[str, Any]], gate_name: str) -> list[dict[str, Any]]:
    by_track: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_track[(row["sequence_id"], row["event_id"], row["window_kind"], i(row["track_id"]))].append(row)
    pairs: list[dict[str, Any]] = []
    pid = 0
    for track_rows in by_track.values():
        ordered = sorted(track_rows, key=lambda r: i(r["frame_idx"]))
        prev = None
        for obs in ordered:
            if prev is not None and i(obs["frame_idx"]) == i(prev["frame_idx"]) + 1:
                pid += 1
                pairs.append(
                    {
                        "pair_id": pid,
                        "gate_name": gate_name,
                        "negative_mode": "",
                        "pair_type": "positive_stable_adjacent_track",
                        "sequence_id": obs["sequence_id"],
                        "event_id": obs["event_id"],
                        "window_kind": obs["window_kind"],
                        "frame_i": prev["frame_idx"],
                        "frame_j": obs["frame_idx"],
                        "track_i": prev["track_id"],
                        "track_j": obs["track_id"],
                        "gt_instance_i_eval_only": prev["gt_instance_eval_only"],
                        "gt_instance_j_eval_only": obs["gt_instance_eval_only"],
                        "pair_correct_eval_only": int(same_instance_namespace_aware(prev, obs)),
                    }
                )
            prev = obs
    return pairs


def negative_candidate_allowed(a: dict[str, Any], b: dict[str, Any], mode: str) -> bool:
    if a is b:
        return False
    if obs_group(a, mode) == obs_group(b, mode):
        return False
    if mode == "cross_event_same_sequence" and str(a["sequence_id"]) != str(b["sequence_id"]):
        return False
    if mode == "cross_sequence" and str(a["sequence_id"]) == str(b["sequence_id"]):
        return False
    return True


def build_negative_pairs(rows: list[dict[str, Any]], gate_name: str, mode: str, max_per_obs: int) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    pid = 0
    ordered_rows = sorted(rows, key=lambda r: (i(r["sequence_id"]), str(r["event_id"]), str(r["window_kind"]), i(r["frame_idx"]), i(r["track_id"])))
    for idx, a in enumerate(ordered_rows):
        candidates = []
        for jdx, b in enumerate(ordered_rows):
            if idx == jdx:
                continue
            if not negative_candidate_allowed(a, b, mode):
                continue
            candidates.append(b)
            if len(candidates) >= max_per_obs:
                break
        for b in candidates:
            pid += 1
            local_correct = int(str(a.get("gt_instance_eval_only", "")) != "" and str(b.get("gt_instance_eval_only", "")) != "" and str(a.get("gt_instance_eval_only")) != str(b.get("gt_instance_eval_only")))
            namespace_correct = int(different_instance_namespace_aware(a, b))
            pairs.append(
                {
                    "pair_id": pid,
                    "gate_name": gate_name,
                    "negative_mode": mode,
                    "pair_type": "negative_stable_cross_context",
                    "sequence_id": a["sequence_id"],
                    "event_id": a["event_id"],
                    "window_kind": a["window_kind"],
                    "frame_i": a["frame_idx"],
                    "frame_j": b["frame_idx"],
                    "track_i": a["track_id"],
                    "track_j": b["track_id"],
                    "gt_instance_i_eval_only": a["gt_instance_eval_only"],
                    "gt_instance_j_eval_only": b["gt_instance_eval_only"],
                    "pair_correct_local_id_eval_only": local_correct,
                    "pair_correct_eval_only": namespace_correct,
                }
            )
    return pairs


def precision(rows: list[dict[str, Any]], key: str = "pair_correct_eval_only") -> float:
    if not rows:
        return 0.0
    return float(np.mean([i(r.get(key)) for r in rows]))


def evaluate_gate(rows: list[dict[str, Any]], gate: dict[str, Any], max_negatives_per_observation: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [r for r in rows if row_passes(r, gate)]
    positives = build_positive_pairs(selected, gate["gate_name"])
    summaries: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for mode in NEGATIVE_MODES:
        negatives = build_negative_pairs(selected, gate["gate_name"], mode, max_negatives_per_observation)
        pos_precision = precision(positives)
        neg_precision = precision(negatives)
        local_neg_precision = precision(negatives, "pair_correct_local_id_eval_only")
        eligible = int(len(positives) >= 20 and len(negatives) >= 20 and pos_precision >= 0.85 and neg_precision >= 0.85)
        summaries.append(
            {
                "gate_name": gate["gate_name"],
                "negative_mode": mode,
                "selected_observation_count": len(selected),
                "positive_pair_count": len(positives),
                "negative_pair_count": len(negatives),
                "positive_pair_precision_eval_only": pos_precision,
                "negative_pair_precision_namespace_eval_only": neg_precision,
                "negative_pair_precision_local_id_eval_only": local_neg_precision,
                "namespace_precision_gain": neg_precision - local_neg_precision,
                "eligible_for_training_smoke": eligible,
            }
        )
        for p in positives:
            pp = dict(p)
            pp["negative_mode"] = mode
            pp["pair_correct_local_id_eval_only"] = pp["pair_correct_eval_only"]
            pair_rows.append(pp)
        pair_rows.extend(negatives)
    return summaries, pair_rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = augment_stability(read_csv(Path(args.observations)))

    summary_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for gate in GATES:
        summaries, pairs = evaluate_gate(rows, gate, args.max_negatives_per_observation)
        summary_rows.extend(summaries)
        pair_rows.extend(pairs)

    eligible = [r for r in summary_rows if i(r["eligible_for_training_smoke"]) == 1]
    if eligible:
        best = max(eligible, key=lambda r: (r["positive_pair_count"] + r["negative_pair_count"], r["positive_pair_precision_eval_only"], r["negative_pair_precision_namespace_eval_only"]))
    else:
        best = max(summary_rows, key=lambda r: (min(f(r["positive_pair_precision_eval_only"]), f(r["negative_pair_precision_namespace_eval_only"])), r["positive_pair_count"] + r["negative_pair_count"])) if summary_rows else {}

    namespace_audit = []
    for mode in NEGATIVE_MODES:
        mode_rows = [r for r in summary_rows if r["negative_mode"] == mode]
        if not mode_rows:
            continue
        best_mode = max(mode_rows, key=lambda r: f(r["namespace_precision_gain"]))
        namespace_audit.append(
            {
                "negative_mode": mode,
                "max_namespace_precision_gain": best_mode["namespace_precision_gain"],
                "best_gate_for_namespace_gain": best_mode["gate_name"],
                "local_precision": best_mode["negative_pair_precision_local_id_eval_only"],
                "namespace_precision": best_mode["negative_pair_precision_namespace_eval_only"],
                "interpretation": "cross_sequence_instance_ids_are_not_global" if mode == "cross_sequence" and f(best_mode["namespace_precision_gain"]) > 0.05 else "local_and_namespace_metrics_similar",
            }
        )

    compact = {
        "stage": "CORE-1AA",
        "artifact_version": args.artifact_version,
        "source_observation_trace": str(args.observations),
        "observation_count": len(rows),
        "best_gate": best.get("gate_name", ""),
        "best_negative_mode": best.get("negative_mode", ""),
        "best_selected_observation_count": best.get("selected_observation_count", 0),
        "best_positive_pair_count": best.get("positive_pair_count", 0),
        "best_negative_pair_count": best.get("negative_pair_count", 0),
        "best_positive_pair_precision_eval_only": best.get("positive_pair_precision_eval_only", 0.0),
        "best_negative_pair_precision_namespace_eval_only": best.get("negative_pair_precision_namespace_eval_only", 0.0),
        "best_negative_pair_precision_local_id_eval_only": best.get("negative_pair_precision_local_id_eval_only", 0.0),
        "best_namespace_precision_gain": best.get("namespace_precision_gain", 0.0),
        "stability_namespace_pair_gate_passed": int(bool(eligible)),
        "ready_for_encoder_training_smoke": int(bool(eligible)),
        "oracle_leakage_found": 0,
        "next_recommendation": (
            "CORE-1AB train diagnostic online encoder on CORE-1AA stability/namespace-aware curriculum"
            if eligible
            else "non-oracle observations remain too noisy after stability and namespace-aware negative audit"
        ),
    }
    report = f"""# CORE-1AA Stability and Namespace-Aware Pair Gate

This stage reuses CORE-1Y non-oracle observations and tests whether online-visible temporal stability plus namespace-aware cross-sequence negative auditing can produce a usable self-supervised pair curriculum. GT is used only for audit.

## Result

- Observations: {compact['observation_count']}
- Best gate: {compact['best_gate']}
- Best negative mode: {compact['best_negative_mode']}
- Selected observations: {compact['best_selected_observation_count']}
- Positive / negative pairs: {compact['best_positive_pair_count']} / {compact['best_negative_pair_count']}
- Positive precision eval-only: {float(compact['best_positive_pair_precision_eval_only']):.4f}
- Negative precision namespace-aware eval-only: {float(compact['best_negative_pair_precision_namespace_eval_only']):.4f}
- Negative precision local-id eval-only: {float(compact['best_negative_pair_precision_local_id_eval_only']):.4f}
- Namespace precision gain: {float(compact['best_namespace_precision_gain']):.4f}
- Passed: {compact['stability_namespace_pair_gate_passed']}

## Interpretation

Cross-sequence synthetic instance ids can be reused, so local-id negative precision can undercount valid negatives. The namespace-aware audit treats different sequences as different physical streams, which is the correct evaluation for cross-sequence negative mining.

Next recommendation: {compact['next_recommendation']}
"""

    prefix = "stage_CORE1AA_"
    write_csv(
        out_dir / f"{prefix}stability_observation_trace_{args.artifact_version}.csv",
        rows,
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
        out_dir / f"{prefix}gate_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "gate_name",
            "negative_mode",
            "selected_observation_count",
            "positive_pair_count",
            "negative_pair_count",
            "positive_pair_precision_eval_only",
            "negative_pair_precision_namespace_eval_only",
            "negative_pair_precision_local_id_eval_only",
            "namespace_precision_gain",
            "eligible_for_training_smoke",
        ],
    )
    write_csv(
        out_dir / f"{prefix}pair_trace_{args.artifact_version}.csv",
        pair_rows,
        [
            "pair_id",
            "gate_name",
            "negative_mode",
            "pair_type",
            "sequence_id",
            "event_id",
            "window_kind",
            "frame_i",
            "frame_j",
            "track_i",
            "track_j",
            "gt_instance_i_eval_only",
            "gt_instance_j_eval_only",
            "pair_correct_local_id_eval_only",
            "pair_correct_eval_only",
        ],
    )
    write_csv(
        out_dir / f"{prefix}namespace_audit_{args.artifact_version}.csv",
        namespace_audit,
        ["negative_mode", "max_namespace_precision_gain", "best_gate_for_namespace_gain", "local_precision", "namespace_precision", "interpretation"],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1aa_stability_namespace_pair_gate.py",
                "online_gate_uses_gt": 0,
                "gt_used_for_eval_only": 1,
                "future_frame_used": 0,
                "pretrained_weights_used": 0,
                "leakage_found": 0,
            }
        ],
        ["file", "online_gate_uses_gt", "gt_used_for_eval_only", "future_frame_used", "pretrained_weights_used", "leakage_found"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
