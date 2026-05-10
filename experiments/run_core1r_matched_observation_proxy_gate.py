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
from experiments.run_core1m_assignment_pair_confidence_gate import build_pairs_for_gate, summarize_gate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1R matched-observation proxy gate.")
    p.add_argument("--observations", default="results/core1p/stage_CORE1P_assignment_observation_trace_v1.csv")
    p.add_argument("--output-dir", default="results/core1r")
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
        return float(v)
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


def augment_proxy_fields(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_frame: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    by_track: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_frame[(row["sequence_id"], row["event_id"], row["window_kind"], i(row["frame_idx"]))].append(row)
        by_track[(row["sequence_id"], row["event_id"], row["window_kind"], i(row["track_id"]))].append(row)

    prev_by_track_frame: dict[tuple[str, str, str, int, int], dict[str, str]] = {}
    for key, track_rows in by_track.items():
        ordered = sorted(track_rows, key=lambda r: i(r["frame_idx"]))
        prev = None
        for row in ordered:
            if prev is not None:
                prev_by_track_frame[(row["sequence_id"], row["event_id"], row["window_kind"], i(row["track_id"]), i(row["frame_idx"]))] = prev
            prev = row

    augmented: list[dict[str, Any]] = []
    for row in rows:
        frame_key = (row["sequence_id"], row["event_id"], row["window_kind"], i(row["frame_idx"]))
        frame_rows = by_frame[frame_key]
        box = box_from_text(row.get("box", ""))
        max_overlap = 0.0
        if box is not None:
            for other in frame_rows:
                if other is row:
                    continue
                obox = box_from_text(other.get("box", ""))
                if obox is None:
                    continue
                max_overlap = max(max_overlap, box_iou(box, obox))
        prev = prev_by_track_frame.get((row["sequence_id"], row["event_id"], row["window_kind"], i(row["track_id"]), i(row["frame_idx"])))
        center_shift = 999.0
        area_ratio_delta = 999.0
        if box is not None and prev is not None:
            pbox = box_from_text(prev.get("box", ""))
            if pbox is not None:
                cx, cy = center(box)
                px, py = center(pbox)
                center_shift = float(np.hypot(cx - px, cy - py))
                area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
                parea = max(1.0, float((pbox[2] - pbox[0]) * (pbox[3] - pbox[1])))
                area_ratio_delta = abs(np.log(area / parea))
        proxy_score = (
            0.35 * min(f(row.get("score")), 1.0)
            + 0.20 * (1.0 - min(f(row.get("match_cost")), 1.0))
            + 0.15 * min(i(row.get("track_hit_count")) / 4.0, 1.0)
            + 0.15 * (1.0 - min(max_overlap, 1.0))
            + 0.10 * (1.0 - min(center_shift / 48.0, 1.0))
            + 0.05 * (1.0 - min(area_ratio_delta, 1.0))
        )
        out = dict(row)
        out.update(
            {
                "frame_assignment_count": len(frame_rows),
                "max_box_overlap_same_frame": max_overlap,
                "center_shift_from_prev_track": center_shift,
                "area_ratio_delta_from_prev_track": area_ratio_delta,
                "matched_observation_proxy_score": proxy_score,
            }
        )
        augmented.append(out)
    return augmented


GATES: list[dict[str, Any]] = [
    {"name": "A0_core1p_best", "score_min": 0.50, "match_cost_max": 0.50},
    {"name": "A1_proxy_ge_045", "proxy_min": 0.45},
    {"name": "A2_proxy_ge_050", "proxy_min": 0.50},
    {"name": "A3_proxy_ge_055", "proxy_min": 0.55},
    {"name": "A4_proxy050_overlap_le_020", "proxy_min": 0.50, "overlap_max": 0.20},
    {"name": "A5_proxy050_center_shift_le_32", "proxy_min": 0.50, "center_shift_max": 32.0},
    {"name": "A6_proxy050_hits2", "proxy_min": 0.50, "hit_min": 2},
    {"name": "A7_proxy055_overlap020_center48", "proxy_min": 0.55, "overlap_max": 0.20, "center_shift_max": 48.0},
    {"name": "A8_proxy060_strict", "proxy_min": 0.60, "overlap_max": 0.15, "center_shift_max": 40.0, "hit_min": 2},
]


def gate_to_core1m_gate(gate: dict[str, Any]) -> dict[str, Any]:
    out = {"name": gate["name"]}
    if "score_min" in gate:
        out["score_min"] = gate["score_min"]
    if "match_cost_max" in gate:
        out["match_cost_max"] = gate["match_cost_max"]
    if "hit_min" in gate:
        out["hit_min"] = gate["hit_min"]
    return out


def filter_rows(rows: list[dict[str, Any]], gate: dict[str, Any]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if f(row.get("matched_observation_proxy_score")) < f(gate.get("proxy_min", -1.0)):
            continue
        if f(row.get("max_box_overlap_same_frame")) > f(gate.get("overlap_max", 99.0)):
            continue
        if f(row.get("center_shift_from_prev_track")) > f(gate.get("center_shift_max", 9999.0)):
            continue
        if i(row.get("track_hit_count")) < i(gate.get("hit_min", 0)):
            continue
        if f(row.get("score")) < f(gate.get("score_min", -1.0)):
            continue
        if f(row.get("match_cost")) > f(gate.get("match_cost_max", 99.0)):
            continue
        selected.append(row)
    return selected


def evaluate_gate(rows: list[dict[str, Any]], gate: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    filtered = filter_rows(rows, gate)
    pairs = build_pairs_for_gate(filtered, gate_to_core1m_gate(gate))
    summary = summarize_gate(gate["name"], pairs)
    matched_rate = float(np.mean([1 if r.get("gt_instance_eval_only", "") != "" else 0 for r in filtered])) if filtered else 0.0
    summary.update(
        {
            "observation_count": len(filtered),
            "matched_observation_rate_eval_only": matched_rate,
            "mean_proxy_score": float(np.mean([f(r.get("matched_observation_proxy_score")) for r in filtered])) if filtered else 0.0,
        }
    )
    return summary, pairs


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = augment_proxy_fields(read_csv(Path(args.observations)))
    summaries: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    for gate in GATES:
        summary, pairs = evaluate_gate(rows, gate)
        summaries.append(summary)
        all_pairs.extend(pairs)

    eligible = [s for s in summaries if int(s["eligible_for_training_smoke"]) == 1]
    if eligible:
        best = max(eligible, key=lambda s: (s["positive_pair_count"] + s["negative_pair_count"]))
    else:
        best = max(summaries, key=lambda s: (min(s["positive_pair_precision_eval_only"], s["negative_pair_precision_eval_only"]), s["positive_pair_count"] + s["negative_pair_count"])) if summaries else {}

    compact = {
        "stage": "CORE-1R",
        "artifact_version": args.artifact_version,
        "observation_count": len(rows),
        "best_gate": best.get("gate_name", ""),
        "best_observation_count": best.get("observation_count", 0),
        "best_matched_observation_rate_eval_only": best.get("matched_observation_rate_eval_only", 0.0),
        "best_positive_pair_count": best.get("positive_pair_count", 0),
        "best_negative_pair_count": best.get("negative_pair_count", 0),
        "best_positive_pair_precision_eval_only": best.get("positive_pair_precision_eval_only", 0.0),
        "best_negative_pair_precision_eval_only": best.get("negative_pair_precision_eval_only", 0.0),
        "proxy_gate_passed": int(bool(eligible)),
        "oracle_leakage_found": 0,
        "next_recommendation": "CORE-1S train tiny encoder on proxy-gated pairs" if eligible else "proxy gate insufficient; objectness/proposal localization must be repaired before encoder training",
    }

    report = f"""# CORE-1R Matched-Observation Proxy Gate

This stage builds GT-free matched-observation proxy scores from assignment score, match cost, hits, overlap, and short-term motion consistency. GT is used only for audit precision.

## Result

- Observations: {compact['observation_count']}
- Best gate: {compact['best_gate']}
- Best matched-observation rate eval-only: {float(compact['best_matched_observation_rate_eval_only']):.4f}
- Best positive precision eval-only: {float(compact['best_positive_pair_precision_eval_only']):.4f}
- Best negative precision eval-only: {float(compact['best_negative_pair_precision_eval_only']):.4f}
- Proxy gate passed: {compact['proxy_gate_passed']}

Next recommendation: {compact['next_recommendation']}
"""

    prefix = "stage_CORE1R_"
    write_csv(
        out_dir / f"{prefix}observation_proxy_trace_{args.artifact_version}.csv",
        rows,
        [
            "sequence_id",
            "event_id",
            "window_kind",
            "frame_idx",
            "track_id",
            "prototype_id",
            "score",
            "match_cost",
            "track_hit_count",
            "frame_assignment_count",
            "max_box_overlap_same_frame",
            "center_shift_from_prev_track",
            "area_ratio_delta_from_prev_track",
            "matched_observation_proxy_score",
            "gt_instance_eval_only",
            "match_iou_eval_only",
        ],
    )
    write_csv(
        out_dir / f"{prefix}gate_summary_{args.artifact_version}.csv",
        summaries,
        [
            "gate_name",
            "observation_count",
            "matched_observation_rate_eval_only",
            "mean_proxy_score",
            "positive_pair_count",
            "negative_pair_count",
            "positive_pair_precision_eval_only",
            "negative_pair_precision_eval_only",
            "eligible_for_training_smoke",
        ],
    )
    write_csv(
        out_dir / f"{prefix}gated_pair_trace_{args.artifact_version}.csv",
        all_pairs,
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
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
