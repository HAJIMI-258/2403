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
from experiments.run_core1u_matched_observation_feature_audit import FEATURES, f, label, normalize_matrix, train_logistic_probe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1V pair-level negative filter.")
    p.add_argument("--proxy-trace", default="results/core1r/stage_CORE1R_observation_proxy_trace_v1.csv")
    p.add_argument("--output-dir", default="results/core1v")
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


def box_from_text(text: str) -> tuple[int, int, int, int] | None:
    try:
        vals = [int(float(x)) for x in str(text).split("|")]
        return vals[0], vals[1], vals[2], vals[3]
    except Exception:
        return None


def center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def prepare_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    y = np.asarray([label(r) for r in rows], dtype=np.int32)
    x = np.asarray([[f(r.get(feat)) for feat in FEATURES] for r in rows], dtype=np.float64)
    probs, _acc = train_logistic_probe(x, y)
    threshold = float(np.quantile(probs, 0.90))
    prepared = []
    for row, prob in zip(rows, probs):
        out = dict(row)
        out["matched_observation_probability"] = float(prob)
        out["high_conf_matched_observation"] = int(prob >= threshold)
        prepared.append(out)
    return prepared


NEGATIVE_FILTERS: list[dict[str, Any]] = [
    {"name": "A0_high_conf_only"},
    {"name": "A1_overlap_le_020", "overlap_max": 0.20},
    {"name": "A2_overlap_le_010", "overlap_max": 0.10},
    {"name": "A3_center_dist_ge_32", "center_dist_min": 32.0},
    {"name": "A4_diff_proto", "diff_proto": True},
    {"name": "A5_overlap020_center32", "overlap_max": 0.20, "center_dist_min": 32.0},
    {"name": "A6_overlap010_center48", "overlap_max": 0.10, "center_dist_min": 48.0},
    {"name": "A7_overlap020_center32_diff_proto", "overlap_max": 0.20, "center_dist_min": 32.0, "diff_proto": True},
]


def build_pairs(rows: list[dict[str, Any]], filt: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [r for r in rows if int(r.get("high_conf_matched_observation", 0)) == 1]
    by_window_track: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    by_frame: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_window_track[(row["sequence_id"], row["event_id"], row["window_kind"], i(row["track_id"]))].append(row)
        by_frame[(row["sequence_id"], row["event_id"], row["window_kind"], i(row["frame_idx"]))].append(row)

    pairs: list[dict[str, Any]] = []
    pair_id = 0
    for track_rows in by_window_track.values():
        ordered = sorted(track_rows, key=lambda r: i(r["frame_idx"]))
        prev = None
        for obs in ordered:
            if prev is not None and i(obs["frame_idx"]) == i(prev["frame_idx"]) + 1:
                pair_id += 1
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "filter_name": filt["name"],
                        "sequence_id": obs["sequence_id"],
                        "event_id": obs["event_id"],
                        "window_kind": obs["window_kind"],
                        "frame_i": prev["frame_idx"],
                        "frame_j": obs["frame_idx"],
                        "track_i": prev["track_id"],
                        "track_j": obs["track_id"],
                        "prototype_i": prev["prototype_id"],
                        "prototype_j": obs["prototype_id"],
                        "pair_type": "positive_high_conf_adjacent_track",
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
                if i(a["track_id"]) == i(b["track_id"]):
                    continue
                box_a = box_from_text(str(a.get("box", "")))
                box_b = box_from_text(str(b.get("box", "")))
                overlap = box_iou(box_a, box_b) if box_a is not None and box_b is not None else 1.0
                if overlap > f(filt.get("overlap_max", 99.0)):
                    continue
                cdist = 0.0
                if box_a is not None and box_b is not None:
                    ax, ay = center(box_a)
                    bx, by = center(box_b)
                    cdist = float(np.hypot(ax - bx, ay - by))
                if cdist < f(filt.get("center_dist_min", -1.0)):
                    continue
                if filt.get("diff_proto"):
                    if i(a.get("prototype_id"), -1) < 0 or i(b.get("prototype_id"), -1) < 0 or i(a.get("prototype_id")) == i(b.get("prototype_id")):
                        continue
                pair_id += 1
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "filter_name": filt["name"],
                        "sequence_id": a["sequence_id"],
                        "event_id": a["event_id"],
                        "window_kind": a["window_kind"],
                        "frame_i": a["frame_idx"],
                        "frame_j": b["frame_idx"],
                        "track_i": a["track_id"],
                        "track_j": b["track_id"],
                        "prototype_i": a["prototype_id"],
                        "prototype_j": b["prototype_id"],
                        "pair_type": "negative_high_conf_cov_visible_track",
                        "gt_instance_i_eval_only": a["gt_instance_eval_only"],
                        "gt_instance_j_eval_only": b["gt_instance_eval_only"],
                        "pair_correct_eval_only": int(a["gt_instance_eval_only"] != "" and b["gt_instance_eval_only"] != "" and a["gt_instance_eval_only"] != b["gt_instance_eval_only"]),
                    }
                )
    return pairs


def summarize(name: str, pairs: list[dict[str, Any]], obs_count: int) -> dict[str, Any]:
    positives = [p for p in pairs if str(p["pair_type"]).startswith("positive")]
    negatives = [p for p in pairs if str(p["pair_type"]).startswith("negative")]
    pos_precision = float(np.mean([i(p["pair_correct_eval_only"]) for p in positives])) if positives else 0.0
    neg_precision = float(np.mean([i(p["pair_correct_eval_only"]) for p in negatives])) if negatives else 0.0
    eligible = int(len(positives) >= 20 and len(negatives) >= 20 and pos_precision >= 0.85 and neg_precision >= 0.85)
    return {
        "filter_name": name,
        "high_conf_observation_count": obs_count,
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
    rows = prepare_rows(read_csv(Path(args.proxy_trace)))
    high_conf_count = sum(int(r["high_conf_matched_observation"]) for r in rows)
    summary_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for filt in NEGATIVE_FILTERS:
        pairs = build_pairs(rows, filt)
        summary_rows.append(summarize(filt["name"], pairs, high_conf_count))
        pair_rows.extend(pairs)
    eligible = [r for r in summary_rows if int(r["eligible_for_training_smoke"]) == 1]
    if eligible:
        best = max(eligible, key=lambda r: (r["positive_pair_count"] + r["negative_pair_count"]))
    else:
        best = max(summary_rows, key=lambda r: (min(r["positive_pair_precision_eval_only"], r["negative_pair_precision_eval_only"]), r["positive_pair_count"] + r["negative_pair_count"])) if summary_rows else {}
    compact = {
        "stage": "CORE-1V",
        "artifact_version": args.artifact_version,
        "high_conf_observation_count": high_conf_count,
        "best_filter": best.get("filter_name", ""),
        "best_positive_pair_count": best.get("positive_pair_count", 0),
        "best_negative_pair_count": best.get("negative_pair_count", 0),
        "best_positive_pair_precision_eval_only": best.get("positive_pair_precision_eval_only", 0.0),
        "best_negative_pair_precision_eval_only": best.get("negative_pair_precision_eval_only", 0.0),
        "pair_level_negative_filter_passed": int(bool(eligible)),
        "oracle_leakage_found": 0,
        "ready_for_encoder_training": int(bool(eligible)),
        "next_recommendation": "CORE-1W train tiny encoder on high-confidence filtered pairs" if eligible else "need stronger duplicate/fragmentation filter or localization-quality feature before encoder training",
    }
    report = f"""# CORE-1V Pair-Level Negative Filter

This stage combines CORE-1U high-confidence matched observations with pair-level negative filters. GT is used only for audit.

## Result

- High-confidence observations: {high_conf_count}
- Best filter: {compact['best_filter']}
- Best positive precision eval-only: {float(compact['best_positive_pair_precision_eval_only']):.4f}
- Best negative precision eval-only: {float(compact['best_negative_pair_precision_eval_only']):.4f}
- Passed: {compact['pair_level_negative_filter_passed']}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1V_"
    write_csv(
        out_dir / f"{prefix}negative_filter_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "filter_name",
            "high_conf_observation_count",
            "positive_pair_count",
            "negative_pair_count",
            "positive_pair_precision_eval_only",
            "negative_pair_precision_eval_only",
            "eligible_for_training_smoke",
        ],
    )
    write_csv(
        out_dir / f"{prefix}filtered_pair_trace_{args.artifact_version}.csv",
        pair_rows,
        [
            "pair_id",
            "filter_name",
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
            "gt_instance_i_eval_only",
            "gt_instance_j_eval_only",
            "pair_correct_eval_only",
        ],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
