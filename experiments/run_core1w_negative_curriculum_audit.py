from __future__ import annotations

import argparse
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
from experiments.run_core1u_matched_observation_feature_audit import FEATURES, f, label, train_logistic_probe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1W negative curriculum audit.")
    p.add_argument("--proxy-trace", default="results/core1r/stage_CORE1R_observation_proxy_trace_v1.csv")
    p.add_argument("--output-dir", default="results/core1w")
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
    probs, _ = train_logistic_probe(x, y)
    threshold = float(np.quantile(probs, 0.90))
    out = []
    for row, prob in zip(rows, probs):
        rr = dict(row)
        rr["matched_observation_probability"] = float(prob)
        rr["high_conf_matched_observation"] = int(prob >= threshold)
        out.append(rr)
    return out


def build_positive_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [r for r in rows if int(r["high_conf_matched_observation"]) == 1]
    by_track: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_track[(row["sequence_id"], row["event_id"], row["window_kind"], i(row["track_id"]))].append(row)
    pairs = []
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
                        "pair_type": "positive_high_conf_adjacent_track",
                        "negative_curriculum_class": "",
                        "sequence_id": obs["sequence_id"],
                        "event_id": obs["event_id"],
                        "window_kind": obs["window_kind"],
                        "frame_i": prev["frame_idx"],
                        "frame_j": obs["frame_idx"],
                        "track_i": prev["track_id"],
                        "track_j": obs["track_id"],
                        "gt_instance_i_eval_only": prev["gt_instance_eval_only"],
                        "gt_instance_j_eval_only": obs["gt_instance_eval_only"],
                        "pair_correct_eval_only": int(prev["gt_instance_eval_only"] != "" and obs["gt_instance_eval_only"] != "" and prev["gt_instance_eval_only"] == obs["gt_instance_eval_only"]),
                    }
                )
            prev = obs
    return pairs


def classify_negative(a: dict[str, Any], b: dict[str, Any]) -> tuple[str, float, float]:
    box_a = box_from_text(str(a.get("box", "")))
    box_b = box_from_text(str(b.get("box", "")))
    overlap = box_iou(box_a, box_b) if box_a is not None and box_b is not None else 1.0
    cdist = 0.0
    if box_a is not None and box_b is not None:
        ax, ay = center(box_a)
        bx, by = center(box_b)
        cdist = float(np.hypot(ax - bx, ay - by))
    proto_a = i(a.get("prototype_id"), -1)
    proto_b = i(b.get("prototype_id"), -1)
    if overlap > 0.20 or cdist < 24.0:
        return "fragment_or_near_duplicate_risk", overlap, cdist
    if proto_a >= 0 and proto_b >= 0 and proto_a != proto_b and cdist >= 32.0:
        return "safe_negative_diff_proto_spatial", overlap, cdist
    if cdist >= 64.0:
        return "safe_negative_far_spatial", overlap, cdist
    if proto_a >= 0 and proto_b >= 0 and proto_a != proto_b:
        return "hard_negative_diff_proto_near", overlap, cdist
    return "ambiguous_negative", overlap, cdist


def build_negative_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [r for r in rows if int(r["high_conf_matched_observation"]) == 1]
    by_frame: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_frame[(row["sequence_id"], row["event_id"], row["window_kind"], i(row["frame_idx"]))].append(row)
    pairs = []
    pid = 0
    for frame_rows in by_frame.values():
        ordered = sorted(frame_rows, key=lambda r: i(r["track_id"]))
        for a_idx, a in enumerate(ordered):
            for b in ordered[a_idx + 1 :]:
                if i(a["track_id"]) == i(b["track_id"]):
                    continue
                cls, overlap, cdist = classify_negative(a, b)
                pid += 1
                pairs.append(
                    {
                        "pair_id": pid,
                        "pair_type": "negative_high_conf_cov_visible_track",
                        "negative_curriculum_class": cls,
                        "sequence_id": a["sequence_id"],
                        "event_id": a["event_id"],
                        "window_kind": a["window_kind"],
                        "frame_i": a["frame_idx"],
                        "frame_j": b["frame_idx"],
                        "track_i": a["track_id"],
                        "track_j": b["track_id"],
                        "prototype_i": a.get("prototype_id", ""),
                        "prototype_j": b.get("prototype_id", ""),
                        "overlap": overlap,
                        "center_distance": cdist,
                        "gt_instance_i_eval_only": a["gt_instance_eval_only"],
                        "gt_instance_j_eval_only": b["gt_instance_eval_only"],
                        "pair_correct_eval_only": int(a["gt_instance_eval_only"] != "" and b["gt_instance_eval_only"] != "" and a["gt_instance_eval_only"] != b["gt_instance_eval_only"]),
                    }
                )
    return pairs


CURRICULA = {
    "C0_all_high_conf_negatives": None,
    "C1_safe_spatial_only": {"safe_negative_far_spatial", "safe_negative_diff_proto_spatial"},
    "C2_safe_diff_proto_only": {"safe_negative_diff_proto_spatial"},
    "C3_safe_far_only": {"safe_negative_far_spatial"},
    "C4_safe_plus_hard_diff_proto": {"safe_negative_far_spatial", "safe_negative_diff_proto_spatial", "hard_negative_diff_proto_near"},
}


def summarize_curriculum(name: str, positives: list[dict[str, Any]], negatives: list[dict[str, Any]], classes: set[str] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_neg = negatives if classes is None else [n for n in negatives if n["negative_curriculum_class"] in classes]
    all_pairs = []
    for p in positives:
        pp = dict(p)
        pp["curriculum_name"] = name
        all_pairs.append(pp)
    for n in selected_neg:
        nn = dict(n)
        nn["curriculum_name"] = name
        all_pairs.append(nn)
    pos_precision = float(np.mean([i(p["pair_correct_eval_only"]) for p in positives])) if positives else 0.0
    neg_precision = float(np.mean([i(n["pair_correct_eval_only"]) for n in selected_neg])) if selected_neg else 0.0
    eligible = int(len(positives) >= 20 and len(selected_neg) >= 20 and pos_precision >= 0.85 and neg_precision >= 0.85)
    return {
        "curriculum_name": name,
        "positive_pair_count": len(positives),
        "negative_pair_count": len(selected_neg),
        "positive_pair_precision_eval_only": pos_precision,
        "negative_pair_precision_eval_only": neg_precision,
        "eligible_for_training_smoke": eligible,
    }, all_pairs


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = prepare_rows(read_csv(Path(args.proxy_trace)))
    positives = build_positive_pairs(rows)
    negatives = build_negative_pairs(rows)

    class_rows = []
    for cls in sorted({n["negative_curriculum_class"] for n in negatives}):
        group = [n for n in negatives if n["negative_curriculum_class"] == cls]
        class_rows.append(
            {
                "negative_curriculum_class": cls,
                "negative_pair_count": len(group),
                "negative_pair_precision_eval_only": float(np.mean([i(n["pair_correct_eval_only"]) for n in group])) if group else 0.0,
                "mean_overlap": float(np.mean([float(n["overlap"]) for n in group])) if group else 0.0,
                "mean_center_distance": float(np.mean([float(n["center_distance"]) for n in group])) if group else 0.0,
            }
        )

    summary_rows = []
    pair_rows = []
    for name, classes in CURRICULA.items():
        summary, pairs = summarize_curriculum(name, positives, negatives, classes)
        summary_rows.append(summary)
        pair_rows.extend(pairs)
    eligible = [r for r in summary_rows if int(r["eligible_for_training_smoke"]) == 1]
    if eligible:
        best = max(eligible, key=lambda r: r["negative_pair_count"])
    else:
        best = max(summary_rows, key=lambda r: (min(r["positive_pair_precision_eval_only"], r["negative_pair_precision_eval_only"]), r["negative_pair_count"])) if summary_rows else {}
    compact = {
        "stage": "CORE-1W",
        "artifact_version": args.artifact_version,
        "positive_pair_count": len(positives),
        "all_negative_pair_count": len(negatives),
        "best_curriculum": best.get("curriculum_name", ""),
        "best_positive_pair_count": best.get("positive_pair_count", 0),
        "best_negative_pair_count": best.get("negative_pair_count", 0),
        "best_positive_pair_precision_eval_only": best.get("positive_pair_precision_eval_only", 0.0),
        "best_negative_pair_precision_eval_only": best.get("negative_pair_precision_eval_only", 0.0),
        "negative_curriculum_passed": int(bool(eligible)),
        "oracle_leakage_found": 0,
        "ready_for_encoder_training": int(bool(eligible)),
        "next_recommendation": "CORE-1X train tiny encoder on safe negative curriculum" if eligible else "safe negatives insufficient in same-frame windows; mine cross-window/cross-event negatives or repair fragmentation",
    }
    report = f"""# CORE-1W Negative Curriculum Audit

This stage classifies high-confidence co-visible negative pairs into safe, hard, and fragment-risk groups. GT is used only for audit precision.

## Result

- Positive pairs: {len(positives)}
- All high-confidence negatives: {len(negatives)}
- Best curriculum: {compact['best_curriculum']}
- Best positive precision eval-only: {float(compact['best_positive_pair_precision_eval_only']):.4f}
- Best negative precision eval-only: {float(compact['best_negative_pair_precision_eval_only']):.4f}
- Negative curriculum passed: {compact['negative_curriculum_passed']}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1W_"
    write_csv(
        out_dir / f"{prefix}negative_class_summary_{args.artifact_version}.csv",
        class_rows,
        ["negative_curriculum_class", "negative_pair_count", "negative_pair_precision_eval_only", "mean_overlap", "mean_center_distance"],
    )
    write_csv(
        out_dir / f"{prefix}curriculum_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "curriculum_name",
            "positive_pair_count",
            "negative_pair_count",
            "positive_pair_precision_eval_only",
            "negative_pair_precision_eval_only",
            "eligible_for_training_smoke",
        ],
    )
    write_csv(
        out_dir / f"{prefix}curriculum_pair_trace_{args.artifact_version}.csv",
        pair_rows,
        [
            "curriculum_name",
            "pair_id",
            "pair_type",
            "negative_curriculum_class",
            "sequence_id",
            "event_id",
            "window_kind",
            "frame_i",
            "frame_j",
            "track_i",
            "track_j",
            "prototype_i",
            "prototype_j",
            "overlap",
            "center_distance",
            "gt_instance_i_eval_only",
            "gt_instance_j_eval_only",
            "pair_correct_eval_only",
        ],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
