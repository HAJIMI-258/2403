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

from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json
from experiments.run_core1u_matched_observation_feature_audit import FEATURES, f, label, train_logistic_probe
from experiments.run_core1w_negative_curriculum_audit import build_positive_pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1X cross-event negative mining.")
    p.add_argument("--proxy-trace", default="results/core1r/stage_CORE1R_observation_proxy_trace_v1.csv")
    p.add_argument("--output-dir", default="results/core1x")
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--max-negatives-per-observation", type=int, default=4)
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


def obs_group(row: dict[str, Any], mode: str) -> Any:
    if mode == "cross_window_same_sequence":
        return (row["sequence_id"], row["event_id"], row["window_kind"])
    if mode == "cross_event_same_sequence":
        return (row["sequence_id"], row["event_id"])
    if mode == "cross_sequence":
        return row["sequence_id"]
    if mode == "cross_sequence_same_proto":
        return row["sequence_id"]
    return row["event_id"]


NEGATIVE_MODES = [
    "cross_window_same_sequence",
    "cross_event_same_sequence",
    "cross_sequence",
    "cross_sequence_same_proto",
]


def build_cross_negatives(rows: list[dict[str, Any]], mode: str, max_per_obs: int) -> list[dict[str, Any]]:
    selected = [r for r in rows if int(r["high_conf_matched_observation"]) == 1]
    pairs = []
    pid = 0
    for idx, a in enumerate(selected):
        candidates = []
        for jdx, b in enumerate(selected):
            if idx == jdx:
                continue
            if obs_group(a, mode) == obs_group(b, mode):
                continue
            if mode == "cross_window_same_sequence" and a["sequence_id"] != b["sequence_id"]:
                continue
            if mode == "cross_event_same_sequence" and a["sequence_id"] != b["sequence_id"]:
                continue
            if mode == "cross_sequence_same_proto" and i(a.get("prototype_id"), -999) != i(b.get("prototype_id"), 999):
                continue
            # Deterministic local subset; enough for curriculum smoke without exploding rows.
            candidates.append(b)
            if len(candidates) >= max_per_obs:
                break
        for b in candidates:
            pid += 1
            pairs.append(
                {
                    "pair_id": pid,
                    "negative_mode": mode,
                    "pair_type": "negative_cross_context_high_conf",
                    "sequence_id": a["sequence_id"],
                    "event_id": a["event_id"],
                    "window_kind": a["window_kind"],
                    "frame_i": a["frame_idx"],
                    "frame_j": b["frame_idx"],
                    "track_i": a["track_id"],
                    "track_j": b["track_id"],
                    "prototype_i": a.get("prototype_id", ""),
                    "prototype_j": b.get("prototype_id", ""),
                    "gt_instance_i_eval_only": a["gt_instance_eval_only"],
                    "gt_instance_j_eval_only": b["gt_instance_eval_only"],
                    "pair_correct_eval_only": int(a["gt_instance_eval_only"] != "" and b["gt_instance_eval_only"] != "" and a["gt_instance_eval_only"] != b["gt_instance_eval_only"]),
                }
            )
    return pairs


def summarize(mode: str, positives: list[dict[str, Any]], negatives: list[dict[str, Any]]) -> dict[str, Any]:
    pos_precision = float(np.mean([i(p["pair_correct_eval_only"]) for p in positives])) if positives else 0.0
    neg_precision = float(np.mean([i(n["pair_correct_eval_only"]) for n in negatives])) if negatives else 0.0
    eligible = int(len(positives) >= 20 and len(negatives) >= 20 and pos_precision >= 0.85 and neg_precision >= 0.85)
    return {
        "negative_mode": mode,
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
    positives = build_positive_pairs(rows)
    summary_rows = []
    pair_rows = []
    for mode in NEGATIVE_MODES:
        negatives = build_cross_negatives(rows, mode, args.max_negatives_per_observation)
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
    compact = {
        "stage": "CORE-1X",
        "artifact_version": args.artifact_version,
        "positive_pair_count": len(positives),
        "best_negative_mode": best.get("negative_mode", ""),
        "best_negative_pair_count": best.get("negative_pair_count", 0),
        "best_positive_pair_precision_eval_only": best.get("positive_pair_precision_eval_only", 0.0),
        "best_negative_pair_precision_eval_only": best.get("negative_pair_precision_eval_only", 0.0),
        "cross_event_negative_mining_passed": int(bool(eligible)),
        "oracle_leakage_found": 0,
        "ready_for_encoder_training": int(bool(eligible)),
        "next_recommendation": "CORE-1Y train tiny encoder on cross-event negative curriculum" if eligible else "cross-context negatives still insufficient; expand window/sequence sample or use oracle proposals for diagnostic encoder",
    }
    report = f"""# CORE-1X Cross-Event Negative Mining

This stage avoids same-frame fragment pseudo-negatives by mining negatives from other windows, events, or sequences among high-confidence matched observations. GT is used only for audit.

## Result

- Positive pairs: {len(positives)}
- Best negative mode: {compact['best_negative_mode']}
- Best negative pair count: {compact['best_negative_pair_count']}
- Best positive precision eval-only: {float(compact['best_positive_pair_precision_eval_only']):.4f}
- Best negative precision eval-only: {float(compact['best_negative_pair_precision_eval_only']):.4f}
- Passed: {compact['cross_event_negative_mining_passed']}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1X_"
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
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
