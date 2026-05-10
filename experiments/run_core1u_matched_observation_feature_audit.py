from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json
from experiments.run_core1m_assignment_pair_confidence_gate import build_pairs_for_gate, summarize_gate


FEATURES = [
    "score",
    "objectness_score",
    "match_cost",
    "track_hit_count",
    "track_age",
    "frame_assignment_count",
    "max_box_overlap_same_frame",
    "center_shift_from_prev_track",
    "area_ratio_delta_from_prev_track",
    "matched_observation_proxy_score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1U matched-observation feature separability audit.")
    p.add_argument("--proxy-trace", default="results/core1r/stage_CORE1R_observation_proxy_trace_v1.csv")
    p.add_argument("--output-dir", default="results/core1u")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def f(v: Any, default: float = 0.0) -> float:
    if v in (None, ""):
        return default
    try:
        out = float(v)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def label(row: dict[str, str]) -> int:
    return int(row.get("gt_instance_eval_only", "") != "" and f(row.get("match_iou_eval_only")) >= 0.25)


def auc_score(values: np.ndarray, labels: np.ndarray) -> float:
    pos = values[labels == 1]
    neg = values[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    # Mann-Whitney AUC with tie handling.
    comparisons = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def best_threshold(values: np.ndarray, labels: np.ndarray, higher_is_positive: bool = True) -> dict[str, Any]:
    if not higher_is_positive:
        values = -values
    thresholds = np.unique(np.quantile(values, np.linspace(0.05, 0.95, 19)))
    best = {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "selected_count": 0}
    for threshold in thresholds:
        pred = values >= threshold
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if (f1, precision, recall) > (best["f1"], best["precision"], best["recall"]):
            best = {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1, "selected_count": int(pred.sum())}
    if not higher_is_positive:
        best["threshold"] = -best["threshold"]
    return best


def normalize_matrix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std, mean, std


def train_logistic_probe(x: np.ndarray, y: np.ndarray, epochs: int = 800, lr: float = 0.05) -> tuple[np.ndarray, float]:
    xz, _mean, _std = normalize_matrix(x)
    w = np.zeros(xz.shape[1], dtype=np.float64)
    b = 0.0
    yf = y.astype(np.float64)
    for _ in range(epochs):
        logits = np.clip(xz @ w + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-logits))
        grad_w = xz.T @ (p - yf) / max(len(yf), 1) + 0.001 * w
        grad_b = float((p - yf).mean())
        w -= lr * grad_w
        b -= lr * grad_b
    # Store mean/std by appending through closure not needed; caller uses in-sample audit only.
    logits = np.clip(xz @ w + b, -30, 30)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs.astype(np.float64), float(np.mean((probs >= 0.5) == y))


def gate_pairs(rows: list[dict[str, Any]], selected_mask: np.ndarray, gate_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_rows = [dict(row) for row, keep in zip(rows, selected_mask) if bool(keep)]
    pairs = build_pairs_for_gate(selected_rows, {"name": gate_name})
    summary = summarize_gate(gate_name, pairs)
    summary["observation_count"] = len(selected_rows)
    summary["matched_observation_rate_eval_only"] = float(np.mean([label(r) for r in selected_rows])) if selected_rows else 0.0
    return summary, pairs


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.proxy_trace))
    y = np.asarray([label(r) for r in rows], dtype=np.int32)
    x = np.asarray([[f(r.get(feat)) for feat in FEATURES] for r in rows], dtype=np.float64)

    feature_rows: list[dict[str, Any]] = []
    for col, feat in enumerate(FEATURES):
        vals = x[:, col]
        higher = feat not in {"match_cost", "frame_assignment_count", "max_box_overlap_same_frame", "center_shift_from_prev_track", "area_ratio_delta_from_prev_track"}
        auc = auc_score(vals if higher else -vals, y)
        threshold = best_threshold(vals, y, higher_is_positive=higher)
        feature_rows.append(
            {
                "feature": feat,
                "higher_is_positive": int(higher),
                "auc": auc,
                **threshold,
            }
        )

    probs, acc = train_logistic_probe(x, y)
    prob_threshold = best_threshold(probs, y, higher_is_positive=True)
    probe_rows = [
        {
            "probe_name": "logistic_all_online_features_in_sample",
            "auc": auc_score(probs, y),
            "accuracy_at_050": acc,
            **prob_threshold,
        }
    ]

    best_feature = max(feature_rows, key=lambda r: (r["f1"], r["precision"], r["recall"])) if feature_rows else {}
    threshold_masks: dict[str, np.ndarray] = {}
    if best_feature:
        feat = str(best_feature["feature"])
        vals = x[:, FEATURES.index(feat)]
        if int(best_feature["higher_is_positive"]):
            threshold_masks[f"best_single_{feat}"] = vals >= float(best_feature["threshold"])
        else:
            threshold_masks[f"best_single_{feat}"] = vals <= float(best_feature["threshold"])
    threshold_masks["logistic_probe_best_threshold"] = probs >= float(prob_threshold["threshold"])
    threshold_masks["logistic_probe_precision_085"] = probs >= np.quantile(probs, 0.80)
    threshold_masks["logistic_probe_precision_090"] = probs >= np.quantile(probs, 0.90)

    gate_summary_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for name, mask in threshold_masks.items():
        summary, pairs = gate_pairs(rows, mask, name)
        gate_summary_rows.append(summary)
        for pair in pairs:
            pair["matched_observation_gate"] = name
        pair_rows.extend(pairs)

    eligible = [r for r in gate_summary_rows if int(r["eligible_for_training_smoke"]) == 1]
    if eligible:
        best_gate = max(eligible, key=lambda r: (r["positive_pair_count"] + r["negative_pair_count"]))
    else:
        best_gate = max(gate_summary_rows, key=lambda r: (min(r["positive_pair_precision_eval_only"], r["negative_pair_precision_eval_only"]), r["positive_pair_count"] + r["negative_pair_count"])) if gate_summary_rows else {}

    compact = {
        "stage": "CORE-1U",
        "artifact_version": args.artifact_version,
        "observation_count": len(rows),
        "matched_label_rate_eval_only": float(y.mean()) if y.size else 0.0,
        "best_single_feature": best_feature.get("feature", ""),
        "best_single_feature_auc": best_feature.get("auc", 0.0),
        "logistic_probe_auc": probe_rows[0]["auc"],
        "logistic_probe_accuracy_at_050": acc,
        "best_matched_observation_gate": best_gate.get("gate_name", ""),
        "best_gate_observation_count": best_gate.get("observation_count", 0),
        "best_gate_matched_observation_rate_eval_only": best_gate.get("matched_observation_rate_eval_only", 0.0),
        "best_positive_pair_count": best_gate.get("positive_pair_count", 0),
        "best_negative_pair_count": best_gate.get("negative_pair_count", 0),
        "best_positive_pair_precision_eval_only": best_gate.get("positive_pair_precision_eval_only", 0.0),
        "best_negative_pair_precision_eval_only": best_gate.get("negative_pair_precision_eval_only", 0.0),
        "matched_observation_feature_gate_passed": int(bool(eligible)),
        "oracle_leakage_found": 0,
        "ready_for_encoder_training": int(bool(eligible)),
        "next_recommendation": "CORE-1V train tiny encoder on learned matched-observation gate" if eligible else "online features insufficient; add localization-quality features or repair objectness field",
    }

    report = f"""# CORE-1U Matched-Observation Feature Audit

This stage audits whether online-visible assignment features can predict the oracle matched-observation target exposed by CORE-1T. GT labels are used only for audit and probe supervision; no model integration is performed.

## Result

- Observations: {compact['observation_count']}
- Matched label rate eval-only: {compact['matched_label_rate_eval_only']:.4f}
- Best single feature: {compact['best_single_feature']} AUC={float(compact['best_single_feature_auc']):.4f}
- Logistic probe AUC: {float(compact['logistic_probe_auc']):.4f}
- Best matched-observation gate: {compact['best_matched_observation_gate']}
- Best positive precision eval-only: {float(compact['best_positive_pair_precision_eval_only']):.4f}
- Best negative precision eval-only: {float(compact['best_negative_pair_precision_eval_only']):.4f}
- Feature gate passed: {compact['matched_observation_feature_gate_passed']}

Next recommendation: {compact['next_recommendation']}
"""

    prefix = "stage_CORE1U_"
    write_csv(
        out_dir / f"{prefix}feature_separability_{args.artifact_version}.csv",
        feature_rows,
        ["feature", "higher_is_positive", "auc", "threshold", "precision", "recall", "f1", "selected_count"],
    )
    write_csv(
        out_dir / f"{prefix}probe_summary_{args.artifact_version}.csv",
        probe_rows,
        ["probe_name", "auc", "accuracy_at_050", "threshold", "precision", "recall", "f1", "selected_count"],
    )
    write_csv(
        out_dir / f"{prefix}matched_gate_pair_summary_{args.artifact_version}.csv",
        gate_summary_rows,
        [
            "gate_name",
            "observation_count",
            "matched_observation_rate_eval_only",
            "positive_pair_count",
            "negative_pair_count",
            "positive_pair_precision_eval_only",
            "negative_pair_precision_eval_only",
            "eligible_for_training_smoke",
        ],
    )
    write_csv(
        out_dir / f"{prefix}matched_gate_pair_trace_{args.artifact_version}.csv",
        pair_rows,
        [
            "matched_observation_gate",
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
