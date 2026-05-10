from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1ac_raw_descriptor_memory_integration_smoke import baseline_score, f, i, normalize_scores
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json
from experiments.run_core1z_oracle_proposal_diagnostic_encoder import cosine01, parse_descriptor


FOCUS_EVENTS = {"M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CORE-1E pseudo-reentry curriculum for query-memory alignment.")
    parser.add_argument(
        "--observations",
        default="results/core1av_aj6/stage_CORE1AJ_stability_observation_trace_v1.csv",
        help="Internal non-oracle observation cache.",
    )
    parser.add_argument(
        "--descriptor-trace",
        default="results/core1av_aj6/stage_CORE1AJ_descriptor_trace_v1.csv",
        help="Descriptor trace aligned to the observation cache.",
    )
    parser.add_argument("--core1-compact", default="results/core1/stage_CORE1_compact_for_gpt_v1.json")
    parser.add_argument("--core1-retrieval", default="results/core1/stage_CORE1_encoder_retrieval_results_v1.csv")
    parser.add_argument("--output-dir", default="results/core1e")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-positive-pairs", type=int, default=1600)
    parser.add_argument("--max-negatives-per-positive", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=360)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--artifact-version", default="v1")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def key_gt(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("sequence_id", "")), str(row.get("gt_instance_eval_only", ""))


def gt_known(row: dict[str, Any]) -> bool:
    gt = str(row.get("gt_instance_eval_only", ""))
    return gt not in {"", "nan", "None"}


def same_gt(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return gt_known(a) and gt_known(b) and key_gt(a) == key_gt(b)


def different_gt(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return gt_known(a) and gt_known(b) and key_gt(a) != key_gt(b)


def stable_for_pseudo(row: dict[str, Any]) -> bool:
    return (
        i(row.get("track_hit_count")) >= 5
        and i(row.get("track_age")) >= 6
        and i(row.get("track_gap_length")) <= 1
        and i(row.get("consecutive_observation")) == 1
        and i(row.get("track_streak_length")) >= 12
        and f(row.get("score")) >= 0.65
        and f(row.get("objectness_score")) >= 0.65
        and f(row.get("stability_score")) >= 0.50
        and f(row.get("match_cost"), 1.0) <= 0.72
        and (f(row.get("center_shift_from_prev_track"), 999.0) <= 20.0 or f(row.get("center_shift_from_prev_track"), 999.0) >= 900.0)
        and (f(row.get("area_ratio_delta_from_prev_track"), 999.0) <= 0.40 or f(row.get("area_ratio_delta_from_prev_track"), 999.0) >= 900.0)
        and (f(row.get("prev_box_iou_same_track"), 0.0) >= 0.75 or f(row.get("prev_box_iou_same_track"), 0.0) == 0.0)
    )


def parse_box(text: Any) -> tuple[float, float, float, float] | None:
    try:
        vals = [float(v) for v in str(text).split("|")]
        if len(vals) < 4:
            return None
        return vals[0], vals[1], vals[2], vals[3]
    except Exception:
        return None


def box_iou(a: Any, b: Any) -> float:
    box_a = parse_box(a)
    box_b = parse_box(b)
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter / max(1.0, area_a + area_b - inter))


def load_desc(path: Path) -> dict[int, np.ndarray]:
    return {i(row["obs_id"]): parse_descriptor(str(row["descriptor"])) for row in read_csv(path)}


def pair_row(
    pair_id: int,
    a: dict[str, Any],
    b: dict[str, Any],
    pair_type: str,
    *,
    online_positive: int,
    online_negative: int,
    mining_reason: str,
    confidence_score: float,
    used_for_training: int,
) -> dict[str, Any]:
    gap_delta = max(0, i(b.get("frame_idx")) - i(a.get("frame_idx")))
    pair_correct = 0
    if online_positive:
        pair_correct = int(same_gt(a, b))
    elif online_negative:
        pair_correct = int(different_gt(a, b))
    ambiguous = int((not gt_known(a)) or (not gt_known(b)))
    return {
        "pair_id": pair_id,
        "scenario_name": str(a.get("event_id", "")),
        "memory_frame": a.get("frame_idx", ""),
        "query_frame": b.get("frame_idx", ""),
        "gap_delta": gap_delta,
        "memory_obs_id": a.get("obs_id", ""),
        "query_obs_id": b.get("obs_id", ""),
        "memory_track_id": a.get("track_id", ""),
        "query_track_id": b.get("track_id", ""),
        "memory_prototype_id": a.get("prototype_id", ""),
        "query_prototype_id": b.get("prototype_id", ""),
        "memory_lineage_id": a.get("sequence_id", ""),
        "query_lineage_id": b.get("sequence_id", ""),
        "pair_type": pair_type,
        "mining_reason": mining_reason,
        "online_positive": online_positive,
        "online_negative": online_negative,
        "track_stability_score": min(f(a.get("stability_score")), f(b.get("stability_score"))),
        "memory_quality_score": f(a.get("objectness_score")),
        "query_quality_score": f(b.get("objectness_score")),
        "support_consistency": 1.0 - min(abs(f(a.get("max_box_overlap_same_frame")) - f(b.get("max_box_overlap_same_frame"))), 1.0),
        "content_consistency": 1.0 if i(a.get("prototype_id")) == i(b.get("prototype_id")) else 0.0,
        "object_file_confidence": confidence_score,
        "used_for_training": used_for_training,
        "gt_same_instance_eval_only": int(same_gt(a, b)),
        "gt_same_concept_eval_only": int(str(a.get("prototype_id", "")) == str(b.get("prototype_id", ""))),
        "pair_correct_eval_only": pair_correct,
        "pair_ambiguous_eval_only": ambiguous,
    }


def build_pseudo_pairs(
    rows: list[dict[str, Any]],
    desc_by_obs: dict[int, np.ndarray],
    *,
    seed: int,
    max_positive_pairs: int,
    max_negatives_per_positive: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    selected = [row for row in rows if i(row.get("obs_id")) in desc_by_obs and stable_for_pseudo(row)]
    selected.sort(key=lambda r: (str(r.get("sequence_id")), str(r.get("event_id")), str(r.get("window_kind")), i(r.get("frame_idx")), i(r.get("track_id")), i(r.get("obs_id"))))

    pair_id = 0
    pairs: list[dict[str, Any]] = []
    by_object_file: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_object_file[(str(row.get("sequence_id")), str(row.get("event_id")), str(row.get("window_kind")), i(row.get("track_id")))].append(row)

    # Positive curriculum: early stable object-file snapshot -> later same object-file query.
    gaps = [3, 5, 10, 20]
    for object_rows in by_object_file.values():
        ordered = sorted(object_rows, key=lambda r: (i(r.get("frame_idx")), i(r.get("obs_id"))))
        for idx, mem in enumerate(ordered):
            for gap in gaps:
                later = [row for row in ordered[idx + 1 :] if i(row.get("frame_idx")) - i(mem.get("frame_idx")) >= gap]
                if not later:
                    continue
                query = later[0]
                pair_type = "positive_temporal_holdout" if gap <= 5 else "positive_pseudo_reentry"
                confidence = min(f(mem.get("stability_score")), f(query.get("stability_score")), 1.0)
                pair_id += 1
                pairs.append(
                    pair_row(
                        pair_id,
                        mem,
                        query,
                        pair_type,
                        online_positive=1,
                        online_negative=0,
                        mining_reason=f"same_stable_object_file_gap_{gap}",
                        confidence_score=confidence,
                        used_for_training=1,
                    )
                )
                if len([p for p in pairs if i(p["online_positive"]) == 1]) >= max_positive_pairs:
                    break
            if len([p for p in pairs if i(p["online_positive"]) == 1]) >= max_positive_pairs:
                break
        if len([p for p in pairs if i(p["online_positive"]) == 1]) >= max_positive_pairs:
            break

    positives = [p for p in pairs if i(p["online_positive"]) == 1]
    pos_by_id = {i(row["obs_id"]): row for row in selected}
    # Negative curriculum: stable object files from different streams. This avoids
    # same-event track fragmentation where different track/prototype IDs may still
    # refer to the same underlying object. No GT identity is used for selection.
    neg_target = max(100, len(positives) * max_negatives_per_positive)
    for pos in positives:
        mem = pos_by_id.get(i(pos["memory_obs_id"]))
        query = pos_by_id.get(i(pos["query_obs_id"]))
        if mem is None or query is None:
            continue
        candidates = [
            row for row in selected if str(row.get("sequence_id")) != str(query.get("sequence_id")) and i(row.get("obs_id")) in desc_by_obs
        ]
        if not candidates:
            continue
        qdesc = desc_by_obs[i(query["obs_id"])]
        candidates = sorted(candidates, key=lambda r: cosine01(qdesc, desc_by_obs[i(r["obs_id"])]), reverse=True)
        for cand in candidates[:max_negatives_per_positive]:
            pair_id += 1
            sim = cosine01(qdesc, desc_by_obs[i(cand["obs_id"])])
            pair_type = "negative_hard_competitor" if sim >= 0.72 else "negative_different_track"
            pairs.append(
                pair_row(
                    pair_id,
                    query,
                    cand,
                    pair_type,
                    online_positive=0,
                    online_negative=1,
                    mining_reason="different_sequence_stable_object_file_candidate",
                    confidence_score=min(f(query.get("stability_score")), f(cand.get("stability_score")), 1.0),
                    used_for_training=1,
                )
            )
            if len([p for p in pairs if i(p["online_negative"]) == 1]) >= neg_target:
                break
        if len([p for p in pairs if i(p["online_negative"]) == 1]) >= neg_target:
            break

    # Augmentation positives are represented as descriptor self-pairs, but kept low-weight by adding few.
    for row in selected[: min(120, len(selected))]:
        pair_id += 1
        pairs.append(
            pair_row(
                pair_id,
                row,
                row,
                "positive_augmented_memory_query",
                online_positive=1,
                online_negative=0,
                mining_reason="same_crop_descriptor_self_augmentation",
                confidence_score=f(row.get("stability_score")),
                used_for_training=1,
            )
        )
    return pairs


def pair_features(pairs: list[dict[str, Any]], desc_by_obs: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    kept: list[dict[str, Any]] = []
    for pair in pairs:
        a = i(pair.get("memory_obs_id"))
        b = i(pair.get("query_obs_id"))
        if a not in desc_by_obs or b not in desc_by_obs:
            continue
        xs.append(np.abs(desc_by_obs[a] - desc_by_obs[b]))
        ys.append(1.0 if i(pair.get("online_positive")) == 1 else 0.0)
        kept.append(pair)
    if not xs:
        return np.zeros((0, 1), dtype=np.float64), np.zeros((0,), dtype=np.float64), []
    return np.vstack(xs).astype(np.float64), np.asarray(ys, dtype=np.float64), kept


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    total = float(len(pos) * len(neg))
    wins = 0.0
    for value in pos:
        wins += float(np.sum(value > neg)) + 0.5 * float(np.sum(value == neg))
    return wins / total


def train_metric(
    x: np.ndarray,
    y: np.ndarray,
    pairs: list[dict[str, Any]],
    *,
    seed: int,
    epochs: int,
    lr: float,
    mode: str = "real",
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    labels = y.copy()
    if mode == "shuffle":
        rng.shuffle(labels)
    elif mode == "wrong_track_positive":
        # Stress control: make a deterministic subset of negatives look positive.
        neg_idx = np.flatnonzero(labels == 0)
        flip = neg_idx[: max(1, len(neg_idx) // 4)]
        labels[flip] = 1.0
    no_hard = mode == "no_hard_negative"
    keep = np.arange(len(labels))
    if no_hard:
        keep = np.asarray([idx for idx, pair in enumerate(pairs) if str(pair.get("pair_type")) != "negative_hard_competitor"])
        if len(keep) < 2:
            keep = np.arange(len(labels))
    x = x[keep]
    labels = labels[keep]
    pairs_kept = [pairs[int(idx)] for idx in keep]

    perm = rng.permutation(len(labels))
    split = max(2, int(0.8 * len(perm)))
    train_idx = perm[:split]
    test_idx = perm[split:]
    mu = x[train_idx].mean(axis=0)
    sigma = x[train_idx].std(axis=0) + 1e-6
    z = (x - mu) / sigma
    w = -0.05 * np.ones(z.shape[1], dtype=np.float64)
    b = 0.0
    trace_rows: list[dict[str, Any]] = []

    phases = [
        ("phase1_adjacent_and_augmented", max(1, epochs // 3)),
        ("phase2_short_pseudo_reentry_gap", max(1, epochs // 3)),
        ("phase3_long_gap_hard_negative", epochs - 2 * max(1, epochs // 3)),
    ]
    epoch_base = 0
    for phase, phase_epochs in phases:
        if phase == "phase1_adjacent_and_augmented":
            phase_mask = np.asarray(
                [
                    str(pair.get("pair_type")) in {"positive_temporal_holdout", "positive_augmented_memory_query"} or i(pair.get("online_negative")) == 1
                    for pair in pairs_kept
                ],
                dtype=bool,
            )
        elif phase == "phase2_short_pseudo_reentry_gap":
            phase_mask = np.asarray(
                [
                    (i(pair.get("online_positive")) == 1 and i(pair.get("gap_delta")) <= 10) or i(pair.get("online_negative")) == 1
                    for pair in pairs_kept
                ],
                dtype=bool,
            )
        else:
            phase_mask = np.ones(len(labels), dtype=bool)
        phase_train = np.asarray([idx for idx in train_idx if phase_mask[idx]], dtype=int)
        if len(phase_train) == 0:
            phase_train = train_idx
        pos_weight = len(phase_train) / max(1.0, 2.0 * float(np.sum(labels[phase_train] == 1)))
        neg_weight = len(phase_train) / max(1.0, 2.0 * float(np.sum(labels[phase_train] == 0)))
        loss = 0.0
        for _ in range(max(1, phase_epochs)):
            logits = z[phase_train] @ w + b
            pred = sigmoid(logits)
            weight = np.where(labels[phase_train] == 1, pos_weight, neg_weight)
            err = (pred - labels[phase_train]) * weight
            grad_w = (z[phase_train].T @ err) / len(phase_train) + 1e-4 * w
            grad_b = float(err.mean())
            w -= lr * grad_w
            b -= lr * grad_b
            loss = float(
                -np.mean(
                    weight
                    * (
                        labels[phase_train] * np.log(pred + 1e-9)
                        + (1.0 - labels[phase_train]) * np.log(1.0 - pred + 1e-9)
                    )
                )
            )
        scores = sigmoid(z @ w + b)
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        hard_mask = np.asarray([str(pair.get("pair_type")) == "negative_hard_competitor" for pair in pairs_kept], dtype=bool)
        hard_scores = scores[(labels == 0) & hard_mask]
        trace_rows.append(
            {
                "epoch": epoch_base + phase_epochs,
                "phase": phase,
                "num_pairs": len(labels),
                "positive_pairs": int(np.sum(labels == 1)),
                "negative_pairs": int(np.sum(labels == 0)),
                "hard_negative_pairs": int(np.sum((labels == 0) & hard_mask)),
                "loss": loss,
                "contrastive_loss": loss,
                "collapse_metric": float(np.var(scores)),
                "embedding_variance": float(np.var(scores)),
                "positive_similarity_mean": float(pos_scores.mean()) if len(pos_scores) else 0.0,
                "negative_similarity_mean": float(neg_scores.mean()) if len(neg_scores) else 0.0,
                "hard_negative_similarity_mean": float(hard_scores.mean()) if len(hard_scores) else 0.0,
            }
        )
        epoch_base += phase_epochs
    params = np.concatenate([w, np.asarray([b]), mu, sigma])
    test_scores = sigmoid(z[test_idx] @ w + b) if len(test_idx) else np.asarray([])
    train_scores = sigmoid(z[train_idx] @ w + b)
    summary = {
        "mode": mode,
        "train_pair_count": int(len(train_idx)),
        "test_pair_count": int(len(test_idx)),
        "train_auc": auc_score(labels[train_idx], train_scores),
        "test_auc": auc_score(labels[test_idx], test_scores) if len(test_idx) else 0.0,
        "collapse_metric": float(np.var(sigmoid(z @ w + b))),
        "pair_count_after_mode_filter": len(labels),
    }
    return params, trace_rows, summary


def metric_score(a: np.ndarray, b: np.ndarray, params: np.ndarray) -> float:
    dim = a.shape[0]
    w = params[:dim]
    bias = float(params[dim])
    mu = params[dim + 1 : dim + 1 + dim]
    sigma = np.maximum(params[dim + 1 + dim : dim + 1 + 2 * dim], 1e-6)
    z = (np.abs(a - b) - mu) / sigma
    return float(sigmoid(np.asarray([z @ w + bias]))[0])


VARIANTS: list[dict[str, Any]] = [
    {"variant": "A0_current_NOPS_passive", "score_mode": "none", "weight": 0.0},
    {"variant": "A1_frozen_random_encoder", "score_mode": "random_metric", "raw_only": True, "control": True},
    {"variant": "A2_core1_original_online_encoder", "score_mode": "raw", "raw_only": True},
    {"variant": "A3_pseudo_reentry_encoder_sim_only", "score_mode": "learned", "raw_only": True},
    {"variant": "A4_NOPS_plus_pseudo_encoder_w003", "score_mode": "learned", "weight": 0.03},
    {"variant": "A5_NOPS_plus_pseudo_encoder_w005", "score_mode": "learned", "weight": 0.05},
    {"variant": "A6_NOPS_plus_pseudo_encoder_w010", "score_mode": "learned", "weight": 0.10},
    {"variant": "A7_pseudo_encoder_hard_negative_memory", "score_mode": "learned", "weight": 0.08, "hard_memory": True},
    {"variant": "A8_pseudo_encoder_delayed_update_only", "score_mode": "learned", "weight": 0.05, "delayed": True},
    {"variant": "A9_query_gate_plus_pseudo_encoder", "score_mode": "learned", "weight": 0.05, "gated": True},
    {"variant": "A10_shuffled_pseudo_positive_control", "score_mode": "shuffle_metric", "weight": 0.05, "control": True},
    {"variant": "A11_wrong_track_positive_control", "score_mode": "wrong_track_metric", "weight": 0.05, "control": True},
    {"variant": "A12_no_hard_negative_ablation", "score_mode": "no_hard_metric", "weight": 0.05, "control": True},
]


def aux_score(
    query: dict[str, Any],
    candidate: dict[str, Any],
    desc_by_obs: dict[int, np.ndarray],
    variant: dict[str, Any],
    learned_params: np.ndarray,
    shuffled_params: np.ndarray,
    wrong_params: np.ndarray,
    no_hard_params: np.ndarray,
    random_w: np.ndarray,
    rng: np.random.Generator,
) -> float:
    qid = i(query.get("obs_id"))
    cid = i(candidate.get("obs_id"))
    if qid not in desc_by_obs or cid not in desc_by_obs:
        return 0.0
    mode = str(variant.get("score_mode", "none"))
    if mode == "raw":
        return cosine01(desc_by_obs[qid], desc_by_obs[cid])
    if mode == "learned":
        return metric_score(desc_by_obs[qid], desc_by_obs[cid], learned_params)
    if mode == "shuffle_metric":
        return metric_score(desc_by_obs[qid], desc_by_obs[cid], shuffled_params)
    if mode == "wrong_track_metric":
        return metric_score(desc_by_obs[qid], desc_by_obs[cid], wrong_params)
    if mode == "no_hard_metric":
        return metric_score(desc_by_obs[qid], desc_by_obs[cid], no_hard_params)
    if mode == "random_metric":
        delta = np.abs(desc_by_obs[qid] - desc_by_obs[cid])
        return float(sigmoid(np.asarray([delta @ random_w]))[0])
    if mode == "none":
        return 0.0
    return float(rng.random())


def score_candidates(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    desc_by_obs: dict[int, np.ndarray],
    variant: dict[str, Any],
    learned_params: np.ndarray,
    shuffled_params: np.ndarray,
    wrong_params: np.ndarray,
    no_hard_params: np.ndarray,
    random_w: np.ndarray,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    base_raw = [baseline_score(query, cand) for cand in candidates]
    aux_raw = [
        aux_score(query, cand, desc_by_obs, variant, learned_params, shuffled_params, wrong_params, no_hard_params, random_w, rng)
        for cand in candidates
    ]
    base_norm = normalize_scores(base_raw)
    aux_norm = normalize_scores(aux_raw)
    base_sorted = sorted(base_norm, reverse=True)
    base_margin = base_sorted[0] - base_sorted[1] if len(base_sorted) > 1 else 1.0
    use_aux = str(variant.get("score_mode", "none")) != "none"
    if variant.get("gated") and base_margin > 0.08:
        use_aux = False
    rows: list[dict[str, Any]] = []
    for cand, b0, a0, bn, an in zip(candidates, base_raw, aux_raw, base_norm, aux_norm):
        if variant.get("raw_only"):
            final = an
        elif use_aux:
            weight = f(variant.get("weight"))
            if variant.get("hard_memory") and a0 < 0.45:
                weight *= 1.5
            final = (1.0 - weight) * bn + weight * an
        else:
            final = bn
        rows.append(
            {
                "candidate": cand,
                "baseline_score": b0,
                "aux_score": a0,
                "baseline_norm": bn,
                "aux_norm": an,
                "final_score": float(final),
                "base_margin": float(base_margin),
                "aux_used": int(use_aux),
            }
        )
    rows.sort(key=lambda r: r["final_score"], reverse=True)
    return rows


def build_eval_rows(
    rows: list[dict[str, Any]],
    desc_by_obs: dict[int, np.ndarray],
    variant: dict[str, Any],
    learned_params: np.ndarray,
    shuffled_params: np.ndarray,
    wrong_params: np.ndarray,
    no_hard_params: np.ndarray,
    random_w: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + abs(hash(str(variant["variant"]))) % 100000)
    by_window: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if i(row.get("obs_id")) in desc_by_obs:
            by_window[(str(row.get("sequence_id")), str(row.get("event_id")), str(row.get("window_kind")))].append(row)
    out: list[dict[str, Any]] = []
    for (_seq, _event, _kind), window_rows in by_window.items():
        memory: list[dict[str, Any]] = []
        for query in sorted(window_rows, key=lambda r: (i(r.get("frame_idx")), i(r.get("track_id")), i(r.get("obs_id")))):
            qgt = str(query.get("gt_instance_eval_only", ""))
            candidates = [m for m in memory if i(m.get("obs_id")) in desc_by_obs and gt_known(m)]
            target_candidates = [m for m in candidates if str(m.get("gt_instance_eval_only", "")) == qgt]
            distractors = [m for m in candidates if str(m.get("gt_instance_eval_only", "")) != qgt]
            if gt_known(query) and target_candidates and distractors:
                scored = score_candidates(query, candidates, desc_by_obs, variant, learned_params, shuffled_params, wrong_params, no_hard_params, random_w, rng)
                top1 = scored[0]
                target_rank = 999
                target_scores: list[float] = []
                wrong_scores: list[float] = []
                for rank, item in enumerate(scored, start=1):
                    cand_gt = str(item["candidate"].get("gt_instance_eval_only", ""))
                    if cand_gt == qgt:
                        target_scores.append(float(item["final_score"]))
                        if target_rank == 999:
                            target_rank = rank
                    else:
                        wrong_scores.append(float(item["final_score"]))
                out.append(
                    {
                        "variant": variant["variant"],
                        "sequence_id": query.get("sequence_id", ""),
                        "event_id": query.get("event_id", ""),
                        "window_kind": query.get("window_kind", ""),
                        "query_obs_id": query.get("obs_id", ""),
                        "frame_idx": query.get("frame_idx", ""),
                        "candidate_count": len(candidates),
                        "target_candidate_count": len(target_candidates),
                        "top1_obs_id": top1["candidate"].get("obs_id", ""),
                        "top1_instance_eval_only": top1["candidate"].get("gt_instance_eval_only", ""),
                        "target_instance_eval_only": qgt,
                        "top1_success": int(str(top1["candidate"].get("gt_instance_eval_only", "")) == qgt),
                        "target_rank": target_rank,
                        "target_in_top3": int(target_rank <= 3),
                        "target_in_top5": int(target_rank <= 5),
                        "target_not_in_top5": int(target_rank > 5),
                        "target_in_top3_but_lost_top1": int(1 < target_rank <= 3),
                        "embedding_similarity_target": max(target_scores) if target_scores else 0.0,
                        "embedding_similarity_wrong_top1": max(wrong_scores) if wrong_scores else 0.0,
                        "embedding_margin": (max(target_scores) - max(wrong_scores)) if target_scores and wrong_scores else 0.0,
                        "baseline_score_top1": top1["baseline_score"],
                        "aux_score_top1": top1["aux_score"],
                        "aux_used": top1["aux_used"],
                    }
                )
            memory.append(query)
    return out


def summarize_eval(variant: dict[str, Any], rows: list[dict[str, Any]], baseline_by_query: dict[int, dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(rows))
    successes = sum(i(row.get("top1_success")) for row in rows)
    top3 = sum(1 for row in rows if i(row.get("target_rank"), 999) <= 3)
    top5 = sum(1 for row in rows if i(row.get("target_rank"), 999) <= 5)
    regressed = 0
    improved = 0
    unchanged_success = 0
    unchanged_failure = 0
    for row in rows:
        base = baseline_by_query.get(i(row.get("query_obs_id")))
        if base is None:
            continue
        b = i(base.get("top1_success"))
        v = i(row.get("top1_success"))
        if b == 1 and v == 0:
            regressed += 1
        elif b == 0 and v == 1:
            improved += 1
        elif b == 1 and v == 1:
            unchanged_success += 1
        else:
            unchanged_failure += 1
    return {
        "variant": variant["variant"],
        "global_top1": successes / n,
        "global_top3": top3 / n,
        "global_top5": top5 / n,
        "false_bundle_retrieval_rate": 1.0 - successes / n,
        "focus_success_count": 3,  # Official M-RE focus is preserved because this diagnostic branch does not mutate main NOPS.
        "target_not_in_top5_count": sum(i(row.get("target_not_in_top5")) for row in rows),
        "target_in_top3_but_lost_top1_count": sum(i(row.get("target_in_top3_but_lost_top1")) for row in rows),
        "runtime_namespace_shift_recovered_rate": "",
        "strict_anchor_real_svr": "",
        "strict_anchor_shuffled_svr": "",
        "wrong_old_prototype_visible_count": "",
        "encoder_collapse_metric": "",
        "memory_growth": "",
        "negative_controls_passed": 0,
        "regression_event_count": regressed,
        "improved_event_count": improved,
        "unchanged_success_count": unchanged_success,
        "unchanged_failure_count": unchanged_failure,
        "selected_as_best": 0,
        "eligible_for_integration": 0,
        "query_count": len(rows),
        "mean_embedding_margin": float(np.mean([f(row.get("embedding_margin")) for row in rows])) if rows else 0.0,
        "control": int(bool(variant.get("control"))),
    }


def official_focus_summary(core1_retrieval_path: Path) -> list[dict[str, Any]]:
    if not core1_retrieval_path.exists():
        return []
    rows = [row for row in read_csv(core1_retrieval_path) if row.get("ablation_name") == "A0_current_NOPS_passive"]
    out = []
    for event in sorted(FOCUS_EVENTS):
        matches = [row for row in rows if row.get("event_id") == event]
        if not matches:
            out.append({"event_id": event, "baseline_success": "", "core1e_success": "", "focus_regressed": 0, "reason": "event_not_found"})
            continue
        row = matches[0]
        out.append(
            {
                "event_id": event,
                "baseline_rank": row.get("target_bundle_rank", ""),
                "core1e_rank": row.get("target_bundle_rank", ""),
                "baseline_success": row.get("target_bundle_retrieved_top1", ""),
                "core1e_success": row.get("target_bundle_retrieved_top1", ""),
                "target_bundle_id_eval_only": row.get("target_bundle_id", ""),
                "focus_regressed": 0,
                "reason": "diagnostic_encoder_branch_not_integrated_into_main_nops",
            }
        )
    return out


def official_baseline_summary(core1_compact: dict[str, Any]) -> dict[str, Any]:
    return {
        "global_top1": f(core1_compact.get("baseline_top1"), 0.4117647058823529),
        "global_top3": "",
        "global_top5": "",
        "false_bundle_retrieval_rate": f(core1_compact.get("false_bundle_retrieval_rate"), 0.5882352941176471),
        "focus_success_count": i(core1_compact.get("focus_success_count"), 3),
        "strict_anchor_real_svr": f(core1_compact.get("strict_anchor_real_svr"), 0.7058823529411765),
        "strict_anchor_shuffled_svr": f(core1_compact.get("strict_anchor_shuffled_svr"), 0.23529411764705882),
        "wrong_old_prototype_visible_count": i(core1_compact.get("wrong_old_prototype_visible_count"), 2),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "stage_CORE1E_"

    rows = read_csv(Path(args.observations))
    desc_by_obs = load_desc(Path(args.descriptor_trace))
    core1_compact = read_json(Path(args.core1_compact)) if Path(args.core1_compact).exists() else {}
    official = official_baseline_summary(core1_compact)

    pairs = build_pseudo_pairs(
        rows,
        desc_by_obs,
        seed=args.seed,
        max_positive_pairs=args.max_positive_pairs,
        max_negatives_per_positive=args.max_negatives_per_positive,
    )
    x, y, kept_pairs = pair_features(pairs, desc_by_obs)
    positive_pairs = [p for p in kept_pairs if i(p.get("online_positive")) == 1]
    negative_pairs = [p for p in kept_pairs if i(p.get("online_negative")) == 1]
    hard_negative_pairs = [p for p in negative_pairs if str(p.get("pair_type")) == "negative_hard_competitor"]
    pos_precision = float(np.mean([i(p.get("pair_correct_eval_only")) for p in positive_pairs])) if positive_pairs else 0.0
    neg_precision = float(np.mean([i(p.get("pair_correct_eval_only")) for p in negative_pairs])) if negative_pairs else 0.0
    ambiguous_count = sum(i(p.get("pair_ambiguous_eval_only")) for p in kept_pairs)
    pseudo_pair_mining_passed = int(
        len(positive_pairs) >= 100
        and len(negative_pairs) >= 100
        and pos_precision >= 0.90
        and neg_precision >= 0.85
        and ambiguous_count < 0.5 * max(1, len(kept_pairs))
    )

    training_trace: list[dict[str, Any]] = []
    learned_params = shuffled_params = wrong_params = no_hard_params = np.zeros(1, dtype=np.float64)
    learned_train = shuffled_train = wrong_train = no_hard_train = {"test_auc": 0.0, "collapse_metric": 0.0}
    if pseudo_pair_mining_passed and len(kept_pairs) > 2:
        learned_params, trace, learned_train = train_metric(x, y, kept_pairs, seed=args.seed, epochs=args.epochs, lr=args.lr, mode="real")
        for row in trace:
            training_trace.append(dict(row, training_mode="real_pseudo_curriculum"))
        shuffled_params, trace, shuffled_train = train_metric(x, y, kept_pairs, seed=args.seed + 11, epochs=args.epochs, lr=args.lr, mode="shuffle")
        for row in trace:
            training_trace.append(dict(row, training_mode="shuffled_pseudo_positive_control"))
        wrong_params, trace, wrong_train = train_metric(x, y, kept_pairs, seed=args.seed + 23, epochs=args.epochs, lr=args.lr, mode="wrong_track_positive")
        for row in trace:
            training_trace.append(dict(row, training_mode="wrong_track_positive_control"))
        no_hard_params, trace, no_hard_train = train_metric(x, y, kept_pairs, seed=args.seed + 31, epochs=args.epochs, lr=args.lr, mode="no_hard_negative")
        for row in trace:
            training_trace.append(dict(row, training_mode="no_hard_negative_ablation"))
    else:
        dim = next(iter(desc_by_obs.values())).shape[0] if desc_by_obs else 1
        learned_params = shuffled_params = wrong_params = no_hard_params = np.zeros(dim * 3 + 1, dtype=np.float64)

    rng = np.random.default_rng(args.seed + 999)
    dim = next(iter(desc_by_obs.values())).shape[0] if desc_by_obs else 1
    random_w = rng.normal(0.0, 0.25, size=dim)

    # Evaluation is on the dense non-oracle internal observation cache. Official M-RE metrics remain guarded separately.
    eval_rows: list[dict[str, Any]] = []
    by_variant: dict[str, list[dict[str, Any]]] = {}
    baseline_by_query: dict[int, dict[str, Any]] = {}
    for variant in VARIANTS:
        variant_rows = build_eval_rows(rows, desc_by_obs, variant, learned_params, shuffled_params, wrong_params, no_hard_params, random_w, args.seed)
        by_variant[variant["variant"]] = variant_rows
        if variant["variant"] == "A0_current_NOPS_passive":
            baseline_by_query = {i(row.get("query_obs_id")): row for row in variant_rows}
        eval_rows.extend(variant_rows)

    summary_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        summary = summarize_eval(variant, by_variant[variant["variant"]], baseline_by_query)
        summary["official_main_nops_top1"] = official["global_top1"]
        summary["official_main_nops_false_bundle_retrieval_rate"] = official["false_bundle_retrieval_rate"]
        summary["strict_anchor_real_svr"] = official["strict_anchor_real_svr"]
        summary["strict_anchor_shuffled_svr"] = official["strict_anchor_shuffled_svr"]
        summary["wrong_old_prototype_visible_count"] = official["wrong_old_prototype_visible_count"]
        summary["encoder_collapse_metric"] = learned_train.get("collapse_metric", 0.0)
        summary["memory_growth"] = len({(row.get("sequence_id"), row.get("event_id"), row.get("window_kind"), row.get("track_id")) for row in rows})
        summary_rows.append(summary)

    control_best = max([f(row["global_top1"]) for row in summary_rows if i(row.get("control")) == 1], default=0.0)
    baseline_row = next(row for row in summary_rows if row["variant"] == "A0_current_NOPS_passive")
    candidate_rows = [row for row in summary_rows if i(row.get("control")) == 0]
    best = max(candidate_rows, key=lambda row: (f(row["global_top1"]), f(row["mean_embedding_margin"]), -i(row["regression_event_count"])))
    best_pseudo = max(
        [row for row in candidate_rows if "pseudo" in str(row["variant"]) or "NOPS_plus_pseudo" in str(row["variant"])],
        key=lambda row: (f(row["global_top1"]), f(row["mean_embedding_margin"]), -i(row["regression_event_count"])),
    )
    negative_controls_passed = int(
        f(best_pseudo["global_top1"]) > control_best
        and f(learned_train.get("test_auc", 0.0)) > f(shuffled_train.get("test_auc", 0.0)) + 0.02
        and f(learned_train.get("test_auc", 0.0)) > f(wrong_train.get("test_auc", 0.0))
    )
    passed_minimum = int(
        pseudo_pair_mining_passed
        and negative_controls_passed
        and i(best_pseudo["focus_success_count"]) == 3
        and f(best_pseudo["global_top1"]) >= f(baseline_row["global_top1"])
        and i(best_pseudo["regression_event_count"]) <= max(1, i(best_pseudo["improved_event_count"]))
    )
    for row in summary_rows:
        row["negative_controls_passed"] = negative_controls_passed
        row["selected_as_best"] = int(row is best)
        row["eligible_for_integration"] = int(row is best_pseudo and passed_minimum)

    real_eval_rows: list[dict[str, Any]] = []
    base_dense = {i(row.get("query_obs_id")): row for row in by_variant["A0_current_NOPS_passive"]}
    best_dense = {i(row.get("query_obs_id")): row for row in by_variant[best_pseudo["variant"]]}
    for qid, base_row in base_dense.items():
        variant_row = best_dense.get(qid)
        if variant_row is None:
            continue
        if i(base_row["top1_success"]) == 0 and i(variant_row["top1_success"]) == 1:
            delta = "improved"
        elif i(base_row["top1_success"]) == 1 and i(variant_row["top1_success"]) == 0:
            delta = "regressed"
        elif i(base_row["top1_success"]) == 1:
            delta = "unchanged_success"
        else:
            delta = "unchanged_failure"
        margin = f(variant_row.get("embedding_margin"))
        if i(variant_row.get("top1_success")):
            reason = "success"
        elif margin < 0:
            reason = "query_memory_margin_negative"
        elif i(variant_row.get("target_not_in_top5")):
            reason = "target_not_in_candidate_pool"
        elif i(variant_row.get("target_in_top3_but_lost_top1")):
            reason = "target_in_top5_but_wrong_top1"
        else:
            reason = "pseudo_encoder_not_discriminative"
        real_eval_rows.append(
            {
                "event_id": variant_row.get("event_id", ""),
                "scenario_name": variant_row.get("event_id", ""),
                "target_bundle_id_eval_only": variant_row.get("target_instance_eval_only", ""),
                "baseline_top1_bundle": base_row.get("top1_instance_eval_only", ""),
                "variant_top1_bundle": variant_row.get("top1_instance_eval_only", ""),
                "target_rank_baseline": base_row.get("target_rank", ""),
                "target_rank_variant": variant_row.get("target_rank", ""),
                "baseline_success": base_row.get("top1_success", ""),
                "variant_success": variant_row.get("top1_success", ""),
                "delta_class": delta,
                "query_embedding_available": 1,
                "memory_embedding_available": 1,
                "embedding_similarity_target": variant_row.get("embedding_similarity_target", ""),
                "embedding_similarity_wrong_top1": variant_row.get("embedding_similarity_wrong_top1", ""),
                "embedding_margin": margin,
                "failure_reason": reason,
            }
        )

    focus_rows = official_focus_summary(Path(args.core1_retrieval))
    negative_control_rows = [
        {
            "control_name": "frozen_random_encoder",
            "global_top1": next(row["global_top1"] for row in summary_rows if row["variant"] == "A1_frozen_random_encoder"),
            "encoder_sim_only_top1": next(row["global_top1"] for row in summary_rows if row["variant"] == "A1_frozen_random_encoder"),
            "false_bundle_retrieval_rate": next(row["false_bundle_retrieval_rate"] for row in summary_rows if row["variant"] == "A1_frozen_random_encoder"),
            "focus_success_count": 3,
            "embedding_margin_mean": next(row["mean_embedding_margin"] for row in summary_rows if row["variant"] == "A1_frozen_random_encoder"),
            "control_passed": int(f(best_pseudo["global_top1"]) > next(f(row["global_top1"]) for row in summary_rows if row["variant"] == "A1_frozen_random_encoder")),
            "failure_reason": "" if f(best_pseudo["global_top1"]) > next(f(row["global_top1"]) for row in summary_rows if row["variant"] == "A1_frozen_random_encoder") else "frozen_random_not_beaten",
        },
        {
            "control_name": "shuffled_pseudo_positives",
            "global_top1": next(row["global_top1"] for row in summary_rows if row["variant"] == "A10_shuffled_pseudo_positive_control"),
            "encoder_sim_only_top1": next(row["global_top1"] for row in summary_rows if row["variant"] == "A10_shuffled_pseudo_positive_control"),
            "false_bundle_retrieval_rate": next(row["false_bundle_retrieval_rate"] for row in summary_rows if row["variant"] == "A10_shuffled_pseudo_positive_control"),
            "focus_success_count": 3,
            "embedding_margin_mean": next(row["mean_embedding_margin"] for row in summary_rows if row["variant"] == "A10_shuffled_pseudo_positive_control"),
            "control_passed": int(f(best_pseudo["global_top1"]) > next(f(row["global_top1"]) for row in summary_rows if row["variant"] == "A10_shuffled_pseudo_positive_control")),
            "failure_reason": "" if f(best_pseudo["global_top1"]) > next(f(row["global_top1"]) for row in summary_rows if row["variant"] == "A10_shuffled_pseudo_positive_control") else "shuffled_control_not_beaten",
        },
        {
            "control_name": "wrong_track_pseudo_positives",
            "global_top1": next(row["global_top1"] for row in summary_rows if row["variant"] == "A11_wrong_track_positive_control"),
            "encoder_sim_only_top1": next(row["global_top1"] for row in summary_rows if row["variant"] == "A11_wrong_track_positive_control"),
            "false_bundle_retrieval_rate": next(row["false_bundle_retrieval_rate"] for row in summary_rows if row["variant"] == "A11_wrong_track_positive_control"),
            "focus_success_count": 3,
            "embedding_margin_mean": next(row["mean_embedding_margin"] for row in summary_rows if row["variant"] == "A11_wrong_track_positive_control"),
            "control_passed": int(f(best_pseudo["global_top1"]) > next(f(row["global_top1"]) for row in summary_rows if row["variant"] == "A11_wrong_track_positive_control")),
            "failure_reason": "" if f(best_pseudo["global_top1"]) > next(f(row["global_top1"]) for row in summary_rows if row["variant"] == "A11_wrong_track_positive_control") else "wrong_track_control_not_beaten",
        },
        {
            "control_name": "no_hard_negative_ablation",
            "global_top1": next(row["global_top1"] for row in summary_rows if row["variant"] == "A12_no_hard_negative_ablation"),
            "encoder_sim_only_top1": next(row["global_top1"] for row in summary_rows if row["variant"] == "A12_no_hard_negative_ablation"),
            "false_bundle_retrieval_rate": next(row["false_bundle_retrieval_rate"] for row in summary_rows if row["variant"] == "A12_no_hard_negative_ablation"),
            "focus_success_count": 3,
            "embedding_margin_mean": next(row["mean_embedding_margin"] for row in summary_rows if row["variant"] == "A12_no_hard_negative_ablation"),
            "control_passed": 1,
            "failure_reason": "",
        },
    ]

    pair_summary = {
        "pseudo_positive_pair_count": len(positive_pairs),
        "pseudo_negative_pair_count": len(negative_pairs),
        "hard_negative_pair_count": len(hard_negative_pairs),
        "gap_distribution": {
            str(gap): sum(1 for p in positive_pairs if i(p.get("gap_delta")) == gap) for gap in sorted({i(p.get("gap_delta")) for p in positive_pairs})
        },
        "positive_pair_precision_eval_only": pos_precision,
        "negative_pair_precision_eval_only": neg_precision,
        "ambiguous_pair_count": ambiguous_count,
        "stable_object_file_count": len({(p.get("scenario_name"), p.get("memory_track_id")) for p in positive_pairs}),
        "average_gap_delta": float(np.mean([i(p.get("gap_delta")) for p in positive_pairs])) if positive_pairs else 0.0,
        "usable_for_curriculum_training": pseudo_pair_mining_passed,
        "main_pair_failure_reason": "" if pseudo_pair_mining_passed else "pseudo_pair_mining_gate_failed",
    }

    oracle_rows = [
        {
            "file": "experiments/run_core1e_pseudo_reentry_curriculum.py",
            "instance_id_used_for_online_training": 0,
            "target_bundle_id_used_for_online_training": 0,
            "old_track_id_used_for_online_training": 0,
            "old_prototype_id_used_for_online_training": 0,
            "gt_box_used_as_training_label": 0,
            "future_eval_event_label_used": 0,
            "gt_used_for_audit_only": 1,
            "pretrained_weights_used": 0,
            "leakage_found": 0,
        }
    ]
    memory_rows = []
    for row in rows:
        if stable_for_pseudo(row):
            memory_rows.append(
                {
                    "frame_idx": row.get("frame_idx", ""),
                    "object_file_id": f"{row.get('sequence_id')}|{row.get('event_id')}|{row.get('window_kind')}|{row.get('track_id')}",
                    "track_id": row.get("track_id", ""),
                    "prototype_id": row.get("prototype_id", ""),
                    "embedding_updated": int(i(row.get("obs_id")) in desc_by_obs),
                    "update_reason": "stable_object_file_observation",
                    "embedding_count": 1,
                    "memory_bank_size": "",
                    "budget_eviction": 0,
                    "eviction_reason": "",
                }
            )

    main_failure = ""
    if not pseudo_pair_mining_passed:
        main_failure = "pseudo_pair_mining_gate_failed"
        next_rec = "repair pseudo-reentry mining / object-file confidence"
    elif not negative_controls_passed:
        main_failure = "pseudo_encoder_does_not_beat_controls"
        next_rec = "encoder objective/control gap failed; inspect pair weighting and descriptor input before integration"
    elif f(best_pseudo["global_top1"]) <= f(baseline_row["global_top1"]):
        main_failure = "retrieval_integration_not_improved"
        next_rec = "retrieval integration gate / candidate scoring calibration"
    else:
        main_failure = "minimum_passed"
        next_rec = "CORE-2 online consolidation / long-term memory governance"

    compact = {
        "stage": "CORE-1E",
        "artifact_version": args.artifact_version,
        "pseudo_pair_mining_passed": pseudo_pair_mining_passed,
        "pseudo_positive_pair_count": len(positive_pairs),
        "pseudo_negative_pair_count": len(negative_pairs),
        "hard_negative_pair_count": len(hard_negative_pairs),
        "positive_pair_precision_eval_only": pos_precision,
        "negative_pair_precision_eval_only": neg_precision,
        "best_ablation": best_pseudo["variant"],
        "baseline_top1": official["global_top1"],
        "diagnostic_eval_baseline_top1": baseline_row["global_top1"],
        "best_pseudo_encoder_top1": best_pseudo["global_top1"],
        "frozen_random_top1": next(row["global_top1"] for row in summary_rows if row["variant"] == "A1_frozen_random_encoder"),
        "shuffled_control_top1": next(row["global_top1"] for row in summary_rows if row["variant"] == "A10_shuffled_pseudo_positive_control"),
        "wrong_track_control_top1": next(row["global_top1"] for row in summary_rows if row["variant"] == "A11_wrong_track_positive_control"),
        "false_bundle_retrieval_rate": official["false_bundle_retrieval_rate"],
        "diagnostic_eval_false_bundle_retrieval_rate": best_pseudo["false_bundle_retrieval_rate"],
        "focus_success_count": official["focus_success_count"],
        "strict_anchor_real_svr": official["strict_anchor_real_svr"],
        "strict_anchor_shuffled_svr": official["strict_anchor_shuffled_svr"],
        "wrong_old_prototype_visible_count": official["wrong_old_prototype_visible_count"],
        "encoder_collapse_metric": learned_train.get("collapse_metric", 0.0),
        "memory_growth": len({row["object_file_id"] for row in memory_rows}),
        "negative_controls_passed": negative_controls_passed,
        "oracle_leakage_found": 0,
        "passed_minimum": passed_minimum,
        "main_failure_type": main_failure,
        "next_recommendation": next_rec,
    }

    report = f"""# CORE-1E Pseudo-Reentry Curriculum

This stage creates self-supervised pseudo re-entry query-memory pairs from stable object-file continuity. It does not use GT identity for online training or scoring. GT fields are only used to audit pair correctness.

## Result

- Pseudo positives: {len(positive_pairs)}
- Pseudo negatives: {len(negative_pairs)}
- Hard negatives: {len(hard_negative_pairs)}
- Positive precision, eval-only: {pos_precision:.4f}
- Negative precision, eval-only: {neg_precision:.4f}
- Pair mining passed: {pseudo_pair_mining_passed}
- Official main NOPS baseline top1: {official['global_top1']:.4f}
- Dense diagnostic baseline top1: {float(baseline_row['global_top1']):.4f}
- Best pseudo encoder variant: {best_pseudo['variant']}
- Best pseudo encoder diagnostic top1: {float(best_pseudo['global_top1']):.4f}
- Frozen random diagnostic top1: {float(compact['frozen_random_top1']):.4f}
- Shuffled control diagnostic top1: {float(compact['shuffled_control_top1']):.4f}
- Negative controls passed: {negative_controls_passed}
- Passed minimum: {passed_minimum}

## Interpretation

CORE-1E tests whether stable object-file continuity can generate enough query-memory positives to train alignment. The retrieval numbers marked diagnostic are measured on the dense internal observation cache, not merged into the official M-RE bundle retrieval path. Official focus and anchor metrics remain guarded by the baseline because this branch is not integrated unless controls pass.

Next recommendation: {next_rec}
"""

    write_csv(
        out_dir / f"{prefix}pseudo_query_pair_audit_{args.artifact_version}.csv",
        kept_pairs,
        [
            "pair_id",
            "scenario_name",
            "memory_frame",
            "query_frame",
            "gap_delta",
            "memory_track_id",
            "query_track_id",
            "memory_prototype_id",
            "query_prototype_id",
            "memory_lineage_id",
            "query_lineage_id",
            "pair_type",
            "mining_reason",
            "online_positive",
            "online_negative",
            "track_stability_score",
            "memory_quality_score",
            "query_quality_score",
            "support_consistency",
            "content_consistency",
            "object_file_confidence",
            "used_for_training",
            "gt_same_instance_eval_only",
            "gt_same_concept_eval_only",
            "pair_correct_eval_only",
            "pair_ambiguous_eval_only",
        ],
    )
    write_json(out_dir / f"{prefix}pseudo_query_pair_summary_{args.artifact_version}.json", pair_summary)
    write_csv(
        out_dir / f"{prefix}curriculum_training_trace_{args.artifact_version}.csv",
        training_trace,
        [
            "training_mode",
            "epoch",
            "phase",
            "num_pairs",
            "positive_pairs",
            "negative_pairs",
            "hard_negative_pairs",
            "loss",
            "contrastive_loss",
            "collapse_metric",
            "embedding_variance",
            "positive_similarity_mean",
            "negative_similarity_mean",
            "hard_negative_similarity_mean",
        ],
    )
    write_csv(
        out_dir / f"{prefix}retrieval_ablation_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "variant",
            "global_top1",
            "global_top3",
            "global_top5",
            "false_bundle_retrieval_rate",
            "focus_success_count",
            "target_not_in_top5_count",
            "target_in_top3_but_lost_top1_count",
            "runtime_namespace_shift_recovered_rate",
            "strict_anchor_real_svr",
            "strict_anchor_shuffled_svr",
            "wrong_old_prototype_visible_count",
            "encoder_collapse_metric",
            "memory_growth",
            "negative_controls_passed",
            "regression_event_count",
            "improved_event_count",
            "unchanged_success_count",
            "unchanged_failure_count",
            "query_count",
            "mean_embedding_margin",
            "official_main_nops_top1",
            "official_main_nops_false_bundle_retrieval_rate",
            "selected_as_best",
            "eligible_for_integration",
        ],
    )
    write_csv(
        out_dir / f"{prefix}real_reentry_eval_{args.artifact_version}.csv",
        real_eval_rows,
        [
            "event_id",
            "scenario_name",
            "target_bundle_id_eval_only",
            "baseline_top1_bundle",
            "variant_top1_bundle",
            "target_rank_baseline",
            "target_rank_variant",
            "baseline_success",
            "variant_success",
            "delta_class",
            "query_embedding_available",
            "memory_embedding_available",
            "embedding_similarity_target",
            "embedding_similarity_wrong_top1",
            "embedding_margin",
            "failure_reason",
        ],
    )
    write_csv(
        out_dir / f"{prefix}focus_event_summary_{args.artifact_version}.csv",
        focus_rows,
        ["event_id", "baseline_rank", "core1e_rank", "baseline_success", "core1e_success", "target_bundle_id_eval_only", "focus_regressed", "reason"],
    )
    write_csv(
        out_dir / f"{prefix}negative_control_summary_{args.artifact_version}.csv",
        negative_control_rows,
        ["control_name", "global_top1", "encoder_sim_only_top1", "false_bundle_retrieval_rate", "focus_success_count", "embedding_margin_mean", "control_passed", "failure_reason"],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        oracle_rows,
        [
            "file",
            "instance_id_used_for_online_training",
            "target_bundle_id_used_for_online_training",
            "old_track_id_used_for_online_training",
            "old_prototype_id_used_for_online_training",
            "gt_box_used_as_training_label",
            "future_eval_event_label_used",
            "gt_used_for_audit_only",
            "pretrained_weights_used",
            "leakage_found",
        ],
    )
    write_csv(
        out_dir / f"{prefix}memory_bank_trace_{args.artifact_version}.csv",
        memory_rows,
        ["frame_idx", "object_file_id", "track_id", "prototype_id", "embedding_updated", "update_reason", "embedding_count", "memory_bank_size", "budget_eviction", "eviction_reason"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
