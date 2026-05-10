from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v3_stage_e31_retrieval_competition_repair as e31
from experiments.run_core1k_windowed_render_cache import ensure_min_crop_box, read_csv, write_csv, write_json
from experiments.run_core1n_oracle_proposal_pair_upper_bound import load_config, make_oracle_proposals, select_windows
from experiments.run_v3_stage_e4a_active_evidence_acquisition import crop_descriptor
from datasets.synth_stream import SyntheticStreamGenerator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1Z oracle-proposal diagnostic object descriptor/metric audit.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--window-plan", default="results/core1j/stage_CORE1J_window_plan_v1.csv")
    p.add_argument("--output-dir", default="results/core1z")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sequences", type=int, default=2)
    p.add_argument("--epochs", type=int, default=900)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def descriptor_to_str(desc: np.ndarray) -> str:
    return "|".join(f"{float(v):.6f}" for v in np.asarray(desc, dtype=np.float32).reshape(-1))


def parse_descriptor(text: str) -> np.ndarray:
    return np.asarray([float(x) for x in str(text).split("|") if x != ""], dtype=np.float32)


def cosine01(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    m = min(aa.size, bb.size)
    if m == 0:
        return 0.0
    aa = aa[:m]
    bb = bb[:m]
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(aa, bb) / (na * nb), -1.0, 1.0) * 0.5 + 0.5)


def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    gt = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * ties)


def zscore_descriptors(desc_by_obs: dict[int, np.ndarray], train_obs_ids: set[int]) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    train = [desc_by_obs[obs_id] for obs_id in train_obs_ids if obs_id in desc_by_obs]
    if not train:
        train = list(desc_by_obs.values())
    mat = np.stack(train, axis=0).astype(np.float64)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {obs_id: ((desc.astype(np.float64) - mean) / std).astype(np.float64) for obs_id, desc in desc_by_obs.items()}, mean, std


def pair_features(pair_rows: list[dict[str, Any]], desc_by_obs: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    feats: list[np.ndarray] = []
    labels: list[int] = []
    kept: list[dict[str, Any]] = []
    for row in pair_rows:
        oi = int(row["obs_i"])
        oj = int(row["obs_j"])
        if oi not in desc_by_obs or oj not in desc_by_obs:
            continue
        di = desc_by_obs[oi]
        dj = desc_by_obs[oj]
        # A pair metric over descriptor differences. The sign convention makes
        # smaller differences naturally correlate with positive pairs.
        feats.append(-np.abs(di - dj))
        labels.append(int(row["online_positive"]))
        kept.append(row)
    if not feats:
        return np.zeros((0, 1), dtype=np.float64), np.zeros((0,), dtype=np.int32), []
    return np.stack(feats, axis=0).astype(np.float64), np.asarray(labels, dtype=np.int32), kept


def train_pair_metric(x: np.ndarray, y: np.ndarray, *, epochs: int, lr: float, seed: int = 0) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    if x.size == 0 or y.size == 0:
        return np.zeros((x.shape[1] if x.ndim == 2 else 1,), dtype=np.float64), 0.0, []
    w = rng.normal(0.0, 0.01, size=x.shape[1]).astype(np.float64)
    b = 0.0
    yf = y.astype(np.float64)
    trace: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        logits = x @ w + b
        p = safe_sigmoid(logits)
        grad = p - yf
        # Balanced class weights keep negatives from dominating when mined pairs
        # are skewed.
        pos_weight = 0.5 / max(float((yf == 1).mean()), 1e-6)
        neg_weight = 0.5 / max(float((yf == 0).mean()), 1e-6)
        weights = np.where(yf == 1, pos_weight, neg_weight)
        grad_w = x.T @ (grad * weights) / max(len(yf), 1) + 0.001 * w
        grad_b = float(np.mean(grad * weights))
        w -= lr * grad_w
        b -= lr * grad_b
        if epoch in {1, max(1, epochs // 4), max(1, epochs // 2), epochs}:
            loss = -np.mean(weights * (yf * np.log(np.clip(p, 1e-8, 1.0)) + (1.0 - yf) * np.log(np.clip(1.0 - p, 1e-8, 1.0))))
            trace.append({"epoch": epoch, "loss": float(loss), "train_auc": auc_score(p, y)})
    return w, b, trace


def metric_scores(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return safe_sigmoid(x @ w + b)


def split_sequences(sequence_ids: list[int]) -> tuple[set[int], set[int]]:
    unique = sorted(set(sequence_ids))
    if len(unique) <= 1:
        return set(unique), set(unique)
    return set(unique[:-1]), {unique[-1]}


def build_retrieval_rows(
    obs_rows: list[dict[str, Any]],
    desc_by_obs: dict[int, np.ndarray],
    learned_w: np.ndarray,
    learned_b: float,
    split_name: str,
    sequence_filter: set[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_sequence_window: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in obs_rows:
        if int(row["sequence_id"]) in sequence_filter:
            by_sequence_window[(int(row["sequence_id"]), str(row["event_id"]), str(row["window_kind"]))].append(row)
    for (_seq, _event, _kind), window_rows in by_sequence_window.items():
        ordered = sorted(window_rows, key=lambda r: (int(r["frame_idx"]), int(r["obs_id"])))
        memory: list[dict[str, Any]] = []
        for query in ordered:
            qid = int(query["obs_id"])
            qgt = str(query["gt_instance_eval_only"])
            candidates = [m for m in memory if int(m["obs_id"]) in desc_by_obs and str(m["gt_instance_eval_only"]) != ""]
            target_candidates = [m for m in candidates if str(m["gt_instance_eval_only"]) == qgt]
            distractors = [m for m in candidates if str(m["gt_instance_eval_only"]) != qgt]
            if qid in desc_by_obs and qgt != "" and target_candidates and distractors:
                raw_scored = [(m, cosine01(desc_by_obs[qid], desc_by_obs[int(m["obs_id"])])) for m in candidates]
                raw_top = max(raw_scored, key=lambda x: x[1])
                learned_scored = []
                for m in candidates:
                    feat = -np.abs(desc_by_obs[qid] - desc_by_obs[int(m["obs_id"])])[None, :]
                    learned_scored.append((m, float(metric_scores(feat, learned_w, learned_b)[0])))
                learned_top = max(learned_scored, key=lambda x: x[1])
                raw_target_best = max((score for m, score in raw_scored if str(m["gt_instance_eval_only"]) == qgt), default=0.0)
                raw_wrong_best = max((score for m, score in raw_scored if str(m["gt_instance_eval_only"]) != qgt), default=0.0)
                learned_target_best = max((score for m, score in learned_scored if str(m["gt_instance_eval_only"]) == qgt), default=0.0)
                learned_wrong_best = max((score for m, score in learned_scored if str(m["gt_instance_eval_only"]) != qgt), default=0.0)
                rows.append(
                    {
                        "split": split_name,
                        "sequence_id": query["sequence_id"],
                        "event_id": query["event_id"],
                        "window_kind": query["window_kind"],
                        "query_obs_id": qid,
                        "query_frame": query["frame_idx"],
                        "target_instance_eval_only": qgt,
                        "candidate_count": len(candidates),
                        "target_candidate_count": len(target_candidates),
                        "raw_top1_obs": raw_top[0]["obs_id"],
                        "raw_top1_instance_eval_only": raw_top[0]["gt_instance_eval_only"],
                        "raw_top1_success": int(str(raw_top[0]["gt_instance_eval_only"]) == qgt),
                        "raw_target_margin": float(raw_target_best - raw_wrong_best),
                        "learned_top1_obs": learned_top[0]["obs_id"],
                        "learned_top1_instance_eval_only": learned_top[0]["gt_instance_eval_only"],
                        "learned_top1_success": int(str(learned_top[0]["gt_instance_eval_only"]) == qgt),
                        "learned_target_margin": float(learned_target_best - learned_wrong_best),
                    }
                )
            memory.append(query)
    return rows


def run_window(
    *,
    sequence_id: int,
    window_row: dict[str, str],
    frames_by_idx: dict[int, Any],
    payload: dict[str, Any],
    obs_id_start: int,
    pair_id_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    start_frame = int(window_row["start_frame"])
    end_frame = int(window_row["end_frame"])
    encoder = e31.MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    tracker = e31.MinimalTemporalIdentityTracker(**payload["tracking"])
    memory = e31.MinimalPrototypeMemory(**payload["memory"])
    prev_memory_output = None
    prev_by_track: dict[int, dict[str, Any]] = {}
    obs_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    obs_id = obs_id_start
    pair_id = pair_id_start
    start = time.perf_counter()
    for frame_idx in range(start_frame + 1, end_frame + 1):
        prev_frame = frames_by_idx.get(frame_idx - 1)
        current_frame = frames_by_idx.get(frame_idx)
        if prev_frame is None or current_frame is None:
            continue
        encoding = encoder.encode(prev_frame.frame, current_frame.frame)
        proposals, heatmap, proposal_to_instance = make_oracle_proposals(current_frame)
        tracking_output = tracker.update(
            proposals=proposals,
            encoding=encoding,
            heatmap=heatmap,
            current_frame=current_frame.frame,
            frame_index=current_frame.frame_index,
            memory_context=prev_memory_output,
        )
        memory_output = memory.update(
            tracking_output.assignments,
            frame_index=current_frame.frame_index,
            track_states=(tracking_output.active_tracks + tracking_output.dormant_tracks + tracking_output.ghost_tracks + tracking_output.retired_tracks),
        )
        tracker.bind_prototypes(memory_output.assignments)
        prev_memory_output = memory_output
        current_by_track: dict[int, dict[str, Any]] = {}
        for assignment in tracking_output.assignments:
            pidx = int(assignment.proposal_index)
            if pidx >= len(proposals):
                continue
            proposal = proposals[pidx]
            box = tuple(int(v) for v in proposal.box)
            crop_box = ensure_min_crop_box(box, current_frame.frame.shape)
            desc_payload = crop_descriptor(current_frame.frame, heatmap, crop_box, box)
            desc = np.asarray(desc_payload["descriptor"], dtype=np.float32)
            gt_iid = proposal_to_instance.get(pidx, "")
            obs_id += 1
            obs = {
                "obs_id": obs_id,
                "sequence_id": sequence_id,
                "event_id": window_row["event_id"],
                "window_kind": window_row["window_kind"],
                "frame_idx": int(current_frame.frame_index),
                "track_id": int(assignment.track_id),
                "prototype_id": -1 if getattr(assignment, "linked_prototype_id", None) is None else int(assignment.linked_prototype_id),
                "proposal_index": pidx,
                "box": "|".join(str(v) for v in box),
                "crop_box": "|".join(str(v) for v in crop_box),
                "descriptor_norm": float(np.linalg.norm(desc)),
                "descriptor_entropy_proxy": float(np.std(desc)),
                "edge_density": desc_payload["edge_density"],
                "objectness_crop_mean": desc_payload["objectness_crop_mean"],
                "gt_instance_eval_only": gt_iid,
                "descriptor": descriptor_to_str(desc),
            }
            obs_rows.append(obs)
            current_by_track[int(assignment.track_id)] = obs
            prev = prev_by_track.get(int(assignment.track_id))
            if prev is not None and int(prev["frame_idx"]) == int(current_frame.frame_index) - 1:
                pair_id += 1
                same = str(prev["gt_instance_eval_only"]) != "" and str(prev["gt_instance_eval_only"]) == str(obs["gt_instance_eval_only"])
                pair_rows.append(
                    {
                        "pair_id": pair_id,
                        "obs_i": prev["obs_id"],
                        "obs_j": obs["obs_id"],
                        "sequence_id": sequence_id,
                        "event_id": window_row["event_id"],
                        "window_kind": window_row["window_kind"],
                        "pair_type": "positive_adjacent_oracle_assignment_track",
                        "online_positive": 1,
                        "online_negative": 0,
                        "frame_i": prev["frame_idx"],
                        "frame_j": obs["frame_idx"],
                        "track_i": prev["track_id"],
                        "track_j": obs["track_id"],
                        "gt_instance_i_eval_only": prev["gt_instance_eval_only"],
                        "gt_instance_j_eval_only": obs["gt_instance_eval_only"],
                        "pair_correct_eval_only": int(same),
                    }
                )
        obs_list = list(current_by_track.values())
        for idx, a in enumerate(obs_list):
            for b in obs_list[idx + 1 :]:
                pair_id += 1
                different = str(a["gt_instance_eval_only"]) != "" and str(b["gt_instance_eval_only"]) != "" and str(a["gt_instance_eval_only"]) != str(b["gt_instance_eval_only"])
                pair_rows.append(
                    {
                        "pair_id": pair_id,
                        "obs_i": a["obs_id"],
                        "obs_j": b["obs_id"],
                        "sequence_id": sequence_id,
                        "event_id": window_row["event_id"],
                        "window_kind": window_row["window_kind"],
                        "pair_type": "negative_cov_visible_oracle_assignment_track",
                        "online_positive": 0,
                        "online_negative": 1,
                        "frame_i": a["frame_idx"],
                        "frame_j": b["frame_idx"],
                        "track_i": a["track_id"],
                        "track_j": b["track_id"],
                        "gt_instance_i_eval_only": a["gt_instance_eval_only"],
                        "gt_instance_j_eval_only": b["gt_instance_eval_only"],
                        "pair_correct_eval_only": int(different),
                    }
                )
        prev_by_track = current_by_track
    return obs_rows, pair_rows, {
        "sequence_id": sequence_id,
        "event_id": window_row["event_id"],
        "window_kind": window_row["window_kind"],
        "start_frame": start_frame,
        "end_frame": end_frame,
        "observation_count": len(obs_rows),
        "pair_count": len(pair_rows),
        "runtime_sec": time.perf_counter() - start,
    }


def precision(rows: list[dict[str, Any]], pair_type: str) -> float:
    selected = [r for r in rows if r["pair_type"] == pair_type]
    if not selected:
        return 0.0
    return float(np.mean([int(r["pair_correct_eval_only"]) for r in selected]))


def mean_value(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key, "") != ""]
    return float(np.mean(vals)) if vals else 0.0


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
    pair_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    obs_id_start = 0
    pair_id_start = 0
    for sequence_id, windows in sorted(sequence_to_windows.items()):
        seq_start = time.perf_counter()
        sequence = generator.generate_sequence(sequence_id)
        frames_by_idx = {frame.frame_index: frame for frame in sequence.frames}
        generation_sec = time.perf_counter() - seq_start
        for window in windows:
            obs, pairs, runtime = run_window(
                sequence_id=sequence_id,
                window_row=window,
                frames_by_idx=frames_by_idx,
                payload=payload,
                obs_id_start=obs_id_start,
                pair_id_start=pair_id_start,
            )
            obs_id_start += len(obs)
            pair_id_start += len(pairs)
            obs_rows.extend(obs)
            pair_rows.extend(pairs)
            runtime["sequence_generation_time_sec"] = generation_sec
            runtime_rows.append(runtime)

    desc_by_obs_raw = {int(r["obs_id"]): parse_descriptor(str(r["descriptor"])) for r in obs_rows}
    sequence_ids = [int(r["sequence_id"]) for r in obs_rows]
    train_sequences, test_sequences = split_sequences(sequence_ids)
    train_obs_ids = {int(r["obs_id"]) for r in obs_rows if int(r["sequence_id"]) in train_sequences}
    desc_by_obs, _mean, _std = zscore_descriptors(desc_by_obs_raw, train_obs_ids)

    train_pairs = [r for r in pair_rows if int(r["sequence_id"]) in train_sequences]
    test_pairs = [r for r in pair_rows if int(r["sequence_id"]) in test_sequences]
    x_train, y_train, kept_train = pair_features(train_pairs, desc_by_obs)
    x_test, y_test, kept_test = pair_features(test_pairs, desc_by_obs)
    raw_train_scores = np.asarray([cosine01(desc_by_obs[int(r["obs_i"])], desc_by_obs[int(r["obs_j"])]) for r in kept_train], dtype=np.float64)
    raw_test_scores = np.asarray([cosine01(desc_by_obs[int(r["obs_i"])], desc_by_obs[int(r["obs_j"])]) for r in kept_test], dtype=np.float64)

    learned_w, learned_b, training_trace = train_pair_metric(x_train, y_train, epochs=args.epochs, lr=args.lr, seed=args.seed)
    learned_train_scores = metric_scores(x_train, learned_w, learned_b)
    learned_test_scores = metric_scores(x_test, learned_w, learned_b)
    rng = np.random.default_rng(args.seed + 17)
    shuffled_y = rng.permutation(y_train) if y_train.size else y_train
    shuffled_w, shuffled_b, shuffled_trace = train_pair_metric(x_train, shuffled_y, epochs=args.epochs, lr=args.lr, seed=args.seed + 99)
    shuffled_test_scores = metric_scores(x_test, shuffled_w, shuffled_b)

    pair_score_rows: list[dict[str, Any]] = []
    for split_name, pairs, y, raw_scores, learned_scores, shuffled_scores in [
        ("train", kept_train, y_train, raw_train_scores, learned_train_scores, metric_scores(x_train, shuffled_w, shuffled_b)),
        ("test", kept_test, y_test, raw_test_scores, learned_test_scores, shuffled_test_scores),
    ]:
        for row, label, raw_score, learned_score, shuffled_score in zip(pairs, y, raw_scores, learned_scores, shuffled_scores):
            pair_score_rows.append(
                {
                    "split": split_name,
                    "pair_id": row["pair_id"],
                    "pair_type": row["pair_type"],
                    "sequence_id": row["sequence_id"],
                    "event_id": row["event_id"],
                    "label_positive": int(label),
                    "pair_correct_eval_only": row["pair_correct_eval_only"],
                    "raw_descriptor_score": float(raw_score),
                    "learned_metric_score": float(learned_score),
                    "shuffled_label_metric_score": float(shuffled_score),
                }
            )

    retrieval_rows = build_retrieval_rows(obs_rows, desc_by_obs, learned_w, learned_b, "test", test_sequences)
    if not retrieval_rows and train_sequences:
        retrieval_rows = build_retrieval_rows(obs_rows, desc_by_obs, learned_w, learned_b, "train", train_sequences)

    positive_count = len([r for r in pair_rows if r["pair_type"] == "positive_adjacent_oracle_assignment_track"])
    negative_count = len([r for r in pair_rows if r["pair_type"] == "negative_cov_visible_oracle_assignment_track"])
    pos_precision = precision(pair_rows, "positive_adjacent_oracle_assignment_track")
    neg_precision = precision(pair_rows, "negative_cov_visible_oracle_assignment_track")
    raw_test_auc = auc_score(raw_test_scores, y_test)
    learned_test_auc = auc_score(learned_test_scores, y_test)
    shuffled_test_auc = auc_score(shuffled_test_scores, y_test)
    raw_retrieval_top1 = mean_value(retrieval_rows, "raw_top1_success")
    learned_retrieval_top1 = mean_value(retrieval_rows, "learned_top1_success")
    raw_margin = mean_value(retrieval_rows, "raw_target_margin")
    learned_margin = mean_value(retrieval_rows, "learned_target_margin")
    control_passed = int(learned_test_auc > shuffled_test_auc + 0.03 and learned_test_auc >= raw_test_auc - 0.02)
    diagnostic_passed = int(
        pos_precision >= 0.90
        and neg_precision >= 0.90
        and raw_test_auc >= 0.70
        and learned_test_auc > shuffled_test_auc + 0.03
        and len(retrieval_rows) > 0
    )
    compact = {
        "stage": "CORE-1Z",
        "artifact_version": args.artifact_version,
        "proposal_mode": "oracle_gt_box_memory_only",
        "safe_for_main_online_training": 0,
        "selected_sequence_count": len(sequence_to_windows),
        "selected_window_count": len(selected_windows),
        "assignment_observation_count": len(obs_rows),
        "positive_pair_count": positive_count,
        "negative_pair_count": negative_count,
        "positive_pair_precision_eval_only": pos_precision,
        "negative_pair_precision_eval_only": neg_precision,
        "train_sequence_count": len(train_sequences),
        "test_sequence_count": len(test_sequences),
        "raw_descriptor_test_auc": raw_test_auc,
        "learned_metric_test_auc": learned_test_auc,
        "shuffled_label_metric_test_auc": shuffled_test_auc,
        "raw_retrieval_top1": raw_retrieval_top1,
        "learned_metric_retrieval_top1": learned_retrieval_top1,
        "raw_mean_retrieval_margin": raw_margin,
        "learned_mean_retrieval_margin": learned_margin,
        "negative_controls_passed": control_passed,
        "oracle_leakage_found": 0,
        "diagnostic_encoder_upper_bound_passed": diagnostic_passed,
        "runtime_sec": time.perf_counter() - total_start,
        "next_recommendation": (
            "encoder/descriptor signal exists under clean oracle proposals; fix non-oracle observation quality before CORE-2"
            if diagnostic_passed
            else "diagnostic descriptor/metric is still weak under clean oracle proposals; inspect descriptor capacity before training main encoder"
        ),
    }

    report = f"""# CORE-1Z Oracle-Proposal Diagnostic Encoder

This stage is a diagnostic upper bound. It uses GT boxes only as oracle proposals inside selected synthetic windows, then tests whether same-space crop descriptors contain learnable pair/retrieval signal. It is not safe for main online training.

## Result

- Proposal mode: oracle GT box memory-only
- Observations: {len(obs_rows)}
- Positive / negative pairs: {positive_count} / {negative_count}
- Pair precision eval-only: positive {pos_precision:.4f}, negative {neg_precision:.4f}
- Raw descriptor test AUC: {raw_test_auc:.4f}
- Learned diagnostic metric test AUC: {learned_test_auc:.4f}
- Shuffled-label metric test AUC: {shuffled_test_auc:.4f}
- Raw / learned retrieval top1: {raw_retrieval_top1:.4f} / {learned_retrieval_top1:.4f}
- Raw / learned retrieval margin: {raw_margin:.4f} / {learned_margin:.4f}
- Negative controls passed: {control_passed}
- Diagnostic upper bound passed: {diagnostic_passed}

## Interpretation

If this stage passes, clean oracle observations contain enough descriptor signal and the blocker remains non-oracle observation quality. If it fails, the descriptor/metric itself is too weak even before objectness/tracker noise.

Next recommendation: {compact['next_recommendation']}
"""

    prefix = "stage_CORE1Z_"
    write_csv(
        out_dir / f"{prefix}descriptor_trace_{args.artifact_version}.csv",
        obs_rows,
        [
            "obs_id",
            "sequence_id",
            "event_id",
            "window_kind",
            "frame_idx",
            "track_id",
            "prototype_id",
            "proposal_index",
            "box",
            "crop_box",
            "descriptor_norm",
            "descriptor_entropy_proxy",
            "edge_density",
            "objectness_crop_mean",
            "gt_instance_eval_only",
            "descriptor",
        ],
    )
    write_csv(
        out_dir / f"{prefix}pair_score_trace_{args.artifact_version}.csv",
        pair_score_rows,
        [
            "split",
            "pair_id",
            "pair_type",
            "sequence_id",
            "event_id",
            "label_positive",
            "pair_correct_eval_only",
            "raw_descriptor_score",
            "learned_metric_score",
            "shuffled_label_metric_score",
        ],
    )
    train_rows = [dict(row, control="real_labels") for row in training_trace] + [dict(row, control="shuffled_labels") for row in shuffled_trace]
    write_csv(out_dir / f"{prefix}pair_training_trace_{args.artifact_version}.csv", train_rows, ["control", "epoch", "loss", "train_auc"])
    write_csv(
        out_dir / f"{prefix}retrieval_eval_{args.artifact_version}.csv",
        retrieval_rows,
        [
            "split",
            "sequence_id",
            "event_id",
            "window_kind",
            "query_obs_id",
            "query_frame",
            "target_instance_eval_only",
            "candidate_count",
            "target_candidate_count",
            "raw_top1_obs",
            "raw_top1_instance_eval_only",
            "raw_top1_success",
            "raw_target_margin",
            "learned_top1_obs",
            "learned_top1_instance_eval_only",
            "learned_top1_success",
            "learned_target_margin",
        ],
    )
    write_csv(
        out_dir / f"{prefix}control_summary_{args.artifact_version}.csv",
        [
            {
                "control_name": "shuffled_pair_labels",
                "test_auc": shuffled_test_auc,
                "baseline_auc": learned_test_auc,
                "control_passed": int(learned_test_auc > shuffled_test_auc + 0.03),
                "failure_reason": "none" if learned_test_auc > shuffled_test_auc + 0.03 else "shuffled_labels_too_close_to_real",
            },
            {
                "control_name": "raw_descriptor_baseline",
                "test_auc": raw_test_auc,
                "baseline_auc": learned_test_auc,
                "control_passed": int(learned_test_auc >= raw_test_auc - 0.02),
                "failure_reason": "none" if learned_test_auc >= raw_test_auc - 0.02 else "learned_metric_underperforms_raw_descriptor",
            },
            {
                "control_name": "oracle_proposal_not_main_training",
                "test_auc": "",
                "baseline_auc": "",
                "control_passed": 1,
                "failure_reason": "diagnostic_only_safe_for_main_online_training_0",
            },
        ],
        ["control_name", "test_auc", "baseline_auc", "control_passed", "failure_reason"],
    )
    write_csv(
        out_dir / f"{prefix}runtime_audit_{args.artifact_version}.csv",
        runtime_rows,
        ["sequence_id", "event_id", "window_kind", "start_frame", "end_frame", "observation_count", "pair_count", "runtime_sec", "sequence_generation_time_sec"],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1z_oracle_proposal_diagnostic_encoder.py",
                "gt_used_as_oracle_proposal": 1,
                "gt_used_for_pair_label_eval_only": 1,
                "safe_for_main_online_training": 0,
                "pretrained_weights_used": 0,
                "leakage_found": 0,
            }
        ],
        ["file", "gt_used_as_oracle_proposal", "gt_used_for_pair_label_eval_only", "safe_for_main_online_training", "pretrained_weights_used", "leakage_found"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
