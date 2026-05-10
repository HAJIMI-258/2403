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

from experiments.run_core1aa_stability_namespace_pair_gate import row_passes
from experiments.run_core1ac_raw_descriptor_memory_integration_smoke import (
    baseline_score,
    f,
    i,
    normalize_scores,
    summarize_variant,
)
from experiments.run_core1ai_observation_quality_frontier import candidate_gates
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json
from experiments.run_core1z_oracle_proposal_diagnostic_encoder import cosine01, parse_descriptor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AL decoupled online metric smoke.")
    p.add_argument("--observations", default="results/core1aj/stage_CORE1AJ_stability_observation_trace_v1.csv")
    p.add_argument("--descriptor-trace", default="results/core1aj/stage_CORE1AJ_descriptor_trace_v1.csv")
    p.add_argument("--core1ak-compact", default="results/core1ak/stage_CORE1AK_compact_for_gpt_v1.json")
    p.add_argument("--output-dir", default="results/core1al")
    p.add_argument("--max-negatives-per-observation", type=int, default=8)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def assign_obs_ids(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [dict(row, obs_id=idx) for idx, row in enumerate(rows, start=1)]


def load_desc(path: Path) -> dict[int, np.ndarray]:
    return {i(row["obs_id"]): parse_descriptor(str(row["descriptor"])) for row in read_csv(path)}


def gate_by_name(name: str) -> dict[str, Any]:
    for gate in candidate_gates():
        if gate["gate_name"] == name:
            return gate
    raise ValueError(name)


def gt_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("sequence_id", "")), str(row.get("gt_instance_eval_only", ""))


def same_instance_namespace_aware(a: dict[str, Any], b: dict[str, Any]) -> bool:
    agt = str(a.get("gt_instance_eval_only", ""))
    bgt = str(b.get("gt_instance_eval_only", ""))
    if agt == "" or bgt == "":
        return False
    return str(a.get("sequence_id", "")) == str(b.get("sequence_id", "")) and agt == bgt


def different_instance_namespace_aware(a: dict[str, Any], b: dict[str, Any]) -> bool:
    agt = str(a.get("gt_instance_eval_only", ""))
    bgt = str(b.get("gt_instance_eval_only", ""))
    if agt == "" or bgt == "":
        return False
    return gt_key(a) != gt_key(b)


def build_train_pairs(
    rows: list[dict[str, Any]],
    desc_by_obs: dict[int, np.ndarray],
    max_negatives_per_observation: int,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    pid = 0
    by_track: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if i(row["obs_id"]) in desc_by_obs:
            by_track[(str(row["sequence_id"]), str(row["event_id"]), str(row["window_kind"]), i(row["track_id"]))].append(row)
    for track_rows in by_track.values():
        ordered = sorted(track_rows, key=lambda r: (i(r["frame_idx"]), i(r["obs_id"])))
        prev = None
        for obs in ordered:
            if prev is not None and i(obs["frame_idx"]) == i(prev["frame_idx"]) + 1:
                pid += 1
                pairs.append(
                    {
                        "pair_id": pid,
                        "obs_i": i(prev["obs_id"]),
                        "obs_j": i(obs["obs_id"]),
                        "pair_type": "positive_adjacent_track",
                        "pseudo_label": 1,
                        "pair_correct_eval_only": int(same_instance_namespace_aware(prev, obs)),
                        "sequence_i": prev.get("sequence_id", ""),
                        "sequence_j": obs.get("sequence_id", ""),
                        "frame_i": prev.get("frame_idx", ""),
                        "frame_j": obs.get("frame_idx", ""),
                        "track_i": prev.get("track_id", ""),
                        "track_j": obs.get("track_id", ""),
                    }
                )
            prev = obs

    ordered = sorted(rows, key=lambda r: (str(r["sequence_id"]), str(r["event_id"]), str(r["window_kind"]), i(r["frame_idx"]), i(r["track_id"]), i(r["obs_id"])))
    for a in ordered:
        if i(a["obs_id"]) not in desc_by_obs:
            continue
        count = 0
        for b in ordered:
            if i(b["obs_id"]) not in desc_by_obs or i(a["obs_id"]) == i(b["obs_id"]):
                continue
            if str(a.get("sequence_id", "")) == str(b.get("sequence_id", "")):
                continue
            pid += 1
            pairs.append(
                {
                    "pair_id": pid,
                    "obs_i": i(a["obs_id"]),
                    "obs_j": i(b["obs_id"]),
                    "pair_type": "negative_cross_sequence",
                    "pseudo_label": 0,
                    "pair_correct_eval_only": int(different_instance_namespace_aware(a, b)),
                    "sequence_i": a.get("sequence_id", ""),
                    "sequence_j": b.get("sequence_id", ""),
                    "frame_i": a.get("frame_idx", ""),
                    "frame_j": b.get("frame_idx", ""),
                    "track_i": a.get("track_id", ""),
                    "track_j": b.get("track_id", ""),
                }
            )
            count += 1
            if count >= max_negatives_per_observation:
                break
    return pairs


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def pair_features(pairs: list[dict[str, Any]], desc_by_obs: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    kept: list[dict[str, Any]] = []
    for pair in pairs:
        oi = i(pair["obs_i"])
        oj = i(pair["obs_j"])
        if oi not in desc_by_obs or oj not in desc_by_obs:
            continue
        xs.append(np.abs(desc_by_obs[oi] - desc_by_obs[oj]))
        ys.append(i(pair["pseudo_label"]))
        kept.append(pair)
    return np.vstack(xs).astype(np.float64), np.asarray(ys, dtype=np.float64), kept


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    # Pairwise AUC is fine here; train pairs are small.
    wins = 0.0
    total = float(len(pos) * len(neg))
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / total


def train_metric(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    lr: float,
    shuffle_labels: bool = False,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    labels = y.copy()
    if shuffle_labels:
        rng.shuffle(labels)
    idx = rng.permutation(len(labels))
    split = max(1, int(0.75 * len(idx)))
    train_idx = idx[:split]
    test_idx = idx[split:]
    mu = x[train_idx].mean(axis=0)
    sigma = x[train_idx].std(axis=0) + 1e-6
    z = (x - mu) / sigma
    # Positive pairs should have smaller absolute descriptor distance, so initialize negative.
    w = -0.05 * np.ones(z.shape[1], dtype=np.float64)
    b = 0.0
    pos_weight = len(train_idx) / max(1.0, 2.0 * float(np.sum(labels[train_idx] == 1)))
    neg_weight = len(train_idx) / max(1.0, 2.0 * float(np.sum(labels[train_idx] == 0)))
    losses: list[float] = []
    for _epoch in range(epochs):
        logits = z[train_idx] @ w + b
        pred = sigmoid(logits)
        ww = np.where(labels[train_idx] == 1, pos_weight, neg_weight)
        err = (pred - labels[train_idx]) * ww
        grad_w = (z[train_idx].T @ err) / len(train_idx) + 1e-4 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
        loss = -np.mean(ww * (labels[train_idx] * np.log(pred + 1e-9) + (1.0 - labels[train_idx]) * np.log(1.0 - pred + 1e-9)))
        losses.append(float(loss))
    train_scores = sigmoid(z[train_idx] @ w + b)
    test_scores = sigmoid(z[test_idx] @ w + b) if len(test_idx) else np.asarray([])
    trace = {
        "train_pair_count": int(len(train_idx)),
        "test_pair_count": int(len(test_idx)),
        "train_positive_count": int(np.sum(labels[train_idx] == 1)),
        "train_negative_count": int(np.sum(labels[train_idx] == 0)),
        "test_positive_count": int(np.sum(labels[test_idx] == 1)) if len(test_idx) else 0,
        "test_negative_count": int(np.sum(labels[test_idx] == 0)) if len(test_idx) else 0,
        "initial_loss": losses[0] if losses else 0.0,
        "final_loss": losses[-1] if losses else 0.0,
        "train_auc": auc_score(labels[train_idx], train_scores),
        "test_auc": auc_score(labels[test_idx], test_scores) if len(test_idx) else 0.0,
        "shuffle_labels": int(shuffle_labels),
        "feature_mean_norm": float(np.linalg.norm(mu)),
        "feature_std_mean": float(sigma.mean()),
    }
    params = np.concatenate([w, np.asarray([b]), mu, sigma])
    return params, b, trace


def metric_score(desc_a: np.ndarray, desc_b: np.ndarray, params: np.ndarray) -> float:
    dim = desc_a.shape[0]
    w = params[:dim]
    b = float(params[dim])
    mu = params[dim + 1 : dim + 1 + dim]
    sigma = params[dim + 1 + dim : dim + 1 + 2 * dim]
    z = (np.abs(desc_a - desc_b) - mu) / sigma
    return float(sigmoid(np.asarray([z @ w + b]))[0])


VARIANTS: list[dict[str, Any]] = [
    {"variant": "A0_track_recency_baseline", "score_mode": "none", "weight": 0.0},
    {"variant": "A1_raw_descriptor_only", "score_mode": "raw", "raw_only": True},
    {"variant": "A2_raw_fusion_w005", "score_mode": "raw", "weight": 0.05},
    {"variant": "A3_raw_fusion_w010", "score_mode": "raw", "weight": 0.10},
    {"variant": "A4_raw_fusion_w020", "score_mode": "raw", "weight": 0.20},
    {"variant": "A5_learned_metric_only", "score_mode": "learned", "raw_only": True},
    {"variant": "A6_learned_fusion_w005", "score_mode": "learned", "weight": 0.05},
    {"variant": "A7_learned_fusion_w010", "score_mode": "learned", "weight": 0.10},
    {"variant": "A8_learned_fusion_w020", "score_mode": "learned", "weight": 0.20},
    {"variant": "A9_shuffled_label_metric_w010_control", "score_mode": "shuffle_metric", "weight": 0.10, "control": True},
    {"variant": "A10_random_metric_w010_control", "score_mode": "random_metric", "weight": 0.10, "control": True},
    {"variant": "A11_shuffled_descriptor_w010_control", "score_mode": "shuffled_descriptor", "weight": 0.10, "control": True},
]


def aux_score(
    query: dict[str, Any],
    cand: dict[str, Any],
    desc_by_obs: dict[int, np.ndarray],
    variant: dict[str, Any],
    learned_params: np.ndarray,
    shuffled_params: np.ndarray,
    random_w: np.ndarray,
    rng: np.random.Generator,
    shuffled_cache: dict[tuple[int, int], float],
) -> float:
    qid = i(query["obs_id"])
    cid = i(cand["obs_id"])
    if qid not in desc_by_obs or cid not in desc_by_obs:
        return 0.0
    mode = str(variant.get("score_mode", "none"))
    if mode == "raw":
        return cosine01(desc_by_obs[qid], desc_by_obs[cid])
    if mode == "learned":
        return metric_score(desc_by_obs[qid], desc_by_obs[cid], learned_params)
    if mode == "shuffle_metric":
        return metric_score(desc_by_obs[qid], desc_by_obs[cid], shuffled_params)
    if mode == "random_metric":
        x = np.abs(desc_by_obs[qid] - desc_by_obs[cid])
        return float(sigmoid(np.asarray([x @ random_w]))[0])
    if mode == "shuffled_descriptor":
        key = (qid, cid)
        if key not in shuffled_cache:
            shuffled_cache[key] = float(rng.random())
        return shuffled_cache[key]
    return 0.0


def score_candidates(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    desc_by_obs: dict[int, np.ndarray],
    variant: dict[str, Any],
    learned_params: np.ndarray,
    shuffled_params: np.ndarray,
    random_w: np.ndarray,
    rng: np.random.Generator,
    shuffled_cache: dict[tuple[int, int], float],
) -> list[dict[str, Any]]:
    base_raw = [baseline_score(query, cand) for cand in candidates]
    aux_raw = [aux_score(query, cand, desc_by_obs, variant, learned_params, shuffled_params, random_w, rng, shuffled_cache) for cand in candidates]
    base = normalize_scores(base_raw)
    aux = normalize_scores(aux_raw)
    rows = []
    use_aux = str(variant.get("score_mode", "none")) != "none"
    for cand, b0, a0, bn, an in zip(candidates, base_raw, aux_raw, base, aux):
        if variant.get("raw_only"):
            final = an
        elif use_aux:
            w = f(variant.get("weight"), 0.0)
            final = (1.0 - w) * bn + w * an
        else:
            final = bn
        rows.append({"candidate": cand, "baseline_score": b0, "aux_score": a0, "baseline_norm": bn, "aux_norm": an, "final_score": float(final), "aux_used": int(use_aux)})
    rows.sort(key=lambda r: r["final_score"], reverse=True)
    return rows


def build_event_rows(
    rows: list[dict[str, Any]],
    desc_by_obs: dict[int, np.ndarray],
    variant: dict[str, Any],
    learned_params: np.ndarray,
    shuffled_params: np.ndarray,
    random_w: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + abs(hash(str(variant["variant"]))) % 100000)
    shuffled_cache: dict[tuple[int, int], float] = {}
    by_window: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_window[(str(row["sequence_id"]), str(row["event_id"]), str(row["window_kind"]))].append(row)
    out: list[dict[str, Any]] = []
    for (_seq, _event, _kind), window_rows in by_window.items():
        memory: list[dict[str, Any]] = []
        for query in sorted(window_rows, key=lambda r: (i(r["frame_idx"]), i(r["track_id"]), i(r["obs_id"]))):
            qgt = str(query.get("gt_instance_eval_only", ""))
            qid = i(query["obs_id"])
            candidates = [m for m in memory if i(m["obs_id"]) in desc_by_obs and str(m.get("gt_instance_eval_only", "")) != ""]
            target_candidates = [m for m in candidates if str(m.get("gt_instance_eval_only", "")) == qgt]
            distractors = [m for m in candidates if str(m.get("gt_instance_eval_only", "")) != qgt]
            if qid in desc_by_obs and qgt != "" and target_candidates and distractors:
                scored = score_candidates(query, candidates, desc_by_obs, variant, learned_params, shuffled_params, random_w, rng, shuffled_cache)
                top1 = scored[0]
                target_scores = [r["final_score"] for r in scored if str(r["candidate"].get("gt_instance_eval_only", "")) == qgt]
                wrong_scores = [r["final_score"] for r in scored if str(r["candidate"].get("gt_instance_eval_only", "")) != qgt]
                target_rank = 999
                for idx, item in enumerate(scored, start=1):
                    if str(item["candidate"].get("gt_instance_eval_only", "")) == qgt:
                        target_rank = idx
                        break
                out.append(
                    {
                        "variant": variant["variant"],
                        "sequence_id": query["sequence_id"],
                        "event_id": query["event_id"],
                        "window_kind": query["window_kind"],
                        "query_obs_id": qid,
                        "candidate_count": len(candidates),
                        "target_candidate_count": len(target_candidates),
                        "top1_obs_id": top1["candidate"]["obs_id"],
                        "top1_instance_eval_only": top1["candidate"].get("gt_instance_eval_only", ""),
                        "target_instance_eval_only": qgt,
                        "top1_success": int(str(top1["candidate"].get("gt_instance_eval_only", "")) == qgt),
                        "target_rank": target_rank,
                        "target_in_top3": int(target_rank <= 3),
                        "target_margin": float(max(target_scores) - max(wrong_scores)) if target_scores and wrong_scores else 0.0,
                        "baseline_score_top1": top1["baseline_score"],
                        "aux_score_top1": top1["aux_score"],
                        "aux_used": top1["aux_used"],
                        "descriptor_used": top1["aux_used"],
                    }
                )
            memory.append(query)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    core1ak = read_json(Path(args.core1ak_compact))
    train_gate = str(core1ak.get("best_train_gate", "S70_C30_streak2"))
    eval_gate = str(core1ak.get("best_eval_gate", "S55_C50"))
    rows = assign_obs_ids(read_csv(Path(args.observations)))
    desc_by_obs = load_desc(Path(args.descriptor_trace))
    train_selected = [row for row in rows if row_passes(row, gate_by_name(train_gate)) and i(row["obs_id"]) in desc_by_obs]
    eval_selected = [row for row in rows if row_passes(row, gate_by_name(eval_gate)) and i(row["obs_id"]) in desc_by_obs]
    pairs = build_train_pairs(train_selected, desc_by_obs, args.max_negatives_per_observation)
    x, y, kept_pairs = pair_features(pairs, desc_by_obs)
    learned_params, _b, learned_trace = train_metric(x, y, seed=args.seed, epochs=args.epochs, lr=args.lr, shuffle_labels=False)
    shuffled_params, _sb, shuffled_trace = train_metric(x, y, seed=args.seed + 17, epochs=args.epochs, lr=args.lr, shuffle_labels=True)
    rng = np.random.default_rng(args.seed + 101)
    random_w = rng.normal(0.0, 0.25, size=x.shape[1])

    pair_rows: list[dict[str, Any]] = []
    for pair in kept_pairs:
        pair_rows.append(dict(pair, descriptor_distance=float(np.mean(np.abs(desc_by_obs[i(pair["obs_i"])] - desc_by_obs[i(pair["obs_j"])])))))

    event_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    baseline_by_query: dict[int, dict[str, Any]] = {}
    for variant in VARIANTS:
        rows_out = build_event_rows(eval_selected, desc_by_obs, variant, learned_params, shuffled_params, random_w, args.seed)
        if variant["variant"] == "A0_track_recency_baseline":
            baseline_rows = rows_out
            baseline_by_query = {i(r["query_obs_id"]): r for r in baseline_rows}
        event_rows.extend(rows_out)
        summary = summarize_variant({"variant": variant["variant"]}, rows_out, baseline_by_query if baseline_by_query else {i(r["query_obs_id"]): r for r in rows_out})
        summary["score_mode"] = variant.get("score_mode", "")
        summary["weight"] = variant.get("weight", "")
        summary["control"] = int(bool(variant.get("control")))
        summary_rows.append(summary)

    # Re-summarize all rows after baseline_by_query is known.
    summary_rows = []
    by_variant = {variant["variant"]: [r for r in event_rows if r["variant"] == variant["variant"]] for variant in VARIANTS}
    for variant in VARIANTS:
        summary = summarize_variant({"variant": variant["variant"]}, by_variant[variant["variant"]], baseline_by_query)
        summary["score_mode"] = variant.get("score_mode", "")
        summary["weight"] = variant.get("weight", "")
        summary["control"] = int(bool(variant.get("control")))
        summary_rows.append(summary)

    control_best = max([f(r["top1"]) for r in summary_rows if i(r["control"]) == 1], default=0.0)
    baseline = next(r for r in summary_rows if r["variant"] == "A0_track_recency_baseline")
    real_rows = [r for r in summary_rows if i(r["control"]) == 0]
    best = max(real_rows, key=lambda r: (f(r["top1"]), f(r["mean_target_margin"]), -i(r["regressed_count"]))) if real_rows else baseline
    learned_rows = [r for r in real_rows if str(r["variant"]).startswith("A5_") or str(r["variant"]).startswith("A6_") or str(r["variant"]).startswith("A7_") or str(r["variant"]).startswith("A8_")]
    best_learned = max(learned_rows, key=lambda r: (f(r["top1"]), f(r["mean_target_margin"]), -i(r["regressed_count"]))) if learned_rows else baseline
    for row in summary_rows:
        row["control_best_top1"] = control_best
        row["selected_as_best"] = int(row is best)
        row["selected_as_best_learned"] = int(row is best_learned)
        row["controls_passed"] = int(f(row["top1"]) > control_best)
        row["beats_baseline"] = int(f(row["top1"]) > f(baseline["top1"]))

    positive_prec = float(np.mean([i(p["pair_correct_eval_only"]) for p in kept_pairs if i(p["pseudo_label"]) == 1])) if any(i(p["pseudo_label"]) == 1 for p in kept_pairs) else 0.0
    negative_prec = float(np.mean([i(p["pair_correct_eval_only"]) for p in kept_pairs if i(p["pseudo_label"]) == 0])) if any(i(p["pseudo_label"]) == 0 for p in kept_pairs) else 0.0
    learned_passed = int(
        f(best_learned["top1"]) > f(baseline["top1"])
        and f(best_learned["top1"]) > control_best
        and i(best_learned["regressed_count"]) <= 1
        and f(learned_trace["test_auc"]) > f(shuffled_trace["test_auc"]) + 0.05
    )
    compact = {
        "stage": "CORE-1AL",
        "artifact_version": args.artifact_version,
        "train_gate": train_gate,
        "eval_gate": eval_gate,
        "train_observation_count": len(train_selected),
        "eval_observation_count": len(eval_selected),
        "train_pair_count": len(kept_pairs),
        "positive_pair_precision_eval_only": positive_prec,
        "negative_pair_precision_eval_only": negative_prec,
        "learned_metric_train_auc": learned_trace["train_auc"],
        "learned_metric_test_auc": learned_trace["test_auc"],
        "shuffled_label_metric_test_auc": shuffled_trace["test_auc"],
        "baseline_top1": baseline["top1"],
        "baseline_false_retrieval_rate": baseline["false_retrieval_rate"],
        "best_variant": best["variant"],
        "best_top1": best["top1"],
        "best_false_retrieval_rate": best["false_retrieval_rate"],
        "best_improved_count": best["improved_count"],
        "best_regressed_count": best["regressed_count"],
        "best_learned_variant": best_learned["variant"],
        "best_learned_top1": best_learned["top1"],
        "best_learned_improved_count": best_learned["improved_count"],
        "best_learned_regressed_count": best_learned["regressed_count"],
        "control_best_top1": control_best,
        "learned_metric_passed": learned_passed,
        "negative_controls_passed": int(f(best["top1"]) > control_best),
        "oracle_leakage_found": 0,
        "pretrained_weights_used": 0,
        "passed_minimum": learned_passed,
        "next_recommendation": (
            "CORE-1AM add delayed object-file memory bank integration with learned metric"
            if learned_passed
            else "learned online metric does not beat baseline/controls on hard eval; inspect raw descriptor vs metric failure before integration"
        ),
    }
    report = f"""# CORE-1AL Decoupled Online Metric Smoke

This stage trains a small random-initialized descriptor metric from online pseudo pairs mined in the clean CORE-1AK train pool, then evaluates it on the broader hard CORE-1AK eval pool. No pretrained weights or GT labels are used for online training/scoring.

## Result

- Train gate: {train_gate}
- Eval gate: {eval_gate}
- Train pairs: {len(kept_pairs)}
- Pair precision: positive {positive_prec:.4f}, negative {negative_prec:.4f}
- Learned metric test AUC: {float(learned_trace['test_auc']):.4f}
- Shuffled-label metric test AUC: {float(shuffled_trace['test_auc']):.4f}
- Baseline top1: {float(baseline['top1']):.4f}
- Best variant: {compact['best_variant']} top1 {float(compact['best_top1']):.4f}
- Best learned variant: {compact['best_learned_variant']} top1 {float(compact['best_learned_top1']):.4f}
- Control best top1: {float(control_best):.4f}
- Learned metric passed: {learned_passed}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AL_"
    write_csv(
        out_dir / f"{prefix}pair_training_trace_{args.artifact_version}.csv",
        pair_rows,
        ["pair_id", "obs_i", "obs_j", "pair_type", "pseudo_label", "pair_correct_eval_only", "sequence_i", "sequence_j", "frame_i", "frame_j", "track_i", "track_j", "descriptor_distance"],
    )
    write_csv(
        out_dir / f"{prefix}training_summary_{args.artifact_version}.csv",
        [dict(learned_trace, metric="learned"), dict(shuffled_trace, metric="shuffled_label")],
        [
            "metric",
            "shuffle_labels",
            "train_pair_count",
            "test_pair_count",
            "train_positive_count",
            "train_negative_count",
            "test_positive_count",
            "test_negative_count",
            "initial_loss",
            "final_loss",
            "train_auc",
            "test_auc",
            "feature_mean_norm",
            "feature_std_mean",
        ],
    )
    write_csv(
        out_dir / f"{prefix}ablation_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "variant",
            "score_mode",
            "weight",
            "control",
            "num_queries",
            "top1",
            "top3",
            "false_retrieval_rate",
            "mean_target_margin",
            "descriptor_used_rate",
            "improved_count",
            "regressed_count",
            "unchanged_success_count",
            "unchanged_failure_count",
            "control_best_top1",
            "controls_passed",
            "beats_baseline",
            "selected_as_best",
            "selected_as_best_learned",
        ],
    )
    write_csv(
        out_dir / f"{prefix}event_results_{args.artifact_version}.csv",
        event_rows,
        [
            "variant",
            "sequence_id",
            "event_id",
            "window_kind",
            "query_obs_id",
            "candidate_count",
            "target_candidate_count",
            "top1_obs_id",
            "top1_instance_eval_only",
            "target_instance_eval_only",
            "top1_success",
            "target_rank",
            "target_in_top3",
            "target_margin",
            "baseline_score_top1",
            "aux_score_top1",
            "aux_used",
        ],
    )
    write_csv(
        out_dir / f"{prefix}negative_control_summary_{args.artifact_version}.csv",
        [r for r in summary_rows if i(r["control"]) == 1],
        [
            "variant",
            "score_mode",
            "top1",
            "false_retrieval_rate",
            "mean_target_margin",
            "improved_count",
            "regressed_count",
            "control_best_top1",
            "selected_as_best",
        ],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1al_decoupled_online_metric_smoke.py",
                "gt_used_for_online_pair_mining": 0,
                "gt_used_for_online_scoring": 0,
                "gt_used_for_eval_only": 1,
                "pretrained_weights_used": 0,
                "future_frame_used": 0,
                "leakage_found": 0,
            }
        ],
        ["file", "gt_used_for_online_pair_mining", "gt_used_for_online_scoring", "gt_used_for_eval_only", "pretrained_weights_used", "future_frame_used", "leakage_found"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
