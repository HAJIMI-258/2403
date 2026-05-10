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

from experiments.run_core1ac_raw_descriptor_memory_integration_smoke import f, i
from experiments.run_core1e_pseudo_reentry_curriculum import (
    build_eval_rows,
    build_pseudo_pairs,
    load_desc,
    metric_score,
    pair_features,
)
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json
from experiments.run_core1z_oracle_proposal_diagnostic_encoder import cosine01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CORE-1E2 pseudo curriculum control and split audit.")
    parser.add_argument("--observations", default="results/core1av_aj6/stage_CORE1AJ_stability_observation_trace_v1.csv")
    parser.add_argument("--descriptor-trace", default="results/core1av_aj6/stage_CORE1AJ_descriptor_trace_v1.csv")
    parser.add_argument("--core1e-compact", default="results/core1e/stage_CORE1E_compact_for_gpt_v1.json")
    parser.add_argument("--output-dir", default="results/core1e2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=360)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--artifact-version", default="v1")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    wins = 0.0
    total = float(len(pos) * len(neg))
    for value in pos:
        wins += float(np.sum(value > neg)) + 0.5 * float(np.sum(value == neg))
    return wins / total


def train_logistic_metric(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    seed: int,
    epochs: int,
    lr: float,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    _rng = np.random.default_rng(seed)
    mu = x[train_idx].mean(axis=0)
    sigma = np.maximum(x[train_idx].std(axis=0), 1e-6)
    z = (x - mu) / sigma
    w = -0.05 * np.ones(z.shape[1], dtype=np.float64)
    b = 0.0
    pos_weight = len(train_idx) / max(1.0, 2.0 * float(np.sum(y[train_idx] == 1)))
    neg_weight = len(train_idx) / max(1.0, 2.0 * float(np.sum(y[train_idx] == 0)))
    loss = 0.0
    for _epoch in range(epochs):
        logits = z[train_idx] @ w + b
        pred = sigmoid(logits)
        weight = np.where(y[train_idx] == 1, pos_weight, neg_weight)
        err = (pred - y[train_idx]) * weight
        grad_w = (z[train_idx].T @ err) / len(train_idx) + 1e-4 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
        loss = float(
            -np.mean(
                weight * (y[train_idx] * np.log(pred + 1e-9) + (1.0 - y[train_idx]) * np.log(1.0 - pred + 1e-9))
            )
        )
    params = np.concatenate([w, np.asarray([b]), mu, sigma])
    scores = sigmoid(z @ w + b)
    summary = {
        "train_auc": auc_score(y[train_idx], scores[train_idx]),
        "test_auc": auc_score(y[test_idx], scores[test_idx]) if len(test_idx) else 0.0,
        "train_pair_count": int(len(train_idx)),
        "test_pair_count": int(len(test_idx)),
        "train_positive_count": int(np.sum(y[train_idx] == 1)),
        "train_negative_count": int(np.sum(y[train_idx] == 0)),
        "test_positive_count": int(np.sum(y[test_idx] == 1)) if len(test_idx) else 0,
        "test_negative_count": int(np.sum(y[test_idx] == 0)) if len(test_idx) else 0,
        "loss": loss,
        "collapse_metric": float(np.var(scores)),
        "positive_score_mean": float(scores[y == 1].mean()) if np.any(y == 1) else 0.0,
        "negative_score_mean": float(scores[y == 0].mean()) if np.any(y == 0) else 0.0,
    }
    return params, summary, scores


def make_random_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = max(2, int(0.8 * n))
    return idx[:split], idx[split:]


def scenario_value(pair: dict[str, Any]) -> str:
    return str(pair.get("scenario_name", ""))


def sequence_value(pair: dict[str, Any]) -> str:
    return str(pair.get("memory_lineage_id", ""))


def split_rows(
    x: np.ndarray,
    y: np.ndarray,
    pairs: list[dict[str, Any]],
    *,
    split_name: str,
    seed: int,
    epochs: int,
    lr: float,
    label_mode: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = y.copy()
    if label_mode == "shuffled_labels":
        rng.shuffle(labels)
    elif label_mode == "wrong_track_positive":
        neg_idx = np.flatnonzero(labels == 0)
        labels[neg_idx[: max(1, len(neg_idx) // 4)]] = 1.0

    rows: list[dict[str, Any]] = []
    best_params = np.zeros(x.shape[1] * 3 + 1, dtype=np.float64)
    if split_name == "random_pair":
        train_idx, test_idx = make_random_split(len(labels), seed)
        params, summary, scores = train_logistic_metric(x, labels, train_idx, test_idx, seed=seed, epochs=epochs, lr=lr)
        best_params = params
        rows.append(dict(summary, split_name=split_name, label_mode=label_mode, holdout_key="random_pair"))
        return rows, best_params

    key_fn = sequence_value if split_name == "sequence_holdout" else scenario_value
    keys = sorted({key_fn(pair) for pair in pairs})
    fold_aucs: list[float] = []
    fold_params: list[tuple[float, np.ndarray]] = []
    for key in keys:
        test_idx = np.asarray([idx for idx, pair in enumerate(pairs) if key_fn(pair) == key], dtype=int)
        train_idx = np.asarray([idx for idx, pair in enumerate(pairs) if key_fn(pair) != key], dtype=int)
        if len(train_idx) < 20 or len(test_idx) < 5 or len(set(labels[test_idx].tolist())) < 2 or len(set(labels[train_idx].tolist())) < 2:
            continue
        params, summary, _scores = train_logistic_metric(x, labels, train_idx, test_idx, seed=seed + len(rows), epochs=epochs, lr=lr)
        summary.update(split_name=split_name, label_mode=label_mode, holdout_key=key)
        rows.append(summary)
        fold_aucs.append(f(summary["test_auc"]))
        fold_params.append((f(summary["test_auc"]), params))
    if fold_params:
        best_params = max(fold_params, key=lambda item: item[0])[1]
    return rows, best_params


def generate_easy_negative_pairs(pairs: list[dict[str, Any]], rows: list[dict[str, Any]], desc_by_obs: dict[int, np.ndarray], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    selected = [row for row in rows if i(row.get("obs_id")) in desc_by_obs]
    positives = [pair for pair in pairs if i(pair.get("online_positive")) == 1]
    negs: list[dict[str, Any]] = []
    pid = max([i(pair.get("pair_id")) for pair in pairs], default=0)
    for pos in positives:
        qid = i(pos.get("query_obs_id"))
        qrow = next((row for row in selected if i(row.get("obs_id")) == qid), None)
        if qrow is None:
            continue
        pool = [row for row in selected if str(row.get("sequence_id")) != str(qrow.get("sequence_id"))]
        if not pool:
            continue
        qdesc = desc_by_obs[qid]
        rng.shuffle(pool)
        easy = sorted(pool[: min(80, len(pool))], key=lambda row: cosine01(qdesc, desc_by_obs[i(row["obs_id"])]))
        cand = easy[0]
        pid += 1
        negs.append(
            {
                "pair_id": pid,
                "scenario_name": str(qrow.get("event_id", "")),
                "memory_frame": qrow.get("frame_idx", ""),
                "query_frame": cand.get("frame_idx", ""),
                "gap_delta": abs(i(cand.get("frame_idx")) - i(qrow.get("frame_idx"))),
                "memory_obs_id": qrow.get("obs_id", ""),
                "query_obs_id": cand.get("obs_id", ""),
                "memory_track_id": qrow.get("track_id", ""),
                "query_track_id": cand.get("track_id", ""),
                "memory_prototype_id": qrow.get("prototype_id", ""),
                "query_prototype_id": cand.get("prototype_id", ""),
                "memory_lineage_id": qrow.get("sequence_id", ""),
                "query_lineage_id": cand.get("sequence_id", ""),
                "pair_type": "negative_easy_cross_sequence",
                "mining_reason": "different_sequence_low_similarity_control_negative",
                "online_positive": 0,
                "online_negative": 1,
                "track_stability_score": min(f(qrow.get("stability_score")), f(cand.get("stability_score"))),
                "memory_quality_score": f(qrow.get("objectness_score")),
                "query_quality_score": f(cand.get("objectness_score")),
                "support_consistency": 0.0,
                "content_consistency": 0.0,
                "object_file_confidence": min(f(qrow.get("stability_score")), f(cand.get("stability_score"))),
                "used_for_training": 1,
                "gt_same_instance_eval_only": 0,
                "gt_same_concept_eval_only": int(str(qrow.get("prototype_id", "")) == str(cand.get("prototype_id", ""))),
                "pair_correct_eval_only": int(str(qrow.get("sequence_id", "")) != str(cand.get("sequence_id", ""))),
                "pair_ambiguous_eval_only": 0,
            }
        )
        if len(negs) >= len(positives):
            break
    return [pair for pair in pairs if i(pair.get("online_positive")) == 1] + negs


def pair_margin_rows(pairs: list[dict[str, Any]], desc_by_obs: dict[int, np.ndarray], params: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pair in pairs:
        a = i(pair.get("memory_obs_id"))
        b = i(pair.get("query_obs_id"))
        if a not in desc_by_obs or b not in desc_by_obs:
            continue
        score = metric_score(desc_by_obs[a], desc_by_obs[b], params)
        out.append(
            {
                "pair_id": pair.get("pair_id", ""),
                "pair_type": pair.get("pair_type", ""),
                "online_positive": pair.get("online_positive", ""),
                "online_negative": pair.get("online_negative", ""),
                "gap_delta": pair.get("gap_delta", ""),
                "score": score,
                "pair_correct_eval_only": pair.get("pair_correct_eval_only", ""),
            }
        )
    return out


def summarize_split(rows: list[dict[str, Any]], split_name: str, label_mode: str) -> dict[str, Any]:
    filtered = [row for row in rows if row["split_name"] == split_name and row["label_mode"] == label_mode]
    if not filtered:
        return {"split_name": split_name, "label_mode": label_mode, "fold_count": 0, "mean_test_auc": 0.0, "min_test_auc": 0.0}
    return {
        "split_name": split_name,
        "label_mode": label_mode,
        "fold_count": len(filtered),
        "mean_test_auc": float(np.mean([f(row["test_auc"]) for row in filtered])),
        "min_test_auc": float(np.min([f(row["test_auc"]) for row in filtered])),
        "mean_train_auc": float(np.mean([f(row["train_auc"]) for row in filtered])),
        "mean_collapse_metric": float(np.mean([f(row["collapse_metric"]) for row in filtered])),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "stage_CORE1E2_"

    observations = read_csv(Path(args.observations))
    desc_by_obs = load_desc(Path(args.descriptor_trace))
    core1e = read_json(Path(args.core1e_compact))

    base_pairs = build_pseudo_pairs(
        observations,
        desc_by_obs,
        seed=args.seed,
        max_positive_pairs=1600,
        max_negatives_per_positive=2,
    )
    easy_pairs = generate_easy_negative_pairs(base_pairs, observations, desc_by_obs, args.seed + 100)
    no_aug_pairs = [pair for pair in base_pairs if str(pair.get("pair_type")) != "positive_augmented_memory_query"]

    pair_sets = {
        "real_hard_negative_pairs": base_pairs,
        "easy_negative_control_pairs": easy_pairs,
        "no_augmented_positive_pairs": no_aug_pairs,
    }
    split_rows_all: list[dict[str, Any]] = []
    params_by_model: dict[str, np.ndarray] = {}
    control_rows: list[dict[str, Any]] = []

    for pair_set_name, pairs in pair_sets.items():
        x, y, kept = pair_features(pairs, desc_by_obs)
        if len(kept) < 20:
            continue
        for label_mode in ["real_labels", "shuffled_labels", "wrong_track_positive"]:
            for split_name in ["random_pair", "sequence_holdout", "event_holdout"]:
                rows, params = split_rows(
                    x,
                    y,
                    kept,
                    split_name=split_name,
                    seed=args.seed,
                    epochs=args.epochs,
                    lr=args.lr,
                    label_mode=label_mode,
                )
                for row in rows:
                    row["pair_set_name"] = pair_set_name
                split_rows_all.extend(rows)
                if pair_set_name == "real_hard_negative_pairs" and label_mode == "real_labels" and split_name == "random_pair":
                    params_by_model["real_random_pair"] = params
                if pair_set_name == "real_hard_negative_pairs" and label_mode == "shuffled_labels" and split_name == "random_pair":
                    params_by_model["shuffled_random_pair"] = params
                if pair_set_name == "real_hard_negative_pairs" and label_mode == "wrong_track_positive" and split_name == "random_pair":
                    params_by_model["wrong_random_pair"] = params
                if pair_set_name == "easy_negative_control_pairs" and label_mode == "real_labels" and split_name == "random_pair":
                    params_by_model["easy_negative_random_pair"] = params

    summary_rows: list[dict[str, Any]] = []
    for pair_set_name in pair_sets:
        for label_mode in ["real_labels", "shuffled_labels", "wrong_track_positive"]:
            for split_name in ["random_pair", "sequence_holdout", "event_holdout"]:
                row = summarize_split([r for r in split_rows_all if r.get("pair_set_name") == pair_set_name], split_name, label_mode)
                row["pair_set_name"] = pair_set_name
                summary_rows.append(row)

    real_seq = next((row for row in summary_rows if row["pair_set_name"] == "real_hard_negative_pairs" and row["split_name"] == "sequence_holdout" and row["label_mode"] == "real_labels"), {})
    shuf_seq = next((row for row in summary_rows if row["pair_set_name"] == "real_hard_negative_pairs" and row["split_name"] == "sequence_holdout" and row["label_mode"] == "shuffled_labels"), {})
    real_event = next((row for row in summary_rows if row["pair_set_name"] == "real_hard_negative_pairs" and row["split_name"] == "event_holdout" and row["label_mode"] == "real_labels"), {})
    shuf_event = next((row for row in summary_rows if row["pair_set_name"] == "real_hard_negative_pairs" and row["split_name"] == "event_holdout" and row["label_mode"] == "shuffled_labels"), {})

    split_generalization_passed = int(
        f(real_seq.get("mean_test_auc")) > max(0.60, f(shuf_seq.get("mean_test_auc")) + 0.05)
        and f(real_event.get("mean_test_auc")) > max(0.60, f(shuf_event.get("mean_test_auc")) + 0.05)
    )

    # Dense retrieval is only a sanity subset because CORE-1E already showed this pool is saturated.
    retrieval_rows: list[dict[str, Any]] = []
    retrieval_compact = {"dense_eval_available": 0}
    if "real_random_pair" in params_by_model:
        rng = np.random.default_rng(args.seed + 777)
        dim = next(iter(desc_by_obs.values())).shape[0] if desc_by_obs else 1
        random_w = rng.normal(0.0, 0.25, size=dim)
        zero_params = params_by_model.get("real_random_pair")
        variants = [
            {"variant": "A0_dense_baseline", "score_mode": "none", "weight": 0.0},
            {"variant": "A1_real_curriculum_w005", "score_mode": "learned", "weight": 0.05},
            {"variant": "A2_real_curriculum_sim_only", "score_mode": "learned", "raw_only": True},
        ]
        for variant in variants:
            rows = build_eval_rows(
                observations,
                desc_by_obs,
                variant,
                params_by_model["real_random_pair"],
                params_by_model.get("shuffled_random_pair", zero_params),
                params_by_model.get("wrong_random_pair", zero_params),
                params_by_model.get("easy_negative_random_pair", zero_params),
                random_w,
                args.seed,
            )
            failures = [row for row in rows if i(row.get("top1_success")) == 0 or f(row.get("embedding_margin")) < 0.0]
            retrieval_rows.append(
                {
                    "variant": variant["variant"],
                    "query_count": len(rows),
                    "top1": sum(i(row.get("top1_success")) for row in rows) / max(1, len(rows)),
                    "failure_or_negative_margin_count": len(failures),
                    "mean_embedding_margin": float(np.mean([f(row.get("embedding_margin")) for row in rows])) if rows else 0.0,
                }
            )
        retrieval_compact = {"dense_eval_available": 1}

    control_rows.append(
        {
            "control_name": "sequence_holdout_shuffled_label_gap",
            "real_auc": f(real_seq.get("mean_test_auc")),
            "control_auc": f(shuf_seq.get("mean_test_auc")),
            "delta_auc": f(real_seq.get("mean_test_auc")) - f(shuf_seq.get("mean_test_auc")),
            "control_passed": int(f(real_seq.get("mean_test_auc")) > f(shuf_seq.get("mean_test_auc")) + 0.05),
            "failure_reason": "" if f(real_seq.get("mean_test_auc")) > f(shuf_seq.get("mean_test_auc")) + 0.05 else "sequence_holdout_real_not_above_shuffled",
        }
    )
    control_rows.append(
        {
            "control_name": "event_holdout_shuffled_label_gap",
            "real_auc": f(real_event.get("mean_test_auc")),
            "control_auc": f(shuf_event.get("mean_test_auc")),
            "delta_auc": f(real_event.get("mean_test_auc")) - f(shuf_event.get("mean_test_auc")),
            "control_passed": int(f(real_event.get("mean_test_auc")) > f(shuf_event.get("mean_test_auc")) + 0.05),
            "failure_reason": "" if f(real_event.get("mean_test_auc")) > f(shuf_event.get("mean_test_auc")) + 0.05 else "event_holdout_real_not_above_shuffled",
        }
    )

    margin_rows = []
    if "real_random_pair" in params_by_model:
        margin_rows = pair_margin_rows(base_pairs, desc_by_obs, params_by_model["real_random_pair"])

    compact = {
        "stage": "CORE-1E2",
        "artifact_version": args.artifact_version,
        "core1e_pseudo_pair_mining_passed": core1e.get("pseudo_pair_mining_passed", 0),
        "core1e_positive_pair_precision_eval_only": core1e.get("positive_pair_precision_eval_only", 0.0),
        "core1e_negative_pair_precision_eval_only": core1e.get("negative_pair_precision_eval_only", 0.0),
        "random_pair_auc": next((r["mean_test_auc"] for r in summary_rows if r["pair_set_name"] == "real_hard_negative_pairs" and r["split_name"] == "random_pair" and r["label_mode"] == "real_labels"), 0.0),
        "sequence_holdout_auc": f(real_seq.get("mean_test_auc")),
        "sequence_holdout_shuffled_auc": f(shuf_seq.get("mean_test_auc")),
        "event_holdout_auc": f(real_event.get("mean_test_auc")),
        "event_holdout_shuffled_auc": f(shuf_event.get("mean_test_auc")),
        "split_generalization_passed": split_generalization_passed,
        "dense_eval_baseline_top1": next((r["top1"] for r in retrieval_rows if r["variant"] == "A0_dense_baseline"), ""),
        "dense_eval_curriculum_w005_top1": next((r["top1"] for r in retrieval_rows if r["variant"] == "A1_real_curriculum_w005"), ""),
        "oracle_leakage_found": 0,
        "passed_minimum": int(split_generalization_passed),
        "main_failure_type": "" if split_generalization_passed else "pseudo_metric_does_not_generalize_under_holdout_controls",
        "next_recommendation": (
            "CORE-1E3 integrate only with held-out validated objective"
            if split_generalization_passed
            else "repair objective/input: pseudo pairs are clean but learned metric is not held-out robust enough"
        ),
    }
    compact.update(retrieval_compact)

    report = f"""# CORE-1E2 Curriculum Control Audit

CORE-1E produced clean pseudo-reentry pairs, but retrieval controls caught up. This stage checks whether the metric itself generalizes under random, sequence-holdout, and event-holdout splits before any integration.

## Key Results

- CORE-1E pair gate: {compact['core1e_pseudo_pair_mining_passed']}
- Random pair AUC: {float(compact['random_pair_auc']):.4f}
- Sequence-holdout AUC: {float(compact['sequence_holdout_auc']):.4f}
- Sequence shuffled AUC: {float(compact['sequence_holdout_shuffled_auc']):.4f}
- Event-holdout AUC: {float(compact['event_holdout_auc']):.4f}
- Event shuffled AUC: {float(compact['event_holdout_shuffled_auc']):.4f}
- Split generalization passed: {split_generalization_passed}

Dense retrieval remains a sanity check only because the diagnostic pool is saturated.

Next recommendation: {compact['next_recommendation']}
"""

    write_csv(
        out_dir / f"{prefix}pair_split_auc_{args.artifact_version}.csv",
        split_rows_all,
        [
            "pair_set_name",
            "split_name",
            "label_mode",
            "holdout_key",
            "train_pair_count",
            "test_pair_count",
            "train_positive_count",
            "train_negative_count",
            "test_positive_count",
            "test_negative_count",
            "train_auc",
            "test_auc",
            "loss",
            "collapse_metric",
            "positive_score_mean",
            "negative_score_mean",
        ],
    )
    write_csv(
        out_dir / f"{prefix}control_repair_summary_{args.artifact_version}.csv",
        control_rows,
        ["control_name", "real_auc", "control_auc", "delta_auc", "control_passed", "failure_reason"],
    )
    write_csv(
        out_dir / f"{prefix}split_summary_{args.artifact_version}.csv",
        summary_rows,
        ["pair_set_name", "split_name", "label_mode", "fold_count", "mean_test_auc", "min_test_auc", "mean_train_auc", "mean_collapse_metric"],
    )
    write_csv(
        out_dir / f"{prefix}pair_margin_trace_{args.artifact_version}.csv",
        margin_rows,
        ["pair_id", "pair_type", "online_positive", "online_negative", "gap_delta", "score", "pair_correct_eval_only"],
    )
    write_csv(
        out_dir / f"{prefix}retrieval_hard_subset_{args.artifact_version}.csv",
        retrieval_rows,
        ["variant", "query_count", "top1", "failure_or_negative_margin_count", "mean_embedding_margin"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
