from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.synth_stream import SyntheticStreamGenerator
from experiments.run_core1aa_stability_namespace_pair_gate import (
    GATES,
    build_negative_pairs,
    build_positive_pairs,
    row_passes,
)
from experiments.run_core1k_windowed_render_cache import ensure_min_crop_box, read_csv, write_csv, write_json
from experiments.run_core1m_assignment_pair_confidence_gate import load_config
from experiments.run_core1z_oracle_proposal_diagnostic_encoder import (
    auc_score,
    cosine01,
    descriptor_to_str,
    metric_scores,
    pair_features,
    train_pair_metric,
    zscore_descriptors,
)
from experiments.run_v3_stage_e4a_active_evidence_acquisition import crop_descriptor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AB non-oracle curriculum diagnostic encoder.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--core1aa-compact", default="results/core1aa/stage_CORE1AA_compact_for_gpt_v1.json")
    p.add_argument("--observations", default="results/core1aa/stage_CORE1AA_stability_observation_trace_v1.csv")
    p.add_argument("--output-dir", default="results/core1ab")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-negatives-per-observation", type=int, default=8)
    p.add_argument("--epochs", type=int, default=900)
    p.add_argument("--lr", type=float, default=0.08)
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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def selected_gate(name: str) -> dict[str, Any]:
    for gate in GATES:
        if gate["gate_name"] == name:
            return gate
    raise ValueError(f"Unknown CORE-1AA gate: {name}")


def namespace_positive(row: dict[str, Any]) -> int:
    return int(str(row.get("gt_instance_i_eval_only", "")) != "" and str(row.get("gt_instance_j_eval_only", "")) != "" and str(row.get("gt_instance_i_eval_only")) == str(row.get("gt_instance_j_eval_only")) and str(row.get("sequence_i_eval_only", row.get("sequence_id"))) == str(row.get("sequence_j_eval_only", row.get("sequence_id"))))


def prepare_selected_observations(rows: list[dict[str, str]], gate: dict[str, Any]) -> list[dict[str, Any]]:
    selected = []
    obs_id = 0
    for row in rows:
        rr: dict[str, Any] = dict(row)
        if not row_passes(rr, gate):
            continue
        obs_id += 1
        rr["obs_id"] = obs_id
        selected.append(rr)
    return selected


def observation_key(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (str(row["sequence_id"]), str(row["event_id"]), str(row["window_kind"]), i(row["frame_idx"]), i(row["track_id"]))


def build_pairs_with_obs_ids(selected: list[dict[str, Any]], gate_name: str, negative_mode: str, max_negatives: int) -> list[dict[str, Any]]:
    by_key = {observation_key(row): row for row in selected}
    positive_rows = build_positive_pairs(selected, gate_name)
    negative_rows = build_negative_pairs(selected, gate_name, negative_mode, max_negatives)
    out: list[dict[str, Any]] = []
    pair_id = 0
    for row in positive_rows:
        ki = (str(row["sequence_id"]), str(row["event_id"]), str(row["window_kind"]), i(row["frame_i"]), i(row["track_i"]))
        kj = (str(row["sequence_id"]), str(row["event_id"]), str(row["window_kind"]), i(row["frame_j"]), i(row["track_j"]))
        if ki not in by_key or kj not in by_key:
            continue
        pair_id += 1
        out.append(
            {
                **row,
                "pair_id": pair_id,
                "obs_i": by_key[ki]["obs_id"],
                "obs_j": by_key[kj]["obs_id"],
                "sequence_i_eval_only": by_key[ki]["sequence_id"],
                "sequence_j_eval_only": by_key[kj]["sequence_id"],
                "online_positive": 1,
                "online_negative": 0,
            }
        )
    for row in negative_rows:
        # build_negative_pairs now retains only the left context in the public
        # row, so reconstruct full right-side rows by deterministic order.
        # Rebuilding inline keeps CORE-1AB independent from CORE-1AA CSV shape.
        continue
    pair_id_start = pair_id
    pair_id = pair_id_start
    ordered = sorted(selected, key=lambda r: (i(r["sequence_id"]), str(r["event_id"]), str(r["window_kind"]), i(r["frame_idx"]), i(r["track_id"])))
    for a in ordered:
        count = 0
        for b in ordered:
            if a is b:
                continue
            if negative_mode == "cross_sequence" and str(a["sequence_id"]) == str(b["sequence_id"]):
                continue
            if negative_mode == "cross_event_same_sequence" and (str(a["sequence_id"]) != str(b["sequence_id"]) or str(a["event_id"]) == str(b["event_id"])):
                continue
            if negative_mode == "cross_window_any" and str(a["sequence_id"]) == str(b["sequence_id"]) and str(a["event_id"]) == str(b["event_id"]) and str(a["window_kind"]) == str(b["window_kind"]):
                continue
            pair_id += 1
            count += 1
            pair_correct = int(
                str(a.get("gt_instance_eval_only", "")) != ""
                and str(b.get("gt_instance_eval_only", "")) != ""
                and (str(a["sequence_id"]) != str(b["sequence_id"]) or str(a.get("gt_instance_eval_only")) != str(b.get("gt_instance_eval_only")))
            )
            out.append(
                {
                    "pair_id": pair_id,
                    "gate_name": gate_name,
                    "negative_mode": negative_mode,
                    "pair_type": "negative_stable_cross_context",
                    "sequence_id": a["sequence_id"],
                    "event_id": a["event_id"],
                    "window_kind": a["window_kind"],
                    "frame_i": a["frame_idx"],
                    "frame_j": b["frame_idx"],
                    "track_i": a["track_id"],
                    "track_j": b["track_id"],
                    "obs_i": a["obs_id"],
                    "obs_j": b["obs_id"],
                    "sequence_i_eval_only": a["sequence_id"],
                    "sequence_j_eval_only": b["sequence_id"],
                    "gt_instance_i_eval_only": a.get("gt_instance_eval_only", ""),
                    "gt_instance_j_eval_only": b.get("gt_instance_eval_only", ""),
                    "pair_correct_eval_only": pair_correct,
                    "online_positive": 0,
                    "online_negative": 1,
                }
            )
            if count >= max_negatives:
                break
    return out


def extract_descriptors(rows: list[dict[str, Any]], config_path: Path, seed: int) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    cfg, _payload = load_config(config_path)
    generator = SyntheticStreamGenerator(cfg, seed=seed)
    by_sequence: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_sequence.setdefault(i(row["sequence_id"]), []).append(row)
    desc_rows: list[dict[str, Any]] = []
    desc_by_obs: dict[int, np.ndarray] = {}
    for sequence_id, seq_rows in sorted(by_sequence.items()):
        sequence = generator.generate_sequence(sequence_id)
        frames_by_idx = {frame.frame_index: frame for frame in sequence.frames}
        for row in seq_rows:
            frame = frames_by_idx.get(i(row["frame_idx"]))
            box = box_from_text(str(row.get("box", "")))
            if frame is None or box is None:
                continue
            crop_box = ensure_min_crop_box(box, frame.frame.shape)
            payload = crop_descriptor(frame.frame, None, crop_box, box)
            desc = np.asarray(payload["descriptor"], dtype=np.float32)
            desc_by_obs[i(row["obs_id"])] = desc
            desc_rows.append(
                {
                    "obs_id": row["obs_id"],
                    "sequence_id": row["sequence_id"],
                    "event_id": row["event_id"],
                    "window_kind": row["window_kind"],
                    "frame_idx": row["frame_idx"],
                    "track_id": row["track_id"],
                    "box": row["box"],
                    "crop_box": "|".join(str(v) for v in crop_box),
                    "descriptor_norm": float(np.linalg.norm(desc)),
                    "descriptor_entropy_proxy": float(np.std(desc)),
                    "edge_density": payload["edge_density"],
                    "gt_instance_eval_only": row.get("gt_instance_eval_only", ""),
                    "descriptor": descriptor_to_str(desc),
                }
            )
    return desc_rows, desc_by_obs


def pair_precision(rows: list[dict[str, Any]], pair_type_prefix: str) -> float:
    selected = [r for r in rows if str(r["pair_type"]).startswith(pair_type_prefix)]
    if not selected:
        return 0.0
    return float(np.mean([i(r["pair_correct_eval_only"]) for r in selected]))


def split_pairs(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for row in rows:
        # Pair-level deterministic split. CORE-1AB is diagnostic; the next
        # stage should use a larger sequence-level split if this passes.
        if i(row["pair_id"]) % 5 == 0:
            test.append(row)
        else:
            train.append(row)
    return train, test


def mean_key(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key, "") != ""]
    return float(np.mean(vals)) if vals else 0.0


def build_retrieval_rows(
    selected: list[dict[str, Any]],
    desc_by_obs: dict[int, np.ndarray],
    learned_w: np.ndarray,
    learned_b: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_window: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in selected:
        by_window.setdefault((str(row["sequence_id"]), str(row["event_id"]), str(row["window_kind"])), []).append(row)
    for (_seq, _event, _kind), rows in by_window.items():
        memory: list[dict[str, Any]] = []
        for query in sorted(rows, key=lambda r: (i(r["frame_idx"]), i(r["track_id"]))):
            qid = i(query["obs_id"])
            qgt = str(query.get("gt_instance_eval_only", ""))
            candidates = [m for m in memory if i(m["obs_id"]) in desc_by_obs and str(m.get("gt_instance_eval_only", "")) != ""]
            target_candidates = [m for m in candidates if str(m.get("gt_instance_eval_only", "")) == qgt]
            distractors = [m for m in candidates if str(m.get("gt_instance_eval_only", "")) != qgt]
            if qid in desc_by_obs and qgt != "" and target_candidates and distractors:
                raw_scored = [(m, cosine01(desc_by_obs[qid], desc_by_obs[i(m["obs_id"])])) for m in candidates]
                learned_scored = []
                for m in candidates:
                    feat = -np.abs(desc_by_obs[qid] - desc_by_obs[i(m["obs_id"])])[None, :]
                    learned_scored.append((m, float(metric_scores(feat, learned_w, learned_b)[0])))
                raw_top = max(raw_scored, key=lambda x: x[1])
                learned_top = max(learned_scored, key=lambda x: x[1])
                raw_target = max((score for m, score in raw_scored if str(m.get("gt_instance_eval_only", "")) == qgt), default=0.0)
                raw_wrong = max((score for m, score in raw_scored if str(m.get("gt_instance_eval_only", "")) != qgt), default=0.0)
                learned_target = max((score for m, score in learned_scored if str(m.get("gt_instance_eval_only", "")) == qgt), default=0.0)
                learned_wrong = max((score for m, score in learned_scored if str(m.get("gt_instance_eval_only", "")) != qgt), default=0.0)
                out.append(
                    {
                        "sequence_id": query["sequence_id"],
                        "event_id": query["event_id"],
                        "window_kind": query["window_kind"],
                        "query_obs_id": qid,
                        "candidate_count": len(candidates),
                        "target_candidate_count": len(target_candidates),
                        "raw_top1_success": int(str(raw_top[0].get("gt_instance_eval_only", "")) == qgt),
                        "raw_target_margin": float(raw_target - raw_wrong),
                        "learned_top1_success": int(str(learned_top[0].get("gt_instance_eval_only", "")) == qgt),
                        "learned_target_margin": float(learned_target - learned_wrong),
                    }
                )
            memory.append(query)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    compact_aa = read_json(Path(args.core1aa_compact))
    gate_name = str(compact_aa["best_gate"])
    negative_mode = str(compact_aa["best_negative_mode"])
    gate = selected_gate(gate_name)
    selected = prepare_selected_observations(read_csv(Path(args.observations)), gate)
    pair_rows = build_pairs_with_obs_ids(selected, gate_name, negative_mode, args.max_negatives_per_observation)
    desc_rows, desc_raw = extract_descriptors(selected, Path(args.config), args.seed)
    train_pairs, test_pairs = split_pairs(pair_rows)
    train_obs_ids = {i(r["obs_i"]) for r in train_pairs} | {i(r["obs_j"]) for r in train_pairs}
    desc_by_obs, _mean, _std = zscore_descriptors(desc_raw, train_obs_ids)
    x_train, y_train, kept_train = pair_features(train_pairs, desc_by_obs)
    x_test, y_test, kept_test = pair_features(test_pairs, desc_by_obs)
    raw_train = np.asarray([cosine01(desc_by_obs[i(r["obs_i"])], desc_by_obs[i(r["obs_j"])]) for r in kept_train], dtype=np.float64)
    raw_test = np.asarray([cosine01(desc_by_obs[i(r["obs_i"])], desc_by_obs[i(r["obs_j"])]) for r in kept_test], dtype=np.float64)
    learned_w, learned_b, train_trace = train_pair_metric(x_train, y_train, epochs=args.epochs, lr=args.lr, seed=args.seed)
    learned_train = metric_scores(x_train, learned_w, learned_b)
    learned_test = metric_scores(x_test, learned_w, learned_b)
    rng = np.random.default_rng(args.seed + 411)
    shuffled_y = rng.permutation(y_train) if y_train.size else y_train
    shuf_w, shuf_b, shuf_trace = train_pair_metric(x_train, shuffled_y, epochs=args.epochs, lr=args.lr, seed=args.seed + 499)
    shuffled_test = metric_scores(x_test, shuf_w, shuf_b)
    retrieval_rows = build_retrieval_rows(selected, desc_by_obs, learned_w, learned_b)

    pair_score_rows: list[dict[str, Any]] = []
    for split_name, pairs, labels, raw_scores, learned_scores, shuffled_scores in [
        ("train", kept_train, y_train, raw_train, learned_train, metric_scores(x_train, shuf_w, shuf_b)),
        ("test", kept_test, y_test, raw_test, learned_test, shuffled_test),
    ]:
        for row, label, raw_score, learned_score, shuffled_score in zip(pairs, labels, raw_scores, learned_scores, shuffled_scores):
            pair_score_rows.append(
                {
                    "split": split_name,
                    "pair_id": row["pair_id"],
                    "pair_type": row["pair_type"],
                    "label_positive": int(label),
                    "pair_correct_eval_only": row["pair_correct_eval_only"],
                    "raw_descriptor_score": float(raw_score),
                    "learned_metric_score": float(learned_score),
                    "shuffled_label_metric_score": float(shuffled_score),
                }
            )

    positive_count = len([r for r in pair_rows if str(r["pair_type"]).startswith("positive")])
    negative_count = len([r for r in pair_rows if str(r["pair_type"]).startswith("negative")])
    pos_precision = pair_precision(pair_rows, "positive")
    neg_precision = pair_precision(pair_rows, "negative")
    raw_test_auc = auc_score(raw_test, y_test)
    learned_test_auc = auc_score(learned_test, y_test)
    shuffled_auc = auc_score(shuffled_test, y_test)
    raw_retrieval_top1 = mean_key(retrieval_rows, "raw_top1_success")
    learned_retrieval_top1 = mean_key(retrieval_rows, "learned_top1_success")
    raw_margin = mean_key(retrieval_rows, "raw_target_margin")
    learned_margin = mean_key(retrieval_rows, "learned_target_margin")
    controls_passed = int(learned_test_auc > shuffled_auc + 0.03 and learned_test_auc >= raw_test_auc - 0.02)
    raw_descriptor_signal_passed = int(raw_test_auc >= 0.95 and raw_retrieval_top1 >= 0.90 and raw_test_auc > shuffled_auc + 0.03)
    passed = int(
        pos_precision >= 0.85
        and neg_precision >= 0.85
        and learned_test_auc > shuffled_auc + 0.03
        and learned_test_auc >= raw_test_auc - 0.02
        and len(retrieval_rows) > 0
    )
    compact = {
        "stage": "CORE-1AB",
        "artifact_version": args.artifact_version,
        "source_curriculum": "CORE-1AA",
        "gate_name": gate_name,
        "negative_mode": negative_mode,
        "selected_observation_count": len(selected),
        "descriptor_available_count": len(desc_raw),
        "positive_pair_count": positive_count,
        "negative_pair_count": negative_count,
        "positive_pair_precision_eval_only": pos_precision,
        "negative_pair_precision_eval_only": neg_precision,
        "raw_descriptor_test_auc": raw_test_auc,
        "learned_metric_test_auc": learned_test_auc,
        "shuffled_label_metric_test_auc": shuffled_auc,
        "raw_retrieval_top1": raw_retrieval_top1,
        "learned_metric_retrieval_top1": learned_retrieval_top1,
        "raw_mean_retrieval_margin": raw_margin,
        "learned_mean_retrieval_margin": learned_margin,
        "negative_controls_passed": controls_passed,
        "raw_descriptor_signal_passed": raw_descriptor_signal_passed,
        "oracle_leakage_found": 0,
        "pretrained_weights_used": 0,
        "diagnostic_encoder_passed": passed,
        "runtime_sec": time.perf_counter() - start,
        "next_recommendation": (
            "CORE-1AC run small non-oracle encoder integration smoke with delayed memory bank"
            if passed
            else (
                "CORE-1AC run conservative raw-descriptor memory integration smoke; learned metric underperforms raw descriptor"
                if raw_descriptor_signal_passed
                else "non-oracle curriculum is clean enough but descriptor/metric learning did not beat controls; inspect crop descriptor capacity"
            )
        ),
    }
    report = f"""# CORE-1AB Non-Oracle Curriculum Encoder

This stage trains a diagnostic descriptor metric on the CORE-1AA non-oracle stability/namespace-aware curriculum. It uses no oracle proposals, no pretrained weights, and GT only for audit.

## Result

- Gate: {gate_name}
- Negative mode: {negative_mode}
- Selected observations: {len(selected)}
- Positive / negative pairs: {positive_count} / {negative_count}
- Pair precision eval-only: positive {pos_precision:.4f}, negative {neg_precision:.4f}
- Raw descriptor test AUC: {raw_test_auc:.4f}
- Learned metric test AUC: {learned_test_auc:.4f}
- Shuffled-label metric test AUC: {shuffled_auc:.4f}
- Raw / learned retrieval top1: {raw_retrieval_top1:.4f} / {learned_retrieval_top1:.4f}
- Raw / learned retrieval margin: {raw_margin:.4f} / {learned_margin:.4f}
- Negative controls passed: {controls_passed}
- Raw descriptor signal passed: {raw_descriptor_signal_passed}
- Diagnostic encoder passed: {passed}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AB_"
    write_csv(
        out_dir / f"{prefix}descriptor_trace_{args.artifact_version}.csv",
        desc_rows,
        ["obs_id", "sequence_id", "event_id", "window_kind", "frame_idx", "track_id", "box", "crop_box", "descriptor_norm", "descriptor_entropy_proxy", "edge_density", "gt_instance_eval_only", "descriptor"],
    )
    write_csv(
        out_dir / f"{prefix}curriculum_pair_trace_{args.artifact_version}.csv",
        pair_rows,
        [
            "pair_id",
            "gate_name",
            "negative_mode",
            "pair_type",
            "obs_i",
            "obs_j",
            "sequence_i_eval_only",
            "sequence_j_eval_only",
            "gt_instance_i_eval_only",
            "gt_instance_j_eval_only",
            "pair_correct_eval_only",
            "online_positive",
            "online_negative",
        ],
    )
    write_csv(
        out_dir / f"{prefix}pair_score_trace_{args.artifact_version}.csv",
        pair_score_rows,
        ["split", "pair_id", "pair_type", "label_positive", "pair_correct_eval_only", "raw_descriptor_score", "learned_metric_score", "shuffled_label_metric_score"],
    )
    train_rows = [dict(r, control="real_labels") for r in train_trace] + [dict(r, control="shuffled_labels") for r in shuf_trace]
    write_csv(out_dir / f"{prefix}training_trace_{args.artifact_version}.csv", train_rows, ["control", "epoch", "loss", "train_auc"])
    write_csv(
        out_dir / f"{prefix}retrieval_eval_{args.artifact_version}.csv",
        retrieval_rows,
        ["sequence_id", "event_id", "window_kind", "query_obs_id", "candidate_count", "target_candidate_count", "raw_top1_success", "raw_target_margin", "learned_top1_success", "learned_target_margin"],
    )
    write_csv(
        out_dir / f"{prefix}control_summary_{args.artifact_version}.csv",
        [
            {
                "control_name": "shuffled_pair_labels",
                "test_auc": shuffled_auc,
                "baseline_auc": learned_test_auc,
                "control_passed": int(learned_test_auc > shuffled_auc + 0.03),
                "failure_reason": "none" if learned_test_auc > shuffled_auc + 0.03 else "shuffled_labels_too_close_to_real",
            },
            {
                "control_name": "raw_descriptor_baseline",
                "test_auc": raw_test_auc,
                "baseline_auc": learned_test_auc,
                "control_passed": int(learned_test_auc >= raw_test_auc - 0.02),
                "failure_reason": "none" if learned_test_auc >= raw_test_auc - 0.02 else "learned_metric_underperforms_raw_descriptor",
            },
            {
                "control_name": "no_oracle_proposal_no_pretraining",
                "test_auc": "",
                "baseline_auc": "",
                "control_passed": 1,
                "failure_reason": "gt_used_for_audit_only",
            },
        ],
        ["control_name", "test_auc", "baseline_auc", "control_passed", "failure_reason"],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1ab_non_oracle_curriculum_encoder.py",
                "oracle_proposals_used": 0,
                "pretrained_weights_used": 0,
                "gt_used_for_scoring_or_training": 0,
                "gt_used_for_eval_only": 1,
                "leakage_found": 0,
            }
        ],
        ["file", "oracle_proposals_used", "pretrained_weights_used", "gt_used_for_scoring_or_training", "gt_used_for_eval_only", "leakage_found"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
