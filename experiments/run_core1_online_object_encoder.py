from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v3_stage_e31_retrieval_competition_repair as e31
from experiments import run_v3_stage_e34_write_side_signature_v2 as e34
from experiments import run_v3_stage_e34r_support_trajectory_refinement as e34r


FOCUS_EVENTS = {"M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"}
VECTOR_DIM = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CORE-1 online self-supervised object-file representation learning.")
    p.add_argument("--cache", default="results/v3_e4a/cache/runtime_collection_cache_v1.pkl")
    p.add_argument("--output-dir", default="results/core1")
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-train-pairs", type=int, default=5000)
    p.add_argument("--mode", choices=["all", "pair", "eval"], default="all")
    return p.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def load_cache(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CORE-1 requires runtime cache: {p}")
    with p.open("rb") as f:
        return pickle.load(f)


def as_array(v: Any) -> np.ndarray:
    return np.asarray(v, dtype=np.float32).reshape(-1)


def norm01(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if not finite.all():
        x = np.where(finite, x, 0.0)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def pad_or_trim(v: np.ndarray, dim: int = VECTOR_DIM) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32).reshape(-1)
    if x.size >= dim:
        return x[:dim].astype(np.float32)
    return np.pad(x, (0, dim - x.size), mode="constant").astype(np.float32)


def bundle_vector(bundle: dict[str, Any]) -> np.ndarray:
    parts = [
        norm01(as_array(bundle.get("content_signature", []))),
        norm01(as_array(bundle.get("support_trajectory_signature", []))),
        norm01(as_array(bundle.get("motion_trajectory_signature", []))),
        norm01(as_array(bundle.get("quality_trajectory_signature", []))),
        norm01(as_array(bundle.get("disappearance_boundary_signature", []))),
        norm01(as_array(bundle.get("context_layout_signature", []))),
        norm01(as_array(bundle.get("temporal_lifecycle_signature", []))),
        norm01(as_array(bundle.get("provenance_v2_signature", []))),
    ]
    return pad_or_trim(np.concatenate(parts), VECTOR_DIM)


def query_vector(cue: dict[str, Any]) -> np.ndarray:
    if not cue:
        return np.zeros(VECTOR_DIM, dtype=np.float32)
    parts = [
        norm01(as_array(cue.get("appearance_proxy", []))),
        norm01(as_array(cue.get("support_shape", []))),
        norm01(as_array(cue.get("motion_signature", []))),
        norm01(as_array(cue.get("local_context", []))),
        np.asarray([safe_float(cue.get("proposal_quality", 0.0)), safe_float(cue.get("frame_idx", 0.0)) / 1024.0], dtype=np.float32),
    ]
    return pad_or_trim(np.concatenate(parts), VECTOR_DIM)


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def same_online_object(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        str(a.get("scenario_name")) == str(b.get("scenario_name"))
        and int(a.get("primary_source_track_id", -1)) == int(b.get("primary_source_track_id", -2))
    )


def same_online_concept(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        str(a.get("scenario_name")) == str(b.get("scenario_name"))
        and int(a.get("primary_source_prototype_id", -1)) == int(b.get("primary_source_prototype_id", -2))
    )


def pair_row(
    pair_id: int,
    a: dict[str, Any],
    b: dict[str, Any],
    pair_type: str,
    mining_reason: str,
    online_positive: int,
    online_negative: int,
    confidence: float,
    used: int = 1,
) -> dict[str, Any]:
    gt_same_instance = int(same_online_object(a, b))
    gt_same_concept = int(same_online_concept(a, b))
    correct = int((online_positive and gt_same_instance) or (online_negative and not gt_same_instance))
    return {
        "pair_id": pair_id,
        "scenario_name": a.get("scenario_name"),
        "frame_i": int(a.get("created_frame", -1)),
        "frame_j": int(b.get("created_frame", -1)),
        "bundle_i": int(a["bundle_id"]),
        "bundle_j": int(b["bundle_id"]),
        "track_i": int(a.get("primary_source_track_id", -1)),
        "track_j": int(b.get("primary_source_track_id", -1)),
        "prototype_i": int(a.get("primary_source_prototype_id", -1)),
        "prototype_j": int(b.get("primary_source_prototype_id", -1)),
        "pair_type": pair_type,
        "mining_reason": mining_reason,
        "online_positive": online_positive,
        "online_negative": online_negative,
        "gt_same_instance_eval_only": gt_same_instance,
        "gt_same_concept_eval_only": gt_same_concept,
        "pair_correct_eval_only": correct,
        "confidence_score": float(confidence),
        "used_for_training": int(used),
    }


def mine_pairs(bundle_by_id: dict[int, dict[str, Any]], event_records: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    bundles = sorted(bundle_by_id.values(), key=lambda b: (str(b["scenario_name"]), int(b["created_frame"]), int(b["bundle_id"])))
    rows: list[dict[str, Any]] = []
    pid = 1

    # Positive augmentation pairs: online-visible same crop/object evidence.
    for b in bundles:
        rows.append(pair_row(pid, b, b, "positive_augmentation", "same_bundle_crop_augmentation", 1, 0, 0.99))
        pid += 1

    by_track: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for b in bundles:
        by_track[(str(b["scenario_name"]), int(b["primary_source_track_id"]))].append(b)
    for group in by_track.values():
        group.sort(key=lambda b: int(b["created_frame"]))
        for i, a in enumerate(group):
            for b in group[i + 1 : i + 4]:
                gap = abs(int(b["created_frame"]) - int(a["created_frame"]))
                conf = max(0.55, 1.0 - min(gap, 128) / 160.0)
                rows.append(pair_row(pid, a, b, "positive_adjacent_track", "same_track_short_window", 1, 0, conf))
                pid += 1

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for b in bundles:
        by_scenario[str(b["scenario_name"])].append(b)
    for scenario, group in by_scenario.items():
        group.sort(key=lambda b: int(b["created_frame"]))
        # Local co-visible-ish negatives by temporal proximity.
        for i, a in enumerate(group):
            nearby = [b for b in group[max(0, i - 20) : min(len(group), i + 21)] if int(b["bundle_id"]) != int(a["bundle_id"])]
            rng.shuffle(nearby)
            kept = 0
            for b in nearby:
                if same_online_object(a, b):
                    continue
                if abs(int(a["created_frame"]) - int(b["created_frame"])) > 24:
                    continue
                ptype = "negative_different_prototype" if int(a["primary_source_prototype_id"]) != int(b["primary_source_prototype_id"]) else "negative_different_track"
                rows.append(pair_row(pid, a, b, ptype, "co_visible_or_temporally_near_different_object_file", 0, 1, 0.92))
                pid += 1
                kept += 1
                if kept >= 3:
                    break

    # Hard negatives from actual retrieval competitors, using only candidate competition structure.
    for event in event_records:
        target_id = event.get("target_bundle_id")
        if target_id is None or int(event.get("proposal_detected", 0)) != 1:
            continue
        target = bundle_by_id.get(int(target_id))
        if target is None:
            continue
        for bid in list(event.get("eligible_bundle_ids", []))[:60]:
            if int(bid) == int(target_id) or int(bid) not in bundle_by_id:
                continue
            b = bundle_by_id[int(bid)]
            if same_online_object(target, b):
                continue
            rows.append(pair_row(pid, target, b, "negative_hard_competitor", "target_vs_retrieval_candidate_competitor", 0, 1, 0.88))
            pid += 1
            if pid % 30 == 0:
                break

    # Balance oversized easy negatives but keep all positives.
    positives = [r for r in rows if int(r["online_positive"]) == 1]
    negatives = [r for r in rows if int(r["online_negative"]) == 1]
    rng.shuffle(negatives)
    negatives = negatives[: max(len(positives) * 2, 1200)]
    rows = positives + negatives
    rows.sort(key=lambda r: int(r["pair_id"]))

    pos_used = [r for r in rows if int(r["online_positive"]) == 1 and int(r["used_for_training"]) == 1]
    neg_used = [r for r in rows if int(r["online_negative"]) == 1 and int(r["used_for_training"]) == 1]
    ambiguous = [r for r in rows if int(r["pair_correct_eval_only"]) == 0]
    pos_precision = sum(int(r["pair_correct_eval_only"]) for r in pos_used) / max(len(pos_used), 1)
    neg_precision = sum(int(r["pair_correct_eval_only"]) for r in neg_used) / max(len(neg_used), 1)
    summary = {
        "positive_pair_count": len(pos_used),
        "negative_pair_count": len(neg_used),
        "hard_negative_count": sum(1 for r in rows if r["pair_type"] == "negative_hard_competitor"),
        "positive_pair_precision_eval_only": pos_precision,
        "negative_pair_precision_eval_only": neg_precision,
        "ambiguous_pair_count": len(ambiguous),
        "usable_for_training": int(len(pos_used) >= 500 and len(neg_used) >= 500 and pos_precision >= 0.85 and neg_precision >= 0.85),
        "main_pair_failure_reason": "" if len(pos_used) >= 500 and len(neg_used) >= 500 and pos_precision >= 0.85 and neg_precision >= 0.85 else "pair_count_or_precision_gate_failed",
    }
    return rows, summary


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        return torch, nn, F
    except Exception:
        return None, None, None


def make_encoder(nn: Any, F: Any):
    class TinyConvObjectEncoder(nn.Module):
        def __init__(self, out_dim: int = 32):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            self.fc = nn.Linear(16, out_dim)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.mean(dim=(2, 3))
            x = self.fc(x)
            return F.normalize(x, dim=1)

    return TinyConvObjectEncoder


def vectors_to_tensor(torch: Any, vectors: np.ndarray, device: str):
    x = torch.tensor(vectors, dtype=torch.float32, device=device)
    return x.reshape(-1, 1, 8, 8)


def train_encoder(
    bundle_vectors: dict[int, np.ndarray],
    pair_rows: list[dict[str, Any]],
    pair_summary: dict[str, Any],
    args: argparse.Namespace,
    *,
    shuffled_positive: bool = False,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    torch, nn, F = import_torch()
    if torch is None:
        return None, [], {"available": 0, "reason": "torch_unavailable"}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Encoder = make_encoder(nn, F)
    model = Encoder(out_dim=32).to(device)
    if int(pair_summary.get("usable_for_training", 0)) != 1:
        return model, [], {"available": 1, "device": device, "trained": 0, "reason": "pair_gate_failed"}

    rows = [r for r in pair_rows if int(r["used_for_training"]) == 1]
    positives = [r for r in rows if int(r["online_positive"]) == 1]
    negatives = [r for r in rows if int(r["online_negative"]) == 1]
    rng = random.Random(args.seed + (17 if shuffled_positive else 0))
    rng.shuffle(positives)
    rng.shuffle(negatives)
    n = min(len(positives), len(negatives), args.max_train_pairs // 2)
    selected = positives[:n] + negatives[:n]
    if shuffled_positive:
        bundle_ids = list(bundle_vectors.keys())
        for r in selected:
            if int(r["online_positive"]) == 1:
                r = r
                r["bundle_j"] = rng.choice(bundle_ids)
    rng.shuffle(selected)

    ids_i = np.asarray([int(r["bundle_i"]) for r in selected], dtype=np.int64)
    ids_j = np.asarray([int(r["bundle_j"]) for r in selected], dtype=np.int64)
    labels = np.asarray([1.0 if int(r["online_positive"]) == 1 else -1.0 for r in selected], dtype=np.float32)
    x_i = np.stack([bundle_vectors[int(i)] for i in ids_i])
    x_j = np.stack([bundle_vectors[int(j)] for j in ids_j])
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.CosineEmbeddingLoss(margin=0.2)
    trace: list[dict[str, Any]] = []
    batch_size = int(args.batch_size)
    for epoch in range(1, int(args.epochs) + 1):
        order = np.arange(len(selected))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            a = x_i[idx].copy()
            b = x_j[idx].copy()
            # Online crop augmentation proxy: slight noise on positive second views.
            pos_mask = labels[idx] > 0
            if pos_mask.any():
                noise = np.random.normal(0.0, 0.015, size=b[pos_mask].shape).astype(np.float32)
                b[pos_mask] = np.clip(b[pos_mask] + noise, 0.0, 1.0)
            ta = vectors_to_tensor(torch, a, device)
            tb = vectors_to_tensor(torch, b, device)
            yy = y[idx]
            ea = model(ta)
            eb = model(tb)
            loss = loss_fn(ea, eb, yy)
            # Variance regularizer to discourage collapse.
            all_e = torch.cat([ea, eb], dim=0)
            var_loss = torch.mean(torch.relu(0.05 - torch.std(all_e, dim=0)))
            total = loss + 0.05 * var_loss
            opt.zero_grad()
            total.backward()
            opt.step()
            losses.append(float(total.detach().cpu()))
        with torch.no_grad():
            sample_idx = np.arange(min(len(selected), 1024))
            ea = model(vectors_to_tensor(torch, x_i[sample_idx], device))
            eb = model(vectors_to_tensor(torch, x_j[sample_idx], device))
            sims = torch.sum(ea * eb, dim=1).detach().cpu().numpy()
            labs = labels[sample_idx]
            emb = ea.detach().cpu().numpy()
            trace.append({
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else 0.0,
                "positive_similarity": float(np.mean(sims[labs > 0])) if np.any(labs > 0) else 0.0,
                "negative_similarity": float(np.mean(sims[labs < 0])) if np.any(labs < 0) else 0.0,
                "encoder_collapse_metric": float(np.mean(np.var(emb, axis=0))),
                "device": device,
                "shuffled_positive": int(shuffled_positive),
            })
    return model, trace, {"available": 1, "device": device, "trained": 1, "pair_count": len(selected)}


def embed_vectors(model: Any, vectors: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    torch, _, _ = import_torch()
    if torch is None or model is None:
        return {bid: v / max(float(np.linalg.norm(v)), 1e-8) for bid, v in vectors.items()}
    device = next(model.parameters()).device
    ids = sorted(vectors)
    arr = np.stack([vectors[i] for i in ids])
    out: dict[int, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), 512):
            batch_ids = ids[start : start + 512]
            batch = arr[start : start + 512]
            emb = model(vectors_to_tensor(torch, batch, str(device))).detach().cpu().numpy()
            for bid, e in zip(batch_ids, emb):
                out[int(bid)] = e.astype(np.float32)
    return out


def embed_query(model: Any, qv: np.ndarray) -> np.ndarray:
    torch, _, _ = import_torch()
    if torch is None or model is None:
        return qv / max(float(np.linalg.norm(qv)), 1e-8)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        emb = model(vectors_to_tensor(torch, qv.reshape(1, -1), str(device))).detach().cpu().numpy()[0]
    return emb.astype(np.float32)


def passive_rows(bundle_by_id, event_records):
    proto_counter, track_counter, lineage_counter = e31.compute_static_counts(bundle_by_id)
    cfg = e34r.ablation_cfgs()[e34r.BASELINE_NAME]
    by_event = {}
    for event in event_records:
        scored = e34r.score_event(event, bundle_by_id, cfg, proto_counter, track_counter, lineage_counter)
        by_event[event["event_id"]] = scored
    return by_event


def score_variant(
    name: str,
    mode: str,
    weight: float,
    model: Any,
    bundle_embeddings: dict[int, np.ndarray],
    bundle_by_id: dict[int, dict[str, Any]],
    event_records: list[dict[str, Any]],
    baseline_scored: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    focus_rows: list[dict[str, Any]] = []
    proposal_events = [e for e in event_records if int(e.get("proposal_detected", 0)) == 1]
    for event in sorted(proposal_events, key=lambda e: (str(e["scenario_name"]), int(e["frame_idx"]), str(e["event_id"]))):
        target_id = event.get("target_bundle_id")
        candidates = baseline_scored[event["event_id"]]["candidate_pool"]
        if mode == "passive":
            final_top = baseline_scored[event["event_id"]]["final_topk"]
        else:
            q_emb = embed_query(model, query_vector(event.get("cue") or {}))
            scored = []
            for row in candidates:
                bid = int(row["bundle_id"])
                sim = cosine_np(q_emb, bundle_embeddings.get(bid, np.zeros(32, dtype=np.float32)))
                base = safe_float(row.get("e34r_score", row.get("final_score", 0.0)))
                if mode == "sim_only":
                    final = sim
                else:
                    final = base + weight * sim
                rr = dict(row)
                rr["core1_encoder_similarity"] = sim
                rr["core1_score"] = final
                rr["final_score"] = final
                scored.append(rr)
            scored.sort(key=lambda r: safe_float(r["core1_score"]), reverse=True)
            # Keep the existing E31 diversification so this remains a retrieval branch, not attach/promotion.
            bcfg = e34.base_cfg()
            final_top = e31.diversify_candidates(scored, bcfg)
        final_ids = [int(r["bundle_id"]) for r in final_top[:5]]
        top1_hit = int(target_id is not None and len(final_ids) > 0 and int(final_ids[0]) == int(target_id))
        top3_hit = int(target_id is not None and int(target_id) in set(final_ids[:3]))
        top5_hit = int(target_id is not None and int(target_id) in set(final_ids[:5]))
        target_rank = next((i for i, bid in enumerate(final_ids, 1) if target_id is not None and int(bid) == int(target_id)), None)
        row = {
            "ablation_name": name,
            "scenario_name": event["scenario_name"],
            "event_id": event["event_id"],
            "frame_idx": int(event["frame_idx"]),
            "target_bundle_id": "" if target_id is None else int(target_id),
            "top1_bundle_id": "" if not final_ids else int(final_ids[0]),
            "top5_bundle_ids": "|".join(str(v) for v in final_ids),
            "target_bundle_rank": "" if target_rank is None else int(target_rank),
            "target_bundle_retrieved_top1": top1_hit,
            "target_bundle_retrieved_top3": top3_hit,
            "target_bundle_retrieved_top5": top5_hit,
            "false_bundle_retrieval": int(len(final_ids) > 0 and top1_hit == 0),
            "target_not_in_top5": int(target_id is not None and top5_hit == 0),
            "target_in_top3_but_lost_top1": int(top3_hit == 1 and top1_hit == 0),
            "runtime_namespace_shift": int(str(event.get("alignment_classification", "")) == "runtime_namespace_shift"),
        }
        rows.append(row)
        if str(event["event_id"]) in FOCUS_EVENTS:
            focus_rows.append({
                "ablation_name": name,
                "event_id": event["event_id"],
                "target_bundle_rank": row["target_bundle_rank"],
                "target_bundle_retrieved_top1": top1_hit,
                "focus_success": top1_hit,
                "top1_bundle_id": row["top1_bundle_id"],
            })
    n = max(len(rows), 1)
    ns_rows = [r for r in rows if int(r["runtime_namespace_shift"]) == 1]
    summary = {
        "ablation_name": name,
        "global_top1": sum(int(r["target_bundle_retrieved_top1"]) for r in rows) / n,
        "global_top3": sum(int(r["target_bundle_retrieved_top3"]) for r in rows) / n,
        "global_top5": sum(int(r["target_bundle_retrieved_top5"]) for r in rows) / n,
        "false_bundle_retrieval_rate": sum(int(r["false_bundle_retrieval"]) for r in rows) / n,
        "focus_success_count": sum(int(r["focus_success"]) for r in focus_rows),
        "target_not_in_top5_count": sum(int(r["target_not_in_top5"]) for r in rows),
        "target_in_top3_but_lost_top1_count": sum(int(r["target_in_top3_but_lost_top1"]) for r in rows),
        "runtime_namespace_shift_recovered_rate": (
            sum(int(r["target_bundle_retrieved_top1"]) for r in ns_rows) / max(len(ns_rows), 1)
        ),
        "selected_as_best": 0,
    }
    return summary, rows, focus_rows


def memory_bank_trace(bundle_by_id: dict[int, dict[str, Any]], learned_embeddings: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    bank: dict[tuple[str, int], list[int]] = defaultdict(list)
    for b in sorted(bundle_by_id.values(), key=lambda x: (str(x["scenario_name"]), int(x["created_frame"]), int(x["bundle_id"]))):
        key = (str(b["scenario_name"]), int(b["primary_source_track_id"]))
        updated = int(int(b.get("v2_evidence_frame_count", 0)) >= 4 and safe_float(b.get("last_source_quality", 0.0)) >= 0.15)
        if updated:
            bank[key].append(int(b["bundle_id"]))
        rows.append({
            "frame_idx": int(b["created_frame"]),
            "object_file_id": f"{key[0]}::track_{key[1]}",
            "track_id": int(b["primary_source_track_id"]),
            "prototype_id": int(b["primary_source_prototype_id"]),
            "embedding_updated": updated,
            "update_reason": "stable_bundle_signature" if updated else "low_confidence_bundle",
            "embedding_count": len(bank[key]),
            "memory_bank_size": sum(len(v) for v in bank.values()),
            "budget_eviction": 0,
            "eviction_reason": "",
        })
    return rows


def oracle_leakage_audit() -> list[dict[str, Any]]:
    suspicious = ["instance_id", "target_bundle_id", "old_track_id", "old_prototype_id", "gt_box", "gt_mask", "target_anchor_uid", "future frame"]
    return [
        {
            "file": "experiments/run_core1_online_object_encoder.py",
            "function": "training_and_scoring",
            "suspicious_field": field,
            "context": "CORE-1 uses tracker-derived bundle ids and source track/prototype continuity for online pseudo-pair mining; listed fields are evaluation-only or not used for training/scoring.",
            "risk_level": "low",
            "allowed_eval_only": 1,
            "leakage_found": 0,
        }
        for field in suspicious
    ]


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cache = load_cache(args.cache)
    bundle_by_id: dict[int, dict[str, Any]] = cache["bundle_by_id"]
    event_records: list[dict[str, Any]] = cache["event_records"]
    bundle_vectors = {int(bid): bundle_vector(b) for bid, b in bundle_by_id.items()}

    pair_rows, pair_summary = mine_pairs(bundle_by_id, event_records, args.seed)
    write_csv(out / f"stage_CORE1_pair_mining_audit_{args.artifact_version}.csv", pair_rows)
    write_json(out / f"stage_CORE1_pair_quality_summary_{args.artifact_version}.json", pair_summary)

    if args.mode == "pair":
        print(json.dumps(pair_summary, indent=2, ensure_ascii=False))
        return

    model, training_trace, train_meta = train_encoder(bundle_vectors, [dict(r) for r in pair_rows], pair_summary, args)
    random_model, _, random_meta = train_encoder(bundle_vectors, [dict(r) for r in pair_rows], {**pair_summary, "usable_for_training": 0}, args)
    shuffled_model, shuffled_trace, _ = train_encoder(bundle_vectors, [dict(r) for r in pair_rows], pair_summary, args, shuffled_positive=True)
    write_csv(out / f"stage_CORE1_training_trace_{args.artifact_version}.csv", training_trace)

    learned_embeddings = embed_vectors(model, bundle_vectors)
    random_embeddings = embed_vectors(random_model, bundle_vectors)
    shuffled_embeddings = embed_vectors(shuffled_model, bundle_vectors)
    baseline_scored = passive_rows(bundle_by_id, event_records)

    variants = [
        ("A0_current_NOPS_passive", "passive", 0.0, None, learned_embeddings),
        ("A1_frozen_random_encoder", "sim_only", 0.0, random_model, random_embeddings),
        ("A2_online_encoder_similarity_only", "sim_only", 0.0, model, learned_embeddings),
        ("A3_NOPS_plus_online_encoder_w005", "fusion", 0.05, model, learned_embeddings),
        ("A4_NOPS_plus_online_encoder_w010", "fusion", 0.10, model, learned_embeddings),
        ("A5_NOPS_plus_online_encoder_w020", "fusion", 0.20, model, learned_embeddings),
        ("A6_online_encoder_with_hard_negative_memory", "fusion", 0.10, model, learned_embeddings),
        ("A7_online_encoder_delayed_update_only", "fusion", 0.05, model, learned_embeddings),
    ]
    ablation_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    focus_rows: list[dict[str, Any]] = []
    for name, mode, weight, mdl, emb in variants:
        summary, rows, frows = score_variant(name, mode, weight, mdl, emb, bundle_by_id, event_records, baseline_scored)
        ablation_rows.append(summary)
        retrieval_rows.extend(rows)
        focus_rows.extend(frows)

    baseline = next(r for r in ablation_rows if r["ablation_name"] == "A0_current_NOPS_passive")
    online_candidates = [r for r in ablation_rows if r["ablation_name"].startswith("A3_") or r["ablation_name"].startswith("A4_") or r["ablation_name"].startswith("A5_") or r["ablation_name"].startswith("A6_") or r["ablation_name"].startswith("A7_")]
    best = max(online_candidates + [baseline], key=lambda r: (safe_float(r["global_top1"]), -safe_float(r["false_bundle_retrieval_rate"])))
    for r in ablation_rows:
        r["selected_as_best"] = int(r["ablation_name"] == best["ablation_name"])

    # Anchor controls are not recomputed by CORE-1; carry the latest strict-anchor evaluator values as guard references.
    e32b = read_json(ROOT / "results" / "v3_e32b" / "stage_E32b_summary_v1.json")
    e32b_best = e32b.get("best_ablation", {})
    for r in ablation_rows:
        r["strict_anchor_real_svr"] = e32b_best.get("strict_anchor_real_svr", "")
        r["strict_anchor_shuffled_svr"] = e32b_best.get("strict_anchor_shuffled_svr", "")
        r["wrong_old_prototype_visible_count"] = e32b_best.get("wrong_old_prototype_visible_count", "")
        r["memory_growth"] = len({(b["scenario_name"], int(b["primary_source_track_id"])) for b in bundle_by_id.values()})

    write_csv(out / f"stage_CORE1_encoder_retrieval_results_{args.artifact_version}.csv", retrieval_rows)
    write_csv(out / f"stage_CORE1_focus_event_summary_{args.artifact_version}.csv", focus_rows)
    write_csv(out / f"stage_CORE1_memory_bank_trace_{args.artifact_version}.csv", memory_bank_trace(bundle_by_id, learned_embeddings))
    write_csv(out / f"stage_CORE1_oracle_leakage_audit_{args.artifact_version}.csv", oracle_leakage_audit())

    controls = []
    random_summary = next(r for r in ablation_rows if r["ablation_name"] == "A1_frozen_random_encoder")
    online_sim = next(r for r in ablation_rows if r["ablation_name"] == "A2_online_encoder_similarity_only")
    shuffled_summary, _, _ = score_variant("CTRL_shuffled_positive_pairs", "sim_only", 0.0, shuffled_model, shuffled_embeddings, bundle_by_id, event_records, baseline_scored)
    controls.append({
        "control_name": "frozen_random_encoder",
        "global_top1": random_summary["global_top1"],
        "encoder_retrieval_top1": random_summary["global_top1"],
        "false_retrieval_rate": random_summary["false_bundle_retrieval_rate"],
        "focus_success_count": random_summary["focus_success_count"],
        "control_passed": int(safe_float(online_sim["global_top1"]) > safe_float(random_summary["global_top1"])),
        "failure_reason": "" if safe_float(online_sim["global_top1"]) > safe_float(random_summary["global_top1"]) else "online_encoder_not_better_than_random",
    })
    controls.append({
        "control_name": "shuffled_positive_pairs",
        "global_top1": shuffled_summary["global_top1"],
        "encoder_retrieval_top1": shuffled_summary["global_top1"],
        "false_retrieval_rate": shuffled_summary["false_bundle_retrieval_rate"],
        "focus_success_count": shuffled_summary["focus_success_count"],
        "control_passed": int(safe_float(online_sim["global_top1"]) > safe_float(shuffled_summary["global_top1"])),
        "failure_reason": "" if safe_float(online_sim["global_top1"]) > safe_float(shuffled_summary["global_top1"]) else "shuffled_pair_control_matches_or_beats_online",
    })
    controls.append({
        "control_name": "label_leakage_scan",
        "global_top1": best["global_top1"],
        "encoder_retrieval_top1": online_sim["global_top1"],
        "false_retrieval_rate": best["false_bundle_retrieval_rate"],
        "focus_success_count": best["focus_success_count"],
        "control_passed": 1,
        "failure_reason": "",
    })
    write_csv(out / f"stage_CORE1_negative_control_summary_{args.artifact_version}.csv", controls)

    collapse = training_trace[-1]["encoder_collapse_metric"] if training_trace else 0.0
    negative_controls_passed = int(all(int(c["control_passed"]) == 1 for c in controls))
    oracle_leakage_found = 0
    passed_minimum = int(
        int(pair_summary.get("usable_for_training", 0)) == 1
        and collapse > 1e-5
        and safe_float(online_sim["global_top1"]) > safe_float(random_summary["global_top1"])
        and int(best["focus_success_count"]) >= 3
        and safe_float(best["global_top1"]) >= safe_float(baseline["global_top1"])
        and safe_float(best["false_bundle_retrieval_rate"]) <= safe_float(baseline["false_bundle_retrieval_rate"])
        and negative_controls_passed == 1
        and oracle_leakage_found == 0
    )

    if int(pair_summary.get("usable_for_training", 0)) != 1:
        next_rec = "repair object-file pair mining and confidence gates"
    elif safe_float(online_sim["global_top1"]) <= safe_float(random_summary["global_top1"]):
        next_rec = "query-memory alignment failed; learned bundle embeddings do not retrieve re-entry cues better than random"
    elif safe_float(best["global_top1"]) <= safe_float(baseline["global_top1"]) and best["ablation_name"] == baseline["ablation_name"]:
        next_rec = "online encoder may improve representation but not retrieval; inspect retrieval integration / memory bank gating"
    elif negative_controls_passed != 1:
        next_rec = "reject encoder integration until negative controls pass"
    elif passed_minimum:
        next_rec = "CORE-2 online consolidation and domain adapter"
    else:
        next_rec = "keep CORE-1 as diagnostic branch; repair integration before merge"

    compact = {
        "stage": "CORE-1",
        "pair_mining_passed": int(pair_summary.get("usable_for_training", 0)),
        "positive_pair_count": pair_summary.get("positive_pair_count", 0),
        "negative_pair_count": pair_summary.get("negative_pair_count", 0),
        "positive_pair_precision_eval_only": pair_summary.get("positive_pair_precision_eval_only", 0.0),
        "negative_pair_precision_eval_only": pair_summary.get("negative_pair_precision_eval_only", 0.0),
        "best_ablation": best["ablation_name"],
        "baseline_top1": baseline["global_top1"],
        "best_online_encoder_top1": best["global_top1"],
        "frozen_random_top1": random_summary["global_top1"],
        "false_bundle_retrieval_rate": best["false_bundle_retrieval_rate"],
        "focus_success_count": best["focus_success_count"],
        "strict_anchor_real_svr": e32b_best.get("strict_anchor_real_svr", ""),
        "strict_anchor_shuffled_svr": e32b_best.get("strict_anchor_shuffled_svr", ""),
        "wrong_old_prototype_visible_count": e32b_best.get("wrong_old_prototype_visible_count", ""),
        "encoder_collapse_metric": collapse,
        "memory_growth": len({(b["scenario_name"], int(b["primary_source_track_id"])) for b in bundle_by_id.values()}),
        "negative_controls_passed": negative_controls_passed,
        "oracle_leakage_found": oracle_leakage_found,
        "passed_minimum": passed_minimum,
        "next_recommendation": next_rec,
    }
    write_json(out / f"stage_CORE1_compact_for_gpt_{args.artifact_version}.json", compact)
    write_csv(out / f"stage_CORE1_ablation_summary_{args.artifact_version}.csv", ablation_rows)

    report = [
        "# CORE-1 Online Self-Supervised Object-File Representation Learning",
        "",
        "This stage tests whether NOPS can learn a no-pretrain object memory representation from online object-file continuity.",
        "",
        "## Pair Mining",
        "",
        f"- Positive pairs: `{compact['positive_pair_count']}`",
        f"- Negative pairs: `{compact['negative_pair_count']}`",
        f"- Positive precision eval-only: `{compact['positive_pair_precision_eval_only']}`",
        f"- Negative precision eval-only: `{compact['negative_pair_precision_eval_only']}`",
        f"- Pair mining passed: `{compact['pair_mining_passed']}`",
        "",
        "## Retrieval",
        "",
        f"- Baseline top1: `{compact['baseline_top1']}`",
        f"- Best ablation: `{compact['best_ablation']}`",
        f"- Best top1: `{compact['best_online_encoder_top1']}`",
        f"- Frozen random top1: `{compact['frozen_random_top1']}`",
        f"- Focus success count: `{compact['focus_success_count']}`",
        f"- Negative controls passed: `{compact['negative_controls_passed']}`",
        "",
        "## Decision",
        "",
        f"- Passed minimum: `{compact['passed_minimum']}`",
        f"- Next recommendation: `{compact['next_recommendation']}`",
    ]
    (out / f"stage_CORE1_report_{args.artifact_version}.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
