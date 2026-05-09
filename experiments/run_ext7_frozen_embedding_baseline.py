from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ext1_utils import write_csv
from experiments.run_ext4a_full_pixel_appearance_validation import (
    as_float,
    as_int,
    build_history,
    candidate_components,
    clamp_crop,
    image_path,
    load_pixel_ready_events,
    score_variant,
    state_at,
    zscores,
)
from experiments.run_ext5_multicategory_full_pixel_validation import category_from_seq


VARIANTS = [
    "A0_geometry_passive",
    "A1_external_trajectory_branch",
    "A2_embedding_nn",
    "A3_geometry_plus_embedding_w005",
    "A4_geometry_plus_embedding_w010",
    "A5_geometry_plus_embedding_w020",
    "A6_external_branch_plus_embedding_w005",
    "A7_external_branch_plus_embedding_w010",
    "A8_external_branch_plus_embedding_w020",
]

CONTROL_MODES = [
    "real",
    "within_event_shuffled_embedding",
    "category_shuffled_embedding",
    "random_vector_control",
    "wrong_candidate_binding_control",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXT-7 frozen visual embedding external baseline.")
    p.add_argument("--lagot-root", default="data/external/lagot_annotations")
    p.add_argument("--lasot-root", default="data/external/lasot")
    p.add_argument("--ext4-dir", default="results/ext4")
    p.add_argument("--output-dir", default="results/ext7")
    p.add_argument("--embedding-model", default="resnet18", choices=["resnet18", "resnet50", "clip", "dinov2"])
    p.add_argument("--bootstrap-samples", type=int, default=300)
    p.add_argument("--device", default="auto")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stable_seed(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


class FrozenEmbedder:
    def __init__(self, model_name: str, device_arg: str = "auto") -> None:
        self.model_name = model_name
        self.available = False
        self.unavailable_reason = ""
        self.device_name = "cpu"
        self.dim = 0
        self.model = None
        self.transform = None
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
        except Exception as exc:  # pragma: no cover - environment guard
            self.unavailable_reason = f"torch_or_torchvision_unavailable:{exc!r}"
            return
        if model_name in {"clip", "dinov2"}:
            self.unavailable_reason = f"{model_name}_not_configured_in_local_ext7_runner"
            return
        if device_arg == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_arg
        try:
            if model_name == "resnet18":
                weights = models.ResNet18_Weights.DEFAULT
                base = models.resnet18(weights=weights)
            elif model_name == "resnet50":
                weights = models.ResNet50_Weights.DEFAULT
                base = models.resnet50(weights=weights)
            else:
                self.unavailable_reason = f"unsupported_model:{model_name}"
                return
            self.transform = weights.transforms()
            self.model = nn.Sequential(*list(base.children())[:-1]).to(device).eval()
            self.device_name = device
            self.available = True
        except Exception as exc:
            self.unavailable_reason = f"model_load_failed:{exc!r}"

    def embed_crop(self, path: Path, box: tuple[float, float, float, float]) -> np.ndarray | None:
        if not self.available or self.model is None or self.transform is None:
            return None
        try:
            import torch
            with Image.open(path) as img:
                img = img.convert("RGB")
                crop = img.crop(clamp_crop(box, img.width, img.height))
            tensor = self.transform(crop).unsqueeze(0).to(self.device_name)
            with torch.no_grad():
                feat = self.model(tensor).flatten(1).detach().cpu().numpy()[0].astype(np.float32)
        except Exception:
            return None
        norm = float(np.linalg.norm(feat))
        if not np.isfinite(norm) or norm <= 1e-8:
            return None
        self.dim = int(feat.shape[0])
        return feat / norm


def cosine_np(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return -1.0
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8))


def score_embedding_variant(variant: str, comps: dict[str, dict[str, Any]], emb_scores: dict[str, float]) -> dict[str, float]:
    geom = {iid: float(c["nops_geometry_score"]) for iid, c in comps.items()}
    ext = {iid: float(c["external_trajectory_heavy_score"]) for iid, c in comps.items()}
    z_geom = zscores(geom)
    z_ext = zscores(ext)
    z_emb = zscores(emb_scores)
    if variant == "A0_geometry_passive":
        return geom
    if variant == "A1_external_trajectory_branch":
        return ext
    if variant == "A2_embedding_nn":
        return emb_scores
    if variant == "A3_geometry_plus_embedding_w005":
        return {iid: z_geom.get(iid, 0.0) + 0.05 * z_emb.get(iid, 0.0) for iid in comps}
    if variant == "A4_geometry_plus_embedding_w010":
        return {iid: z_geom.get(iid, 0.0) + 0.10 * z_emb.get(iid, 0.0) for iid in comps}
    if variant == "A5_geometry_plus_embedding_w020":
        return {iid: z_geom.get(iid, 0.0) + 0.20 * z_emb.get(iid, 0.0) for iid in comps}
    if variant == "A6_external_branch_plus_embedding_w005":
        return {iid: z_ext.get(iid, 0.0) + 0.05 * z_emb.get(iid, 0.0) for iid in comps}
    if variant == "A7_external_branch_plus_embedding_w010":
        return {iid: z_ext.get(iid, 0.0) + 0.10 * z_emb.get(iid, 0.0) for iid in comps}
    if variant == "A8_external_branch_plus_embedding_w020":
        return {iid: z_ext.get(iid, 0.0) + 0.20 * z_emb.get(iid, 0.0) for iid in comps}
    raise ValueError(variant)


def collect_payloads(args: argparse.Namespace, embedder: FrozenEmbedder) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lagot, event_rows, _ = load_pixel_ready_events(args)
    hist_cache: dict[str, dict[str, list[tuple[int, tuple[float, float, float, float], Any]]]] = {}
    emb_cache: dict[tuple[str, tuple[float, float, float, float]], np.ndarray | None] = {}
    payloads: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for row in event_rows:
        seq = row["sequence_id"]
        seq_dir = Path(row["seq_dir"])
        if seq not in hist_cache:
            hist_cache[seq] = build_history(lagot, seq)
        hist = hist_cache[seq]
        target_id = row["instance_id"]
        reappear = int(row["reappear_frame"])
        target_state = state_at(hist, target_id, reappear)
        if target_state is None:
            continue
        img_path = image_path(seq_dir, reappear)
        if not img_path.exists():
            continue
        with Image.open(img_path) as img:
            width, height = img.width, img.height
        query_box = target_state[1]
        q_key = (str(img_path), tuple(float(x) for x in query_box))
        if q_key not in emb_cache:
            emb_cache[q_key] = embedder.embed_crop(img_path, query_box)
        query_emb = emb_cache[q_key]
        quality_rows.append(quality_row(row, "query", target_id, embedder, img_path, query_box, query_emb))
        if query_emb is None:
            continue
        candidates = {iid: states for iid, states in hist.items() if any(st[0] < reappear for st in states)}
        comps: dict[str, dict[str, Any]] = {}
        emb_scores: dict[str, float] = {}
        for iid, states in candidates.items():
            comp = candidate_components(query_box, states, float(width), float(height), reappear)
            if not comp:
                continue
            last_frame = int(comp["last_frame"])
            last_box = comp["last_box"]
            last_path = image_path(seq_dir, last_frame)
            c_key = (str(last_path), tuple(float(x) for x in last_box))
            if c_key not in emb_cache:
                emb_cache[c_key] = embedder.embed_crop(last_path, last_box)
            cand_emb = emb_cache[c_key]
            quality_rows.append(quality_row(row, "candidate", iid, embedder, last_path, last_box, cand_emb))
            if cand_emb is None:
                continue
            comps[iid] = comp
            emb_scores[iid] = cosine_np(query_emb, cand_emb)
        if target_id not in comps:
            continue
        payloads.append({
            "event_id": row["event_id"],
            "sequence_id": seq,
            "category": category_from_seq(seq),
            "target_id": target_id,
            "gap_length": row.get("gap_length", ""),
            "candidate_count": len(comps),
            "comps": comps,
            "embedding_scores": emb_scores,
        })
    return payloads, quality_rows


def quality_row(
    event_row: dict[str, Any],
    role: str,
    candidate_id: str,
    embedder: FrozenEmbedder,
    path: Path,
    box: tuple[float, float, float, float],
    emb: np.ndarray | None,
) -> dict[str, Any]:
    norm = float(np.linalg.norm(emb)) if emb is not None else 0.0
    return {
        "event_id": event_row.get("event_id", ""),
        "category": category_from_seq(event_row.get("sequence_id", "")),
        "sequence_id": event_row.get("sequence_id", ""),
        "crop_role": role,
        "candidate_id_eval_only": candidate_id,
        "embedding_model": embedder.model_name,
        "embedding_available": int(emb is not None),
        "embedding_norm": norm,
        "embedding_dim": int(emb.shape[0]) if emb is not None else 0,
        "embedding_degenerate": int(emb is None or not np.isfinite(norm) or norm <= 1e-8),
        "crop_path": str(path),
        "crop_box": json.dumps([float(x) for x in box]),
        "missing_reason": "" if emb is not None else "embedding_unavailable_or_crop_failed",
    }


def shuffled_scores(scores: dict[str, float], seed_text: str) -> dict[str, float]:
    ids = list(scores.keys())
    vals = list(scores.values())
    rng = random.Random(stable_seed(seed_text))
    rng.shuffle(vals)
    return dict(zip(ids, vals))


def category_shuffled(payloads: list[dict[str, Any]], payload: dict[str, Any], idx: int) -> dict[str, float]:
    ids = list(payload["embedding_scores"].keys())
    n = len(payloads)
    source = None
    for off in range(1, n + 1):
        cand = payloads[(idx + off * 23) % n]
        if cand["category"] != payload["category"] and cand["embedding_scores"]:
            source = cand
            break
    if source is None:
        return shuffled_scores(payload["embedding_scores"], payload["event_id"] + ":category")
    vals = list(source["embedding_scores"].values())
    rng = random.Random(stable_seed(payload["event_id"] + ":category"))
    rng.shuffle(vals)
    return {iid: vals[i % len(vals)] for i, iid in enumerate(ids)}


def random_vector_scores(payload: dict[str, Any]) -> dict[str, float]:
    return {
        iid: random.Random(stable_seed(payload["event_id"] + ":" + iid + ":random_vector")).uniform(-1.0, 1.0)
        for iid in payload["embedding_scores"]
    }


def control_scores(payloads: list[dict[str, Any]], payload: dict[str, Any], idx: int, mode: str) -> dict[str, float]:
    if mode == "real":
        return payload["embedding_scores"]
    if mode == "within_event_shuffled_embedding":
        return shuffled_scores(payload["embedding_scores"], payload["event_id"] + ":within")
    if mode == "category_shuffled_embedding":
        return category_shuffled(payloads, payload, idx)
    if mode == "random_vector_control":
        return random_vector_scores(payload)
    if mode == "wrong_candidate_binding_control":
        ids = list(payload["embedding_scores"].keys())
        vals = list(payload["embedding_scores"].values())
        if len(vals) > 1:
            vals = vals[1:] + vals[:1]
        return dict(zip(ids, vals))
    raise ValueError(mode)


def evaluate_mode(payloads: list[dict[str, Any]], mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    for idx, payload in enumerate(payloads):
        emb_scores = control_scores(payloads, payload, idx, mode)
        comps = payload["comps"]
        target_id = payload["target_id"]
        geom_scores = score_embedding_variant("A0_geometry_passive", comps, emb_scores)
        ranked_geom = sorted(geom_scores.items(), key=lambda x: x[1], reverse=True)
        wrong_id = ranked_geom[0][0] if ranked_geom and ranked_geom[0][0] != target_id else (ranked_geom[1][0] if len(ranked_geom) > 1 else "")
        margin = emb_scores.get(target_id, 0.0) - emb_scores.get(wrong_id, 0.0)
        margin_rows.append({
            "control_mode": mode,
            "event_id": payload["event_id"],
            "category": payload["category"],
            "sequence_id": payload["sequence_id"],
            "target_instance_id_eval_only": target_id,
            "wrong_top1_id_eval_only": wrong_id,
            "target_embedding_score": emb_scores.get(target_id, 0.0),
            "wrong_embedding_score": emb_scores.get(wrong_id, 0.0),
            "embedding_margin_target_minus_wrong": margin,
            "embedding_margin_positive": int(margin > 0),
        })
        for variant in VARIANTS:
            scores = score_embedding_variant(variant, comps, emb_scores)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ranked_ids = [iid for iid, _ in ranked]
            top1 = ranked_ids[0] if ranked_ids else ""
            top1_hit = int(top1 == target_id)
            top3_hit = int(target_id in ranked_ids[:3])
            top5_hit = int(target_id in ranked_ids[:5])
            if top1_hit:
                reason = "success"
            elif target_id not in ranked_ids[:5]:
                reason = "target_not_in_top5"
            elif margin < 0:
                reason = "embedding_negative_margin"
            elif variant == "A2_embedding_nn":
                reason = "embedding_collision"
            else:
                reason = "target_in_top5_but_wrong_top1"
            event_rows.append({
                "control_mode": mode,
                "event_id": payload["event_id"],
                "category": payload["category"],
                "sequence_id": payload["sequence_id"],
                "variant": variant,
                "target_instance_id_eval_only": target_id,
                "predicted_memory_id": top1,
                "top1": top1_hit,
                "top3": top3_hit,
                "top5": top5_hit,
                "false_retrieval": int(not top1_hit),
                "gap_length": payload.get("gap_length", ""),
                "candidate_count": payload["candidate_count"],
                "failure_reason": reason,
            })
    return event_rows, margin_rows


def summarize_events(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        by_key[(row["control_mode"], row["variant"])].append(row)
    out = []
    for mode in CONTROL_MODES:
        for variant in VARIANTS:
            rows = by_key.get((mode, variant), [])
            n = len(rows)
            out.append({
                "control_mode": mode,
                "variant": variant,
                "num_events": n,
                "global_top1": sum(as_int(r["top1"]) for r in rows) / max(n, 1),
                "global_top3": sum(as_int(r["top3"]) for r in rows) / max(n, 1),
                "global_top5": sum(as_int(r["top5"]) for r in rows) / max(n, 1),
                "false_retrieval_rate": sum(as_int(r["false_retrieval"]) for r in rows) / max(n, 1),
            })
    return out


def summarize_margins(margin_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in margin_rows:
        by_mode[row["control_mode"]].append(row)
    out = []
    for mode in CONTROL_MODES:
        rows = by_mode.get(mode, [])
        vals = [as_float(r["embedding_margin_target_minus_wrong"]) for r in rows]
        out.append({
            "control_mode": mode,
            "num_events": len(rows),
            "embedding_margin_positive_rate": sum(1 for v in vals if v > 0) / max(len(vals), 1),
            "mean_embedding_margin": sum(vals) / max(len(vals), 1),
        })
    return out


def lookup(summary: list[dict[str, Any]], mode: str, variant: str) -> dict[str, Any]:
    return next((r for r in summary if r["control_mode"] == mode and r["variant"] == variant), {})


def best_of(summary: list[dict[str, Any]], mode: str, variants: list[str]) -> dict[str, Any]:
    return max((lookup(summary, mode, v) for v in variants), key=lambda r: as_float(r.get("global_top1")))


def control_summary(summary: list[dict[str, Any]], margin_summary: list[dict[str, Any]], embedding_model: str) -> list[dict[str, Any]]:
    rows = []
    real_geom = as_float(lookup(summary, "real", "A0_geometry_passive").get("global_top1"))
    real_ext = as_float(lookup(summary, "real", "A1_external_trajectory_branch").get("global_top1"))
    for mode in CONTROL_MODES:
        geom_best = best_of(summary, mode, ["A3_geometry_plus_embedding_w005", "A4_geometry_plus_embedding_w010", "A5_geometry_plus_embedding_w020"])
        ext_best = best_of(summary, mode, ["A6_external_branch_plus_embedding_w005", "A7_external_branch_plus_embedding_w010", "A8_external_branch_plus_embedding_w020"])
        emb = lookup(summary, mode, "A2_embedding_nn")
        margins = next((r for r in margin_summary if r["control_mode"] == mode), {})
        rows.append({
            "control_name": mode,
            "embedding_model": embedding_model,
            "num_events": geom_best.get("num_events", 0),
            "geometry_plus_embedding_top1": geom_best.get("global_top1", 0),
            "external_branch_plus_embedding_top1": ext_best.get("global_top1", 0),
            "embedding_nn_top1": emb.get("global_top1", 0),
            "mean_embedding_margin": margins.get("mean_embedding_margin", 0),
            "embedding_margin_positive_rate": margins.get("embedding_margin_positive_rate", 0),
            "control_passed": int(mode != "real"),
            "failure_reason": "",
            "geometry_gain_vs_real_base": as_float(geom_best.get("global_top1")) - real_geom,
            "external_gain_vs_real_base": as_float(ext_best.get("global_top1")) - real_ext,
        })
    real_row = rows[0]
    max_control_geom = max(as_float(r["geometry_gain_vs_real_base"]) for r in rows[1:])
    max_control_ext = max(as_float(r["external_gain_vs_real_base"]) for r in rows[1:])
    real_geom_gain = as_float(real_row["geometry_gain_vs_real_base"])
    real_ext_gain = as_float(real_row["external_gain_vs_real_base"])
    real_pass = int(real_geom_gain > max_control_geom + 1e-12 and real_ext_gain >= max_control_ext - 1e-12)
    real_row["control_passed"] = real_pass
    real_row["failure_reason"] = "" if real_pass else "real_gain_not_above_shuffled_or_random_controls"
    return rows


def bootstrap_delta(event_rows: list[dict[str, Any]], mode: str, baseline_variant: str, test_variant: str, samples: int, seed: int = 7) -> dict[str, Any]:
    by_variant: dict[str, dict[str, int]] = defaultdict(dict)
    for row in event_rows:
        if row["control_mode"] != mode:
            continue
        by_variant[row["variant"]][row["event_id"]] = as_int(row["top1"])
    base = by_variant.get(baseline_variant, {})
    test = by_variant.get(test_variant, {})
    event_ids = sorted(set(base) & set(test))
    if not event_ids:
        return {"delta_top1": 0.0, "bootstrap_mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_value_or_permutation_rate": 1.0, "significant": 0, "effect_size": 0.0}
    diffs = np.array([test[e] - base[e] for e in event_ids], dtype=np.float64)
    delta = float(diffs.mean())
    rng = np.random.default_rng(seed)
    boots = []
    n = len(diffs)
    for _ in range(max(samples, 1)):
        idx = rng.integers(0, n, size=n)
        boots.append(float(diffs[idx].mean()))
    arr = np.array(boots)
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    # One-sided sign-flip permutation for positive improvement.
    signs = rng.choice([-1.0, 1.0], size=(max(samples, 1), n))
    perm = (signs * diffs).mean(axis=1)
    p = float((np.sum(perm >= delta) + 1) / (len(perm) + 1))
    return {
        "delta_top1": delta,
        "bootstrap_mean_delta": float(arr.mean()),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value_or_permutation_rate": p,
        "significant": int(ci_low > 0.0 and p < 0.05),
        "effect_size": delta / max(float(diffs.std()), 1e-8),
    }


def significance_rows(event_rows: list[dict[str, Any]], summary: list[dict[str, Any]], samples: int) -> list[dict[str, Any]]:
    geom_best = best_of(summary, "real", ["A3_geometry_plus_embedding_w005", "A4_geometry_plus_embedding_w010", "A5_geometry_plus_embedding_w020"])["variant"]
    ext_best = best_of(summary, "real", ["A6_external_branch_plus_embedding_w005", "A7_external_branch_plus_embedding_w010", "A8_external_branch_plus_embedding_w020"])["variant"]
    comparisons = [
        ("embedding_nn vs geometry_passive", "real", "A0_geometry_passive", "A2_embedding_nn"),
        ("geometry_plus_embedding_best vs geometry_passive", "real", "A0_geometry_passive", geom_best),
        ("external_branch_plus_embedding_best vs external_branch", "real", "A1_external_trajectory_branch", ext_best),
        ("real_embedding_gain vs within_event_shuffled_gain", "within_event_shuffled_embedding", "A0_geometry_passive", geom_best),
        ("real_embedding_gain vs category_shuffled_gain", "category_shuffled_embedding", "A0_geometry_passive", geom_best),
        ("real_embedding_gain vs random_vector_gain", "random_vector_control", "A0_geometry_passive", geom_best),
    ]
    out = []
    for name, mode, base, test in comparisons:
        row = bootstrap_delta(event_rows, mode, base, test, samples=samples, seed=stable_seed(name) % 100000)
        row.update({
            "comparison": name,
            "baseline_variant": base,
            "test_variant": test,
            "control_mode": mode,
        })
        out.append(row)
    return out


def category_summary(event_rows: list[dict[str, Any]], margin_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    real_rows = [r for r in event_rows if r["control_mode"] == "real"]
    real_margins = [r for r in margin_rows if r["control_mode"] == "real"]
    by_cat_var: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in real_rows:
        by_cat_var[(row["category"], row["variant"])].append(row)
    margins_by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in real_margins:
        margins_by_cat[row["category"]].append(row)
    out = []
    for cat in sorted({r["category"] for r in real_rows}):
        def top1(variant: str) -> float:
            rows = by_cat_var.get((cat, variant), [])
            return sum(as_int(r["top1"]) for r in rows) / max(len(rows), 1)
        geom = top1("A0_geometry_passive")
        ext = top1("A1_external_trajectory_branch")
        emb = top1("A2_embedding_nn")
        geom_emb = max(top1(v) for v in ["A3_geometry_plus_embedding_w005", "A4_geometry_plus_embedding_w010", "A5_geometry_plus_embedding_w020"])
        ext_emb = max(top1(v) for v in ["A6_external_branch_plus_embedding_w005", "A7_external_branch_plus_embedding_w010", "A8_external_branch_plus_embedding_w020"])
        vals = [as_float(r["embedding_margin_target_minus_wrong"]) for r in margins_by_cat.get(cat, [])]
        if len(vals) < 10:
            rec = "insufficient_events"
        elif ext_emb > ext:
            rec = "embedding_helpful"
        elif geom_emb > geom and ext_emb >= ext:
            rec = "embedding_helpful"
        elif ext_emb < ext or geom_emb < geom:
            rec = "embedding_hurts"
        elif emb <= geom and emb <= ext:
            rec = "geometry_sufficient"
        else:
            rec = "embedding_non_discriminative"
        out.append({
            "category": cat,
            "num_events": len(by_cat_var.get((cat, "A0_geometry_passive"), [])),
            "geometry_passive_top1": geom,
            "external_branch_top1": ext,
            "embedding_nn_top1": emb,
            "geometry_plus_embedding_best_top1": geom_emb,
            "external_branch_plus_embedding_best_top1": ext_emb,
            "embedding_margin_positive_rate": sum(1 for v in vals if v > 0) / max(len(vals), 1),
            "mean_embedding_margin": sum(vals) / max(len(vals), 1),
            "embedding_helped_geometry": int(geom_emb > geom),
            "embedding_hurt_geometry": int(geom_emb < geom),
            "embedding_helped_external_branch": int(ext_emb > ext),
            "embedding_hurt_external_branch": int(ext_emb < ext),
            "recommended_use": rec,
        })
    return out


def failure_taxonomy(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "event_id": r["event_id"],
            "category": r["category"],
            "sequence_id": r["sequence_id"],
            "variant": r["variant"],
            "failure_reason": r["failure_reason"],
        }
        for r in event_rows
        if r["control_mode"] == "real"
    ]


def usage_decision(compact: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"decision_item": "embedding_as_main_nops_method", "allowed": 0, "reason": "frozen pretrained embedding violates no-pretrain main method constraint"},
        {"decision_item": "embedding_as_external_baseline", "allowed": 1, "reason": "valid external full-pixel baseline"},
        {"decision_item": "embedding_as_external_branch_auxiliary", "allowed": int(compact["embedding_safe_as_external_branch_auxiliary"]), "reason": "requires clean controls and significant external branch gain"},
        {"decision_item": "embedding_as_main_merge", "allowed": 0, "reason": "never allowed in EXT-7"},
        {"decision_item": "requires_more_pixels", "allowed": int(compact["num_events"] < 500), "reason": "current subset is still small for final paper claim"},
        {"decision_item": "requires_better_controls", "allowed": int(not compact["embedding_controls_passed"]), "reason": "controls must pass before integration"},
    ]


def unavailable_outputs(args: argparse.Namespace, embedder: FrozenEmbedder) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compact = {
        "stage": "EXT-7",
        "embedding_model": args.embedding_model,
        "model_unavailable": 1,
        "unavailable_reason": embedder.unavailable_reason,
        "embedding_safe_for_main_merge": 0,
        "next_recommendation": "install/download model weights or choose an available embedding model",
    }
    write_json(out_dir / "stage_EXT7_compact_for_gpt_v1.json", compact)
    (out_dir / "stage_EXT7_report_v1.md").write_text(
        "# EXT-7 Frozen Embedding Baseline\n\n"
        f"Model unavailable: `{embedder.unavailable_reason}`\n",
        encoding="utf-8",
    )
    print(json.dumps(compact, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = FrozenEmbedder(args.embedding_model, args.device)
    if not embedder.available:
        unavailable_outputs(args, embedder)
        return
    payloads, quality_rows = collect_payloads(args, embedder)
    all_events: list[dict[str, Any]] = []
    all_margins: list[dict[str, Any]] = []
    for mode in CONTROL_MODES:
        events, margins = evaluate_mode(payloads, mode)
        all_events.extend(events)
        all_margins.extend(margins)
    summary = summarize_events(all_events)
    margins = summarize_margins(all_margins)
    controls = control_summary(summary, margins, args.embedding_model)
    sig = significance_rows(all_events, summary, args.bootstrap_samples)
    cats = category_summary(all_events, all_margins)
    failures = failure_taxonomy(all_events)

    real_geom = lookup(summary, "real", "A0_geometry_passive")
    real_ext = lookup(summary, "real", "A1_external_trajectory_branch")
    real_emb = lookup(summary, "real", "A2_embedding_nn")
    geom_best = best_of(summary, "real", ["A3_geometry_plus_embedding_w005", "A4_geometry_plus_embedding_w010", "A5_geometry_plus_embedding_w020"])
    ext_best = best_of(summary, "real", ["A6_external_branch_plus_embedding_w005", "A7_external_branch_plus_embedding_w010", "A8_external_branch_plus_embedding_w020"])
    real_margin = next(r for r in margins if r["control_mode"] == "real")
    real_control = next(r for r in controls if r["control_name"] == "real")
    meaningful_sig = any(as_int(r["significant"]) for r in sig[:3])
    external_gain = as_float(ext_best.get("global_top1")) - as_float(real_ext.get("global_top1"))
    safe_aux = int(as_int(real_control["control_passed"]) and meaningful_sig and external_gain > 0)
    available_rate = sum(as_int(r["embedding_available"]) for r in quality_rows) / max(len(quality_rows), 1)
    compact = {
        "stage": "EXT-7",
        "embedding_model": args.embedding_model,
        "method_family": "frozen_pretrained_embedding_baseline",
        "num_events": len(payloads),
        "embedding_available_rate": available_rate,
        "device": embedder.device_name,
        "geometry_passive_top1": as_float(real_geom.get("global_top1")),
        "external_branch_top1": as_float(real_ext.get("global_top1")),
        "embedding_nn_top1": as_float(real_emb.get("global_top1")),
        "geometry_plus_embedding_best_top1": as_float(geom_best.get("global_top1")),
        "external_branch_plus_embedding_best_top1": as_float(ext_best.get("global_top1")),
        "embedding_margin_positive_rate": as_float(real_margin["embedding_margin_positive_rate"]),
        "mean_embedding_margin": as_float(real_margin["mean_embedding_margin"]),
        "embedding_controls_passed": as_int(real_control["control_passed"]),
        "significance_passed": int(meaningful_sig),
        "embedding_safe_as_external_baseline": int(available_rate >= 0.90),
        "embedding_safe_as_external_branch_auxiliary": safe_aux,
        "embedding_safe_for_main_merge": 0,
        "category_results": cats,
        "next_recommendation": (
            "define isolated embedding-assisted external branch" if safe_aux
            else "do not use embedding fusion; keep as diagnostic external baseline"
        ),
    }

    for row in summary:
        row["embedding_model"] = args.embedding_model
    write_csv(out_dir / "stage_EXT7_ablation_summary_v1.csv", summary)
    write_csv(out_dir / "stage_EXT7_event_results_v1.csv", all_events)
    write_csv(out_dir / "stage_EXT7_embedding_margin_trace_v1.csv", all_margins)
    write_csv(out_dir / "stage_EXT7_embedding_quality_v1.csv", quality_rows)
    write_csv(out_dir / "stage_EXT7_control_summary_v1.csv", controls)
    write_csv(out_dir / "stage_EXT7_significance_audit_v1.csv", sig)
    write_csv(out_dir / "stage_EXT7_category_summary_v1.csv", cats)
    write_csv(out_dir / "stage_EXT7_failure_taxonomy_v1.csv", failures)
    write_csv(out_dir / "stage_EXT7_usage_decision_v1.csv", usage_decision(compact))
    write_json(out_dir / "stage_EXT7_compact_for_gpt_v1.json", compact)
    report = [
        "# EXT-7 Frozen Embedding External Baseline",
        "",
        "## Result",
        "",
        f"- Embedding model: `{args.embedding_model}`",
        f"- Device: `{embedder.device_name}`",
        f"- Events: `{compact['num_events']}`",
        f"- Geometry passive top1: `{compact['geometry_passive_top1']:.4f}`",
        f"- External branch top1: `{compact['external_branch_top1']:.4f}`",
        f"- Embedding NN top1: `{compact['embedding_nn_top1']:.4f}`",
        f"- Geometry + embedding best top1: `{compact['geometry_plus_embedding_best_top1']:.4f}`",
        f"- External branch + embedding best top1: `{compact['external_branch_plus_embedding_best_top1']:.4f}`",
        f"- Mean embedding margin: `{compact['mean_embedding_margin']:.4f}`",
        f"- Controls passed: `{compact['embedding_controls_passed']}`",
        f"- Significance passed: `{compact['significance_passed']}`",
        "",
        "## Decision",
        "",
        compact["next_recommendation"],
        "",
        "Frozen embeddings are recorded as an external pretrained baseline only, not as the NOPS main method.",
    ]
    (out_dir / "stage_EXT7_report_v1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
