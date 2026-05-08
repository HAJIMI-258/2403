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
    cosine,
    crop_descriptor,
    image_path,
    load_pixel_ready_events,
    normalized_hist,
    score_variant,
    state_at,
    zscores,
)
from experiments.run_ext5_multicategory_full_pixel_validation import category_from_seq


VARIANTS = [
    "A0_nops_geometry_passive",
    "A1_raw_appearance_nn",
    "A2_strong_descriptor_nn",
    "A3_geometry_plus_strong_w005",
    "A4_geometry_plus_strong_w010",
    "A5_geometry_plus_strong_w020",
    "A6_external_trajectory_heavy",
    "A7_external_trajectory_plus_strong_w005",
    "A8_external_trajectory_plus_strong_w010",
    "A9_external_trajectory_plus_strong_w020",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXT-6 stronger local descriptor validation.")
    p.add_argument("--lagot-root", default="data/external/lagot_annotations")
    p.add_argument("--lasot-root", default="data/external/lasot")
    p.add_argument("--ext4-dir", default="results/ext4")
    p.add_argument("--output-dir", default="results/ext6")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stable_seed(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def _grid_hist(gray: np.ndarray, bins: int = 8, cells: int = 4) -> np.ndarray:
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    ori = (np.arctan2(gy, gx) + math.pi) / (2 * math.pi)
    parts: list[np.ndarray] = []
    h, w = gray.shape
    for cy in range(cells):
        for cx in range(cells):
            y1, y2 = int(cy * h / cells), int((cy + 1) * h / cells)
            x1, x2 = int(cx * w / cells), int((cx + 1) * w / cells)
            hist, _ = np.histogram(
                ori[y1:y2, x1:x2].ravel(),
                bins=bins,
                range=(0.0, 1.0),
                weights=mag[y1:y2, x1:x2].ravel(),
            )
            hist = hist.astype(np.float32)
            if hist.sum() > 0:
                hist /= hist.sum()
            parts.append(hist)
    return np.concatenate(parts)


def _lbp_hist(gray: np.ndarray) -> np.ndarray:
    center = gray[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint8)
    offsets = [
        (-1, -1), (-1, 0), (-1, 1), (0, 1),
        (1, 1), (1, 0), (1, -1), (0, -1),
    ]
    for bit, (dy, dx) in enumerate(offsets):
        neigh = gray[1 + dy: gray.shape[0] - 1 + dy, 1 + dx: gray.shape[1] - 1 + dx]
        code |= ((neigh >= center).astype(np.uint8) << bit)
    hist, _ = np.histogram(code.ravel(), bins=32, range=(0, 256))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def _grid_color_moments(arr: np.ndarray, cells: int = 4) -> np.ndarray:
    parts: list[float] = []
    h, w, _ = arr.shape
    for cy in range(cells):
        for cx in range(cells):
            y1, y2 = int(cy * h / cells), int((cy + 1) * h / cells)
            x1, x2 = int(cx * w / cells), int((cx + 1) * w / cells)
            patch = arr[y1:y2, x1:x2]
            parts.extend(float(v) for v in patch.mean(axis=(0, 1)))
            parts.extend(float(v) for v in patch.std(axis=(0, 1)))
    return np.array(parts, dtype=np.float32)


def strong_crop_descriptor(path: Path, box: tuple[float, float, float, float]) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            crop = img.crop(clamp_crop(box, img.width, img.height)).resize((64, 64))
    except Exception:
        return None
    arr = np.asarray(crop).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    raw = crop_descriptor(path, box)
    if raw is None:
        return None
    gray_eq = gray - float(gray.mean())
    gray_std = float(gray.std())
    if gray_std > 1e-6:
        gray_eq = gray_eq / gray_std
    gray_eq = np.clip((gray_eq + 3.0) / 6.0, 0.0, 1.0)
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    gradient_stats = np.array([
        float(mag.mean()),
        float(mag.std()),
        float(np.percentile(mag, 75)),
        float(np.percentile(mag, 90)),
        float(gray.mean()),
        float(gray.std()),
        float(normalized_hist(gray_eq.ravel(), 16, 0.0, 1.0).max()),
    ], dtype=np.float32)
    vec = np.concatenate([
        raw,
        _grid_hist(gray, bins=8, cells=4),
        _lbp_hist(gray),
        _grid_color_moments(arr, cells=4),
        gradient_stats,
    ])
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return vec.astype(np.float32) / norm


def score_strong_variant(variant: str, comps: dict[str, dict[str, Any]], raw_scores: dict[str, float], strong_scores: dict[str, float]) -> dict[str, float]:
    geom = {iid: float(c["nops_geometry_score"]) for iid, c in comps.items()}
    ext = {iid: float(c["external_trajectory_heavy_score"]) for iid, c in comps.items()}
    z_geom = zscores(geom)
    z_ext = zscores(ext)
    z_strong = zscores(strong_scores)
    if variant == "A0_nops_geometry_passive":
        return geom
    if variant == "A1_raw_appearance_nn":
        return raw_scores
    if variant == "A2_strong_descriptor_nn":
        return strong_scores
    if variant == "A3_geometry_plus_strong_w005":
        return {iid: z_geom.get(iid, 0.0) + 0.05 * z_strong.get(iid, 0.0) for iid in comps}
    if variant == "A4_geometry_plus_strong_w010":
        return {iid: z_geom.get(iid, 0.0) + 0.10 * z_strong.get(iid, 0.0) for iid in comps}
    if variant == "A5_geometry_plus_strong_w020":
        return {iid: z_geom.get(iid, 0.0) + 0.20 * z_strong.get(iid, 0.0) for iid in comps}
    if variant == "A6_external_trajectory_heavy":
        return ext
    if variant == "A7_external_trajectory_plus_strong_w005":
        return {iid: z_ext.get(iid, 0.0) + 0.05 * z_strong.get(iid, 0.0) for iid in comps}
    if variant == "A8_external_trajectory_plus_strong_w010":
        return {iid: z_ext.get(iid, 0.0) + 0.10 * z_strong.get(iid, 0.0) for iid in comps}
    if variant == "A9_external_trajectory_plus_strong_w020":
        return {iid: z_ext.get(iid, 0.0) + 0.20 * z_strong.get(iid, 0.0) for iid in comps}
    raise ValueError(variant)


def collect_payloads(args: argparse.Namespace) -> list[dict[str, Any]]:
    lagot, event_rows, _ = load_pixel_ready_events(args)
    hist_cache: dict[str, dict[str, list[tuple[int, tuple[float, float, float, float], Any]]]] = {}
    raw_cache: dict[tuple[str, tuple[float, float, float, float]], Any] = {}
    strong_cache: dict[tuple[str, tuple[float, float, float, float]], Any] = {}
    payloads: list[dict[str, Any]] = []
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
        if q_key not in raw_cache:
            raw_cache[q_key] = crop_descriptor(img_path, query_box)
        if q_key not in strong_cache:
            strong_cache[q_key] = strong_crop_descriptor(img_path, query_box)
        query_raw = raw_cache[q_key]
        query_strong = strong_cache[q_key]
        if query_raw is None or query_strong is None:
            continue
        candidates = {iid: states for iid, states in hist.items() if any(st[0] < reappear for st in states)}
        comps: dict[str, dict[str, Any]] = {}
        raw_scores: dict[str, float] = {}
        strong_scores: dict[str, float] = {}
        for iid, states in candidates.items():
            c = candidate_components(query_box, states, float(width), float(height), reappear)
            if not c:
                continue
            last_frame = int(c["last_frame"])
            last_box = c["last_box"]
            last_path = image_path(seq_dir, last_frame)
            key = (str(last_path), tuple(float(x) for x in last_box))
            if key not in raw_cache:
                raw_cache[key] = crop_descriptor(last_path, last_box)
            if key not in strong_cache:
                strong_cache[key] = strong_crop_descriptor(last_path, last_box)
            raw_desc = raw_cache[key]
            strong_desc = strong_cache[key]
            if raw_desc is None or strong_desc is None:
                continue
            comps[iid] = c
            raw_scores[iid] = cosine(query_raw, raw_desc)
            strong_scores[iid] = cosine(query_strong, strong_desc)
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
            "raw_scores": raw_scores,
            "strong_scores": strong_scores,
        })
    return payloads


def shuffle_within_event(scores: dict[str, float], event_id: str, salt: str) -> dict[str, float]:
    ids = list(scores.keys())
    vals = list(scores.values())
    rng = random.Random(stable_seed(event_id + ":" + salt))
    rng.shuffle(vals)
    return dict(zip(ids, vals))


def category_shuffled(payloads: list[dict[str, Any]], payload: dict[str, Any], idx: int) -> dict[str, float]:
    ids = list(payload["strong_scores"].keys())
    source = None
    n = len(payloads)
    for off in range(1, n + 1):
        cand = payloads[(idx + off * 19) % n]
        if cand["category"] != payload["category"] and cand["strong_scores"]:
            source = cand
            break
    if source is None:
        return shuffle_within_event(payload["strong_scores"], payload["event_id"], "category")
    vals = list(source["strong_scores"].values())
    rng = random.Random(stable_seed(payload["event_id"] + ":category"))
    rng.shuffle(vals)
    return {iid: vals[i % len(vals)] for i, iid in enumerate(ids)}


def evaluate_mode(payloads: list[dict[str, Any]], mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    for idx, payload in enumerate(payloads):
        if mode == "real":
            strong_scores = payload["strong_scores"]
        elif mode == "within_event_shuffled":
            strong_scores = shuffle_within_event(payload["strong_scores"], payload["event_id"], "within")
        elif mode == "category_shuffled":
            strong_scores = category_shuffled(payloads, payload, idx)
        else:
            raise ValueError(mode)
        target_id = payload["target_id"]
        comps = payload["comps"]
        a0_scores = score_variant("A0_nops_geometry_passive", comps, payload["raw_scores"])
        ranked_a0 = sorted(a0_scores.items(), key=lambda x: x[1], reverse=True)
        wrong_id = ranked_a0[0][0] if ranked_a0 and ranked_a0[0][0] != target_id else (ranked_a0[1][0] if len(ranked_a0) > 1 else "")
        raw_margin = payload["raw_scores"].get(target_id, 0.0) - payload["raw_scores"].get(wrong_id, 0.0)
        strong_margin = strong_scores.get(target_id, 0.0) - strong_scores.get(wrong_id, 0.0)
        margin_rows.append({
            "control_mode": mode,
            "event_id": payload["event_id"],
            "category": payload["category"],
            "target_instance_id_eval_only": target_id,
            "wrong_top1_id_eval_only": wrong_id,
            "raw_margin_target_minus_wrong": raw_margin,
            "strong_margin_target_minus_wrong": strong_margin,
            "strong_margin_positive": int(strong_margin > 0),
            "strong_beats_raw_margin": int(strong_margin > raw_margin),
        })
        for variant in VARIANTS:
            scores = score_strong_variant(variant, comps, payload["raw_scores"], strong_scores)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ranked_ids = [iid for iid, _ in ranked]
            top1 = ranked_ids[0] if ranked_ids else ""
            top1_hit = int(top1 == target_id)
            top3_hit = int(target_id in ranked_ids[:3])
            top5_hit = int(target_id in ranked_ids[:5])
            event_rows.append({
                "control_mode": mode,
                "event_id": payload["event_id"],
                "sequence_id": payload["sequence_id"],
                "category": payload["category"],
                "variant": variant,
                "target_instance_id_eval_only": target_id,
                "predicted_memory_id": top1,
                "top1": top1_hit,
                "top3": top3_hit,
                "top5": top5_hit,
                "false_retrieval": int(not top1_hit),
                "gap_length": payload.get("gap_length", ""),
                "candidate_count": payload["candidate_count"],
            })
    return event_rows, margin_rows


def summarize(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        by_key[(row["control_mode"], row["variant"])].append(row)
    out: list[dict[str, Any]] = []
    for mode in ("real", "within_event_shuffled", "category_shuffled"):
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


def margin_summary(margin_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in margin_rows:
        by_mode[row["control_mode"]].append(row)
    out = []
    for mode, rows in sorted(by_mode.items()):
        strong = [as_float(r["strong_margin_target_minus_wrong"]) for r in rows]
        raw = [as_float(r["raw_margin_target_minus_wrong"]) for r in rows]
        out.append({
            "control_mode": mode,
            "num_events": len(rows),
            "raw_margin_positive_rate": sum(1 for v in raw if v > 0) / max(len(raw), 1),
            "mean_raw_margin": sum(raw) / max(len(raw), 1),
            "strong_margin_positive_rate": sum(1 for v in strong if v > 0) / max(len(strong), 1),
            "mean_strong_margin": sum(strong) / max(len(strong), 1),
            "strong_beats_raw_margin_rate": sum(as_int(r["strong_beats_raw_margin"]) for r in rows) / max(len(rows), 1),
        })
    return out


def lookup(summary: list[dict[str, Any]], mode: str, variant: str) -> dict[str, Any]:
    return next((r for r in summary if r["control_mode"] == mode and r["variant"] == variant), {})


def category_summary(event_rows: list[dict[str, Any]], margin_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    real_rows = [r for r in event_rows if r["control_mode"] == "real"]
    real_margins = [r for r in margin_rows if r["control_mode"] == "real"]
    by_cat_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in real_rows:
        by_cat_variant[(row["category"], row["variant"])].append(row)
    margins_by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in real_margins:
        margins_by_cat[row["category"]].append(row)
    cats = sorted({r["category"] for r in real_rows})
    out = []
    for cat in cats:
        def top1(variant: str) -> float:
            rows = by_cat_variant.get((cat, variant), [])
            return sum(as_int(r["top1"]) for r in rows) / max(len(rows), 1)

        geom = top1("A0_nops_geometry_passive")
        geom_strong = max(top1(v) for v in ("A3_geometry_plus_strong_w005", "A4_geometry_plus_strong_w010", "A5_geometry_plus_strong_w020"))
        ext = top1("A6_external_trajectory_heavy")
        ext_strong = max(top1(v) for v in ("A7_external_trajectory_plus_strong_w005", "A8_external_trajectory_plus_strong_w010", "A9_external_trajectory_plus_strong_w020"))
        mrows = margins_by_cat.get(cat, [])
        vals = [as_float(r["strong_margin_target_minus_wrong"]) for r in mrows]
        out.append({
            "category": cat,
            "num_events": len(by_cat_variant.get((cat, "A0_nops_geometry_passive"), [])),
            "geometry_passive_top1": geom,
            "geometry_plus_strong_best_top1": geom_strong,
            "external_branch_top1": ext,
            "external_branch_plus_strong_best_top1": ext_strong,
            "strong_margin_positive_rate": sum(1 for v in vals if v > 0) / max(len(vals), 1),
            "mean_strong_margin": sum(vals) / max(len(vals), 1),
            "strong_helped_geometry": int(geom_strong > geom),
            "strong_helped_external_branch": int(ext_strong > ext),
        })
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = collect_payloads(args)
    all_events: list[dict[str, Any]] = []
    all_margins: list[dict[str, Any]] = []
    for mode in ("real", "within_event_shuffled", "category_shuffled"):
        events, margins = evaluate_mode(payloads, mode)
        all_events.extend(events)
        all_margins.extend(margins)
    summary = summarize(all_events)
    margins = margin_summary(all_margins)
    cats = category_summary(all_events, all_margins)

    real_geom = lookup(summary, "real", "A0_nops_geometry_passive")
    real_geom_strong = max(
        (lookup(summary, "real", v) for v in ("A3_geometry_plus_strong_w005", "A4_geometry_plus_strong_w010", "A5_geometry_plus_strong_w020")),
        key=lambda r: as_float(r.get("global_top1")),
    )
    real_ext = lookup(summary, "real", "A6_external_trajectory_heavy")
    real_ext_strong = max(
        (lookup(summary, "real", v) for v in ("A7_external_trajectory_plus_strong_w005", "A8_external_trajectory_plus_strong_w010", "A9_external_trajectory_plus_strong_w020")),
        key=lambda r: as_float(r.get("global_top1")),
    )
    shuf_geom_strong = max(
        (lookup(summary, "within_event_shuffled", v) for v in ("A3_geometry_plus_strong_w005", "A4_geometry_plus_strong_w010", "A5_geometry_plus_strong_w020")),
        key=lambda r: as_float(r.get("global_top1")),
    )
    cat_geom_strong = max(
        (lookup(summary, "category_shuffled", v) for v in ("A3_geometry_plus_strong_w005", "A4_geometry_plus_strong_w010", "A5_geometry_plus_strong_w020")),
        key=lambda r: as_float(r.get("global_top1")),
    )
    shuf_ext_strong = max(
        (lookup(summary, "within_event_shuffled", v) for v in ("A7_external_trajectory_plus_strong_w005", "A8_external_trajectory_plus_strong_w010", "A9_external_trajectory_plus_strong_w020")),
        key=lambda r: as_float(r.get("global_top1")),
    )
    cat_ext_strong = max(
        (lookup(summary, "category_shuffled", v) for v in ("A7_external_trajectory_plus_strong_w005", "A8_external_trajectory_plus_strong_w010", "A9_external_trajectory_plus_strong_w020")),
        key=lambda r: as_float(r.get("global_top1")),
    )

    real_geom_gain = as_float(real_geom_strong.get("global_top1")) - as_float(real_geom.get("global_top1"))
    real_ext_gain = as_float(real_ext_strong.get("global_top1")) - as_float(real_ext.get("global_top1"))
    shuf_geom_gain = as_float(shuf_geom_strong.get("global_top1")) - as_float(lookup(summary, "within_event_shuffled", "A0_nops_geometry_passive").get("global_top1"))
    cat_geom_gain = as_float(cat_geom_strong.get("global_top1")) - as_float(lookup(summary, "category_shuffled", "A0_nops_geometry_passive").get("global_top1"))
    shuf_ext_gain = as_float(shuf_ext_strong.get("global_top1")) - as_float(lookup(summary, "within_event_shuffled", "A6_external_trajectory_heavy").get("global_top1"))
    cat_ext_gain = as_float(cat_ext_strong.get("global_top1")) - as_float(lookup(summary, "category_shuffled", "A6_external_trajectory_heavy").get("global_top1"))

    real_margin = next(r for r in margins if r["control_mode"] == "real")
    controls_passed = int(
        real_geom_gain > max(shuf_geom_gain, cat_geom_gain) + 1e-12
        and real_ext_gain >= max(shuf_ext_gain, cat_ext_gain) - 1e-12
        and as_float(real_margin["mean_strong_margin"]) > 0
    )
    safe_external = int(controls_passed and real_ext_gain > 0)
    compact = {
        "stage": "EXT-6",
        "num_events": len(payloads),
        "geometry_passive_top1": as_float(real_geom.get("global_top1")),
        "geometry_plus_strong_best_top1": as_float(real_geom_strong.get("global_top1")),
        "external_branch_top1": as_float(real_ext.get("global_top1")),
        "external_branch_plus_strong_best_top1": as_float(real_ext_strong.get("global_top1")),
        "strong_geometry_gain": real_geom_gain,
        "strong_external_gain": real_ext_gain,
        "within_event_shuffled_geometry_gain": shuf_geom_gain,
        "category_shuffled_geometry_gain": cat_geom_gain,
        "within_event_shuffled_external_gain": shuf_ext_gain,
        "category_shuffled_external_gain": cat_ext_gain,
        "strong_margin_positive_rate": as_float(real_margin["strong_margin_positive_rate"]),
        "mean_strong_margin": as_float(real_margin["mean_strong_margin"]),
        "strong_controls_passed": controls_passed,
        "strong_descriptor_safe_for_external_branch": safe_external,
        "strong_descriptor_safe_for_main_merge": 0,
        "next_recommendation": (
            "consider descriptor-gated external branch only" if safe_external
            else "strong local descriptor is not reliable enough for integration"
        ),
    }

    write_csv(out_dir / "stage_EXT6_ablation_summary_v1.csv", summary)
    write_csv(out_dir / "stage_EXT6_event_results_v1.csv", all_events)
    write_csv(out_dir / "stage_EXT6_descriptor_margin_trace_v1.csv", all_margins)
    write_csv(out_dir / "stage_EXT6_margin_summary_v1.csv", margins)
    write_csv(out_dir / "stage_EXT6_category_summary_v1.csv", cats)
    write_json(out_dir / "stage_EXT6_compact_for_gpt_v1.json", compact)
    report = [
        "# EXT-6 Stronger Local Descriptor Validation",
        "",
        "## Result",
        "",
        f"- Events: `{compact['num_events']}`",
        f"- Geometry passive top1: `{compact['geometry_passive_top1']:.4f}`",
        f"- Geometry + strong best top1: `{compact['geometry_plus_strong_best_top1']:.4f}`",
        f"- External branch top1: `{compact['external_branch_top1']:.4f}`",
        f"- External branch + strong best top1: `{compact['external_branch_plus_strong_best_top1']:.4f}`",
        f"- Mean strong margin: `{compact['mean_strong_margin']:.4f}`",
        f"- Strong controls passed: `{compact['strong_controls_passed']}`",
        "",
        "## Decision",
        "",
        compact["next_recommendation"],
    ]
    (out_dir / "stage_EXT6_report_v1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
