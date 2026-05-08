from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.external.lagot_adapter import LaGOTAdapter
from experiments.ext1_utils import (
    Box,
    box_descriptor,
    box_iou,
    center_distance,
    l2,
    normalize_distance,
    read_csv,
    size_similarity,
    trajectory_prediction,
    write_csv,
)


VARIANTS = [
    "A0_nops_geometry_passive",
    "A1_appearance_nn",
    "A2_geometry_plus_appearance_w010",
    "A3_geometry_plus_appearance_w020",
    "A4_external_trajectory_heavy",
    "A5_external_trajectory_plus_appearance_w010",
    "A6_external_trajectory_plus_appearance_w020",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXT-4A full-pixel appearance memory-only validation.")
    p.add_argument("--lagot-root", default="data/external/lagot_annotations")
    p.add_argument("--lasot-root", default="data/external/lasot")
    p.add_argument("--ext4-dir", default="results/ext4")
    p.add_argument("--output-dir", default="results/ext4a")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def as_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def lagot_to_lasot_sequence(sequence_id: str) -> str:
    s = sequence_id
    if s.startswith("lagot_"):
        s = s[len("lagot_"):]
    head, sep, tail = s.rpartition("_")
    if sep and tail.isdigit():
        return head
    return s


def find_lasot_sequence_dir(root: Path, sequence_id: str) -> Path | None:
    direct = root / sequence_id
    if (direct / "img").exists():
        return direct
    category = sequence_id.split("-")[0]
    nested = root / category / sequence_id
    if (nested / "img").exists():
        return nested
    if not root.exists():
        return None
    for img_dir in root.rglob("img"):
        if img_dir.is_dir() and img_dir.parent.name == sequence_id:
            return img_dir.parent
    return None


def image_path(seq_dir: Path, frame_idx_1based: int) -> Path:
    return seq_dir / "img" / f"{int(frame_idx_1based):08d}.jpg"


def clamp_crop(box: Box, width: int, height: int, margin: float = 0.08) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    x1 -= w * margin
    x2 += w * margin
    y1 -= h * margin
    y2 += h * margin
    ix1 = max(0, min(width - 1, int(math.floor(x1))))
    iy1 = max(0, min(height - 1, int(math.floor(y1))))
    ix2 = max(ix1 + 1, min(width, int(math.ceil(x2))))
    iy2 = max(iy1 + 1, min(height, int(math.ceil(y2))))
    return ix1, iy1, ix2, iy2


def normalized_hist(vals: np.ndarray, bins: int, lo: float, hi: float) -> np.ndarray:
    hist, _ = np.histogram(vals, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float32)
    denom = float(hist.sum())
    if denom > 0:
        hist /= denom
    return hist


def crop_descriptor(path: Path, box: Box) -> np.ndarray | None:
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
    rgb_hist = [normalized_hist(arr[:, :, c].ravel(), 8, 0.0, 1.0) for c in range(3)]
    gray_hist = normalized_hist(gray.ravel(), 16, 0.0, 1.0)
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    ori = (np.arctan2(gy, gx) + math.pi) / (2 * math.pi)
    edge_hist, _ = np.histogram(ori.ravel(), bins=8, range=(0.0, 1.0), weights=mag.ravel())
    edge_hist = edge_hist.astype(np.float32)
    if edge_hist.sum() > 0:
        edge_hist /= edge_hist.sum()
    stats = np.array([
        arr[:, :, 0].mean(),
        arr[:, :, 1].mean(),
        arr[:, :, 2].mean(),
        arr[:, :, 0].std(),
        arr[:, :, 1].std(),
        arr[:, :, 2].std(),
        mag.mean(),
        mag.std(),
    ], dtype=np.float32)
    vec = np.concatenate([*rgb_hist, gray_hist, edge_hist, stats])
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return vec / norm


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return -1.0
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8))


def zscores(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=np.float64)
    mu = float(arr.mean())
    sigma = float(arr.std())
    if sigma <= 1e-12:
        return {k: 0.0 for k in values}
    return {k: (float(v) - mu) / sigma for k, v in values.items()}


def candidate_components(
    query_box: Box,
    states_all: list[tuple[int, Box, Any]],
    width: float,
    height: float,
    reappear_frame: int,
) -> dict[str, float | Box | int]:
    states = [st for st in states_all if st[0] < reappear_frame]
    if not states:
        return {}
    last_frame, last_box, _ = states[-1]
    pred_box = trajectory_prediction(states[-8:]) or last_box
    q_desc = box_descriptor(query_box, width, height)
    last_desc = box_descriptor(last_box, width, height)
    candidate_age = max(0, int(reappear_frame - last_frame))
    dist_pred = normalize_distance(center_distance(query_box, pred_box), width, height)
    shape_last = -l2(q_desc[2:], last_desc[2:])
    trajectory_score = 0.8 * box_iou(query_box, pred_box) + 0.4 * size_similarity(query_box, pred_box) - dist_pred
    return {
        "last_frame": int(last_frame),
        "last_box": last_box,
        "pred_box": pred_box,
        "candidate_age": float(candidate_age),
        "recency_score": -min(1.0, candidate_age / 200.0),
        "trajectory_score": trajectory_score,
        "shape_score": shape_last,
        "nops_geometry_score": 0.45 * (0.8 * box_iou(query_box, pred_box) - dist_pred) + 0.35 * shape_last + 0.20 * (-min(1.0, candidate_age / 200.0)),
        "external_trajectory_heavy_score": 0.85 * trajectory_score + 0.15 * shape_last,
    }


def score_variant(variant: str, comps: dict[str, dict[str, Any]], app_scores: dict[str, float]) -> dict[str, float]:
    geom = {iid: float(c["nops_geometry_score"]) for iid, c in comps.items()}
    ext = {iid: float(c["external_trajectory_heavy_score"]) for iid, c in comps.items()}
    app = app_scores
    z_geom = zscores(geom)
    z_ext = zscores(ext)
    z_app = zscores(app)
    if variant == "A0_nops_geometry_passive":
        return geom
    if variant == "A1_appearance_nn":
        return app
    if variant == "A2_geometry_plus_appearance_w010":
        return {iid: z_geom.get(iid, 0.0) + 0.10 * z_app.get(iid, 0.0) for iid in comps}
    if variant == "A3_geometry_plus_appearance_w020":
        return {iid: z_geom.get(iid, 0.0) + 0.20 * z_app.get(iid, 0.0) for iid in comps}
    if variant == "A4_external_trajectory_heavy":
        return ext
    if variant == "A5_external_trajectory_plus_appearance_w010":
        return {iid: z_ext.get(iid, 0.0) + 0.10 * z_app.get(iid, 0.0) for iid in comps}
    if variant == "A6_external_trajectory_plus_appearance_w020":
        return {iid: z_ext.get(iid, 0.0) + 0.20 * z_app.get(iid, 0.0) for iid in comps}
    raise ValueError(variant)


def load_pixel_ready_events(args: argparse.Namespace) -> tuple[LaGOTAdapter, list[dict[str, Any]], dict[str, Path]]:
    lagot = LaGOTAdapter(args.lagot_root)
    rows = read_csv(Path(args.ext4_dir) / "stage_EXT4_lagot_lasot_sequence_linkage_v1.csv")
    seq_dirs: dict[str, Path] = {}
    for row in rows:
        if as_int(row.get("pixel_ready")) != 1:
            continue
        seq_dir = Path(row["lasot_sequence_dir"])
        if seq_dir.exists():
            seq_dirs[row["lagot_sequence_id"]] = seq_dir
    event_rows: list[dict[str, Any]] = []
    for lagot_seq, seq_dir in seq_dirs.items():
        for idx, ev in enumerate(lagot.derive_events(lagot_seq)):
            if ev.gap_length < 3:
                continue
            event_rows.append({
                "dataset_name": "lagot_lasot_pixels",
                "sequence_id": lagot_seq,
                "lasot_sequence_id": lagot_to_lasot_sequence(lagot_seq),
                "event_id": f"lagot_lasot_pixels:{lagot_seq}:{idx}:{ev.instance_id}",
                "instance_id": ev.instance_id,
                "disappear_frame": ev.disappear_frame,
                "reappear_frame": ev.reappear_frame,
                "gap_length": ev.gap_length,
                "category_id": ev.metadata.get("category_id", ""),
                "seq_dir": str(seq_dir),
            })
    return lagot, event_rows, seq_dirs


def build_history(lagot: LaGOTAdapter, sequence_id: str) -> dict[str, list[tuple[int, Box, Any]]]:
    hist: dict[str, list[tuple[int, Box, Any]]] = defaultdict(list)
    category = lagot.category_from_sequence(sequence_id)
    for row in lagot._read_gt_rows(sequence_id):
        iid = f"{sequence_id}:{row['instance_id']}"
        hist[iid].append((int(row["frame_idx"]), row["box"], category))
    for states in hist.values():
        states.sort(key=lambda x: x[0])
    return hist


def state_at(hist: dict[str, list[tuple[int, Box, Any]]], iid: str, frame_idx: int) -> tuple[int, Box, Any] | None:
    for state in hist.get(iid, []):
        if state[0] == frame_idx:
            return state
    return None


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    lagot, event_rows, _ = load_pixel_ready_events(args)
    event_out: list[dict[str, Any]] = []
    margin_out: list[dict[str, Any]] = []
    desc_quality: list[dict[str, Any]] = []
    hist_cache: dict[str, dict[str, list[tuple[int, Box, Any]]]] = {}
    desc_cache: dict[tuple[str, int, tuple[int, int, int, int]], np.ndarray | None] = {}

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
        query_desc = crop_descriptor(img_path, query_box)
        if query_desc is None:
            continue
        candidates = {iid: states for iid, states in hist.items() if any(st[0] < reappear for st in states)}
        comps: dict[str, dict[str, Any]] = {}
        app_scores: dict[str, float] = {}
        for iid, states in candidates.items():
            c = candidate_components(query_box, states, float(width), float(height), reappear)
            if not c:
                continue
            last_frame = int(c["last_frame"])
            last_box = c["last_box"]
            assert isinstance(last_box, tuple)
            last_path = image_path(seq_dir, last_frame)
            crop_key = (str(last_path), last_frame, tuple(clamp_crop(last_box, width, height)))
            if crop_key not in desc_cache:
                desc_cache[crop_key] = crop_descriptor(last_path, last_box)
            cand_desc = desc_cache[crop_key]
            if cand_desc is None:
                continue
            comps[iid] = c
            app_scores[iid] = cosine(query_desc, cand_desc)
        if target_id not in comps:
            continue

        target_app = app_scores.get(target_id, -1.0)
        # Current geometry top competitor for margin diagnostics.
        a0_scores = score_variant("A0_nops_geometry_passive", comps, app_scores)
        wrong_ranked = sorted(a0_scores.items(), key=lambda x: x[1], reverse=True)
        wrong_id = wrong_ranked[0][0] if wrong_ranked and wrong_ranked[0][0] != target_id else (wrong_ranked[1][0] if len(wrong_ranked) > 1 else "")
        wrong_app = app_scores.get(wrong_id, -1.0)
        margin_out.append({
            "event_id": row["event_id"],
            "sequence_id": seq,
            "target_instance_id_eval_only": target_id,
            "wrong_top1_id_eval_only": wrong_id,
            "target_appearance_score": target_app,
            "wrong_appearance_score": wrong_app,
            "appearance_margin_target_minus_wrong": target_app - wrong_app,
            "appearance_margin_positive": int((target_app - wrong_app) > 0),
            "target_gap_length": row["gap_length"],
            "candidate_count": len(comps),
        })
        desc_quality.append({
            "event_id": row["event_id"],
            "query_descriptor_available": 1,
            "candidate_descriptor_count": len(app_scores),
            "target_descriptor_available": int(target_id in app_scores),
            "appearance_score_mean": float(np.mean(list(app_scores.values()))) if app_scores else 0.0,
            "appearance_score_std": float(np.std(list(app_scores.values()))) if app_scores else 0.0,
            "descriptor_degenerate": 0,
        })

        for variant in VARIANTS:
            scores = score_variant(variant, comps, app_scores)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ranked_ids = [iid for iid, _ in ranked]
            top1 = ranked_ids[0] if ranked_ids else ""
            top1_hit = int(top1 == target_id)
            top3_hit = int(target_id in ranked_ids[:3])
            top5_hit = int(target_id in ranked_ids[:5])
            if not top5_hit:
                failure = "target_not_in_top5"
            elif not top1_hit:
                failure = "target_in_top5_but_wrong_top1"
            else:
                failure = ""
            event_out.append({
                "dataset_name": row["dataset_name"],
                "sequence_id": seq,
                "lasot_sequence_id": row["lasot_sequence_id"],
                "event_id": row["event_id"],
                "variant": variant,
                "proposal_mode": "oracle_gt_box_memory_only",
                "pixel_mode": "full_pixel_crop_descriptor",
                "target_instance_id_eval_only": target_id,
                "predicted_memory_id": top1,
                "top1": top1_hit,
                "top3": top3_hit,
                "top5": top5_hit,
                "false_retrieval": int(not top1_hit),
                "target_in_top5_but_lost_top1": int(top5_hit and not top1_hit),
                "target_not_in_top5": int(not top5_hit),
                "candidate_count": len(comps),
                "gap_length": row["gap_length"],
                "failure_reason": failure,
            })
    return event_out, margin_out, desc_quality


def summarize(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        by_variant[row["variant"]].append(row)
    out: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows = by_variant.get(variant, [])
        n = len(rows)
        out.append({
            "variant": variant,
            "num_events": n,
            "global_top1": sum(as_int(r["top1"]) for r in rows) / max(n, 1),
            "global_top3": sum(as_int(r["top3"]) for r in rows) / max(n, 1),
            "global_top5": sum(as_int(r["top5"]) for r in rows) / max(n, 1),
            "false_retrieval_rate": sum(as_int(r["false_retrieval"]) for r in rows) / max(n, 1),
            "target_in_top5_but_lost_top1_count": sum(as_int(r["target_in_top5_but_lost_top1"]) for r in rows),
            "target_not_in_top5_count": sum(as_int(r["target_not_in_top5"]) for r in rows),
        })
    best = max(out, key=lambda r: (as_float(r["global_top1"]), as_float(r["global_top3"]), as_float(r["global_top5"])), default=None)
    if best:
        for row in out:
            row["selected_as_best"] = int(row["variant"] == best["variant"])
    return out


def failure_taxonomy(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in event_rows:
        out.append({
            "dataset_name": row["dataset_name"],
            "event_id": row["event_id"],
            "variant": row["variant"],
            "failure_reason": row["failure_reason"],
        })
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    event_rows, margin_rows, desc_quality = evaluate(args)
    summary_rows = summarize(event_rows)
    failure_rows = failure_taxonomy(event_rows)

    best = next((r for r in summary_rows if as_int(r.get("selected_as_best")) == 1), summary_rows[0] if summary_rows else {})
    a0 = next((r for r in summary_rows if r["variant"] == "A0_nops_geometry_passive"), {})
    ext = next((r for r in summary_rows if r["variant"] == "A4_external_trajectory_heavy"), {})
    app_variants = [r for r in summary_rows if "appearance" in r["variant"]]
    best_app = max(app_variants, key=lambda r: as_float(r["global_top1"]), default={})
    margins = [as_float(r["appearance_margin_target_minus_wrong"]) for r in margin_rows]
    positive_rate = sum(1 for m in margins if m > 0) / max(len(margins), 1)
    mean_margin = sum(margins) / max(len(margins), 1)
    appearance_helped = int(as_float(best_app.get("global_top1")) > as_float(a0.get("global_top1")))
    appearance_beats_external_branch = int(as_float(best_app.get("global_top1")) > as_float(ext.get("global_top1")))

    compact = {
        "stage": "EXT-4A",
        "pixel_subset": "LaSOT dog category linked to LaGOT annotations",
        "num_events": as_int(best.get("num_events")),
        "best_variant": best.get("variant", ""),
        "best_top1": as_float(best.get("global_top1")),
        "best_top3": as_float(best.get("global_top3")),
        "best_top5": as_float(best.get("global_top5")),
        "geometry_passive_top1": as_float(a0.get("global_top1")),
        "external_geometry_branch_top1": as_float(ext.get("global_top1")),
        "best_appearance_variant": best_app.get("variant", ""),
        "best_appearance_top1": as_float(best_app.get("global_top1")),
        "appearance_helped_vs_geometry_passive": appearance_helped,
        "appearance_beats_external_geometry_branch": appearance_beats_external_branch,
        "appearance_margin_positive_rate": positive_rate,
        "mean_appearance_margin": mean_margin,
        "full_pixel_validation_ready": 1,
        "safe_main_merge": 0,
        "next_recommendation": (
            "appearance helps current passive but does not beat external geometry branch; run larger multi-category full-pixel validation and descriptor failure audit"
            if appearance_helped and not appearance_beats_external_branch else
            "appearance beats external geometry branch on this subset; run larger multi-category full-pixel validation before integration"
            if appearance_beats_external_branch else
            "appearance crop descriptor does not beat geometry passive on this subset; analyze descriptor/failure taxonomy before integration"
        ),
    }

    report = f"""# EXT-4A Full-Pixel Appearance Validation

## Scope

This is oracle-proposal, memory-only validation on LaSOT dog pixels linked to LaGOT annotations.

It does not test detection or full perception.

## Result

- Events: `{compact["num_events"]}`
- Geometry passive top1: `{compact["geometry_passive_top1"]:.4f}`
- External geometry branch top1: `{compact["external_geometry_branch_top1"]:.4f}`
- Best variant: `{compact["best_variant"]}`
- Best top1: `{compact["best_top1"]:.4f}`
- Best appearance variant: `{compact["best_appearance_variant"]}`
- Best appearance top1: `{compact["best_appearance_top1"]:.4f}`
- Appearance helped vs geometry passive: `{appearance_helped}`
- Appearance beat external geometry branch: `{appearance_beats_external_branch}`
- Mean appearance margin: `{mean_margin:.6f}`
- Appearance margin positive rate: `{positive_rate:.4f}`

## Decision

Do not merge into main NOPS.

Appearance improved the current passive geometry score on this dog subset, but it did not beat the isolated external geometry branch, and the mean target-vs-wrong appearance margin is still negative.

Next: run larger multi-category full-pixel validation and audit appearance descriptor failures before any integration.
"""

    write_csv(out_dir / "stage_EXT4A_ablation_summary_v1.csv", summary_rows)
    write_csv(out_dir / "stage_EXT4A_event_results_v1.csv", event_rows)
    write_csv(out_dir / "stage_EXT4A_appearance_margin_trace_v1.csv", margin_rows)
    write_csv(out_dir / "stage_EXT4A_descriptor_quality_v1.csv", desc_quality)
    write_csv(out_dir / "stage_EXT4A_failure_taxonomy_v1.csv", failure_rows)
    write_json(out_dir / "stage_EXT4A_compact_for_gpt_v1.json", compact)
    (out_dir / "stage_EXT4A_report_v1.md").write_text(report, encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
