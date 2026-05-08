from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ext1_utils import write_csv
from experiments.run_ext4a_full_pixel_appearance_validation import (
    VARIANTS,
    as_float,
    as_int,
    build_history,
    candidate_components,
    cosine,
    crop_descriptor,
    image_path,
    load_pixel_ready_events,
    score_variant,
    state_at,
)
from experiments.run_ext5_multicategory_full_pixel_validation import category_from_seq


APP_VARIANTS = [
    "A2_geometry_plus_appearance_w005",
    "A3_geometry_plus_appearance_w010",
    "A4_geometry_plus_appearance_w020",
    "A6_external_trajectory_plus_appearance_w005",
    "A7_external_trajectory_plus_appearance_w010",
    "A8_external_trajectory_plus_appearance_w020",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXT-5C real appearance shuffle control audit.")
    p.add_argument("--lagot-root", default="data/external/lagot_annotations")
    p.add_argument("--lasot-root", default="data/external/lasot")
    p.add_argument("--ext4-dir", default="results/ext4")
    p.add_argument("--output-dir", default="results/ext5c")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stable_seed(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def zsafe_category(row: dict[str, Any]) -> str:
    return str(row.get("category") or category_from_seq(str(row.get("sequence_id", ""))))


def collect_payloads(args: argparse.Namespace) -> list[dict[str, Any]]:
    lagot, event_rows, _ = load_pixel_ready_events(args)
    hist_cache: dict[str, dict[str, list[tuple[int, tuple[float, float, float, float], Any]]]] = {}
    desc_cache: dict[tuple[str, int, tuple[float, float, float, float]], Any] = {}
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
            last_path = image_path(seq_dir, last_frame)
            key = (str(last_path), last_frame, tuple(float(x) for x in last_box))
            if key not in desc_cache:
                desc_cache[key] = crop_descriptor(last_path, last_box)
            cand_desc = desc_cache[key]
            if cand_desc is None:
                continue
            comps[iid] = c
            app_scores[iid] = cosine(query_desc, cand_desc)
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
            "app_scores": app_scores,
        })
    return payloads


def shuffle_within_event(payload: dict[str, Any]) -> dict[str, float]:
    ids = list(payload["app_scores"].keys())
    values = list(payload["app_scores"].values())
    rng = random.Random(stable_seed(payload["event_id"] + ":within"))
    rng.shuffle(values)
    return dict(zip(ids, values))


def category_shuffled(payloads: list[dict[str, Any]], payload: dict[str, Any], idx: int) -> dict[str, float]:
    ids = list(payload["app_scores"].keys())
    source = None
    n = len(payloads)
    for off in range(1, n + 1):
        cand = payloads[(idx + off * 17) % n]
        if cand["category"] != payload["category"] and cand["app_scores"]:
            source = cand
            break
    if source is None:
        return shuffle_within_event(payload)
    values = list(source["app_scores"].values())
    rng = random.Random(stable_seed(payload["event_id"] + ":category"))
    rng.shuffle(values)
    if not values:
        return {iid: 0.0 for iid in ids}
    return {iid: values[i % len(values)] for i, iid in enumerate(ids)}


def evaluate_mode(payloads: list[dict[str, Any]], mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    for idx, payload in enumerate(payloads):
        if mode == "real":
            app_scores = payload["app_scores"]
        elif mode == "within_event_shuffled":
            app_scores = shuffle_within_event(payload)
        elif mode == "category_shuffled":
            app_scores = category_shuffled(payloads, payload, idx)
        else:
            raise ValueError(mode)
        target_id = payload["target_id"]
        comps = payload["comps"]
        a0_scores = score_variant("A0_nops_geometry_passive", comps, app_scores)
        ranked_a0 = sorted(a0_scores.items(), key=lambda x: x[1], reverse=True)
        wrong_id = ranked_a0[0][0] if ranked_a0 and ranked_a0[0][0] != target_id else (ranked_a0[1][0] if len(ranked_a0) > 1 else "")
        target_app = app_scores.get(target_id, 0.0)
        wrong_app = app_scores.get(wrong_id, 0.0)
        margin_rows.append({
            "control_mode": mode,
            "event_id": payload["event_id"],
            "category": payload["category"],
            "target_instance_id_eval_only": target_id,
            "wrong_top1_id_eval_only": wrong_id,
            "target_appearance_score": target_app,
            "wrong_appearance_score": wrong_app,
            "appearance_margin_target_minus_wrong": target_app - wrong_app,
            "appearance_margin_positive": int((target_app - wrong_app) > 0),
        })
        for variant in VARIANTS:
            scores = score_variant(variant, comps, app_scores)
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
        vals = [as_float(r["appearance_margin_target_minus_wrong"]) for r in rows]
        out.append({
            "control_mode": mode,
            "num_events": len(rows),
            "appearance_margin_positive_rate": sum(1 for v in vals if v > 0) / max(len(vals), 1),
            "mean_appearance_margin": sum(vals) / max(len(vals), 1),
            "severe_negative_margin_count": sum(1 for v in vals if v < -0.25),
        })
    return out


def lookup(summary_rows: list[dict[str, Any]], mode: str, variant: str) -> dict[str, Any]:
    return next((r for r in summary_rows if r["control_mode"] == mode and r["variant"] == variant), {})


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
    summary_rows = summarize(all_events)
    margin_rows = margin_summary(all_margins)

    real_geom = lookup(summary_rows, "real", "A0_nops_geometry_passive")
    real_geom_app = max(
        (lookup(summary_rows, "real", v) for v in APP_VARIANTS if "geometry_plus" in v),
        key=lambda r: as_float(r.get("global_top1")),
    )
    real_ext = lookup(summary_rows, "real", "A5_external_trajectory_heavy")
    real_ext_app = max(
        (lookup(summary_rows, "real", v) for v in APP_VARIANTS if "external_trajectory_plus" in v),
        key=lambda r: as_float(r.get("global_top1")),
    )
    shuffled_geom_app = max(
        (lookup(summary_rows, "within_event_shuffled", v) for v in APP_VARIANTS if "geometry_plus" in v),
        key=lambda r: as_float(r.get("global_top1")),
    )
    cat_geom_app = max(
        (lookup(summary_rows, "category_shuffled", v) for v in APP_VARIANTS if "geometry_plus" in v),
        key=lambda r: as_float(r.get("global_top1")),
    )
    shuffled_ext_app = max(
        (lookup(summary_rows, "within_event_shuffled", v) for v in APP_VARIANTS if "external_trajectory_plus" in v),
        key=lambda r: as_float(r.get("global_top1")),
    )
    cat_ext_app = max(
        (lookup(summary_rows, "category_shuffled", v) for v in APP_VARIANTS if "external_trajectory_plus" in v),
        key=lambda r: as_float(r.get("global_top1")),
    )

    real_gain_geom = as_float(real_geom_app.get("global_top1")) - as_float(real_geom.get("global_top1"))
    shuffled_gain_geom = as_float(shuffled_geom_app.get("global_top1")) - as_float(real_geom.get("global_top1"))
    cat_gain_geom = as_float(cat_geom_app.get("global_top1")) - as_float(real_geom.get("global_top1"))
    real_gain_ext = as_float(real_ext_app.get("global_top1")) - as_float(real_ext.get("global_top1"))
    shuffled_gain_ext = as_float(shuffled_ext_app.get("global_top1")) - as_float(real_ext.get("global_top1"))
    cat_gain_ext = as_float(cat_ext_app.get("global_top1")) - as_float(real_ext.get("global_top1"))
    controls_passed = int(
        shuffled_gain_geom <= real_gain_geom + 1e-12
        and cat_gain_geom <= real_gain_geom + 1e-12
        and shuffled_gain_ext <= max(real_gain_ext, 0.0) + 1e-12
        and cat_gain_ext <= max(real_gain_ext, 0.0) + 1e-12
    )

    compact = {
        "stage": "EXT-5C",
        "num_events": len(payloads),
        "real_geometry_passive_top1": as_float(real_geom.get("global_top1")),
        "real_geometry_plus_appearance_best_top1": as_float(real_geom_app.get("global_top1")),
        "real_external_branch_top1": as_float(real_ext.get("global_top1")),
        "real_external_branch_plus_appearance_best_top1": as_float(real_ext_app.get("global_top1")),
        "real_geometry_appearance_gain": real_gain_geom,
        "shuffled_geometry_appearance_gain": shuffled_gain_geom,
        "category_shuffled_geometry_appearance_gain": cat_gain_geom,
        "real_external_appearance_gain": real_gain_ext,
        "shuffled_external_appearance_gain": shuffled_gain_ext,
        "category_shuffled_external_appearance_gain": cat_gain_ext,
        "appearance_controls_passed": controls_passed,
        "appearance_safe_for_external_branch": int(real_gain_ext > 0 and controls_passed),
        "appearance_safe_for_main_merge": 0,
        "next_recommendation": (
            "controls clean but appearance still does not improve external branch; keep as auxiliary diagnostic"
            if controls_passed and real_gain_ext <= 0
            else "controls indicate appearance gains are not reliable; do not integrate"
            if not controls_passed
            else "appearance may support external branch; require stronger descriptor/profile validation"
        ),
    }

    report = f"""# EXT-5C Appearance Control Audit

## Result

- Events: `{compact["num_events"]}`
- Real geometry appearance gain: `{real_gain_geom:.4f}`
- Shuffled geometry appearance gain: `{shuffled_gain_geom:.4f}`
- Category-shuffled geometry appearance gain: `{cat_gain_geom:.4f}`
- Real external-branch appearance gain: `{real_gain_ext:.4f}`
- Shuffled external-branch appearance gain: `{shuffled_gain_ext:.4f}`
- Category-shuffled external-branch appearance gain: `{cat_gain_ext:.4f}`
- Controls passed: `{controls_passed}`

## Decision

Appearance is not safe for main NOPS merge.

{compact["next_recommendation"]}
"""

    write_csv(out_dir / "stage_EXT5C_control_ablation_summary_v1.csv", summary_rows)
    write_csv(out_dir / "stage_EXT5C_control_event_results_v1.csv", all_events)
    write_csv(out_dir / "stage_EXT5C_control_margin_trace_v1.csv", all_margins)
    write_csv(out_dir / "stage_EXT5C_margin_summary_v1.csv", margin_rows)
    write_json(out_dir / "stage_EXT5C_compact_for_gpt_v1.json", compact)
    (out_dir / "stage_EXT5C_report_v1.md").write_text(report, encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
