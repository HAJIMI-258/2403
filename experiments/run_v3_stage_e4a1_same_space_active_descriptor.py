from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator
from experiments import run_v3_stage_e31_retrieval_competition_repair as e31
from experiments import run_v3_stage_e32b_safe_top3_rerank as e32b
from experiments import run_v3_stage_e34r_support_trajectory_refinement as e34r
from experiments import run_v3_stage_e4a_active_evidence_acquisition as e4a
from experiments.phase3r_utils import build_phase3_scenario_map, load_config_payload


FOCUS_EVENT_IDS = {"M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"}
PASSIVE_BASELINE = {
    "global_top1": 0.4117647058823529,
    "global_top5": 0.7647058823529411,
    "false_bundle_retrieval_rate": 0.5882352941176471,
    "focus_success_count": 3,
    "mean_active_evidence_margin": -0.004707469659693101,
}
E4AD_BASELINE = {
    "descriptor_margin_positive_rate": 0.23333333333333334,
    "same_space_margin_positive_rate": 0.28888888888888886,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run v3 Stage E4A.1 same-space active evidence descriptor.")
    p.add_argument("--cache", default="results/v3_e4a/cache/runtime_collection_cache_v1.pkl")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--output-dir", default="results/v3_e4a1")
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def descriptor_stats(desc: np.ndarray | None) -> dict[str, float | int]:
    if desc is None:
        return {
            "descriptor_norm": 0.0,
            "descriptor_entropy": 0.0,
            "descriptor_variance": 0.0,
            "descriptor_degenerate": 1,
        }
    d = np.asarray(desc, dtype=np.float32).reshape(-1)
    if d.size == 0:
        return {
            "descriptor_norm": 0.0,
            "descriptor_entropy": 0.0,
            "descriptor_variance": 0.0,
            "descriptor_degenerate": 1,
        }
    hist = np.abs(d) / max(float(np.abs(d).sum()), 1e-8)
    return {
        "descriptor_norm": float(np.linalg.norm(d)),
        "descriptor_entropy": float(-(hist * np.log(hist + 1e-8)).sum()),
        "descriptor_variance": float(np.var(d)),
        "descriptor_degenerate": int(float(np.linalg.norm(d)) < 1e-6 or float(np.var(d)) < 1e-6),
    }


def geomean(values: list[float]) -> float:
    vals = [max(1e-6, min(1.0, float(v))) for v in values]
    if not vals:
        return 0.0
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def proposal_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(area_a + area_b - inter, 1e-8))


def reconstruct_source_box(bundle: dict[str, Any], frame_shape: tuple[int, int, int] | tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    support = np.asarray(bundle["support_trajectory_signature"], dtype=np.float32).reshape(-1)
    disp = np.asarray(bundle["disappearance_boundary_signature"], dtype=np.float32).reshape(-1)
    bw = float(support[8] * 128.0) if support.size > 8 else 16.0
    bh = float(support[9] * 128.0) if support.size > 9 else 16.0
    cx = float(disp[1] * w) if disp.size > 1 and disp[1] > 0 else w * 0.5
    cy = float(disp[2] * h) if disp.size > 2 and disp[2] > 0 else h * 0.5
    x1 = max(0, int(round(cx - bw / 2.0)))
    y1 = max(0, int(round(cy - bh / 2.0)))
    x2 = min(w, int(round(cx + bw / 2.0)))
    y2 = min(h, int(round(cy + bh / 2.0)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return (x1, y1, x2, y2)


def mode_for_policy(policy: str) -> str:
    if "random" in policy:
        return "random"
    if "neighbor" in policy:
        return "neighbor"
    if "objectness" in policy:
        return "objectness"
    if "boundary" in policy or "support" in policy:
        return "boundary"
    return "boundary"


def descriptor_types_for_policy(policy: str) -> list[str]:
    if policy == "combined":
        return ["boundary", "neighbor"]
    return [mode_for_policy(policy)]


def load_source_frames(config_path: str, bundle_by_id: dict[int, dict[str, Any]], seed: int) -> dict[str, dict[int, np.ndarray]]:
    scenario_map = build_phase3_scenario_map(config_path)
    needed: dict[str, set[int]] = {}
    for b in bundle_by_id.values():
        needed.setdefault(str(b["scenario_name"]), set()).add(int(b["last_source_frame"]))
    frames: dict[str, dict[int, np.ndarray]] = {}
    for scenario_name, frame_ids in needed.items():
        sequence = SyntheticStreamGenerator(scenario_map[scenario_name], seed=seed).generate_sequence(0)
        by_frame: dict[int, np.ndarray] = {}
        for fr in sequence.frames:
            idx = int(fr.frame_index)
            if idx in frame_ids:
                by_frame[idx] = np.asarray(fr.frame)
        frames[scenario_name] = by_frame
    return frames


def crop_with_mode(frame: np.ndarray, box: tuple[int, int, int, int], mode: str, event_key: str, proposals=None) -> tuple[int, int, int, int]:
    return e4a.crop_box_around(box, frame.shape, mode, proposals or [], event_key)


def build_historical_store(cache: dict[str, Any], config_path: str, seed: int):
    bundle_by_id = cache["bundle_by_id"]
    source_frames = load_source_frames(config_path, bundle_by_id, seed)
    store: dict[int, dict[str, dict[str, Any]]] = {}
    trace_rows: list[dict[str, Any]] = []
    for bid, b in sorted(bundle_by_id.items()):
        scenario = str(b["scenario_name"])
        frame_idx = int(b["last_source_frame"])
        frame = source_frames.get(scenario, {}).get(frame_idx)
        store[int(bid)] = {}
        for dtype in ("center", "boundary", "neighbor", "objectness", "combined"):
            available = frame is not None
            missing = "" if available else "historical_frame_missing"
            desc_vec = None
            crop_box = ""
            if available:
                source_box = reconstruct_source_box(b, frame.shape)
                if dtype == "combined":
                    modes = ["boundary", "neighbor"]
                elif dtype == "center":
                    modes = ["center"]
                else:
                    modes = [dtype]
                descs = []
                crop_boxes = []
                for mode in modes:
                    # Historical descriptor intentionally uses the same crop descriptor
                    # function as current evidence. Heatmap/proposals are omitted on both
                    # sides for same-space compatibility in this stage.
                    cbox = crop_with_mode(frame, source_box, mode, f"hist:{bid}:{dtype}", [])
                    d = e4a.crop_descriptor(frame, None, cbox, source_box)
                    descs.append(d["descriptor"])
                    crop_boxes.append(cbox)
                desc_vec = np.mean(np.stack(descs), axis=0).astype(np.float32)
                crop_box = ";".join("|".join(str(v) for v in c) for c in crop_boxes)
            stats = descriptor_stats(desc_vec)
            store[int(bid)][dtype] = {
                "descriptor": desc_vec,
                "available": int(available),
                "missing_reason": missing,
                "crop_box": crop_box,
                **stats,
            }
            trace_rows.append({
                "bundle_id": int(bid),
                "memory_anchor_id": b["memory_anchor_id"],
                "scenario_name": scenario,
                "source_track_id": int(b["primary_source_track_id"]),
                "source_prototype_id": int(b["primary_source_prototype_id"]),
                "created_frame": int(b["created_frame"]),
                "last_source_frame": frame_idx,
                "descriptor_type": dtype,
                "historical_crop_box": crop_box,
                "historical_descriptor_available": int(available),
                "historical_descriptor_norm": stats["descriptor_norm"],
                "historical_descriptor_entropy": stats["descriptor_entropy"],
                "historical_descriptor_variance": stats["descriptor_variance"],
                "historical_descriptor_degenerate": stats["descriptor_degenerate"],
                "missing_reason": missing,
            })
    return store, trace_rows


def current_descriptor_for_event(event: dict[str, Any], context: dict[str, Any], policy: str):
    pbox = tuple(int(v) for v in event["cue"]["box"])
    rows = []
    descs = []
    for dtype in descriptor_types_for_policy(policy):
        cbox = crop_with_mode(context["frame"], pbox, dtype, f"{event['event_id']}:{policy}", context.get("proposals", []))
        # Same-space scoring uses frame-only descriptors on both historical and current sides.
        desc = e4a.crop_descriptor(context["frame"], None, cbox, pbox)
        descs.append(desc["descriptor"])
        neighbor_count = sum(
            1
            for p in context.get("proposals", [])
            if proposal_iou(tuple(p["box"]), cbox) > 0.01 and tuple(p["box"]) != tuple(pbox)
        )
        stats = descriptor_stats(desc["descriptor"])
        rows.append({
            "event_id": event["event_id"],
            "policy_name": policy,
            "descriptor_type": dtype,
            "current_crop_box": "|".join(str(v) for v in cbox),
            "current_descriptor_norm": stats["descriptor_norm"],
            "current_descriptor_entropy": stats["descriptor_entropy"],
            "current_descriptor_variance": stats["descriptor_variance"],
            "current_descriptor_degenerate": stats["descriptor_degenerate"],
            "crop_overlaps_proposal": proposal_iou(cbox, pbox),
            "crop_contains_neighbor": int(neighbor_count > 0),
            "objectness_crop_mean": desc["objectness_crop_mean"],
            "edge_density": desc["edge_density"],
        })
    return np.mean(np.stack(descs), axis=0).astype(np.float32), rows, ("combined" if policy == "combined" else descriptor_types_for_policy(policy)[0])


def ablations() -> dict[str, dict[str, Any]]:
    entries = {"A0_passive_E34r_baseline": {"policy": "passive", "weight": 0.0}}
    idx = 1
    policies = [
        ("random", "random"),
        ("boundary", "support_boundary"),
        ("neighbor", "neighbor_context"),
        ("memory_uncertainty", "memory_uncertainty"),
        ("combined", "combined"),
    ]
    for weight in (0.03, 0.05):
        for label, policy in policies:
            entries[f"A{idx}_{label}_same_space_w{int(weight * 1000):03d}"] = {"policy": policy, "weight": weight}
            idx += 1
    for weight in (0.08, 0.10):
        for label, policy in (("memory_uncertainty", "memory_uncertainty"), ("combined", "combined")):
            entries[f"A{idx}_{label}_same_space_w{int(weight * 1000):03d}"] = {"policy": policy, "weight": weight}
            idx += 1
    return entries


def score_with_same_space(
    event: dict[str, Any],
    scored: dict[str, Any],
    current_desc: np.ndarray | None,
    descriptor_type: str,
    hist_store: dict[int, dict[str, dict[str, Any]]],
    weight: float,
):
    if current_desc is None or weight <= 0:
        return list(scored["final_topk"]), []
    active_rows = []
    margin_trace = []
    for row in scored["candidate_pool"]:
        bid = int(row["bundle_id"])
        href = hist_store.get(bid, {}).get(descriptor_type)
        if href is None or not int(href["available"]):
            compat = 0.0
        else:
            compat = e4a.cosine(current_desc, href["descriptor"])
        rr = dict(row)
        rr["same_space_compatibility"] = compat
        rr["same_space_score"] = safe_float(row.get("e34r_score", row.get("final_score"))) + weight * compat
        active_rows.append(rr)
    active_rows.sort(key=lambda r: r["same_space_score"], reverse=True)
    final_top = e31.diversify_candidates([dict(r, final_score=r["same_space_score"]) for r in active_rows], e34r.e34.base_cfg())
    target_id = event["target_bundle_id"]
    passive_top1 = scored["final_topk"][0] if scored["final_topk"] else None
    target_row = next((r for r in active_rows if target_id is not None and int(r["bundle_id"]) == int(target_id)), None)
    wrong_row = next((r for r in active_rows if passive_top1 is not None and int(r["bundle_id"]) == int(passive_top1["bundle_id"])), None)
    if target_id is not None and passive_top1 is not None:
        t_ref = hist_store.get(int(target_id), {}).get(descriptor_type)
        w_ref = hist_store.get(int(passive_top1["bundle_id"]), {}).get(descriptor_type)
        target_available = int(t_ref is not None and int(t_ref["available"]) == 1)
        wrong_available = int(w_ref is not None and int(w_ref["available"]) == 1)
        target_compat = e4a.cosine(current_desc, t_ref["descriptor"]) if target_available else 0.0
        wrong_compat = e4a.cosine(current_desc, w_ref["descriptor"]) if wrong_available else 0.0
        target_degen = int(t_ref["descriptor_degenerate"]) if target_available else 1
        wrong_degen = int(w_ref["descriptor_degenerate"]) if wrong_available else 1
        current_degen = int(descriptor_stats(current_desc)["descriptor_degenerate"])
        margin = target_compat - wrong_compat
        if not target_available:
            reason = "target_historical_descriptor_missing"
        elif not wrong_available:
            reason = "wrong_historical_descriptor_missing"
        elif current_degen:
            reason = "current_descriptor_degenerate"
        elif target_degen:
            reason = "target_descriptor_degenerate"
        elif wrong_degen:
            reason = "wrong_descriptor_degenerate"
        elif margin > 0:
            reason = "same_space_discriminative"
        else:
            reason = "same_space_not_discriminative"
        margin_trace.append({
            "event_id": event["event_id"],
            "policy_name": descriptor_type,
            "descriptor_type": descriptor_type,
            "target_bundle_id": int(target_id),
            "wrong_top1_bundle_id": int(passive_top1["bundle_id"]),
            "target_descriptor_available": target_available,
            "wrong_descriptor_available": wrong_available,
            "target_same_space_compat": target_compat,
            "wrong_same_space_compat": wrong_compat,
            "same_space_margin_target_minus_wrong": margin,
            "pseudo_ref_margin_from_E4AD_if_available": "",
            "same_space_beats_pseudo_ref": "",
            "target_descriptor_degenerate": target_degen,
            "wrong_descriptor_degenerate": wrong_degen,
            "current_descriptor_degenerate": current_degen,
            "margin_positive": int(margin > 0),
            "failure_reason": reason,
            "target_candidate_same_space_compatibility": "" if target_row is None else target_row.get("same_space_compatibility", ""),
            "wrong_candidate_same_space_compatibility": "" if wrong_row is None else wrong_row.get("same_space_compatibility", ""),
        })
    return final_top, margin_trace


def evaluate_ablation(name: str, cfg: dict[str, Any], cache, hist_store, proto_counter, track_counter, lineage_counter):
    bundle_by_id = cache["bundle_by_id"]
    passive_cfg = e34r.ablation_cfgs()[e34r.BASELINE_NAME]
    retrieval_rows: list[dict[str, Any]] = []
    current_trace: list[dict[str, Any]] = []
    margin_trace: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    focus_rows: list[dict[str, Any]] = []
    for event in sorted(cache["event_records"], key=lambda r: (r["scenario_name"], int(r["frame_idx"]), r["event_id"])):
        if int(event["proposal_detected"]) != 1:
            continue
        scored = e34r.score_event(event, bundle_by_id, passive_cfg, proto_counter, track_counter, lineage_counter)
        passive_top = list(scored["final_topk"])
        final_top = passive_top
        uncertainty = e4a.uncertainty_from_rows(passive_top)
        context = cache["proposal_context"].get(f"{event['scenario_name']}:{int(event['frame_idx'])}")
        active_triggered = int(cfg["policy"] != "passive" and uncertainty["trigger"] == 1 and context is not None and event["cue"] is not None)
        descriptor_type = ""
        current_desc = None
        if active_triggered:
            current_desc, cur_rows, descriptor_type = current_descriptor_for_event(event, context, str(cfg["policy"]))
            current_trace.extend(cur_rows)
            final_top, margins = score_with_same_space(event, scored, current_desc, descriptor_type, hist_store, float(cfg["weight"]))
            margin_trace.extend([dict(m, ablation_name=name) for m in margins])
        target_id = event["target_bundle_id"]
        final_ids = [int(r["bundle_id"]) for r in final_top]
        passive_ids = [int(r["bundle_id"]) for r in passive_top]
        top1_hit = int(target_id is not None and final_ids[:1] == [int(target_id)])
        top3_hit = int(target_id is not None and int(target_id) in set(final_ids[:3]))
        top5_hit = int(target_id is not None and int(target_id) in set(final_ids[:5]))
        passive_top1_hit = int(target_id is not None and passive_ids[:1] == [int(target_id)])
        false_retrieval = int(final_ids and top1_hit == 0)
        active_resolved = int(passive_top1_hit == 0 and top1_hit == 1)
        active_false = int(passive_top1_hit == 1 and top1_hit == 0)
        row = {
            "ablation_name": name,
            "event_id": event["event_id"],
            "scenario_name": event["scenario_name"],
            "proposal_detected": int(event["proposal_detected"]),
            "target_bundle_id": "" if target_id is None else int(target_id),
            "target_bundle_retrieved_top1": top1_hit,
            "target_bundle_retrieved_top3": top3_hit,
            "target_bundle_retrieved_top5": top5_hit,
            "pattern_completion_success": top1_hit,
            "false_bundle_retrieval": false_retrieval,
            "top1_bundle_id": "" if not final_top else int(final_top[0]["bundle_id"]),
            "top5_bundle_ids": "|".join(str(v) for v in final_ids[:5]),
            "active_triggered": active_triggered,
            "descriptor_type": descriptor_type,
        }
        retrieval_rows.append(row)
        compare_rows.append({
            "ablation_name": name,
            "event_id": event["event_id"],
            "passive_top1_hit": passive_top1_hit,
            "active_top1_hit": top1_hit,
            "passive_top1_bundle": "" if not passive_top else int(passive_top[0]["bundle_id"]),
            "active_top1_bundle": "" if not final_top else int(final_top[0]["bundle_id"]),
            "active_resolved": active_resolved,
            "active_false_rescue": active_false,
        })
        if event["event_id"] in FOCUS_EVENT_IDS:
            focus_rows.append(row)
        if false_retrieval:
            if not top5_hit:
                reason = "target_not_in_top5"
            elif top3_hit:
                reason = "target_in_top3_but_lost_top1"
            else:
                reason = "target_in_top5_but_lost_top3"
            failure_rows.append({"ablation_name": name, "event_id": event["event_id"], "failure_reason": reason, "active_triggered": active_triggered})
    proposal_rows = retrieval_rows
    margins = [safe_float(r["same_space_margin_target_minus_wrong"]) for r in margin_trace]
    positive = [m for m in margins if m > 0]
    missing = [r for r in margin_trace if int(r["target_descriptor_available"]) == 0 or int(r["wrong_descriptor_available"]) == 0]
    degen = [r for r in margin_trace if int(r["target_descriptor_degenerate"]) == 1 or int(r["wrong_descriptor_degenerate"]) == 1 or int(r["current_descriptor_degenerate"]) == 1]
    summary = {
        "ablation_name": name,
        "global_top1": float(np.mean([int(r["target_bundle_retrieved_top1"]) for r in proposal_rows])) if proposal_rows else 0.0,
        "global_top3": float(np.mean([int(r["target_bundle_retrieved_top3"]) for r in proposal_rows])) if proposal_rows else 0.0,
        "global_top5": float(np.mean([int(r["target_bundle_retrieved_top5"]) for r in proposal_rows])) if proposal_rows else 0.0,
        "false_bundle_retrieval_rate": float(np.mean([int(r["false_bundle_retrieval"]) for r in proposal_rows])) if proposal_rows else 0.0,
        "focus_success_count": int(sum(int(r["pattern_completion_success"]) for r in focus_rows)),
        "regression_event_count": int(sum(int(r["active_false_rescue"]) for r in compare_rows)),
        "active_trigger_rate": float(np.mean([int(r["active_triggered"]) for r in proposal_rows])) if proposal_rows else 0.0,
        "mean_fixation_count": float(np.mean([2 if r["descriptor_type"] == "combined" else (1 if r["active_triggered"] else 0) for r in proposal_rows])) if proposal_rows else 0.0,
        "same_space_margin_positive_rate": 0.0 if not margins else len(positive) / len(margins),
        "mean_same_space_margin": float(np.mean(margins)) if margins else PASSIVE_BASELINE["mean_active_evidence_margin"],
        "mean_margin_gain_vs_E4AD": (float(np.mean(margins)) - PASSIVE_BASELINE["mean_active_evidence_margin"]) if margins else 0.0,
        "active_resolved_event_count": int(sum(int(r["active_resolved"]) for r in compare_rows)),
        "active_false_rescue_count": int(sum(int(r["active_false_rescue"]) for r in compare_rows)),
        "target_not_in_top5_count": int(sum(1 for r in proposal_rows if int(r["target_bundle_retrieved_top5"]) == 0)),
        "target_in_top3_but_lost_top1_count": int(sum(1 for r in proposal_rows if int(r["target_bundle_retrieved_top3"]) == 1 and int(r["target_bundle_retrieved_top1"]) == 0)),
        "competition_removed_target_count": int(sum(1 for r in proposal_rows if int(r["target_bundle_retrieved_top5"]) == 0)),
        "random_same_space_gain": 0.0,
        "memory_uncertainty_same_space_gain": 0.0,
        "descriptor_missing_rate": 0.0 if not margin_trace else len(missing) / len(margin_trace),
        "descriptor_degenerate_rate": 0.0 if not margin_trace else len(degen) / len(margin_trace),
        "selected_as_best": 0,
        "eligible_for_best": 0,
    }
    return {
        "summary": summary,
        "retrieval_rows": retrieval_rows,
        "current_trace": current_trace,
        "margin_trace": margin_trace,
        "compare_rows": compare_rows,
        "failure_rows": failure_rows,
        "focus_rows": focus_rows,
    }


def apply_selection(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(r for r in summaries if r["ablation_name"] == "A0_passive_E34r_baseline")
    random_best = max([r for r in summaries if "random" in r["ablation_name"]] or [baseline], key=lambda r: r["global_top1"])
    mem_best = max([r for r in summaries if "memory_uncertainty" in r["ablation_name"] or "combined" in r["ablation_name"]] or [baseline], key=lambda r: r["global_top1"])
    for row in summaries:
        row["random_same_space_gain"] = float(random_best["global_top1"]) - float(baseline["global_top1"])
        row["memory_uncertainty_same_space_gain"] = float(mem_best["global_top1"]) - float(baseline["global_top1"])
        ok = (
            int(row["focus_success_count"]) == 3
            and float(row["global_top1"]) >= PASSIVE_BASELINE["global_top1"]
            and float(row["false_bundle_retrieval_rate"]) <= PASSIVE_BASELINE["false_bundle_retrieval_rate"]
            and float(row["same_space_margin_positive_rate"]) > E4AD_BASELINE["same_space_margin_positive_rate"]
            and float(row["mean_same_space_margin"]) > PASSIVE_BASELINE["mean_active_evidence_margin"]
            and int(row["active_resolved_event_count"]) >= 1
            and int(row["active_false_rescue_count"]) == 0
            and float(row["descriptor_missing_rate"]) < 0.30
            and float(row["descriptor_degenerate_rate"]) < 0.10
        )
        row["eligible_for_best"] = int(ok)
    eligible = [
        r for r in summaries
        if int(r["eligible_for_best"]) == 1 and float(r["memory_uncertainty_same_space_gain"]) >= float(r["random_same_space_gain"])
    ]
    best = max(eligible, key=lambda r: (r["global_top1"], -r["false_bundle_retrieval_rate"], r["mean_same_space_margin"])) if eligible else baseline
    for row in summaries:
        row["selected_as_best"] = int(row["ablation_name"] == best["ablation_name"])
    return best


def negative_controls(best_cfg: dict[str, Any], cache, hist_store, proto_counter, track_counter, lineage_counter) -> list[dict[str, Any]]:
    # Keep controls compact: compare the chosen real store against a deterministic
    # shuffled historical descriptor store. This detects whether gains depend on
    # correct historical bundle binding rather than generic descriptor statistics.
    bids = sorted(hist_store)
    shifted = bids[1:] + bids[:1]
    shuffled = {bid: hist_store[src] for bid, src in zip(bids, shifted)}
    rows = []
    for name, store in (("real_same_space", hist_store), ("shuffled_historical_descriptor", shuffled)):
        res = evaluate_ablation(name, best_cfg, cache, store, proto_counter, track_counter, lineage_counter)
        s = res["summary"]
        rows.append({
            "control_name": name,
            "global_top1": s["global_top1"],
            "false_bundle_retrieval_rate": s["false_bundle_retrieval_rate"],
            "same_space_margin_positive_rate": s["same_space_margin_positive_rate"],
            "mean_same_space_margin": s["mean_same_space_margin"],
            "focus_success_count": s["focus_success_count"],
            "control_passed": int(name == "real_same_space" or s["global_top1"] <= rows[0]["global_top1"]),
        })
    return rows


def render_report(compact: dict[str, Any]) -> str:
    lines = [
        "# Stage E4A.1 Report",
        "",
        "## Verdict",
        "",
        compact["next_recommendation"],
        "",
        "## Compact",
        "",
        "```json",
        json.dumps(compact, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Interpretation",
        "",
        "This stage tests real same-space historical/current crop descriptors. It does not use target labels for online scoring and does not enter attach or promotion.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with Path(args.cache).open("rb") as f:
        cache = pickle.load(f)
    bundle_by_id = cache["bundle_by_id"]
    proto_counter, track_counter, lineage_counter = e31.compute_static_counts(bundle_by_id)
    hist_store, historical_trace = build_historical_store(cache, args.config, args.seed)
    summaries: list[dict[str, Any]] = []
    all_retrieval: list[dict[str, Any]] = []
    all_current: list[dict[str, Any]] = []
    all_margins: list[dict[str, Any]] = []
    all_compare: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    all_focus: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    for name, cfg in ablations().items():
        res = evaluate_ablation(name, cfg, cache, hist_store, proto_counter, track_counter, lineage_counter)
        results[name] = res
        summaries.append(res["summary"])
        all_retrieval.extend(res["retrieval_rows"])
        all_current.extend(res["current_trace"])
        all_margins.extend(res["margin_trace"])
        all_compare.extend([dict(r, ablation_name=name) for r in res["compare_rows"]])
        all_failures.extend(res["failure_rows"])
        all_focus.extend([r for r in res["focus_rows"] if name == "A0_passive_E34r_baseline" or res["summary"]["selected_as_best"]])
    best = apply_selection(summaries)
    active_summaries = [r for r in summaries if r["ablation_name"] != "A0_passive_E34r_baseline" and "random" not in r["ablation_name"]]
    best_active = max(
        active_summaries,
        key=lambda r: (
            float(r["global_top1"]),
            -float(r["false_bundle_retrieval_rate"]),
            float(r["mean_same_space_margin"]),
            float(r["same_space_margin_positive_rate"]),
        ),
    ) if active_summaries else best
    control_cfg = ablations()[best_active["ablation_name"]] if best["ablation_name"] == "A0_passive_E34r_baseline" else ablations()[best["ablation_name"]]
    controls = negative_controls(control_cfg, cache, hist_store, proto_counter, track_counter, lineage_counter)
    controls_passed = int(all(int(r["control_passed"]) == 1 for r in controls))
    failure_counts = Counter(r["failure_reason"] for r in all_failures)
    passed = bool(best["eligible_for_best"])
    if passed and controls_passed:
        next_rec = "E4A.1 passed: same-space descriptor can be integrated cautiously; next E4A.2 integration / candidate-generation refinement."
    elif float(best_active["active_resolved_event_count"]) == 0 and float(best_active["same_space_margin_positive_rate"]) <= E4AD_BASELINE["same_space_margin_positive_rate"]:
        next_rec = "same-space descriptor is weak: margin improves slightly in best active policy but positive-rate/retrieval gates fail; next E4A.2 stronger local descriptor or real active re-observation."
    elif float(best["mean_same_space_margin"]) <= PASSIVE_BASELINE["mean_active_evidence_margin"]:
        next_rec = "same-space descriptor does not improve margin; next E4A.2 stronger local descriptor or real active re-observation."
    elif float(best["memory_uncertainty_same_space_gain"]) <= float(best["random_same_space_gain"]):
        next_rec = "same-space descriptor helps weakly, but memory-uncertainty policy does not beat random; repair fixation policy / uncertainty trigger."
    elif int(best["active_false_rescue_count"]) > 0:
        next_rec = "active false rescue detected; reduce weight / add safety gate, do not attach."
    else:
        next_rec = "same-space descriptor improves margin but not retrieval; next E4A.2 integration / candidate-generation refinement."
    compact = {
        "stage": "E4A.1",
        "best_ablation": best["ablation_name"],
        "best_active_ablation": best_active["ablation_name"],
        "passed_minimum": passed and bool(controls_passed),
        "global_top1": best["global_top1"],
        "global_top3": best["global_top3"],
        "global_top5": best["global_top5"],
        "false_bundle_retrieval_rate": best["false_bundle_retrieval_rate"],
        "focus_success_count": best["focus_success_count"],
        "same_space_margin_positive_rate": best["same_space_margin_positive_rate"],
        "mean_same_space_margin": best["mean_same_space_margin"],
        "active_resolved_event_count": best["active_resolved_event_count"],
        "active_false_rescue_count": best["active_false_rescue_count"],
        "random_same_space_gain": best["random_same_space_gain"],
        "memory_uncertainty_same_space_gain": best["memory_uncertainty_same_space_gain"],
        "descriptor_missing_rate": best["descriptor_missing_rate"],
        "descriptor_degenerate_rate": best["descriptor_degenerate_rate"],
        "best_active_global_top1": best_active["global_top1"],
        "best_active_false_bundle_retrieval_rate": best_active["false_bundle_retrieval_rate"],
        "best_active_same_space_margin_positive_rate": best_active["same_space_margin_positive_rate"],
        "best_active_mean_same_space_margin": best_active["mean_same_space_margin"],
        "best_active_active_resolved_event_count": best_active["active_resolved_event_count"],
        "best_active_active_false_rescue_count": best_active["active_false_rescue_count"],
        "negative_controls_passed": controls_passed,
        "main_failure_counts": dict(failure_counts),
        "next_recommendation": next_rec,
    }
    e31.write_csv(out / f"stage_E4A1_ablation_summary_{args.artifact_version}.csv", summaries)
    e31.write_csv(out / f"stage_E4A1_historical_descriptor_write_trace_{args.artifact_version}.csv", historical_trace)
    e31.write_csv(out / f"stage_E4A1_current_descriptor_trace_{args.artifact_version}.csv", all_current)
    e31.write_csv(out / f"stage_E4A1_same_space_margin_trace_{args.artifact_version}.csv", all_margins)
    e31.write_csv(out / f"stage_E4A1_passive_vs_active_compare_{args.artifact_version}.csv", all_compare)
    e31.write_csv(out / f"stage_E4A1_fixation_policy_compare_{args.artifact_version}.csv", all_retrieval)
    e31.write_csv(out / f"stage_E4A1_negative_control_summary_{args.artifact_version}.csv", controls)
    e31.write_csv(out / f"stage_E4A1_failure_taxonomy_{args.artifact_version}.csv", all_failures)
    best_focus = [r for r in results[best["ablation_name"]]["focus_rows"]]
    e31.write_csv(out / f"stage_E4A1_focus_event_summary_{args.artifact_version}.csv", best_focus)
    (out / f"stage_E4A1_compact_for_gpt_{args.artifact_version}.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"stage_E4A1_report_{args.artifact_version}.md").write_text(render_report(compact), encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
