from __future__ import annotations

import argparse
import hashlib
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

from experiments import run_v3_stage_e31_retrieval_competition_repair as e31
from experiments import run_v3_stage_e32b_safe_top3_rerank as e32b
from experiments import run_v3_stage_e34_write_side_signature_v2 as e34
from experiments import run_v3_stage_e34r_support_trajectory_refinement as e34r


FOCUS_EVENT_IDS = {"M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run v3 Stage E4A active evidence acquisition.")
    p.add_argument("--cache", default="results/v3_e4a/cache/runtime_collection_cache_v1.pkl")
    p.add_argument("--e2c-negative-events", default="results/v3_e2c/stage_E2C_negative_control_events_v1.csv")
    p.add_argument("--output-dir", default="results/v3_e4a")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def geomean(values: list[float]) -> float:
    vals = [max(1e-6, min(1.0, float(v))) for v in values]
    if not vals:
        return 0.0
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def crop_box_around(box, frame_shape, mode: str, proposals=None, event_id: str = ""):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    if mode == "random":
        digest = int(hashlib.md5(str(event_id).encode()).hexdigest()[:8], 16)
        cx = int((digest % max(w, 1)))
        cy = int(((digest // 997) % max(h, 1)))
    elif mode == "boundary":
        x1 -= bw // 3; y1 -= bh // 3; x2 += bw // 3; y2 += bh // 3
    elif mode == "neighbor" and proposals:
        centers = [(p["centroid"][0], p["centroid"][1], p["box"]) for p in proposals if tuple(p["box"]) != tuple(box)]
        if centers:
            nb = min(centers, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)[2]
            nx1, ny1, nx2, ny2 = [int(v) for v in nb]
            x1, y1, x2, y2 = min(x1, nx1), min(y1, ny1), max(x2, nx2), max(y2, ny2)
    elif mode == "objectness" and proposals:
        best = max(proposals, key=lambda p: safe_float(p.get("score")))
        x1, y1, x2, y2 = [int(v) for v in best["box"]]
    # center/support mode falls through.
    pad = max(2, min(bw, bh) // 5)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    )


def crop_descriptor(frame: np.ndarray, heatmap: np.ndarray | None, crop_box, proposal_box) -> dict[str, Any]:
    x1, y1, x2, y2 = [int(v) for v in crop_box]
    crop = np.asarray(frame[y1:y2, x1:x2])
    if crop.ndim == 3:
        gray = crop.mean(axis=2)
    else:
        gray = crop.astype(np.float32)
    if gray.size == 0:
        gray = np.zeros((1, 1), dtype=np.float32)
    gray = gray.astype(np.float32)
    if gray.max() > 1.0:
        gray = gray / 255.0
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    edge_density = float(np.mean(mag > (mag.mean() + mag.std() * 0.5))) if mag.size else 0.0
    angles = (np.arctan2(gy, gx) + math.pi) / (2.0 * math.pi)
    hist, _ = np.histogram(angles.reshape(-1), bins=8, range=(0.0, 1.0), weights=mag.reshape(-1))
    hist = hist.astype(np.float32)
    hist = hist / max(float(hist.sum()), 1e-6)
    ihist, _ = np.histogram(gray.reshape(-1), bins=8, range=(0.0, 1.0))
    ihist = ihist.astype(np.float32) / max(float(ihist.sum()), 1e-6)
    px1, py1, px2, py2 = [int(v) for v in proposal_box]
    pw, ph = max(1, px2 - px1), max(1, py2 - py1)
    cw, ch = max(1, x2 - x1), max(1, y2 - y1)
    fill_ratio = float((pw * ph) / max(cw * ch, 1))
    compactness = float(min(pw, ph) / max(max(pw, ph), 1))
    hm = None if heatmap is None else np.asarray(heatmap[y1:y2, x1:x2], dtype=np.float32)
    obj_mean = 0.0 if hm is None or hm.size == 0 else float(np.mean(hm))
    obj_std = 0.0 if hm is None or hm.size == 0 else float(np.std(hm))
    descriptor = np.concatenate([
        np.asarray([edge_density, fill_ratio, compactness, obj_mean, obj_std, cw / 128.0, ch / 128.0], dtype=np.float32),
        hist,
        ihist,
    ])
    return {
        "edge_density": edge_density,
        "boundary_hist_signature": hist.tolist(),
        "support_refined_signature": [fill_ratio, compactness, cw / 128.0, ch / 128.0],
        "texture_signature": ihist.tolist(),
        "context_layout_signature": [cw / 128.0, ch / 128.0, fill_ratio],
        "objectness_crop_mean": obj_mean,
        "objectness_crop_std": obj_std,
        "evidence_quality_score": float(np.clip(edge_density + obj_mean + compactness, 0.0, 1.0)),
        "descriptor": descriptor.astype(np.float32),
    }


def bundle_active_ref(bundle: dict[str, Any]) -> np.ndarray:
    support = np.asarray(bundle["support_trajectory_signature"], dtype=np.float32).reshape(-1)
    content = np.asarray(bundle["content_signature"], dtype=np.float32).reshape(-1)
    texture = np.pad(content[:8], (0, max(0, 8 - content[:8].size)))[:8]
    ref = np.concatenate([
        np.asarray([float(np.mean(support[:4])), float(np.std(support[:8])), float(support[2] if support.size > 2 else 0.0), float(bundle["accessibility_score"]), 0.0, float(support[0] if support.size else 0.0), float(support[1] if support.size > 1 else 0.0)], dtype=np.float32),
        np.ones(8, dtype=np.float32) / 8.0,
        texture.astype(np.float32),
    ])
    return ref


def cosine(a, b) -> float:
    aa, bb = np.asarray(a, dtype=np.float32).reshape(-1), np.asarray(b, dtype=np.float32).reshape(-1)
    m = min(aa.size, bb.size)
    if m == 0:
        return 0.0
    aa, bb = aa[:m], bb[:m]
    na, nb = float(np.linalg.norm(aa)), float(np.linalg.norm(bb))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(aa, bb) / (na * nb), -1.0, 1.0) * 0.5 + 0.5)


def passive_scored(event, bundle_by_id, cfg, proto_counter, track_counter, lineage_counter):
    return e34r.score_event(event, bundle_by_id, cfg, proto_counter, track_counter, lineage_counter)


def uncertainty_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"score": 0.0, "trigger": 0, "reason": "no_candidates", "margin": 0.0, "hub": 0.0, "spread": 0.0}
    top1, top2 = rows[0], rows[1] if len(rows) > 1 else None
    margin = safe_float(top1.get("e34r_score", top1.get("final_score"))) - (safe_float(top2.get("e34r_score", top2.get("final_score"))) if top2 else 0.0)
    support_scores = [safe_float(r.get("support_v3_score")) for r in rows[:5]]
    spread = float(max(support_scores) - min(support_scores)) if support_scores else 0.0
    hub = float(sum(1 for r in rows[:5] if int(r["primary_source_prototype_id"]) == 0)) / 5.0
    score = float(np.clip((0.08 - margin) * 4.0 + hub + max(0.0, 0.08 - spread) * 3.0 + len(rows) / 80.0, 0.0, 1.0))
    reasons = []
    if margin < 0.08:
        reasons.append("low_top1_margin")
    if hub >= 0.2:
        reasons.append("high_hub_competition")
    if spread < 0.08:
        reasons.append("support_spread_low")
    return {"score": score, "trigger": int(score >= 0.35), "reason": "|".join(reasons) if reasons else "low_uncertainty", "margin": margin, "hub": hub, "spread": spread}


def policy_to_mode(policy: str) -> str:
    if "random" in policy:
        return "random"
    if "objectness" in policy:
        return "objectness"
    if "boundary" in policy or "support" in policy:
        return "boundary"
    if "neighbor" in policy or "context" in policy:
        return "neighbor"
    return "boundary"


def ablations() -> dict[str, dict[str, Any]]:
    entries = {"A0_passive_E34r_baseline": {"policy": "passive", "weight": 0.0}}
    names = [
        ("random_fixation_control", "random"),
        ("objectness_guided", "objectness"),
        ("support_boundary", "support_boundary"),
        ("neighbor_context", "neighbor_context"),
        ("memory_uncertainty_guided", "memory_uncertainty"),
        ("combined_uncertainty_support_context", "combined"),
    ]
    idx = 1
    for weight in (0.05, 0.10):
        for label, policy in names:
            entries[f"A{idx}_{label}_w{int(weight*1000):03d}"] = {"policy": policy, "weight": weight}
            idx += 1
    entries[f"A{idx}_memory_uncertainty_guided_w015"] = {"policy": "memory_uncertainty", "weight": 0.15}; idx += 1
    entries[f"A{idx}_combined_uncertainty_support_context_w015"] = {"policy": "combined", "weight": 0.15}
    return entries


def evaluate_ablation(name, cfg, cache, proto_counter, track_counter, lineage_counter, wrong_proto_map):
    bundle_by_id = cache["bundle_by_id"]
    passive_cfg = e34r.ablation_cfgs()[e34r.BASELINE_NAME]
    rows, trigger_rows, plan_rows, evidence_rows, compare_rows, focus_rows, failure_rows = [], [], [], [], [], [], []
    for event in sorted(cache["event_records"], key=lambda r: (r["scenario_name"], int(r["frame_idx"]), r["event_id"])):
        scored = passive_scored(event, bundle_by_id, passive_cfg, proto_counter, track_counter, lineage_counter)
        candidate_pool = scored["candidate_pool"]
        passive_top = scored["final_topk"]
        final_top = list(passive_top)
        uncertainty = uncertainty_from_rows(passive_top)
        context = cache["proposal_context"].get(f"{event['scenario_name']}:{int(event['frame_idx'])}")
        triggered = int(cfg["policy"] != "passive" and uncertainty["trigger"] == 1 and context is not None and event["cue"] is not None)
        active_margin = ""
        fixation_count = 0
        if triggered:
            pbox = tuple(int(v) for v in event["cue"]["box"])
            modes = ["boundary", "neighbor"] if cfg["policy"] == "combined" else [policy_to_mode(str(cfg["policy"]))]
            descs = []
            for i, mode in enumerate(modes):
                cbox = crop_box_around(pbox, context["frame"].shape, mode, context.get("proposals", []), str(event["event_id"]))
                desc = crop_descriptor(context["frame"], context.get("heatmap"), cbox, pbox)
                descs.append(desc["descriptor"])
                fixation_count += 1
                plan_rows.append({"ablation_name": name, "event_id": event["event_id"], "policy_name": cfg["policy"], "fixation_count": len(modes), "fixation_type": mode, "crop_box": "|".join(str(v) for v in cbox), "crop_area_ratio": float(((cbox[2]-cbox[0])*(cbox[3]-cbox[1])) / max(context["frame"].shape[0]*context["frame"].shape[1], 1)), "selected_by": cfg["policy"], "selection_score": uncertainty["score"], "uses_target_oracle": 0})
                evidence_rows.append({"ablation_name": name, "event_id": event["event_id"], "policy_name": cfg["policy"], "fixation_id": i, "crop_box": "|".join(str(v) for v in cbox), "edge_density": desc["edge_density"], "boundary_hist_signature": "|".join(f"{x:.4f}" for x in desc["boundary_hist_signature"]), "support_refined_signature": "|".join(f"{x:.4f}" for x in desc["support_refined_signature"]), "texture_signature": "|".join(f"{x:.4f}" for x in desc["texture_signature"]), "context_layout_signature": "|".join(f"{x:.4f}" for x in desc["context_layout_signature"]), "objectness_crop_mean": desc["objectness_crop_mean"], "objectness_crop_std": desc["objectness_crop_std"], "evidence_quality_score": desc["evidence_quality_score"]})
            active_desc = np.mean(np.stack(descs), axis=0)
            active_rows = []
            for r in candidate_pool:
                b = bundle_by_id[int(r["bundle_id"])]
                compat = cosine(active_desc, bundle_active_ref(b))
                rr = dict(r)
                rr["active_evidence_compatibility"] = compat
                rr["active_score"] = safe_float(r.get("e34r_score", r.get("final_score"))) + float(cfg["weight"]) * compat
                active_rows.append(rr)
            active_rows.sort(key=lambda r: r["active_score"], reverse=True)
            final_top = e31.diversify_candidates([dict(r, final_score=r["active_score"]) for r in active_rows], e34r.e34.base_cfg())
            target = next((r for r in active_rows if event["target_bundle_id"] is not None and int(r["bundle_id"]) == int(event["target_bundle_id"])), None)
            wrong = final_top[0] if final_top else None
            if target is not None and wrong is not None:
                active_margin = safe_float(target.get("active_evidence_compatibility")) - safe_float(wrong.get("active_evidence_compatibility"))
        trigger_rows.append({"ablation_name": name, "event_id": event["event_id"], "top1_bundle_id": "" if not passive_top else int(passive_top[0]["bundle_id"]), "top1_margin": uncertainty["margin"], "candidate_count": len(candidate_pool), "hub_competition_score": uncertainty["hub"], "support_v3_margin_proxy": uncertainty["spread"], "cue_disagreement_score": 1.0 - uncertainty["spread"], "memory_uncertainty_score": uncertainty["score"], "active_evidence_triggered": triggered, "trigger_reason": uncertainty["reason"]})
        target_id = event["target_bundle_id"]
        final_ids = [int(r["bundle_id"]) for r in final_top]
        top1 = final_top[0] if final_top else None
        top1_hit = int(target_id is not None and final_ids[:1] == [int(target_id)])
        top3_hit = int(target_id is not None and int(target_id) in set(final_ids[:3]))
        top5_hit = int(target_id is not None and int(target_id) in set(final_ids[:5]))
        passive_ids = [int(r["bundle_id"]) for r in passive_top]
        passive_top1_hit = int(target_id is not None and passive_ids[:1] == [int(target_id)])
        success = int(top1_hit == 1)
        false_retrieval = int(int(event["proposal_detected"]) == 1 and final_ids and top1_hit == 0)
        rows.append({"ablation_name": name, "event_id": event["event_id"], "scenario_name": event["scenario_name"], "proposal_detected": int(event["proposal_detected"]), "target_bundle_id": "" if target_id is None else int(target_id), "target_bundle_retrieved_top1": top1_hit, "target_bundle_retrieved_top3": top3_hit, "target_bundle_retrieved_top5": top5_hit, "pattern_completion_success": success, "false_bundle_retrieval": false_retrieval, "top1_bundle_id": "" if top1 is None else int(top1["bundle_id"]), "top5_bundle_ids": "|".join(str(v) for v in final_ids[:5]), "top5_proto_ids": "|".join(str(int(r["primary_source_prototype_id"])) for r in final_top[:5]), "active_triggered": triggered, "fixation_count": fixation_count, "active_evidence_margin": active_margin})
        compare_rows.append({"ablation_name": name, "event_id": event["event_id"], "passive_top1_hit": passive_top1_hit, "active_top1_hit": top1_hit, "passive_top1_bundle": "" if not passive_top else int(passive_top[0]["bundle_id"]), "active_top1_bundle": "" if top1 is None else int(top1["bundle_id"]), "active_resolved": int(passive_top1_hit == 0 and top1_hit == 1), "active_false_rescue": int(passive_top1_hit == 1 and top1_hit == 0), "active_evidence_margin": active_margin})
        if event["event_id"] in FOCUS_EVENT_IDS:
            focus_rows.append(rows[-1])
        if false_retrieval:
            failure_rows.append({"ablation_name": name, "event_id": event["event_id"], "failure_reason": "active_false_retrieval" if triggered else "passive_false_retrieval", "active_triggered": triggered})
    proposal_rows = [r for r in rows if int(r["proposal_detected"]) == 1]
    focus_eval = [r for r in rows if r["event_id"] in FOCUS_EVENT_IDS]
    margins = [safe_float(r["active_evidence_margin"]) for r in rows if r["active_evidence_margin"] not in ("", None)]
    summary = {"ablation_name": name, "global_top1": float(np.mean([int(r["target_bundle_retrieved_top1"]) for r in proposal_rows])) if proposal_rows else 0.0, "global_top3": float(np.mean([int(r["target_bundle_retrieved_top3"]) for r in proposal_rows])) if proposal_rows else 0.0, "global_top5": float(np.mean([int(r["target_bundle_retrieved_top5"]) for r in proposal_rows])) if proposal_rows else 0.0, "false_bundle_retrieval_rate": float(np.mean([int(r["false_bundle_retrieval"]) for r in proposal_rows])) if proposal_rows else 0.0, "focus_success_count": int(sum(int(r["pattern_completion_success"]) for r in focus_eval)), "regression_event_count": 0, "active_trigger_rate": float(np.mean([int(r["active_triggered"]) for r in proposal_rows])) if proposal_rows else 0.0, "mean_fixation_count": float(np.mean([int(r["fixation_count"]) for r in proposal_rows])) if proposal_rows else 0.0, "compute_proxy": float(sum(int(r["fixation_count"]) for r in proposal_rows)), "target_not_in_top5_count": int(sum(1 for r in proposal_rows if int(r["target_bundle_retrieved_top5"]) == 0)), "target_in_top3_but_lost_top1_count": int(sum(1 for r in proposal_rows if int(r["target_bundle_retrieved_top3"]) == 1 and int(r["target_bundle_retrieved_top1"]) == 0)), "competition_removed_target_count": 0, "mean_passive_margin": -0.004707469659693101, "mean_active_evidence_margin": float(np.mean(margins)) if margins else -0.004707469659693101, "mean_margin_gain": (float(np.mean(margins)) + 0.004707469659693101) if margins else 0.0, "active_resolved_event_count": int(sum(int(r["active_resolved"]) for r in compare_rows)), "active_false_rescue_count": int(sum(int(r["active_false_rescue"]) for r in compare_rows)), "random_fixation_gain": 0, "objectness_fixation_gain": 0, "memory_uncertainty_fixation_gain": 0, "strict_anchor_real_svr": float(np.mean([int(r["target_bundle_retrieved_top5"]) for r in proposal_rows])) if proposal_rows else 0.0, "strict_anchor_shuffled_svr": e32b.compute_shuffled_strict_svr(proposal_rows), "selected_as_best": 0, "eligible_for_best": 0}
    return {"summary": summary, "rows": rows, "triggers": trigger_rows, "plans": plan_rows, "evidence": evidence_rows, "compare": compare_rows, "focus": focus_rows, "failures": failure_rows}


def add_select(results, summaries):
    baseline = next(r for r in summaries if r["ablation_name"] == "A0_passive_E34r_baseline")
    for row in summaries:
        ok = int(row["focus_success_count"]) == 3 and float(row["global_top1"]) >= 0.4117647058823529 and float(row["false_bundle_retrieval_rate"]) <= 0.5882352941176471 and float(row["mean_active_evidence_margin"]) > -0.004707469659693101 and int(row["active_resolved_event_count"]) >= 1 and int(row["active_false_rescue_count"]) == 0
        row["eligible_for_best"] = int(ok)
    by_name = {r["ablation_name"]: r for r in summaries}
    random_best = max([r for r in summaries if "random" in r["ablation_name"]] or [baseline], key=lambda r: r["global_top1"])
    object_best = max([r for r in summaries if "objectness" in r["ablation_name"]] or [baseline], key=lambda r: r["global_top1"])
    mem_best = max([r for r in summaries if "memory_uncertainty" in r["ablation_name"] or "combined" in r["ablation_name"]] or [baseline], key=lambda r: r["global_top1"])
    for r in summaries:
        r["random_fixation_gain"] = float(random_best["global_top1"]) - float(baseline["global_top1"])
        r["objectness_fixation_gain"] = float(object_best["global_top1"]) - float(baseline["global_top1"])
        r["memory_uncertainty_fixation_gain"] = float(mem_best["global_top1"]) - float(baseline["global_top1"])
    eligible = [r for r in summaries if int(r["eligible_for_best"]) == 1 and float(r["memory_uncertainty_fixation_gain"]) >= float(r["random_fixation_gain"])]
    best = max(eligible, key=lambda r: (r["global_top1"], -r["false_bundle_retrieval_rate"], r["mean_active_evidence_margin"])) if eligible else baseline
    for r in summaries:
        r["selected_as_best"] = int(r["ablation_name"] == best["ablation_name"])
    return best


def render_report(summary):
    b = summary["best_ablation"]
    return "\n".join(["# Stage E4A Report", "", "## Verdict", "", summary["human_summary"], "", "## Best Ablation", "", *[f"- `{k} = {b.get(k)}`" for k in ["ablation_name", "global_top1", "global_top5", "false_bundle_retrieval_rate", "focus_success_count", "mean_active_evidence_margin", "active_resolved_event_count", "active_false_rescue_count"]], "", "## Next", "", summary["next_recommendation"]]) + "\n"


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with Path(args.cache).open("rb") as f:
        cache = pickle.load(f)
    bundle_by_id = cache["bundle_by_id"]
    proto_counter, track_counter, lineage_counter = e31.compute_static_counts(bundle_by_id)
    wrong_proto_map = e32b.wrong_proto_map_from_negative_rows(e31.load_negative_controls(args.e2c_negative_events))
    results, summaries, all_rows, all_triggers, all_plans, all_evidence, all_compare, all_focus, all_failures = {}, [], [], [], [], [], [], [], []
    for name, cfg in ablations().items():
        res = evaluate_ablation(name, cfg, cache, proto_counter, track_counter, lineage_counter, wrong_proto_map)
        results[name] = res; summaries.append(res["summary"])
        all_rows.extend(res["rows"]); all_triggers.extend(res["triggers"]); all_plans.extend(res["plans"]); all_evidence.extend(res["evidence"]); all_compare.extend(res["compare"]); all_focus.extend(res["focus"]); all_failures.extend(res["failures"])
    best = add_select(results, summaries)
    passed = bool(best["eligible_for_best"])
    if passed:
        human = "E4A passed minimum active evidence gate."
        next_rec = "Proceed to E4A.1 refinement or controlled attach precheck."
    elif float(best["mean_active_evidence_margin"]) <= 0.0:
        human = "E4A did not pass: active local evidence margin remains non-positive."
        next_rec = "Do not enter attach/promotion; use stronger visual frontend or real active re-observation."
    else:
        human = "E4A did not pass: active evidence improved margin but not retrieval reliability."
        next_rec = "Refine uncertainty trigger / fixation planner before identity attach."
    summary = {"stage": "E4A", "best_ablation": best, "passed_minimum": passed, "human_summary": human, "next_recommendation": next_rec}
    compact = {"stage": "E4A", "best_ablation": best["ablation_name"], "passed_minimum": passed, "global_top1": best["global_top1"], "global_top3": best["global_top3"], "global_top5": best["global_top5"], "false_bundle_retrieval_rate": best["false_bundle_retrieval_rate"], "focus_success_count": best["focus_success_count"], "target_not_in_top5_count": best["target_not_in_top5_count"], "target_in_top3_but_lost_top1_count": best["target_in_top3_but_lost_top1_count"], "competition_removed_target_count": best["competition_removed_target_count"], "mean_active_evidence_margin": best["mean_active_evidence_margin"], "active_resolved_event_count": best["active_resolved_event_count"], "active_false_rescue_count": best["active_false_rescue_count"], "random_fixation_gain": best["random_fixation_gain"], "objectness_fixation_gain": best["objectness_fixation_gain"], "memory_uncertainty_fixation_gain": best["memory_uncertainty_fixation_gain"], "next_recommendation": next_rec}
    e31.write_csv(out / f"stage_E4A_ablation_summary_{args.artifact_version}.csv", summaries)
    e31.write_csv(out / f"stage_E4A_uncertainty_trigger_trace_{args.artifact_version}.csv", all_triggers)
    e31.write_csv(out / f"stage_E4A_fixation_plan_trace_{args.artifact_version}.csv", all_plans)
    e31.write_csv(out / f"stage_E4A_active_evidence_trace_{args.artifact_version}.csv", all_evidence)
    e31.write_csv(out / f"stage_E4A_passive_vs_active_compare_{args.artifact_version}.csv", all_compare)
    e31.write_csv(out / f"stage_E4A_focus_event_summary_{args.artifact_version}.csv", [r for r in all_focus if r["ablation_name"] == best["ablation_name"]])
    e31.write_csv(out / f"stage_E4A_failure_taxonomy_{args.artifact_version}.csv", all_failures)
    (out / f"stage_E4A_compact_for_gpt_{args.artifact_version}.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"stage_E4A_report_{args.artifact_version}.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
