from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v3_stage_e31_retrieval_competition_repair as e31
from experiments import run_v3_stage_e34r_support_trajectory_refinement as e34r
from experiments import run_v3_stage_e4a_active_evidence_acquisition as e4a
from experiments.phase3r_utils import write_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run v3 Stage E4A-D active evidence validity audit.")
    p.add_argument("--cache", default="results/v3_e4a/cache/runtime_collection_cache_v1.pkl")
    p.add_argument("--output-dir", default="results/v3_e4ad")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def descriptor_stats(desc: np.ndarray) -> dict[str, float | int]:
    d = np.asarray(desc, dtype=np.float32).reshape(-1)
    if d.size == 0:
        return {"descriptor_norm": 0.0, "descriptor_entropy": 0.0, "descriptor_variance": 0.0, "descriptor_degenerate_flag": 1}
    hist = np.abs(d) / max(float(np.abs(d).sum()), 1e-8)
    entropy = float(-(hist * np.log(hist + 1e-8)).sum())
    var = float(np.var(d))
    norm = float(np.linalg.norm(d))
    return {"descriptor_norm": norm, "descriptor_entropy": entropy, "descriptor_variance": var, "descriptor_degenerate_flag": int(norm < 1e-6 or var < 1e-6)}


def historical_crop_descriptor(bundle: dict[str, Any]) -> np.ndarray:
    # Same dimensionality as E4A's crop descriptor: 7 scalar shape/objectness
    # proxies + 8 boundary bins + 8 texture bins. This is audit-only.
    support = np.asarray(bundle["support_trajectory_signature"], dtype=np.float32).reshape(-1)
    quality = np.asarray(bundle["quality_trajectory_signature"], dtype=np.float32).reshape(-1)
    content = np.asarray(bundle["content_signature"], dtype=np.float32).reshape(-1)
    scalar = np.zeros(7, dtype=np.float32)
    scalar[0] = float(np.std(support[:8])) if support.size >= 8 else 0.0
    scalar[1] = float(support[3]) if support.size > 3 else 0.0
    scalar[2] = float(min(support[0], support[1]) / max(max(support[0], support[1]), 1e-6)) if support.size > 1 else 0.0
    scalar[3] = float(quality[0]) if quality.size > 0 else float(bundle.get("last_source_quality", 0.0))
    scalar[4] = float(quality[1]) if quality.size > 1 else 0.0
    scalar[5] = float(support[8]) if support.size > 8 else 0.0
    scalar[6] = float(support[9]) if support.size > 9 else 0.0
    boundary = np.ones(8, dtype=np.float32) / 8.0
    texture = np.pad(content[:8], (0, max(0, 8 - content[:8].size)))[:8].astype(np.float32)
    return np.concatenate([scalar, boundary, texture]).astype(np.float32)


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


def audit_policy(policy_name: str, weight: float, cache, proto_counter, track_counter, lineage_counter):
    bundle_by_id = cache["bundle_by_id"]
    passive_cfg = e34r.ablation_cfgs()[e34r.BASELINE_NAME]
    trigger_events, coverage_rows, margin_rows, ref_rows, event_rows = [], [], [], [], []
    for event in sorted(cache["event_records"], key=lambda r: (r["scenario_name"], int(r["frame_idx"]), r["event_id"])):
        if int(event["proposal_detected"]) != 1 or event["cue"] is None:
            continue
        scored = e34r.score_event(event, bundle_by_id, passive_cfg, proto_counter, track_counter, lineage_counter)
        passive_top = scored["final_topk"]
        target_id = event["target_bundle_id"]
        target_row = scored["target_row"]
        top1 = passive_top[0] if passive_top else None
        passive_top1_hit = int(target_id is not None and top1 is not None and int(top1["bundle_id"]) == int(target_id))
        target_in_top5 = int(target_id is not None and any(int(r["bundle_id"]) == int(target_id) for r in passive_top[:5]))
        target_in_top3 = int(target_id is not None and any(int(r["bundle_id"]) == int(target_id) for r in passive_top[:3]))
        target_in_candidate = int(target_row is not None)
        uncertainty = e4a.uncertainty_from_rows(passive_top)
        active_triggered = int(uncertainty["trigger"] == 1 and policy_name != "passive")
        trigger_events.append({
            "row_type": "event",
            "event_id": event["event_id"], "policy_name": policy_name, "passive_success": passive_top1_hit,
            "weight": weight,
            "passive_top1_bundle": "" if top1 is None else int(top1["bundle_id"]),
            "target_bundle_id": "" if target_id is None else int(target_id),
            "target_in_candidate_pool": target_in_candidate,
            "target_in_top5": target_in_top5,
            "target_in_top3_but_lost_top1": int(target_in_top3 == 1 and passive_top1_hit == 0),
            "active_triggered": active_triggered,
            "trigger_reason": uncertainty["reason"],
            "memory_uncertainty_score": uncertainty["score"],
            "top1_margin": uncertainty["margin"],
            "candidate_count": len(scored["candidate_pool"]),
            "hub_competition_score": uncertainty["hub"],
            "support_spread": uncertainty["spread"],
        })
        if not active_triggered:
            event_rows.append({
                "event_id": event["event_id"],
                "policy_name": policy_name,
                "failure_type": classify_failure(False, target_in_candidate, target_in_top5, target_in_top3, passive_top1_hit, 0, 0.0, 0.0),
            })
            continue
        context = cache["proposal_context"].get(f"{event['scenario_name']}:{int(event['frame_idx'])}")
        if context is None:
            continue
        pbox = tuple(int(v) for v in event["cue"]["box"])
        modes = ["boundary", "neighbor"] if policy_name == "combined" else [e4a.policy_to_mode(policy_name)]
        descs = []
        for mode in modes:
            cbox = e4a.crop_box_around(pbox, context["frame"].shape, mode, context.get("proposals", []), str(event["event_id"]))
            desc = e4a.crop_descriptor(context["frame"], context.get("heatmap"), cbox, pbox)
            descs.append(desc["descriptor"])
            neighbor_count = sum(
                1
                for p in context.get("proposals", [])
                if proposal_iou(tuple(p["box"]), cbox) > 0.01 and tuple(p["box"]) != tuple(pbox)
            )
            coverage_rows.append({
                "event_id": event["event_id"], "policy_name": policy_name, "fixation_type": mode,
                "crop_box": "|".join(str(v) for v in cbox), "proposal_box": "|".join(str(v) for v in pbox),
                "target_gt_box_if_available_for_eval_only": "",
                "crop_area_ratio": float(((cbox[2]-cbox[0])*(cbox[3]-cbox[1])) / max(context["frame"].shape[0]*context["frame"].shape[1], 1)),
                "crop_overlaps_proposal": proposal_iou(cbox, pbox),
                "crop_overlaps_target_eval_only": "",
                "crop_contains_neighbor": int(neighbor_count > 0),
                "neighbor_count_in_crop": neighbor_count,
                "objectness_crop_mean": desc["objectness_crop_mean"],
                "objectness_crop_peak": desc["objectness_crop_mean"] + desc["objectness_crop_std"],
                "boundary_coverage_proxy": desc["edge_density"],
            })
        current_desc = np.mean(np.stack(descs), axis=0)
        target_bundle = bundle_by_id.get(int(target_id)) if target_id is not None else None
        wrong_bundle = bundle_by_id.get(int(top1["bundle_id"])) if top1 is not None else None
        target_ref = e4a.bundle_active_ref(target_bundle) if target_bundle is not None else None
        wrong_ref = e4a.bundle_active_ref(wrong_bundle) if wrong_bundle is not None else None
        target_hist = historical_crop_descriptor(target_bundle) if target_bundle is not None else None
        wrong_hist = historical_crop_descriptor(wrong_bundle) if wrong_bundle is not None else None
        compat_target_ref = e4a.cosine(current_desc, target_ref) if target_ref is not None else 0.0
        compat_wrong_ref = e4a.cosine(current_desc, wrong_ref) if wrong_ref is not None else 0.0
        compat_target_hist = e4a.cosine(current_desc, target_hist) if target_hist is not None else 0.0
        compat_wrong_hist = e4a.cosine(current_desc, wrong_hist) if wrong_hist is not None else 0.0
        stats = descriptor_stats(current_desc)
        margin_rows.append({
            "event_id": event["event_id"], "policy_name": policy_name, "target_bundle_id": "" if target_id is None else int(target_id),
            "wrong_top1_bundle_id": "" if top1 is None else int(top1["bundle_id"]),
            "passive_top1_bundle_id": "" if top1 is None else int(top1["bundle_id"]),
            "active_top1_bundle_id": "",
            "compat_current_active_to_target_ref": compat_target_ref,
            "compat_current_active_to_wrong_ref": compat_wrong_ref,
            "active_margin_target_minus_wrong": compat_target_ref - compat_wrong_ref,
            "compat_current_active_to_target_ref_normalized": compat_target_ref,
            "compat_current_active_to_wrong_ref_normalized": compat_wrong_ref,
            "active_margin_normalized": compat_target_ref - compat_wrong_ref,
            **stats,
        })
        same_space_margin = compat_target_hist - compat_wrong_hist
        ref_margin = compat_target_ref - compat_wrong_ref
        ref_rows.append({
            "event_id": event["event_id"], "policy_name": policy_name, "target_bundle_id": "" if target_id is None else int(target_id),
            "wrong_bundle_id": "" if top1 is None else int(top1["bundle_id"]),
            "margin_bundle_active_ref": ref_margin,
            "margin_same_space_historical_crop": same_space_margin,
            "target_same_space_compat": compat_target_hist,
            "wrong_same_space_compat": compat_wrong_hist,
            "target_ref_type": "bundle_active_ref|historical_crop_descriptor",
            "wrong_ref_type": "bundle_active_ref|historical_crop_descriptor",
            "same_space_improves_margin": int(same_space_margin > ref_margin),
            "reference_mismatch_likely": int(same_space_margin > ref_margin and same_space_margin > 0.0),
        })
        event_rows.append({
            "event_id": event["event_id"], "policy_name": policy_name, "failure_type": classify_failure(active_triggered, target_in_candidate, target_in_top5, target_in_top3, passive_top1_hit, stats["descriptor_degenerate_flag"], same_space_margin, ref_margin),
        })
    return trigger_events, coverage_rows, margin_rows, ref_rows, event_rows


def classify_failure(triggered, target_in_candidate, target_in_top5, target_in_top3, passive_top1_hit, degenerate, same_space_margin, ref_margin):
    if passive_top1_hit:
        return "passive_success_no_active_needed"
    if not triggered and not passive_top1_hit:
        return "trigger_missed_failure_event"
    if not target_in_candidate:
        return "target_not_in_candidate_pool"
    if degenerate:
        return "descriptor_degenerate"
    if same_space_margin > ref_margin and same_space_margin > 0:
        return "reference_space_mismatch"
    if ref_margin <= 0 and same_space_margin <= 0:
        return "active_evidence_not_discriminative"
    if not target_in_top5:
        return "passive_evidence_insufficient"
    return "ambiguous_multi_valid_bundle"


def aggregate_trigger_summary(trigger_rows):
    rows = []
    by = defaultdict(list)
    for r in trigger_rows:
        by[r["policy_name"]].append(r)
    for policy, rs in by.items():
        trig = [r for r in rs if int(r["active_triggered"]) == 1]
        rows.append({
            "row_type": "summary",
            "policy_name": policy,
            "weight": rs[0].get("weight", "") if rs else "",
            "num_events": len(rs),
            "proposal_detected_events": len(rs),
            "active_triggered_count": len(trig),
            "active_trigger_rate": 0.0 if not rs else len(trig) / len(rs),
            "triggered_on_passive_failure_count": sum(1 for r in trig if int(r["passive_success"]) == 0),
            "triggered_on_passive_success_count": sum(1 for r in trig if int(r["passive_success"]) == 1),
            "triggered_target_in_candidate_pool_count": sum(1 for r in trig if int(r["target_in_candidate_pool"]) == 1),
            "triggered_target_not_in_candidate_pool_count": sum(1 for r in trig if int(r["target_in_candidate_pool"]) == 0),
            "triggered_target_not_in_top5_count": sum(1 for r in trig if int(r["target_in_top5"]) == 0),
            "triggered_target_in_top3_but_lost_top1_count": sum(1 for r in trig if int(r["target_in_top3_but_lost_top1"]) == 1),
        })
    return rows


def metric_consistency(cache):
    bundle_by_id = cache["bundle_by_id"]
    proto_counter, track_counter, lineage_counter = e31.compute_static_counts(bundle_by_id)
    cfg = e34r.ablation_cfgs()[e34r.BASELINE_NAME]
    proposal = []
    comp_removed = 0
    for event in cache["event_records"]:
        if int(event["proposal_detected"]) != 1:
            continue
        scored = e34r.score_event(event, bundle_by_id, cfg, proto_counter, track_counter, lineage_counter)
        final_ids = [int(r["bundle_id"]) for r in scored["final_topk"]]
        target_id = event["target_bundle_id"]
        top1 = int(target_id is not None and final_ids[:1] == [int(target_id)])
        top3 = int(target_id is not None and int(target_id) in set(final_ids[:3]))
        top5 = int(target_id is not None and int(target_id) in set(final_ids[:5]))
        proposal.append({"top1": top1, "top3": top3, "top5": top5, "false": int(final_ids and top1 == 0)})
        if target_id is not None and not top5 and scored["target_candidate_rank"] is not None:
            comp_removed += 1
    values = {
        "global_top1": float(np.mean([r["top1"] for r in proposal])),
        "global_top3": float(np.mean([r["top3"] for r in proposal])),
        "global_top5": float(np.mean([r["top5"] for r in proposal])),
        "false_bundle_retrieval_rate": float(np.mean([r["false"] for r in proposal])),
        "target_not_in_top5_count": int(sum(1 for r in proposal if r["top5"] == 0)),
        "target_in_top3_but_lost_top1_count": int(sum(1 for r in proposal if r["top3"] == 1 and r["top1"] == 0)),
        "competition_removed_target_count": comp_removed,
    }
    e34r_expected = {
        "global_top1": 0.4117647058823529,
        "global_top3": 0.6470588235294118,
        "global_top5": 0.7647058823529411,
        "false_bundle_retrieval_rate": 0.5882352941176471,
        "target_not_in_top5_count": 4,
        "target_in_top3_but_lost_top1_count": 4,
        "competition_removed_target_count": 4,
    }
    return [{"metric_name": k, "e34r_value": e34r_expected[k], "e4a_recomputed_value": values[k], "matched": int(abs(float(e34r_expected[k]) - float(values[k])) < 1e-9), "difference_reason": "" if abs(float(e34r_expected[k]) - float(values[k])) < 1e-9 else "metric implementation mismatch"} for k in e34r_expected]


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with Path(args.cache).open("rb") as f:
        cache = pickle.load(f)
    proto_counter, track_counter, lineage_counter = e31.compute_static_counts(cache["bundle_by_id"])
    policies = ["random", "objectness", "support_boundary", "neighbor_context", "memory_uncertainty", "combined"]
    all_triggers, all_coverage, all_margins, all_refs, all_failures = [], [], [], [], []
    for policy in policies:
        tr, cov, mar, refs, fails = audit_policy(policy, 0.05, cache, proto_counter, track_counter, lineage_counter)
        all_triggers.extend(tr); all_coverage.extend(cov); all_margins.extend(mar); all_refs.extend(refs); all_failures.extend(fails)
    trigger_summary = aggregate_trigger_summary(all_triggers)
    metric_rows = metric_consistency(cache)
    failure_counts = Counter(r["failure_type"] for r in all_failures)
    descriptor_positive = [r for r in all_margins if safe_float(r["active_margin_target_minus_wrong"]) > 0]
    same_space_positive = [r for r in all_refs if safe_float(r["margin_same_space_historical_crop"]) > 0]
    ref_mismatch = [r for r in all_refs if int(r["reference_mismatch_likely"]) == 1]
    degenerate = [r for r in all_margins if int(r["descriptor_degenerate_flag"]) == 1]
    metric_passed = all(int(r["matched"]) == 1 for r in metric_rows)
    actionable_failure_counts = Counter(
        k for k, v in failure_counts.items() for _ in range(v)
        if k != "passive_success_no_active_needed"
    )
    if not metric_passed:
        next_rec = "修 metric / report，不做新实验。"
    elif ref_mismatch and len(same_space_positive) > len(descriptor_positive):
        next_rec = "E4A.1 same-space active evidence descriptor"
    elif len(degenerate) > 0:
        next_rec = "E4A.1 stronger local descriptor"
    elif failure_counts.get("trigger_missed_failure_event", 0) > 0:
        next_rec = "E4A.1 uncertainty trigger repair"
    else:
        next_rec = "stronger visual frontend or real active re-observation"
    compact = {
        "stage": "E4A-D",
        "trigger_audit_passed": int(sum(r["active_triggered_count"] for r in trigger_summary) > 0),
        "fixation_coverage_passed": int(any(safe_float(r["crop_overlaps_proposal"]) > 0.5 for r in all_coverage)),
        "descriptor_margin_positive_rate": 0.0 if not all_margins else len(descriptor_positive) / len(all_margins),
        "same_space_margin_positive_rate": 0.0 if not all_refs else len(same_space_positive) / len(all_refs),
        "reference_mismatch_event_count": len(ref_mismatch),
        "descriptor_degenerate_event_count": len(degenerate),
        "metric_consistency_passed": int(metric_passed),
        "competition_removed_target_count_recomputed": next((r["e4a_recomputed_value"] for r in metric_rows if r["metric_name"] == "competition_removed_target_count"), None),
        "main_failure_type": actionable_failure_counts.most_common(1)[0][0] if actionable_failure_counts else "",
        "failure_counts": dict(failure_counts),
        "actionable_failure_counts": dict(actionable_failure_counts),
        "next_recommendation": next_rec,
    }
    report = "\n".join(["# Stage E4A-D Report", "", "## Verdict", "", f"`next_recommendation = {next_rec}`", "", "## Compact", "", json.dumps(compact, indent=2, ensure_ascii=False)]) + "\n"
    write_csv(out / f"stage_E4AD_trigger_audit_{args.artifact_version}.csv", trigger_summary + all_triggers)
    write_csv(out / f"stage_E4AD_fixation_coverage_audit_{args.artifact_version}.csv", all_coverage)
    write_csv(out / f"stage_E4AD_descriptor_margin_audit_{args.artifact_version}.csv", all_margins)
    write_csv(out / f"stage_E4AD_reference_space_audit_{args.artifact_version}.csv", all_refs)
    write_csv(out / f"stage_E4AD_metric_consistency_audit_{args.artifact_version}.csv", metric_rows)
    write_csv(out / f"stage_E4AD_failure_taxonomy_{args.artifact_version}.csv", all_failures)
    (out / f"stage_E4AD_compact_for_gpt_{args.artifact_version}.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"stage_E4AD_report_{args.artifact_version}.md").write_text(report, encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
