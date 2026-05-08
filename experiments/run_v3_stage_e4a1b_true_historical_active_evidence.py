from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v3_stage_e31_retrieval_competition_repair as e31
from experiments import run_v3_stage_e34_write_side_signature_v2 as e34
from experiments import run_v3_stage_e34r_support_trajectory_refinement as e34r
from experiments import run_v3_stage_e4a_active_evidence_acquisition as e4a
from experiments import run_v3_stage_e4a1_same_space_active_descriptor as e4a1


FOCUS_EVENT_IDS = {"M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"}
PASSIVE_BASELINE = {
    "global_top1": 0.4117647058823529,
    "global_top5": 0.7647058823529411,
    "false_bundle_retrieval_rate": 0.5882352941176471,
    "focus_success_count": 3,
}
E4A1_ACTIVE_BASELINE = {
    "best_active_same_space_margin_positive_rate": 0.26666666666666666,
    "best_active_mean_same_space_margin": -0.00015783309936523438,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run v3 Stage E4A.1b true historical active evidence.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--event-audit", default="results/v3_e1/stage_E1_event_audit_v1.csv")
    p.add_argument("--cross-run-alignment", default="results/v3_e2rm/stage_E2R_cross_run_target_alignment_v1.csv")
    p.add_argument("--output-dir", default="results/v3_e4a1b")
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--buffer-size", type=int, default=12)
    return p.parse_args()


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def descriptor_stats(desc: np.ndarray | None) -> dict[str, float | int]:
    if desc is None:
        return {"norm": 0.0, "entropy": 0.0, "variance": 0.0, "degenerate": 1}
    d = np.asarray(desc, dtype=np.float32).reshape(-1)
    if d.size == 0:
        return {"norm": 0.0, "entropy": 0.0, "variance": 0.0, "degenerate": 1}
    hist = np.abs(d) / max(float(np.abs(d).sum()), 1e-8)
    return {
        "norm": float(np.linalg.norm(d)),
        "entropy": float(-(hist * np.log(hist + 1e-8)).sum()),
        "variance": float(np.var(d)),
        "degenerate": int(float(np.linalg.norm(d)) < 1e-6 or float(np.var(d)) < 1e-6),
    }


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


def proposals_to_dicts(proposals) -> list[dict[str, Any]]:
    rows = []
    for idx, p in enumerate(proposals):
        rows.append({
            "proposal_id": int(idx),
            "box": tuple(int(v) for v in p.box),
            "centroid": tuple(float(v) for v in p.centroid),
            "score": float(p.score),
        })
    return rows


def best_proposal_for_box(proposals: list[dict[str, Any]], box: tuple[int, int, int, int]) -> dict[str, Any] | None:
    if not proposals:
        return None
    return max(proposals, key=lambda p: proposal_iou(tuple(p["box"]), box))


def descriptor_types() -> tuple[str, ...]:
    return ("center", "boundary", "neighbor", "objectness", "combined", "random")


def modes_for_dtype(dtype: str) -> list[str]:
    if dtype == "combined":
        return ["boundary", "neighbor"]
    if dtype == "center":
        return ["center"]
    return [dtype]


def dtype_for_policy(policy: str) -> str:
    if "random" in policy:
        return "random"
    if "neighbor" in policy:
        return "neighbor"
    if "objectness" in policy:
        return "objectness"
    if "combined" in policy:
        return "combined"
    return "boundary"


def crop_desc_for_dtype(
    frame: np.ndarray,
    heatmap: np.ndarray | None,
    base_box: tuple[int, int, int, int],
    dtype: str,
    key: str,
    proposals: list[dict[str, Any]],
) -> tuple[np.ndarray | None, str, str]:
    if dtype in {"neighbor", "combined"}:
        neighbor_count = sum(1 for p in proposals if tuple(p["box"]) != tuple(base_box))
        if neighbor_count <= 0:
            return None, "", "historical_neighbor_context_missing"
    if dtype in {"objectness"} and (heatmap is None or not proposals):
        return None, "", "historical_objectness_context_missing"
    descs = []
    crop_boxes = []
    for mode in modes_for_dtype(dtype):
        cbox = e4a.crop_box_around(base_box, frame.shape, mode, proposals, key)
        desc = e4a.crop_descriptor(frame, heatmap, cbox, base_box)
        descs.append(desc["descriptor"])
        crop_boxes.append(cbox)
    return np.mean(np.stack(descs), axis=0).astype(np.float32), ";".join("|".join(str(v) for v in c) for c in crop_boxes), ""


def reconstructed_desc_for_dtype(bundle: dict[str, Any], frame: np.ndarray, dtype: str) -> np.ndarray | None:
    try:
        box = e4a1.reconstruct_source_box(bundle, frame.shape)
        desc, _, _ = crop_desc_for_dtype(frame, None, box, dtype, f"recon:{bundle['bundle_id']}:{dtype}", [])
        return desc
    except Exception:
        return None


def build_descriptor_record(
    b: dict[str, Any],
    dtype: str,
    desc: np.ndarray | None,
    crop_box: str,
    source: str,
    missing_reason: str,
    hist_ctx: dict[str, Any] | None,
) -> dict[str, Any]:
    stats = descriptor_stats(desc)
    track_box = "" if hist_ctx is None else "|".join(str(v) for v in hist_ctx["track_box"])
    proposal_box = "" if hist_ctx is None or hist_ctx.get("proposal_box") is None else "|".join(str(v) for v in hist_ctx["proposal_box"])
    overlap_track = 0.0 if hist_ctx is None or not crop_box else proposal_iou(tuple(int(v) for v in crop_box.split(";")[0].split("|")), hist_ctx["track_box"])
    overlap_prop = 0.0 if hist_ctx is None or hist_ctx.get("proposal_box") is None or not crop_box else proposal_iou(tuple(int(v) for v in crop_box.split(";")[0].split("|")), hist_ctx["proposal_box"])
    return {
        "bundle_id": int(b["bundle_id"]),
        "memory_anchor_id": b["memory_anchor_id"],
        "scenario_name": b["scenario_name"],
        "source_track_id": int(b["primary_source_track_id"]),
        "source_prototype_id": int(b["primary_source_prototype_id"]),
        "created_frame": int(b["created_frame"]),
        "last_source_frame": int(b["last_source_frame"]),
        "historical_descriptor_source": source,
        "historical_frame_available": int(hist_ctx is not None),
        "historical_proposal_available": int(hist_ctx is not None and hist_ctx.get("proposal_box") is not None),
        "historical_heatmap_available": int(hist_ctx is not None and hist_ctx.get("heatmap") is not None),
        "historical_neighbor_context_available": int(hist_ctx is not None and len(hist_ctx.get("proposals", [])) > 1),
        "descriptor_type": dtype,
        "historical_crop_box": crop_box,
        "historical_proposal_box": proposal_box,
        "historical_track_box": track_box,
        "historical_crop_overlaps_track_box": overlap_track,
        "historical_crop_overlaps_proposal_box": overlap_prop,
        "historical_descriptor_available": int(desc is not None),
        "historical_descriptor_norm": stats["norm"],
        "historical_descriptor_entropy": stats["entropy"],
        "historical_descriptor_variance": stats["variance"],
        "historical_descriptor_degenerate": stats["degenerate"],
        "missing_reason": missing_reason,
    }


def collect_runtime_true_store(args: argparse.Namespace):
    payload = e31.load_config_payload(args.config)
    scenario_map = e31.build_phase3_scenario_map(args.config)
    event_rows_by_scenario = e31.load_events(args.event_audit)
    alignment_map = e31.load_alignment(args.cross_run_alignment)
    bundle_by_id: dict[int, dict[str, Any]] = {}
    event_records: list[dict[str, Any]] = []
    true_store: dict[int, dict[str, dict[str, Any]]] = {}
    reconstructed_store: dict[int, dict[str, dict[str, Any]]] = {}
    write_rows: list[dict[str, Any]] = []
    descriptor_rows: list[dict[str, Any]] = []
    next_bundle_id = 1
    for scenario_name in e31.SCENARIO_NAMES:
        sequence = e31.SyntheticStreamGenerator(scenario_map[scenario_name], seed=args.seed).generate_sequence(0)
        encoder = e31.MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
        field = e31.MinimalObjectnessField(**payload["field"])
        tracker = e31.MinimalTemporalIdentityTracker(**payload["tracking"])
        memory = e31.MinimalPrototypeMemory(**payload["memory"])
        bundles: list[dict[str, Any]] = []
        prev_tracks: dict[int, dict[str, Any]] = {}
        prev_contexts: dict[int, dict[str, Any]] = {}
        prev_memory_output = None
        frame_shape = tuple(int(v) for v in sequence.frames[0].frame.shape[:2])
        events_at_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=args.buffer_size))
        for ev in event_rows_by_scenario.get(scenario_name, []):
            events_at_frame[int(ev["reappear_frame"])].append(ev)
        for frame_offset in range(1, len(sequence.frames)):
            prev_frame, current_frame = sequence.frames[frame_offset - 1], sequence.frames[frame_offset]
            frame_idx = int(current_frame.frame_index)
            encoding = encoder.encode(prev_frame.frame, current_frame.frame)
            objectness_output = field.compute(encoding)
            proposal_dicts = proposals_to_dicts(objectness_output.proposals)
            tracking_output = tracker.update(
                proposals=objectness_output.proposals,
                encoding=encoding,
                heatmap=objectness_output.heatmap,
                current_frame=current_frame.frame,
                frame_index=current_frame.frame_index,
                memory_context=prev_memory_output,
            )
            memory_output = memory.update(
                tracking_output.assignments,
                frame_index=current_frame.frame_index,
                track_states=(tracking_output.active_tracks + tracking_output.dormant_tracks + tracking_output.ghost_tracks + tracking_output.retired_tracks),
            )
            tracker.apply_concept_gated_resurrection(tracking_output, memory_output, frame_index=current_frame.frame_index, frame_shape=objectness_output.heatmap.shape)
            tracker.bind_prototypes(memory_output.assignments)
            prev_memory_output = memory_output
            all_tracks = tracking_output.active_tracks + tracking_output.dormant_tracks + tracking_output.ghost_tracks + tracking_output.retired_tracks
            current_tracks = {int(t.track_id): e31.track_snap(t) for t in all_tracks}
            centroids = [tuple(float(v) for v in t["centroid"]) for t in current_tracks.values()]
            current_contexts: dict[int, dict[str, Any]] = {}
            for tid, snap in current_tracks.items():
                buffers[tid].append(e34.evidence_from_snap(snap, frame_idx, frame_shape, len(objectness_output.proposals), centroids))
                track_box = tuple(int(v) for v in snap["box"])
                best_prop = best_proposal_for_box(proposal_dicts, track_box)
                current_contexts[tid] = {
                    "frame": np.asarray(current_frame.frame),
                    "heatmap": np.asarray(objectness_output.heatmap, dtype=np.float32),
                    "proposals": proposal_dicts,
                    "track_box": track_box,
                    "proposal_box": None if best_prop is None else tuple(int(v) for v in best_prop["box"]),
                }

            for track_id, prev_t in prev_tracks.items():
                cur_t = current_tracks.get(track_id)
                trigger, source_t = None, None
                if cur_t is None and prev_t["state"] in {"active", "dormant", "ghost"}:
                    trigger, source_t = "track_missing_from_registry", prev_t
                elif cur_t is not None and prev_t["state"] == "active" and cur_t["state"] in {"dormant", "ghost", "retired"}:
                    trigger, source_t = f"active_to_{cur_t['state']}", prev_t
                if trigger is None or source_t is None:
                    continue
                if not e31.bundle_worthy(source_t):
                    write_rows.append({"scenario_name": scenario_name, "frame_idx": frame_idx, "track_id": source_t["track_id"], "prototype_id": source_t["prototype_id"], "lineage_id": source_t["continuity_lineage_id"] if source_t["continuity_lineage_id"] is not None else source_t["lineage_id"], "write_trigger": trigger, "bundle_written": 0, "bundle_id": "", "memory_anchor_id": "", "skip_reason": "not_bundle_worthy", "evidence_frame_count": len(buffers[track_id])})
                    continue
                dedup = (source_t["track_id"], source_t["prototype_id"], source_t["last_seen_frame"])
                if any((e31.b_track(b), e31.b_proto(b), int(b["last_source_frame"])) == dedup for b in bundles):
                    continue
                b = e34.make_bundle_v2(next_bundle_id, scenario_name, source_t, list(buffers[track_id]), trigger, frame_idx, frame_shape, len(objectness_output.proposals))
                next_bundle_id += 1
                b["scenario_name"] = scenario_name
                bundles.append(b)
                bundle_by_id[int(b["bundle_id"])] = b
                hist_ctx = prev_contexts.get(track_id)
                true_store[int(b["bundle_id"])] = {}
                reconstructed_store[int(b["bundle_id"])] = {}
                for dtype in descriptor_types():
                    true_desc, crop_box, missing = None, "", "historical_context_missing"
                    if hist_ctx is not None:
                        true_desc, crop_box, missing = crop_desc_for_dtype(
                            hist_ctx["frame"],
                            hist_ctx["heatmap"],
                            hist_ctx["track_box"],
                            dtype,
                            f"true:{b['bundle_id']}:{dtype}",
                            hist_ctx["proposals"],
                        )
                    true_stats = descriptor_stats(true_desc)
                    true_store[int(b["bundle_id"])][dtype] = {
                        "descriptor": true_desc,
                        "available": int(true_desc is not None),
                        "missing_reason": missing,
                        "crop_box": crop_box,
                        "descriptor_degenerate": true_stats["degenerate"],
                    }
                    descriptor_rows.append(build_descriptor_record(b, dtype, true_desc, crop_box, "true_write", missing, hist_ctx))
                    recon_desc = reconstructed_desc_for_dtype(b, hist_ctx["frame"], dtype) if hist_ctx is not None else None
                    recon_stats = descriptor_stats(recon_desc)
                    reconstructed_store[int(b["bundle_id"])][dtype] = {
                        "descriptor": recon_desc,
                        "available": int(recon_desc is not None),
                        "missing_reason": "" if recon_desc is not None else "reconstructed_descriptor_missing",
                        "descriptor_degenerate": recon_stats["degenerate"],
                    }
                write_rows.append({"scenario_name": scenario_name, "frame_idx": frame_idx, "track_id": source_t["track_id"], "prototype_id": source_t["prototype_id"], "lineage_id": b["canonical_lineage_id"], "write_trigger": trigger, "bundle_written": 1, "bundle_id": b["bundle_id"], "memory_anchor_id": b["memory_anchor_id"], "skip_reason": "", "evidence_frame_count": len(buffers[track_id])})

            if frame_idx in events_at_frame:
                for event in events_at_frame[frame_idx]:
                    proposal_detected = int(e31.si(event.get("proposal_detected"), 0) or 0)
                    target_box = e31.gt_box(current_frame, int(event["instance_id"]))
                    picked = e31.pick_proposal(objectness_output.proposals, target_box) if target_box is not None else None
                    if picked is None:
                        event_records.append({"scenario_name": scenario_name, "event_id": str(event["ledger_event_id"]), "frame_idx": frame_idx, "proposal_detected": proposal_detected, "proposal_id": None, "cue": None, "eligible_bundle_ids": [], "target_bundle_id": None, "target_bundle_exists": 0, "old_track_id": e31.si(event.get("old_track_id"), -1), "old_prototype_id": e31.si(event.get("old_prototype_id"), -1), "alignment_classification": alignment_map.get(str(event["ledger_event_id"]), {}).get("classification", ""), "target_anchor_uid": alignment_map.get(str(event["ledger_event_id"]), {}).get("target_anchor_uid", "")})
                        continue
                    proposal_id, proposal, _ = picked
                    assignment = e31.find_assignment(tracking_output.assignments, proposal_id)
                    cue = e31.cue_from_obs(proposal, assignment, frame_shape, len(objectness_output.proposals), frame_idx)
                    eligible = [b for b in bundles if int(b["created_frame"]) < frame_idx]
                    old_track_id, old_proto_id = e31.si(event.get("old_track_id"), -1), e31.si(event.get("old_prototype_id"), -1)
                    targets = [b for b in eligible if (old_track_id >= 0 and old_track_id in b["source_track_ids"]) or (old_proto_id >= 0 and old_proto_id in b["source_prototype_ids"])]
                    targets.sort(key=lambda bb: (int(bb["created_frame"]), int(bb["bundle_id"])), reverse=True)
                    target_bundle = targets[0] if targets else None
                    event_records.append({"scenario_name": scenario_name, "event_id": str(event["ledger_event_id"]), "frame_idx": frame_idx, "proposal_detected": proposal_detected, "proposal_id": int(proposal_id), "cue": cue, "eligible_bundle_ids": [int(b["bundle_id"]) for b in eligible], "target_bundle_id": None if target_bundle is None else int(target_bundle["bundle_id"]), "target_bundle_exists": int(target_bundle is not None), "old_track_id": old_track_id, "old_prototype_id": old_proto_id, "alignment_classification": alignment_map.get(str(event["ledger_event_id"]), {}).get("classification", ""), "target_anchor_uid": alignment_map.get(str(event["ledger_event_id"]), {}).get("target_anchor_uid", "")})
            prev_tracks = current_tracks
            prev_contexts = current_contexts
    return bundle_by_id, event_records, true_store, reconstructed_store, write_rows, descriptor_rows


def current_descriptor_for_event(event: dict[str, Any], context: dict[str, Any], dtype: str):
    pbox = tuple(int(v) for v in event["cue"]["box"])
    desc, crop_box, missing = crop_desc_for_dtype(
        context["frame"],
        context.get("heatmap"),
        pbox,
        dtype,
        f"current:{event['event_id']}:{dtype}",
        context.get("proposals", []),
    )
    stats = descriptor_stats(desc)
    return desc, {
        "event_id": event["event_id"],
        "descriptor_type": dtype,
        "current_crop_box": crop_box,
        "current_descriptor_available": int(desc is not None),
        "current_descriptor_norm": stats["norm"],
        "current_descriptor_entropy": stats["entropy"],
        "current_descriptor_variance": stats["variance"],
        "current_descriptor_degenerate": stats["degenerate"],
        "missing_reason": missing,
    }


def ablations() -> dict[str, dict[str, Any]]:
    entries = {"A0_passive_E34r_baseline": {"dtype": "passive", "weight": 0.0}}
    specs = [
        ("center", "center", 0.03), ("boundary", "boundary", 0.03), ("neighbor", "neighbor", 0.03), ("objectness", "objectness", 0.03), ("combined", "combined", 0.03),
        ("center", "center", 0.05), ("boundary", "boundary", 0.05), ("neighbor", "neighbor", 0.05), ("objectness", "objectness", 0.05), ("combined", "combined", 0.05),
        ("neighbor", "neighbor", 0.08), ("combined", "combined", 0.08), ("neighbor", "neighbor", 0.10), ("combined", "combined", 0.10),
        ("random", "random", 0.05), ("random", "random", 0.10),
    ]
    for idx, (label, dtype, weight) in enumerate(specs, start=1):
        entries[f"A{idx}_{label}_true_same_space_w{int(weight * 1000):03d}"] = {"dtype": dtype, "weight": weight}
    return entries


def score_event_active(event, scored, current_desc, dtype, true_store, reconstructed_store, bundle_by_id, weight):
    if current_desc is None or weight <= 0:
        return list(scored["final_topk"]), []
    active_rows = []
    for row in scored["candidate_pool"]:
        bid = int(row["bundle_id"])
        ref = true_store.get(bid, {}).get(dtype)
        compat = e4a.cosine(current_desc, ref["descriptor"]) if ref is not None and int(ref["available"]) else 0.0
        rr = dict(row)
        rr["true_same_space_compatibility"] = compat
        rr["true_same_space_score"] = safe_float(row.get("e34r_score", row.get("final_score"))) + float(weight) * compat
        active_rows.append(rr)
    active_rows.sort(key=lambda r: r["true_same_space_score"], reverse=True)
    final_top = e31.diversify_candidates([dict(r, final_score=r["true_same_space_score"]) for r in active_rows], e34r.e34.base_cfg())
    target_id = event["target_bundle_id"]
    passive_top1 = scored["final_topk"][0] if scored["final_topk"] else None
    rows = []
    if target_id is not None and passive_top1 is not None:
        tid, wid = int(target_id), int(passive_top1["bundle_id"])
        tb, wb = bundle_by_id[tid], bundle_by_id[wid]
        true_t = true_store.get(tid, {}).get(dtype)
        true_w = true_store.get(wid, {}).get(dtype)
        recon_t = reconstructed_store.get(tid, {}).get(dtype)
        recon_w = reconstructed_store.get(wid, {}).get(dtype)
        pseudo_target = e4a.cosine(current_desc, e4a.bundle_active_ref(tb))
        pseudo_wrong = e4a.cosine(current_desc, e4a.bundle_active_ref(wb))
        recon_target = e4a.cosine(current_desc, recon_t["descriptor"]) if recon_t is not None and int(recon_t["available"]) else 0.0
        recon_wrong = e4a.cosine(current_desc, recon_w["descriptor"]) if recon_w is not None and int(recon_w["available"]) else 0.0
        true_target = e4a.cosine(current_desc, true_t["descriptor"]) if true_t is not None and int(true_t["available"]) else 0.0
        true_wrong = e4a.cosine(current_desc, true_w["descriptor"]) if true_w is not None and int(true_w["available"]) else 0.0
        true_margin = true_target - true_wrong
        if true_t is None or not int(true_t["available"]) or true_w is None or not int(true_w["available"]):
            reason = "true_descriptor_missing"
        elif int(true_t["descriptor_degenerate"]) or int(true_w["descriptor_degenerate"]):
            reason = "true_descriptor_degenerate"
        elif true_margin > 0:
            reason = "true_same_space_discriminative"
        elif recon_target - recon_wrong > true_margin:
            reason = "reconstructed_was_better"
        elif pseudo_target - pseudo_wrong > true_margin:
            reason = "pseudo_was_better"
        else:
            reason = "true_same_space_not_discriminative"
        rows.append({
            "event_id": event["event_id"],
            "policy_name": dtype,
            "descriptor_type": dtype,
            "target_bundle_id": tid,
            "wrong_top1_bundle_id": wid,
            "pseudo_margin": pseudo_target - pseudo_wrong,
            "reconstructed_same_space_margin": recon_target - recon_wrong,
            "true_same_space_margin": true_margin,
            "pseudo_target_compat": pseudo_target,
            "pseudo_wrong_compat": pseudo_wrong,
            "reconstructed_target_compat": recon_target,
            "reconstructed_wrong_compat": recon_wrong,
            "true_target_compat": true_target,
            "true_wrong_compat": true_wrong,
            "true_beats_pseudo": int(true_margin > (pseudo_target - pseudo_wrong)),
            "true_beats_reconstructed": int(true_margin > (recon_target - recon_wrong)),
            "reference_failure_reason": reason,
        })
    return final_top, rows


def evaluate_ablation(name, cfg, bundle_by_id, event_records, contexts, true_store, reconstructed_store, proto_counter, track_counter, lineage_counter):
    passive_cfg = e34r.ablation_cfgs()[e34r.BASELINE_NAME]
    retrieval_rows: list[dict[str, Any]] = []
    current_rows: list[dict[str, Any]] = []
    ref_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    focus_rows: list[dict[str, Any]] = []
    for event in sorted(event_records, key=lambda r: (r["scenario_name"], int(r["frame_idx"]), r["event_id"])):
        if int(event["proposal_detected"]) != 1:
            continue
        scored = e34r.score_event(event, bundle_by_id, passive_cfg, proto_counter, track_counter, lineage_counter)
        passive_top = list(scored["final_topk"])
        final_top = passive_top
        dtype = str(cfg["dtype"])
        active_triggered = int(dtype != "passive")
        current_desc = None
        if active_triggered:
            context = contexts.get(f"{event['scenario_name']}:{int(event['frame_idx'])}")
            if context is not None and event["cue"] is not None:
                current_desc, cur = current_descriptor_for_event(event, context, dtype)
                cur["ablation_name"] = name
                current_rows.append(cur)
                final_top, rows = score_event_active(event, scored, current_desc, dtype, true_store, reconstructed_store, bundle_by_id, float(cfg["weight"]))
                ref_rows.extend([dict(r, ablation_name=name) for r in rows])
        target_id = event["target_bundle_id"]
        final_ids = [int(r["bundle_id"]) for r in final_top]
        passive_ids = [int(r["bundle_id"]) for r in passive_top]
        top1_hit = int(target_id is not None and final_ids[:1] == [int(target_id)])
        top3_hit = int(target_id is not None and int(target_id) in set(final_ids[:3]))
        top5_hit = int(target_id is not None and int(target_id) in set(final_ids[:5]))
        passive_top1_hit = int(target_id is not None and passive_ids[:1] == [int(target_id)])
        false_ret = int(final_ids and top1_hit == 0)
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
            "false_bundle_retrieval": false_ret,
            "top1_bundle_id": "" if not final_top else int(final_top[0]["bundle_id"]),
            "top5_bundle_ids": "|".join(str(v) for v in final_ids[:5]),
            "active_triggered": active_triggered,
            "descriptor_type": dtype,
        }
        retrieval_rows.append(row)
        compare_rows.append({
            "ablation_name": name,
            "event_id": event["event_id"],
            "passive_top1_hit": passive_top1_hit,
            "active_top1_hit": top1_hit,
            "passive_top1_bundle": "" if not passive_top else int(passive_top[0]["bundle_id"]),
            "active_top1_bundle": "" if not final_top else int(final_top[0]["bundle_id"]),
            "active_resolved": int(passive_top1_hit == 0 and top1_hit == 1),
            "active_false_rescue": int(passive_top1_hit == 1 and top1_hit == 0),
        })
        if event["event_id"] in FOCUS_EVENT_IDS:
            focus_rows.append(row)
        if false_ret:
            reason = "target_not_in_top5" if not top5_hit else ("target_in_top3_but_lost_top1" if top3_hit else "target_in_top5_but_lost_top3")
            failures.append({"ablation_name": name, "event_id": event["event_id"], "failure_reason": reason, "active_triggered": active_triggered})
    margins = [safe_float(r["true_same_space_margin"]) for r in ref_rows]
    pos = [m for m in margins if m > 0]
    true_missing = [r for r in ref_rows if r["reference_failure_reason"] == "true_descriptor_missing"]
    true_degenerate = [r for r in ref_rows if r["reference_failure_reason"] == "true_descriptor_degenerate"]
    summary = {
        "ablation_name": name,
        "global_top1": float(np.mean([int(r["target_bundle_retrieved_top1"]) for r in retrieval_rows])) if retrieval_rows else 0.0,
        "global_top3": float(np.mean([int(r["target_bundle_retrieved_top3"]) for r in retrieval_rows])) if retrieval_rows else 0.0,
        "global_top5": float(np.mean([int(r["target_bundle_retrieved_top5"]) for r in retrieval_rows])) if retrieval_rows else 0.0,
        "false_bundle_retrieval_rate": float(np.mean([int(r["false_bundle_retrieval"]) for r in retrieval_rows])) if retrieval_rows else 0.0,
        "focus_success_count": int(sum(int(r["pattern_completion_success"]) for r in focus_rows)),
        "regression_event_count": int(sum(int(r["active_false_rescue"]) for r in compare_rows)),
        "active_trigger_rate": float(np.mean([int(r["active_triggered"]) for r in retrieval_rows])) if retrieval_rows else 0.0,
        "mean_fixation_count": float(np.mean([2 if r["descriptor_type"] == "combined" else (1 if int(r["active_triggered"]) else 0) for r in retrieval_rows])) if retrieval_rows else 0.0,
        "true_descriptor_available_rate": 0.0 if not ref_rows else 1.0 - len(true_missing) / len(ref_rows),
        "true_descriptor_degenerate_rate": 0.0 if not ref_rows else len(true_degenerate) / len(ref_rows),
        "true_same_space_margin_positive_rate": 0.0 if not margins else len(pos) / len(margins),
        "mean_true_same_space_margin": float(np.mean(margins)) if margins else E4A1_ACTIVE_BASELINE["best_active_mean_same_space_margin"],
        "mean_true_margin_gain_vs_E4A1": (float(np.mean(margins)) - E4A1_ACTIVE_BASELINE["best_active_mean_same_space_margin"]) if margins else 0.0,
        "pseudo_margin_positive_rate": 0.0 if not ref_rows else sum(1 for r in ref_rows if safe_float(r["pseudo_margin"]) > 0) / len(ref_rows),
        "reconstructed_margin_positive_rate": 0.0 if not ref_rows else sum(1 for r in ref_rows if safe_float(r["reconstructed_same_space_margin"]) > 0) / len(ref_rows),
        "active_resolved_event_count": int(sum(int(r["active_resolved"]) for r in compare_rows)),
        "active_false_rescue_count": int(sum(int(r["active_false_rescue"]) for r in compare_rows)),
        "target_not_in_top5_count": int(sum(1 for r in retrieval_rows if int(r["target_bundle_retrieved_top5"]) == 0)),
        "target_in_top3_but_lost_top1_count": int(sum(1 for r in retrieval_rows if int(r["target_bundle_retrieved_top3"]) == 1 and int(r["target_bundle_retrieved_top1"]) == 0)),
        "competition_removed_target_count": int(sum(1 for r in retrieval_rows if int(r["target_bundle_retrieved_top5"]) == 0)),
        "random_true_same_space_gain": 0.0,
        "neighbor_true_same_space_gain": 0.0,
        "combined_true_same_space_gain": 0.0,
        "negative_controls_passed": 0,
        "selected_as_best": 0,
        "eligible_for_best": 0,
    }
    return {
        "summary": summary,
        "retrieval_rows": retrieval_rows,
        "current_rows": current_rows,
        "reference_rows": ref_rows,
        "compare_rows": compare_rows,
        "failure_rows": failures,
        "focus_rows": focus_rows,
    }


def apply_selection(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(r for r in summaries if r["ablation_name"] == "A0_passive_E34r_baseline")
    random_best = max([r for r in summaries if "random" in r["ablation_name"]] or [baseline], key=lambda r: r["global_top1"])
    neighbor_best = max([r for r in summaries if "neighbor" in r["ablation_name"]] or [baseline], key=lambda r: r["global_top1"])
    combined_best = max([r for r in summaries if "combined" in r["ablation_name"]] or [baseline], key=lambda r: r["global_top1"])
    for row in summaries:
        row["random_true_same_space_gain"] = float(random_best["global_top1"]) - float(baseline["global_top1"])
        row["neighbor_true_same_space_gain"] = float(neighbor_best["global_top1"]) - float(baseline["global_top1"])
        row["combined_true_same_space_gain"] = float(combined_best["global_top1"]) - float(baseline["global_top1"])
        ok = (
            int(row["focus_success_count"]) == 3
            and float(row["global_top1"]) >= PASSIVE_BASELINE["global_top1"]
            and float(row["false_bundle_retrieval_rate"]) <= PASSIVE_BASELINE["false_bundle_retrieval_rate"]
            and float(row["true_descriptor_available_rate"]) >= 0.70
            and float(row["true_descriptor_degenerate_rate"]) < 0.10
            and float(row["true_same_space_margin_positive_rate"]) > E4A1_ACTIVE_BASELINE["best_active_same_space_margin_positive_rate"]
            and float(row["mean_true_same_space_margin"]) > E4A1_ACTIVE_BASELINE["best_active_mean_same_space_margin"]
            and int(row["active_resolved_event_count"]) >= 1
            and int(row["active_false_rescue_count"]) == 0
        )
        row["eligible_for_best"] = int(ok)
    eligible = [r for r in summaries if int(r["eligible_for_best"]) == 1]
    best = max(eligible, key=lambda r: (r["global_top1"], -r["false_bundle_retrieval_rate"], r["mean_true_same_space_margin"])) if eligible else baseline
    for row in summaries:
        row["selected_as_best"] = int(row["ablation_name"] == best["ablation_name"])
    return best


def negative_controls(best_cfg, bundle_by_id, event_records, contexts, true_store, reconstructed_store, proto_counter, track_counter, lineage_counter):
    bids = sorted(true_store)
    shifted = bids[1:] + bids[:1]
    shuffled = {bid: true_store[src] for bid, src in zip(bids, shifted)}
    rows = []
    for name, store in (("real_true_same_space", true_store), ("shuffled_true_historical_descriptor", shuffled), ("wrong_descriptor_control", shuffled), ("true_write_only_control", true_store)):
        res = evaluate_ablation(name, best_cfg, bundle_by_id, event_records, contexts, store, reconstructed_store, proto_counter, track_counter, lineage_counter)
        s = res["summary"]
        if not rows:
            control_passed = 1
        elif name == "true_write_only_control":
            control_passed = int(
                safe_float(s["mean_true_same_space_margin"]) >= safe_float(rows[0]["mean_true_same_space_margin"]) - 1e-9
                and safe_float(s["same_space_margin_positive_rate"]) >= safe_float(rows[0]["same_space_margin_positive_rate"]) - 1e-9
            )
        else:
            control_passed = int(
                safe_float(s["global_top1"]) <= safe_float(rows[0]["global_top1"]) + 1e-9
                and safe_float(s["mean_true_same_space_margin"]) < safe_float(rows[0]["mean_true_same_space_margin"])
                and safe_float(s["same_space_margin_positive_rate"]) <= safe_float(rows[0]["same_space_margin_positive_rate"]) + 1e-9
            )
        rows.append({
            "control_name": name,
            "global_top1": s["global_top1"],
            "false_bundle_retrieval_rate": s["false_bundle_retrieval_rate"],
            "same_space_margin_positive_rate": s["true_same_space_margin_positive_rate"],
            "mean_true_same_space_margin": s["mean_true_same_space_margin"],
            "active_resolved_event_count": s["active_resolved_event_count"],
            "active_false_rescue_count": s["active_false_rescue_count"],
            "focus_success_count": s["focus_success_count"],
            "control_passed": control_passed,
        })
    return rows


def render_report(compact: dict[str, Any]) -> str:
    return "\n".join([
        "# Stage E4A.1b Report",
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
    ]) + "\n"


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle_by_id, event_records, true_store, reconstructed_store, write_rows, descriptor_rows = collect_runtime_true_store(args)
    contexts = e4a0_contexts(args.config, event_records, args.seed)
    proto_counter, track_counter, lineage_counter = e31.compute_static_counts(bundle_by_id)
    summaries: list[dict[str, Any]] = []
    all_retrieval: list[dict[str, Any]] = []
    all_current: list[dict[str, Any]] = []
    all_refs: list[dict[str, Any]] = []
    all_compare: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    for name, cfg in ablations().items():
        res = evaluate_ablation(name, cfg, bundle_by_id, event_records, contexts, true_store, reconstructed_store, proto_counter, track_counter, lineage_counter)
        results[name] = res
        summaries.append(res["summary"])
        all_retrieval.extend(res["retrieval_rows"])
        all_current.extend(res["current_rows"])
        all_refs.extend(res["reference_rows"])
        all_compare.extend(res["compare_rows"])
        all_failures.extend(res["failure_rows"])
    best = apply_selection(summaries)
    active = [r for r in summaries if r["ablation_name"] != "A0_passive_E34r_baseline" and "random" not in r["ablation_name"]]
    best_active = max(active, key=lambda r: (r["global_top1"], -r["false_bundle_retrieval_rate"], r["mean_true_same_space_margin"], r["true_same_space_margin_positive_rate"])) if active else best
    control_cfg = ablations()[best_active["ablation_name"]] if best["ablation_name"] == "A0_passive_E34r_baseline" else ablations()[best["ablation_name"]]
    controls = negative_controls(control_cfg, bundle_by_id, event_records, contexts, true_store, reconstructed_store, proto_counter, track_counter, lineage_counter)
    controls_passed = int(all(int(r["control_passed"]) == 1 for r in controls))
    for row in summaries:
        row["negative_controls_passed"] = controls_passed
    failure_counts = Counter(r["failure_reason"] for r in all_failures)
    passed = bool(best["eligible_for_best"] and controls_passed)
    if passed:
        next_rec = "E4A.1b passed: true historical same-space evidence works; next E4A.2 integration / fixation policy refinement."
    elif not controls_passed:
        next_rec = "true same-space margin improves, but negative controls are not clean; do not integrate yet. Tighten descriptor binding/control or move to stronger local descriptor with strict controls."
    elif float(best_active["true_descriptor_available_rate"]) < 0.70:
        next_rec = "fix historical evidence write coverage"
    elif float(best_active["mean_true_same_space_margin"]) <= E4A1_ACTIVE_BASELINE["best_active_mean_same_space_margin"]:
        next_rec = "true same-space margin does not improve over E4A1; next E4A.2 stronger local descriptor or real active re-observation."
    elif int(best_active["active_resolved_event_count"]) == 0:
        next_rec = "true same-space margin improves but no retrieval gain; next E4A.2 candidate integration / safety gate refinement."
    elif float(best_active["neighbor_true_same_space_gain"]) <= float(best_active["random_true_same_space_gain"]) and float(best_active["combined_true_same_space_gain"]) <= float(best_active["random_true_same_space_gain"]):
        next_rec = "fixation policy / uncertainty trigger repair"
    elif int(best_active["active_false_rescue_count"]) > 0:
        next_rec = "reduce weight / add safety gate, do not attach"
    else:
        next_rec = "E4A.2 integration / fixation policy refinement"
    compact = {
        "stage": "E4A.1b",
        "best_ablation": best["ablation_name"],
        "best_active_ablation": best_active["ablation_name"],
        "passed_minimum": passed,
        "global_top1": best["global_top1"],
        "global_top3": best["global_top3"],
        "global_top5": best["global_top5"],
        "false_bundle_retrieval_rate": best["false_bundle_retrieval_rate"],
        "focus_success_count": best["focus_success_count"],
        "true_descriptor_available_rate": best_active["true_descriptor_available_rate"],
        "true_descriptor_degenerate_rate": best_active["true_descriptor_degenerate_rate"],
        "true_same_space_margin_positive_rate": best_active["true_same_space_margin_positive_rate"],
        "mean_true_same_space_margin": best_active["mean_true_same_space_margin"],
        "active_resolved_event_count": best_active["active_resolved_event_count"],
        "active_false_rescue_count": best_active["active_false_rescue_count"],
        "random_true_same_space_gain": best_active["random_true_same_space_gain"],
        "neighbor_true_same_space_gain": best_active["neighbor_true_same_space_gain"],
        "combined_true_same_space_gain": best_active["combined_true_same_space_gain"],
        "negative_controls_passed": controls_passed,
        "main_failure_counts": dict(failure_counts),
        "next_recommendation": next_rec,
    }
    e31.write_csv(out / f"stage_E4A1B_historical_evidence_write_trace_{args.artifact_version}.csv", descriptor_rows)
    e31.write_csv(out / f"stage_E4A1B_historical_crop_descriptor_inventory_{args.artifact_version}.csv", descriptor_rows)
    e31.write_csv(out / f"stage_E4A1B_current_crop_descriptor_trace_{args.artifact_version}.csv", all_current)
    e31.write_csv(out / f"stage_E4A1B_reference_comparison_trace_{args.artifact_version}.csv", all_refs)
    e31.write_csv(out / f"stage_E4A1B_same_space_margin_trace_{args.artifact_version}.csv", all_refs)
    e31.write_csv(out / f"stage_E4A1B_passive_vs_active_compare_{args.artifact_version}.csv", all_compare)
    e31.write_csv(out / f"stage_E4A1B_negative_control_summary_{args.artifact_version}.csv", controls)
    e31.write_csv(out / f"stage_E4A1B_ablation_summary_{args.artifact_version}.csv", summaries)
    e31.write_csv(out / f"stage_E4A1B_failure_taxonomy_{args.artifact_version}.csv", all_failures)
    e31.write_csv(out / f"stage_E4A1B_focus_event_summary_{args.artifact_version}.csv", results[best["ablation_name"]]["focus_rows"])
    (out / f"stage_E4A1B_compact_for_gpt_{args.artifact_version}.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"stage_E4A1B_report_{args.artifact_version}.md").write_text(render_report(compact), encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False))


def e4a0_contexts(config_path: str, event_records: list[dict[str, Any]], seed: int):
    # Same context cache logic as E4A-0, local to avoid depending on cached files.
    payload = e31.load_config_payload(config_path)
    scenario_map = e31.build_phase3_scenario_map(config_path)
    needed: dict[str, set[int]] = defaultdict(set)
    for e in event_records:
        if int(e["proposal_detected"]) == 1:
            needed[str(e["scenario_name"])].add(int(e["frame_idx"]))
    contexts: dict[str, dict[str, Any]] = {}
    for scenario_name, frames_needed in needed.items():
        sequence = e31.SyntheticStreamGenerator(scenario_map[scenario_name], seed=seed).generate_sequence(0)
        encoder = e31.MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
        field = e31.MinimalObjectnessField(**payload["field"])
        for frame_offset in range(1, len(sequence.frames)):
            current_frame = sequence.frames[frame_offset]
            frame_idx = int(current_frame.frame_index)
            if frame_idx not in frames_needed:
                continue
            prev_frame = sequence.frames[frame_offset - 1]
            encoding = encoder.encode(prev_frame.frame, current_frame.frame)
            objectness_output = field.compute(encoding)
            contexts[f"{scenario_name}:{frame_idx}"] = {
                "scenario_name": scenario_name,
                "frame_idx": frame_idx,
                "frame": np.asarray(current_frame.frame),
                "proposals": proposals_to_dicts(objectness_output.proposals),
                "heatmap": np.asarray(objectness_output.heatmap, dtype=np.float32),
            }
    return contexts


if __name__ == "__main__":
    main()
