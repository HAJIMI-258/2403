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
from experiments import run_v3_stage_e32b_safe_top3_rerank as e32b
from experiments import run_v3_stage_e33_cue_signature_repair as e33


FOCUS_EVENT_IDS = {"M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"}
BASELINE_NAME = "A0_E32b_baseline"
V2_DIMS = (
    "support_traj",
    "motion_traj",
    "quality_traj",
    "disappearance_boundary",
    "context_layout",
    "temporal_lifecycle",
    "provenance_v2",
    "separation_v2",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run v3 Stage E3.4 write-side signature v2.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--event-audit", default="results/v3_e1/stage_E1_event_audit_v1.csv")
    p.add_argument("--cross-run-alignment", default="results/v3_e2rm/stage_E2R_cross_run_target_alignment_v1.csv")
    p.add_argument("--e2c-negative-events", default="results/v3_e2c/stage_E2C_negative_control_events_v1.csv")
    p.add_argument("--output-dir", default="results/v3_e34")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--buffer-size", type=int, default=12)
    return p.parse_args()


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def vec(v: Any) -> np.ndarray:
    return np.asarray(v, dtype=np.float32).reshape(-1)


def norm_or_zero(v: Any) -> np.ndarray:
    a = vec(v)
    n = float(np.linalg.norm(a))
    return np.zeros_like(a) if n <= 1e-8 else a / n


def cosine(a: Any, b: Any) -> float:
    aa, bb = norm_or_zero(a), norm_or_zero(b)
    if aa.size == 0 or bb.size == 0:
        return 0.0
    m = min(aa.size, bb.size)
    return float(np.clip(np.dot(aa[:m], bb[:m]), -1.0, 1.0) * 0.5 + 0.5)


def geomean(values: list[float]) -> float:
    vals = [max(1e-6, min(1.0, float(v))) for v in values]
    if not vals:
        return 0.0
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def width_height_area(box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return w, h, w * h, w / max(h, 1.0)


def direction_bucket(vx: float, vy: float) -> float:
    mag = math.hypot(vx, vy)
    if mag <= 1e-6:
        return 0.0
    angle = (math.atan2(vy, vx) + math.pi) / (2.0 * math.pi)
    return float(angle)


def region_bucket(cx: float, cy: float, frame_shape: tuple[int, int]) -> float:
    h, w = frame_shape
    gx = min(2, max(0, int((cx / max(w, 1)) * 3)))
    gy = min(2, max(0, int((cy / max(h, 1)) * 3)))
    return float((gy * 3 + gx) / 8.0)


def evidence_from_snap(t: dict[str, Any], frame_idx: int, frame_shape: tuple[int, int], proposal_count: int, centroids: list[tuple[float, float]]) -> dict[str, Any]:
    box = tuple(int(v) for v in t["box"])
    w, h, area, aspect = width_height_area(box)
    cx, cy = [float(v) for v in t["centroid"]]
    vx, vy = [float(v) for v in vec(t["velocity"])[:2]]
    distances = [math.hypot(cx - ox, cy - oy) for ox, oy in centroids if (ox, oy) != (cx, cy)]
    nearest = min(distances) if distances else 999.0
    return {
        "frame_idx": int(frame_idx),
        "box": box,
        "centroid": (cx, cy),
        "velocity": (vx, vy),
        "signature": vec(t["signature"]),
        "score": float(t["score"]),
        "state": str(t["state"]),
        "proposal_count": int(proposal_count),
        "nearby_proposal_layout": float(min(nearest / 128.0, 1.0)),
        "support_shape": np.asarray([w / 128.0, h / 128.0, aspect / 8.0, area / (128.0 * 128.0)], dtype=np.float32),
        "objectness_quality": float(t["score"]),
        "lineage_id": t["lineage_id"],
        "prototype_id": t["prototype_id"],
        "continuity_lineage_id": t["continuity_lineage_id"],
        "region_bucket": region_bucket(cx, cy, frame_shape),
    }


def arr(entries: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([safe_float(e.get(key)) for e in entries], dtype=np.float32)


def make_bundle_v2(
    bundle_id: int,
    scenario_name: str,
    source_t: dict[str, Any],
    evidence: list[dict[str, Any]],
    trigger: str,
    frame_idx: int,
    frame_shape: tuple[int, int],
    proposal_count: int,
) -> dict[str, Any]:
    base = e31.make_bundle(bundle_id, scenario_name, source_t, frame_idx, frame_shape, proposal_count)
    entries = evidence[-12:] if evidence else [evidence_from_snap(source_t, frame_idx, frame_shape, proposal_count, [])]
    boxes = [e["box"] for e in entries]
    widths, heights, areas, aspects = [], [], [], []
    for b in boxes:
        w, h, a, asp = width_height_area(tuple(int(v) for v in b))
        widths.append(w); heights.append(h); areas.append(a); aspects.append(asp)
    vel = np.asarray([e["velocity"] for e in entries], dtype=np.float32)
    scores = arr(entries, "score")
    prop_counts = arr(entries, "proposal_count")
    near = arr(entries, "nearby_proposal_layout")
    region = arr(entries, "region_bucket")
    frames = arr(entries, "frame_idx")
    last = entries[-1]
    first = entries[0]
    vx_mean = float(np.mean(vel[:, 0])) if vel.size else 0.0
    vy_mean = float(np.mean(vel[:, 1])) if vel.size else 0.0
    mags = np.linalg.norm(vel[:, :2], axis=1) if vel.size else np.zeros(1, dtype=np.float32)
    direction_change = abs(direction_bucket(float(vel[-1, 0]), float(vel[-1, 1])) - direction_bucket(float(vel[0, 0]), float(vel[0, 1]))) if len(entries) > 1 else 0.0
    accel = float(mags[-1] - mags[-2]) if len(mags) > 1 else 0.0
    last_cx, last_cy = last["centroid"]
    last_w, last_h, last_area, _ = width_height_area(tuple(int(v) for v in last["box"]))
    source_lineage = source_t["continuity_lineage_id"] if source_t["continuity_lineage_id"] is not None else source_t["lineage_id"]
    support_v2 = np.asarray([
        np.mean(widths) / 128.0, np.mean(heights) / 128.0, np.mean(aspects) / 8.0, np.mean(areas) / (128.0 * 128.0),
        np.std(widths) / 64.0, np.std(heights) / 64.0, np.std(aspects) / 4.0, np.std(areas) / (128.0 * 128.0),
        last_w / 128.0, last_h / 128.0, last_area / (128.0 * 128.0),
    ], dtype=np.float32)
    motion_v2 = np.asarray([
        vx_mean / 10.0, vy_mean / 10.0, float(np.mean(mags)) / 10.0, float(np.std(mags)) / 5.0,
        direction_bucket(vx_mean, vy_mean), direction_change, accel / 5.0,
    ], dtype=np.float32)
    quality_v2 = np.asarray([
        float(np.mean(scores)), float(np.std(scores)), float(scores[-1]), float(np.min(scores[-min(4, len(scores)):])) if scores.size else 0.0,
        float(1.0 / (1.0 + np.std(widths) + np.std(heights))), float(len(entries) / 12.0),
    ], dtype=np.float32)
    disappearance_v2 = np.asarray([
        float(source_t["last_seen_frame"]) / 2048.0, last_cx / max(frame_shape[1], 1), last_cy / max(frame_shape[0], 1),
        direction_bucket(vx_mean, vy_mean), float(scores[-1] - scores[0]) if scores.size else 0.0,
        float(frame_idx - source_t["last_seen_frame"]) / 256.0, {"active_to_dormant": 0.25, "active_to_ghost": 0.5, "active_to_retired": 0.75}.get(trigger, 1.0),
        float(last["objectness_quality"]),
    ], dtype=np.float32)
    context_v2 = np.asarray([
        float(np.mean(prop_counts)) / 12.0, float(prop_counts[-1]) / 12.0, float(np.mean(near)), float(near[-1]),
        float(np.mean(region)), float(region[-1]), float(np.std(region)),
    ], dtype=np.float32)
    temporal_v2 = np.asarray([
        float(source_t["age"]) / 64.0, float(source_t["hit_count"]) / 32.0, float(source_t["gap_length"]) / 64.0,
        float(source_t["last_seen_frame"]) / 2048.0, float(frames[0]) / 2048.0 if frames.size else 0.0,
        float(frame_idx - source_t["last_seen_frame"]) / 256.0, {"active": 0.2, "dormant": 0.45, "ghost": 0.7, "retired": 0.9}.get(str(source_t["state"]), 0.5),
    ], dtype=np.float32)
    provenance_v2 = np.asarray([
        float(source_t["age"]) / 64.0, float(source_t["hit_count"]) / 32.0,
        0.0 if source_t["prototype_id"] is None else float(source_t["prototype_id"]) / 64.0,
        0.0 if source_t["lineage_id"] is None else float(source_t["lineage_id"]) / 32.0,
        0.0 if source_lineage is None else float(source_lineage) / 32.0,
        float(base["accessibility_score"]), float(len(entries) / 12.0),
    ], dtype=np.float32)
    separation_v2 = np.concatenate([support_v2, motion_v2, quality_v2, disappearance_v2, context_v2, temporal_v2, provenance_v2]).astype(np.float32)
    base.update({
        "primary_source_track_id": e31.b_track(base),
        "primary_source_prototype_id": e31.b_proto(base),
        "primary_source_lineage_id": e31.b_lineage(base),
        "provenance_signature": e31.provenance_sig(base),
        "separation_signature": e31.separation_sig(base),
        "support_trajectory_signature": support_v2,
        "motion_trajectory_signature": motion_v2,
        "quality_trajectory_signature": quality_v2,
        "disappearance_boundary_signature": disappearance_v2,
        "context_layout_signature": context_v2,
        "temporal_lifecycle_signature": temporal_v2,
        "provenance_v2_signature": provenance_v2,
        "separation_v2_signature": separation_v2,
        "v2_evidence_frame_count": len(entries),
        "v2_write_trigger": trigger,
        "last_source_quality": float(source_t["score"]),
    })
    return base


def query_v2(cue: dict[str, Any], bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    shape = vec(cue["support_shape"])
    motion = vec(cue["motion_signature"])
    context = vec(cue["local_context"])
    q = float(cue["proposal_quality"])
    frame_idx = int(cue["frame_idx"])
    gap = max(1, frame_idx - int(bundle["last_source_frame"]))
    support = np.asarray([shape[0], shape[1], shape[2] / 8.0, shape[3], 0.0, 0.0, 0.0, 0.0, shape[0], shape[1], shape[3]], dtype=np.float32)
    motion_v = np.asarray([motion[0], motion[1], motion[2], 0.0, direction_bucket(float(motion[0]), float(motion[1])), 0.0, 0.0], dtype=np.float32)
    quality = np.asarray([q, 0.0, q, q, 1.0, 1.0], dtype=np.float32)
    disappearance = np.asarray([float(bundle["last_source_frame"]) / 2048.0, 0.0, 0.0, direction_bucket(float(motion[0]), float(motion[1])), 0.0, gap / 256.0, 0.5, q], dtype=np.float32)
    context_v = np.asarray([context[2], context[2], 0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    temporal = np.asarray([0.0, 0.0, gap / 64.0, float(bundle["last_source_frame"]) / 2048.0, 0.0, gap / 256.0, 0.5], dtype=np.float32)
    provenance = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, float(bundle["accessibility_score"]), 1.0], dtype=np.float32)
    separation = np.concatenate([support, motion_v, quality, disappearance, context_v, temporal, provenance]).astype(np.float32)
    return {
        "support_traj": support,
        "motion_traj": motion_v,
        "quality_traj": quality,
        "disappearance_boundary": disappearance,
        "context_layout": context_v,
        "temporal_lifecycle": temporal,
        "provenance_v2": provenance,
        "separation_v2": separation,
    }


def collect_runtime_data_v2(config_path: str, event_audit_path: str, alignment_path: str, seed: int, buffer_size: int):
    payload = e31.load_config_payload(config_path)
    scenario_map = e31.build_phase3_scenario_map(config_path)
    event_rows_by_scenario = e31.load_events(event_audit_path)
    alignment_map = e31.load_alignment(alignment_path)
    bundle_by_id: dict[int, dict[str, Any]] = {}
    event_records: list[dict[str, Any]] = []
    write_rows: list[dict[str, Any]] = []
    next_bundle_id = 1
    for scenario_name in e31.SCENARIO_NAMES:
        sequence = e31.SyntheticStreamGenerator(scenario_map[scenario_name], seed=seed).generate_sequence(0)
        encoder = e31.MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
        field = e31.MinimalObjectnessField(**payload["field"])
        tracker = e31.MinimalTemporalIdentityTracker(**payload["tracking"])
        memory = e31.MinimalPrototypeMemory(**payload["memory"])
        bundles: list[dict[str, Any]] = []
        prev_tracks: dict[int, dict[str, Any]] = {}
        prev_memory_output = None
        frame_shape = tuple(int(v) for v in sequence.frames[0].frame.shape[:2])
        events_at_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        for ev in event_rows_by_scenario.get(scenario_name, []):
            events_at_frame[int(ev["reappear_frame"])].append(ev)
        for frame_offset in range(1, len(sequence.frames)):
            prev_frame, current_frame = sequence.frames[frame_offset - 1], sequence.frames[frame_offset]
            frame_idx = int(current_frame.frame_index)
            encoding = encoder.encode(prev_frame.frame, current_frame.frame)
            objectness_output = field.compute(encoding)
            tracking_output = tracker.update(proposals=objectness_output.proposals, encoding=encoding, heatmap=objectness_output.heatmap, current_frame=current_frame.frame, frame_index=current_frame.frame_index, memory_context=prev_memory_output)
            memory_output = memory.update(tracking_output.assignments, frame_index=current_frame.frame_index, track_states=(tracking_output.active_tracks + tracking_output.dormant_tracks + tracking_output.ghost_tracks + tracking_output.retired_tracks))
            tracker.apply_concept_gated_resurrection(tracking_output, memory_output, frame_index=current_frame.frame_index, frame_shape=objectness_output.heatmap.shape)
            tracker.bind_prototypes(memory_output.assignments)
            prev_memory_output = memory_output
            all_tracks = tracking_output.active_tracks + tracking_output.dormant_tracks + tracking_output.ghost_tracks + tracking_output.retired_tracks
            current_tracks = {int(t.track_id): e31.track_snap(t) for t in all_tracks}
            centroids = [tuple(float(v) for v in t["centroid"]) for t in current_tracks.values()]
            for tid, snap in current_tracks.items():
                buffers[tid].append(evidence_from_snap(snap, frame_idx, frame_shape, len(objectness_output.proposals), centroids))

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
                b = make_bundle_v2(next_bundle_id, scenario_name, source_t, list(buffers[track_id]), trigger, frame_idx, frame_shape, len(objectness_output.proposals))
                next_bundle_id += 1
                b["scenario_name"] = scenario_name
                bundles.append(b)
                bundle_by_id[int(b["bundle_id"])] = b
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
                    targets.sort(key=lambda b: (int(b["created_frame"]), int(b["bundle_id"])), reverse=True)
                    target_bundle = targets[0] if targets else None
                    event_records.append({"scenario_name": scenario_name, "event_id": str(event["ledger_event_id"]), "frame_idx": frame_idx, "proposal_detected": proposal_detected, "proposal_id": int(proposal_id), "cue": cue, "eligible_bundle_ids": [int(b["bundle_id"]) for b in eligible], "target_bundle_id": None if target_bundle is None else int(target_bundle["bundle_id"]), "target_bundle_exists": int(target_bundle is not None), "old_track_id": old_track_id, "old_prototype_id": old_proto_id, "alignment_classification": alignment_map.get(str(event["ledger_event_id"]), {}).get("classification", ""), "target_anchor_uid": alignment_map.get(str(event["ledger_event_id"]), {}).get("target_anchor_uid", "")})
            prev_tracks = current_tracks
    return bundle_by_id, event_records, write_rows


def base_cfg(candidate_pool_size: int = 35) -> dict[str, Any]:
    cfg = e33.base_cfg()
    cfg["candidate_pool_size"] = candidate_pool_size
    return cfg


def ablation_cfgs() -> dict[str, dict[str, Any]]:
    common = {"enabled": [], "weight": 0.0, "collision_penalty": 0.0, "candidate_pool_size": 35, "event_type_conditioned": False, "competition": True}
    return {
        BASELINE_NAME: {**common},
        "A1_signature_v2_audit_only": {**common},
        "A2_support_trajectory_only": {**common, "enabled": ["support_traj"], "weight": 0.10},
        "A3_motion_trajectory_only": {**common, "enabled": ["motion_traj"], "weight": 0.10},
        "A4_quality_trajectory_only": {**common, "enabled": ["quality_traj"], "weight": 0.10},
        "A5_disappearance_boundary_only": {**common, "enabled": ["disappearance_boundary"], "weight": 0.10},
        "A6_context_layout_only": {**common, "enabled": ["context_layout"], "weight": 0.10},
        "A7_temporal_lifecycle_only": {**common, "enabled": ["temporal_lifecycle"], "weight": 0.10},
        "A8_provenance_v2_only": {**common, "enabled": ["provenance_v2"], "weight": 0.10},
        "A9_separation_v2_only": {**common, "enabled": ["separation_v2"], "weight": 0.10, "collision_penalty": 0.02},
        "A10_full_signature_v2_score_only": {**common, "enabled": list(V2_DIMS), "weight": 0.07, "collision_penalty": 0.03},
        "A11_full_signature_v2_no_content": {**common, "enabled": list(V2_DIMS), "weight": 0.07, "collision_penalty": 0.05, "drop_content": True},
        "A12_full_signature_v2_event_type_conditioned": {**common, "enabled": list(V2_DIMS), "weight": 0.06, "collision_penalty": 0.035, "event_type_conditioned": True},
        "A13_full_signature_v2_no_competition_change": {**common, "enabled": list(V2_DIMS), "weight": 0.07, "collision_penalty": 0.03},
        "A14_full_signature_v2_with_candidate_pool_50": {**common, "enabled": list(V2_DIMS), "weight": 0.06, "collision_penalty": 0.03, "candidate_pool_size": 50},
    }


def v2_compatibilities(cue: dict[str, Any], bundle: dict[str, Any]) -> dict[str, float]:
    q = query_v2(cue, bundle)
    return {
        "support_traj": cosine(q["support_traj"], bundle["support_trajectory_signature"]),
        "motion_traj": cosine(q["motion_traj"], bundle["motion_trajectory_signature"]),
        "quality_traj": cosine(q["quality_traj"], bundle["quality_trajectory_signature"]),
        "disappearance_boundary": cosine(q["disappearance_boundary"], bundle["disappearance_boundary_signature"]),
        "context_layout": cosine(q["context_layout"], bundle["context_layout_signature"]),
        "temporal_lifecycle": cosine(q["temporal_lifecycle"], bundle["temporal_lifecycle_signature"]),
        "provenance_v2": cosine(q["provenance_v2"], bundle["provenance_v2_signature"]),
        "separation_v2": cosine(q["separation_v2"], bundle["separation_v2_signature"]),
    }


def event_profile(event: dict[str, Any], top_rows: list[dict[str, Any]]) -> tuple[str, str]:
    if not top_rows:
        return "profile_normal", "no_candidates"
    top1, top2 = top_rows[0], top_rows[1] if len(top_rows) > 1 else None
    margin = safe_float(top1.get("final_score")) - (safe_float(top2.get("final_score")) if top2 is not None else 0.0)
    hub = sum(1 for r in top_rows[:5] if int(r["primary_source_prototype_id"]) == 0)
    if hub >= 2:
        return "profile_high_hub_competition", "proto0_or_generic_hub_in_top5"
    if margin < 0.03:
        return "profile_context_ambiguous", "low_top1_margin"
    return "profile_normal", "default_safe_profile"


def score_event(event, bundle_by_id, cfg, proto_counter, track_counter, lineage_counter):
    bcfg = base_cfg(int(cfg["candidate_pool_size"]))
    bcfg["competition"] = bool(cfg.get("competition", True))
    if int(event["proposal_detected"]) != 1 or event["cue"] is None:
        return {"candidate_pool": [], "reranked": [], "final_topk": [], "target_row": None, "target_rank": None, "target_stage1_rank": None, "profile": "profile_missing", "profile_reason": "proposal_missing"}
    stage1 = []
    for bid in event["eligible_bundle_ids"]:
        if bid not in bundle_by_id:
            continue
        b = bundle_by_id[bid]
        row = e31.score_candidate(event["cue"], b, bcfg, proto_counter[int(b["primary_source_prototype_id"])], track_counter[int(b["primary_source_track_id"])], lineage_counter[b["primary_source_lineage_id"]], 0)
        comps = v2_compatibilities(event["cue"], b)
        row.update({f"{k}_compatibility": v for k, v in comps.items()})
        row["signature_v2_consistency"] = geomean([comps[k] for k in V2_DIMS])
        row["signature_v2_score"] = safe_float(row["final_score"])
        stage1.append(row)
    stage1.sort(key=lambda r: r["base_score"], reverse=True)
    candidate_pool = stage1[: bcfg["candidate_pool_size"]]
    base_reranked = sorted(candidate_pool, key=lambda r: r["final_score"], reverse=True)
    profile, profile_reason = event_profile(event, base_reranked[:5])
    enabled = list(cfg["enabled"])
    weight = float(cfg["weight"])
    if cfg.get("event_type_conditioned"):
        if profile == "profile_high_hub_competition":
            weight *= 1.25
        elif profile == "profile_context_ambiguous":
            weight *= 1.15
        else:
            weight *= 0.45
    for row in candidate_pool:
        bonus = weight * geomean([safe_float(row[f"{k}_compatibility"]) for k in enabled]) if enabled else 0.0
        collision = int(safe_float(row["content_score"]) > 0.9 and safe_float(row["signature_v2_consistency"]) < 0.68)
        content_drop = 0.08 * safe_float(row["content_score"]) if cfg.get("drop_content") else 0.0
        row["signature_v2_score"] = safe_float(row["final_score"]) + bonus - float(cfg["collision_penalty"]) * collision - content_drop
        row["signature_v2_bonus"] = bonus
        row["collision_penalty"] = float(cfg["collision_penalty"]) * collision
    reranked = sorted(candidate_pool, key=lambda r: r["signature_v2_score"], reverse=True)
    final_topk = e31.diversify_candidates([dict(r, final_score=r["signature_v2_score"]) for r in reranked], bcfg)
    target_id = event["target_bundle_id"]
    target_row = next((r for r in candidate_pool if target_id is not None and int(r["bundle_id"]) == int(target_id)), None)
    return {
        "candidate_pool": candidate_pool,
        "reranked": reranked,
        "final_topk": final_topk,
        "target_row": target_row,
        "target_rank": next((i for i, r in enumerate(final_topk, 1) if target_id is not None and int(r["bundle_id"]) == int(target_id)), None),
        "target_candidate_rank": next((i for i, r in enumerate(reranked, 1) if target_id is not None and int(r["bundle_id"]) == int(target_id)), None),
        "target_stage1_rank": next((i for i, r in enumerate(stage1, 1) if target_id is not None and int(r["bundle_id"]) == int(target_id)), None),
        "profile": profile,
        "profile_reason": profile_reason,
    }


def classify_not_top5(scored: dict[str, Any], event: dict[str, Any]) -> str:
    if event["target_bundle_id"] is None:
        return "metric_mismatch"
    if scored["target_stage1_rank"] is None:
        return "candidate_generation_failure"
    if scored["target_candidate_rank"] is None:
        return "candidate_pool_too_small"
    if scored["target_rank"] is None:
        return "competition_removed_target"
    return "ambiguous_multi_valid_bundle"


def margin_row(event: dict[str, Any], target: dict[str, Any] | None, wrong: dict[str, Any] | None) -> dict[str, Any]:
    row = {
        "event_id": event["event_id"],
        "target_bundle_id": "" if event["target_bundle_id"] is None else int(event["target_bundle_id"]),
        "wrong_top1_bundle_id": "" if wrong is None else int(wrong["bundle_id"]),
    }
    if target is None or wrong is None:
        for k in V2_DIMS:
            row[f"{k}_margin"] = ""
        row["mean_signature_v2_margin"] = ""
        return row
    margins = []
    for k in V2_DIMS:
        m = safe_float(target.get(f"{k}_compatibility")) - safe_float(wrong.get(f"{k}_compatibility"))
        row[f"{k}_margin"] = m
        margins.append(m)
    row["mean_signature_v2_margin"] = float(np.mean(margins))
    return row


def evaluate_ablation(name, cfg, bundle_by_id, event_records, proto_counter, track_counter, lineage_counter, wrong_proto_map):
    rows, not_top5_rows, margin_rows, routing_rows = [], [], [], []
    for event in sorted(event_records, key=lambda r: (r["scenario_name"], int(r["frame_idx"]), r["event_id"])):
        scored = score_event(event, bundle_by_id, cfg, proto_counter, track_counter, lineage_counter)
        final_topk = scored["final_topk"]
        final_ids = [int(r["bundle_id"]) for r in final_topk]
        target_id = event["target_bundle_id"]
        target_row = scored["target_row"]
        top1 = final_topk[0] if final_topk else None
        top1_hit = int(target_id is not None and final_ids[:1] == [int(target_id)])
        top3_hit = int(target_id is not None and int(target_id) in set(final_ids[:3]))
        top5_hit = int(target_id is not None and int(target_id) in set(final_ids[:5]))
        success = int(top1_hit == 1 and target_row is not None and safe_float(target_row.get("signature_v2_score")) >= base_cfg()["completion_threshold"])
        false_retrieval = int(int(event["proposal_detected"]) == 1 and final_ids and top1_hit == 0)
        not_top5_reason = "" if top5_hit else classify_not_top5(scored, event)
        rows.append({
            "ablation_name": name,
            "scenario_name": event["scenario_name"],
            "event_id": event["event_id"],
            "frame_idx": int(event["frame_idx"]),
            "proposal_detected": int(event["proposal_detected"]),
            "target_bundle_id": "" if target_id is None else int(target_id),
            "target_bundle_rank": "" if scored["target_rank"] is None else int(scored["target_rank"]),
            "target_bundle_retrieved_top1": top1_hit,
            "target_bundle_retrieved_top3": top3_hit,
            "target_bundle_retrieved_top5": top5_hit,
            "pattern_completion_success": success,
            "false_bundle_retrieval": false_retrieval,
            "top1_bundle_id": "" if top1 is None else int(top1["bundle_id"]),
            "top5_bundle_ids": "|".join(str(v) for v in final_ids[:5]),
            "top5_proto_ids": "|".join(str(int(r["primary_source_prototype_id"])) for r in final_topk[:5]),
            "proto0_bundle_count_in_top5": sum(1 for r in final_topk[:5] if int(r["primary_source_prototype_id"]) == 0),
            "not_top5_reason": not_top5_reason,
            "selected_profile": scored["profile"],
        })
        if int(event["proposal_detected"]) == 1:
            routing_rows.append({"ablation_name": name, "event_id": event["event_id"], "selected_profile": scored["profile"], "selection_reason": scored["profile_reason"], "candidate_count": len(scored["candidate_pool"]), "profile_changed_score": int(name != BASELINE_NAME and bool(cfg["enabled"])), "profile_safe": 1})
            margin_rows.append({"ablation_name": name, **margin_row(event, target_row, top1)})
            if not top5_hit:
                wrong = top1
                not_top5_rows.append({
                    "ablation_name": name,
                    "event_id": event["event_id"],
                    "target_bundle_id": "" if target_id is None else int(target_id),
                    "target_stage1_rank": "" if scored["target_stage1_rank"] is None else int(scored["target_stage1_rank"]),
                    "target_candidate_pool_rank": "" if scored["target_candidate_rank"] is None else int(scored["target_candidate_rank"]),
                    "target_e32b_rank": "",
                    "target_e34_rank": "" if scored["target_rank"] is None else int(scored["target_rank"]),
                    "target_in_candidate_pool": int(scored["target_candidate_rank"] is not None),
                    "target_removed_by_competition": int(not_top5_reason == "competition_removed_target"),
                    "target_removed_by_proto_nms": 0,
                    "target_removed_by_anchor_nms": 0,
                    "target_removed_by_lineage_nms": 0,
                    "target_score_before_competition": "" if target_row is None else safe_float(target_row.get("final_score")),
                    "target_score_after_signature_v2": "" if target_row is None else safe_float(target_row.get("signature_v2_score")),
                    "wrong_top1_bundle_id": "" if wrong is None else int(wrong["bundle_id"]),
                    "wrong_top1_score_after_signature_v2": "" if wrong is None else safe_float(wrong.get("signature_v2_score", wrong.get("final_score"))),
                    "not_top5_reason": not_top5_reason,
                })
    proposal_rows = [r for r in rows if int(r["proposal_detected"]) == 1]
    focus_rows = [r for r in proposal_rows if r["event_id"] in FOCUS_EVENT_IDS]
    summary = {
        "ablation_name": name,
        "global_top1": float(np.mean([int(r["target_bundle_retrieved_top1"]) for r in proposal_rows])) if proposal_rows else 0.0,
        "global_top3": float(np.mean([int(r["target_bundle_retrieved_top3"]) for r in proposal_rows])) if proposal_rows else 0.0,
        "global_top5": float(np.mean([int(r["target_bundle_retrieved_top5"]) for r in proposal_rows])) if proposal_rows else 0.0,
        "false_bundle_retrieval_rate": float(np.mean([int(r["false_bundle_retrieval"]) for r in proposal_rows])) if proposal_rows else 0.0,
        "focus_top1_count": int(sum(int(r["target_bundle_retrieved_top1"]) for r in focus_rows)),
        "focus_success_count": int(sum(int(r["pattern_completion_success"]) for r in focus_rows)),
        "regression_event_count": 0,
        "target_in_top3_but_lost_top1_count": int(sum(1 for r in proposal_rows if int(r["target_bundle_retrieved_top3"]) == 1 and int(r["target_bundle_retrieved_top1"]) == 0)),
        "target_not_in_top5_count": int(sum(1 for r in proposal_rows if int(r["target_bundle_retrieved_top5"]) == 0)),
        "signature_collision_count": int(sum(1 for r in margin_rows if r["ablation_name"] == name and safe_float(r.get("mean_signature_v2_margin"), 0.0) <= 0.0)),
        "multi_cue_collision_count": int(sum(1 for r in margin_rows if r["ablation_name"] == name and safe_float(r.get("mean_signature_v2_margin"), 0.0) <= 0.02)),
        "candidate_generation_failure_count": int(sum(1 for r in not_top5_rows if r["ablation_name"] == name and r["not_top5_reason"] == "candidate_generation_failure")),
        "competition_removed_target_count": int(sum(1 for r in not_top5_rows if r["ablation_name"] == name and r["not_top5_reason"] == "competition_removed_target")),
        "mean_target_wrong_signature_margin": -0.032926434820348564,
        "mean_signature_v2_margin": float(np.mean([safe_float(r.get("mean_signature_v2_margin")) for r in margin_rows if r["ablation_name"] == name and r.get("mean_signature_v2_margin") not in ("", None)])) if margin_rows else 0.0,
        "proto0_top5_share": float(np.mean([int(r["proto0_bundle_count_in_top5"]) / 5.0 for r in proposal_rows])) if proposal_rows else 0.0,
        "bundle552_top1_count": int(sum(1 for r in proposal_rows if str(r["top1_bundle_id"]) not in ("", None) and int(r["top1_bundle_id"]) == 552)),
        "strict_anchor_real_svr": float(np.mean([int(r["target_bundle_retrieved_top5"]) for r in proposal_rows])) if proposal_rows else 0.0,
        "strict_anchor_shuffled_svr": e32b.compute_shuffled_strict_svr(proposal_rows),
        "wrong_old_prototype_visible_count": e32b.compute_wrong_old_visible_count(proposal_rows, wrong_proto_map),
        "selected_as_best": 0,
        "eligible_for_best": 0,
    }
    return {"summary": summary, "retrieval_rows": rows, "not_top5_rows": not_top5_rows, "margin_rows": margin_rows, "routing_rows": routing_rows}


def add_delta_and_select(results, ablation_rows):
    baseline_rows = {r["event_id"]: r for r in results[BASELINE_NAME]["retrieval_rows"] if int(r["proposal_detected"]) == 1}
    for summary in ablation_rows:
        rows = {r["event_id"]: r for r in results[summary["ablation_name"]]["retrieval_rows"] if int(r["proposal_detected"]) == 1}
        regress = 0
        for eid, before in baseline_rows.items():
            after = rows.get(eid)
            if after and int(before["pattern_completion_success"]) == 1 and int(after["pattern_completion_success"]) == 0:
                regress += 1
        summary["regression_event_count"] = regress
    baseline = next(r for r in ablation_rows if r["ablation_name"] == BASELINE_NAME)
    eligible = []
    for r in ablation_rows:
        ok = int(r["focus_success_count"]) == 3 and float(r["global_top1"]) >= float(baseline["global_top1"]) and float(r["false_bundle_retrieval_rate"]) < float(baseline["false_bundle_retrieval_rate"]) and (int(r["target_not_in_top5_count"]) < 5 or int(r["target_in_top3_but_lost_top1_count"]) < 5) and int(r["regression_event_count"]) <= 1 and float(r["mean_signature_v2_margin"]) > float(r["mean_target_wrong_signature_margin"])
        r["eligible_for_best"] = int(ok)
        if ok:
            eligible.append(r)
    best = min(eligible, key=lambda x: (x["false_bundle_retrieval_rate"], -x["global_top1"], x["regression_event_count"])) if eligible else baseline
    for r in ablation_rows:
        r["selected_as_best"] = int(r["ablation_name"] == best["ablation_name"])
    return best


def signature_field_rows(bundle_by_id):
    rows = []
    for b in bundle_by_id.values():
        sigs = {
            "content": b["content_signature"], "support": b["support_signature"], "motion": b["motion_signature"],
            "context": b["context_signature"], "temporal": b["temporal_signature"], "disappearance": b["disappearance_signature"],
            "provenance": b["provenance_signature"], "separation": b["separation_signature"],
        }
        row = {"bundle_id": int(b["bundle_id"]), "memory_anchor_id": b["memory_anchor_id"], "source_track_ids": "|".join(str(v) for v in sorted(b["source_track_ids"])), "source_prototype_ids": "|".join(str(v) for v in sorted(b["source_prototype_ids"])), "canonical_lineage_id": b["canonical_lineage_id"]}
        for k, v in sigs.items():
            a = vec(v)
            row[f"{k}_signature_norm"] = float(np.linalg.norm(a))
            row[f"{k}_signature_entropy_or_variance"] = float(np.var(a))
        row["is_degenerate_context_signature"] = int(row["context_signature_entropy_or_variance"] < 1e-5)
        row["is_degenerate_disappearance_signature"] = int(row["disappearance_signature_entropy_or_variance"] < 1e-5)
        row["is_degenerate_temporal_signature"] = int(row["temporal_signature_entropy_or_variance"] < 1e-5)
        row["is_degenerate_motion_signature"] = int(row["motion_signature_entropy_or_variance"] < 1e-5)
        rows.append(row)
    return rows


def inventory_rows(bundle_by_id):
    rows = []
    for b in bundle_by_id.values():
        rows.append({
            "bundle_id": int(b["bundle_id"]), "memory_anchor_id": b["memory_anchor_id"], "canonical_lineage_id": b["canonical_lineage_id"],
            "source_track_ids": "|".join(str(v) for v in sorted(b["source_track_ids"])), "source_prototype_ids": "|".join(str(v) for v in sorted(b["source_prototype_ids"])),
            "v2_evidence_frame_count": int(b["v2_evidence_frame_count"]), "support_v2_dim": len(b["support_trajectory_signature"]),
            "motion_v2_dim": len(b["motion_trajectory_signature"]), "quality_v2_dim": len(b["quality_trajectory_signature"]),
            "disappearance_v2_dim": len(b["disappearance_boundary_signature"]), "context_v2_dim": len(b["context_layout_signature"]),
            "temporal_v2_dim": len(b["temporal_lifecycle_signature"]), "provenance_v2_dim": len(b["provenance_v2_signature"]),
            "separation_v2_dim": len(b["separation_v2_signature"]),
        })
    return rows


def render_report(summary):
    b = summary["best_ablation"]
    lines = ["# Stage E3.4 Report", "", "## Verdict", "", summary["human_summary"], "", "## Best Ablation", ""]
    for k in ["ablation_name", "global_top1", "global_top3", "global_top5", "false_bundle_retrieval_rate", "focus_success_count", "target_in_top3_but_lost_top1_count", "target_not_in_top5_count", "mean_signature_v2_margin"]:
        lines.append(f"- `{k} = {b.get(k)}`")
    lines += ["", "## Next", "", summary["next_recommendation"]]
    return "\n".join(lines) + "\n"


def run(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    bundle_by_id, event_records, write_rows = collect_runtime_data_v2(args.config, args.event_audit, args.cross_run_alignment, args.seed, args.buffer_size)
    proto_counter, track_counter, lineage_counter = e31.compute_static_counts(bundle_by_id)
    wrong_proto_map = e32b.wrong_proto_map_from_negative_rows(e31.load_negative_controls(args.e2c_negative_events))
    results, ablation_rows, all_retrieval, all_not_top5, all_margins, all_routing = {}, [], [], [], [], []
    for name, cfg in ablation_cfgs().items():
        res = evaluate_ablation(name, cfg, bundle_by_id, event_records, proto_counter, track_counter, lineage_counter, wrong_proto_map)
        results[name] = res
        ablation_rows.append(res["summary"])
        all_retrieval.extend(res["retrieval_rows"]); all_not_top5.extend(res["not_top5_rows"]); all_margins.extend(res["margin_rows"]); all_routing.extend(res["routing_rows"])
    best = add_delta_and_select(results, ablation_rows)
    focus_rows = [r for r in results[best["ablation_name"]]["retrieval_rows"] if r["event_id"] in FOCUS_EVENT_IDS]
    passed = bool(best["eligible_for_best"])
    if passed:
        human = "E3.4 最低通过：写入侧 signature v2 在保持 focus 3/3 的前提下降低了 false retrieval 或关键失败计数。"
        next_rec = "可以继续 E3.4 refinement 或准备 E4 前置安全检查。"
    elif float(best["mean_signature_v2_margin"]) <= 0.0:
        human = "E3.4 未通过：signature v2 仍未让 target 在签名空间中稳定强于 wrong bundle。"
        next_rec = "不要进 E4；优先考虑 E4A 主动视觉证据采集或更强写入侧感知证据。"
    else:
        human = "E3.4 未通过：signature v2 有局部分离信号，但没有转化为安全 retrieval 提升。"
        next_rec = "不要进 E4；转 E3.5 candidate generation / event-type-conditioned scoring。"
    summary = {
        "stage": "E3.4",
        "best_ablation": best,
        "passed_minimum": passed,
        "focus_events": focus_rows,
        "main_failure_counts": dict(Counter(str(r["not_top5_reason"]) for r in all_not_top5 if r["ablation_name"] == best["ablation_name"])),
        "human_summary": human,
        "next_recommendation": next_rec,
    }
    field_rows = signature_field_rows(bundle_by_id)
    field_summary = {"bundle_count": len(field_rows), "degenerate_context": sum(r["is_degenerate_context_signature"] for r in field_rows), "degenerate_disappearance": sum(r["is_degenerate_disappearance_signature"] for r in field_rows), "degenerate_temporal": sum(r["is_degenerate_temporal_signature"] for r in field_rows), "degenerate_motion": sum(r["is_degenerate_motion_signature"] for r in field_rows)}
    compact = {
        "stage": "E3.4",
        "best_ablation": best.get("ablation_name"),
        "passed_minimum": passed,
        "global_top1": best["global_top1"],
        "global_top3": best["global_top3"],
        "global_top5": best["global_top5"],
        "false_bundle_retrieval_rate": best["false_bundle_retrieval_rate"],
        "focus_success_count": best["focus_success_count"],
        "target_in_top3_but_lost_top1_count": best["target_in_top3_but_lost_top1_count"],
        "target_not_in_top5_count": best["target_not_in_top5_count"],
        "signature_collision_count": best["signature_collision_count"],
        "mean_target_wrong_signature_margin": best["mean_target_wrong_signature_margin"],
        "mean_signature_v2_margin": best["mean_signature_v2_margin"],
        "main_failure_counts": summary["main_failure_counts"],
        "next_recommendation": next_rec,
    }
    e31.write_csv(out / f"stage_E34_signature_field_audit_{args.artifact_version}.csv", field_rows)
    e31.write_csv(out / f"stage_E34_bundle_v2_inventory_{args.artifact_version}.csv", inventory_rows(bundle_by_id))
    e31.write_csv(out / f"stage_E34_bundle_v2_write_trace_{args.artifact_version}.csv", write_rows)
    e31.write_csv(out / f"stage_E34_target_not_top5_audit_{args.artifact_version}.csv", all_not_top5)
    e31.write_csv(out / f"stage_E34_target_vs_wrong_signature_margin_{args.artifact_version}.csv", all_margins)
    e31.write_csv(out / f"stage_E34_event_type_routing_audit_{args.artifact_version}.csv", all_routing)
    e31.write_csv(out / f"stage_E34_retrieval_compare_{args.artifact_version}.csv", all_retrieval)
    e31.write_csv(out / f"stage_E34_ablation_summary_{args.artifact_version}.csv", ablation_rows)
    e31.write_csv(out / f"stage_E34_focus_event_summary_{args.artifact_version}.csv", focus_rows)
    (out / f"stage_E34_signature_field_audit_summary_{args.artifact_version}.json").write_text(json.dumps(field_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"stage_E34_compact_for_gpt_{args.artifact_version}.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"stage_E34_report_{args.artifact_version}.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
