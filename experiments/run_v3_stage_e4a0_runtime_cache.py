from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator
from experiments import run_v3_stage_e31_retrieval_competition_repair as e31
from experiments import run_v3_stage_e34_write_side_signature_v2 as e34
from experiments import run_v3_stage_e34r_support_trajectory_refinement as e34r
from experiments.phase3r_utils import build_phase3_scenario_map, load_config_payload
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.objectness import MinimalObjectnessField


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build v3 Stage E4A runtime cache.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--event-audit", default="results/v3_e1/stage_E1_event_audit_v1.csv")
    p.add_argument("--cross-run-alignment", default="results/v3_e2rm/stage_E2R_cross_run_target_alignment_v1.csv")
    p.add_argument("--output-dir", default="results/v3_e4a/cache")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--buffer-size", type=int, default=16)
    return p.parse_args()


def arr_to_list(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.astype(float).tolist()
    if isinstance(v, dict):
        return {k: arr_to_list(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [arr_to_list(x) for x in v]
    return v


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_name": event["scenario_name"],
        "event_id": event["event_id"],
        "frame_idx": int(event["frame_idx"]),
        "proposal_detected": int(event["proposal_detected"]),
        "proposal_id": event["proposal_id"],
        "target_bundle_id": event["target_bundle_id"],
        "target_bundle_exists": int(event["target_bundle_exists"]),
        "old_track_id": event["old_track_id"],
        "old_prototype_id": event["old_prototype_id"],
        "alignment_classification": event["alignment_classification"],
        "target_anchor_uid": event["target_anchor_uid"],
        "cue": arr_to_list(event["cue"]) if event["cue"] is not None else None,
    }


def bundle_inventory(bundle_by_id: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for b in bundle_by_id.values():
        rows.append({
            "bundle_id": int(b["bundle_id"]),
            "memory_anchor_id": b["memory_anchor_id"],
            "canonical_lineage_id": b["canonical_lineage_id"],
            "source_track_ids": sorted(int(v) for v in b["source_track_ids"]),
            "source_prototype_ids": sorted(int(v) for v in b["source_prototype_ids"]),
            "source_lineage_ids": sorted(int(v) for v in b["source_lineage_ids"]),
            "created_frame": int(b["created_frame"]),
            "last_source_frame": int(b["last_source_frame"]),
            "v2_evidence_frame_count": int(b.get("v2_evidence_frame_count", 0)),
        })
    return rows


def event_frame_context(config_path: str, event_records: list[dict[str, Any]], seed: int) -> dict[str, dict[str, Any]]:
    payload = load_config_payload(config_path)
    scenario_map = build_phase3_scenario_map(config_path)
    needed: dict[str, set[int]] = {}
    for e in event_records:
        if int(e["proposal_detected"]) != 1:
            continue
        needed.setdefault(str(e["scenario_name"]), set()).add(int(e["frame_idx"]))
    contexts: dict[str, dict[str, Any]] = {}
    for scenario_name, frames_needed in needed.items():
        sequence = SyntheticStreamGenerator(scenario_map[scenario_name], seed=seed).generate_sequence(0)
        encoder = MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
        field = MinimalObjectnessField(**payload["field"])
        for frame_offset in range(1, len(sequence.frames)):
            current_frame = sequence.frames[frame_offset]
            frame_idx = int(current_frame.frame_index)
            if frame_idx not in frames_needed:
                continue
            prev_frame = sequence.frames[frame_offset - 1]
            encoding = encoder.encode(prev_frame.frame, current_frame.frame)
            objectness_output = field.compute(encoding)
            proposals = []
            for idx, p in enumerate(objectness_output.proposals):
                proposals.append({
                    "proposal_id": int(idx),
                    "box": tuple(int(v) for v in p.box),
                    "centroid": tuple(float(v) for v in p.centroid),
                    "score": float(p.score),
                })
            contexts[f"{scenario_name}:{frame_idx}"] = {
                "scenario_name": scenario_name,
                "frame_idx": frame_idx,
                "frame": np.asarray(current_frame.frame),
                "proposals": proposals,
                "heatmap": np.asarray(objectness_output.heatmap, dtype=np.float32),
            }
    return contexts


def passive_scores(bundle_by_id, event_records):
    proto_counter, track_counter, lineage_counter = e31.compute_static_counts(bundle_by_id)
    cfg = e34r.ablation_cfgs()[e34r.BASELINE_NAME]
    rows = []
    for event in event_records:
        scored = e34r.score_event(event, bundle_by_id, cfg, proto_counter, track_counter, lineage_counter)
        top = scored["final_topk"]
        rows.append({
            "event_id": event["event_id"],
            "top5_bundle_ids": [int(r["bundle_id"]) for r in top[:5]],
            "top5_scores": [float(r.get("e34r_score", r.get("final_score", 0.0))) for r in top[:5]],
            "target_bundle_id": event["target_bundle_id"],
            "target_rank": scored["target_rank"],
        })
    return rows


def main() -> None:
    args = parse_args()
    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle_by_id, event_records, _ = e34.collect_runtime_data_v2(args.config, args.event_audit, args.cross_run_alignment, args.seed, args.buffer_size)
    contexts = event_frame_context(args.config, event_records, args.seed)
    passive = passive_scores(bundle_by_id, event_records)
    cache = {
        "cache_version": args.artifact_version,
        "config": args.config,
        "seed": args.seed,
        "bundle_by_id": bundle_by_id,
        "event_records": event_records,
        "proposal_context": contexts,
        "e34r_passive_scores": passive,
    }
    with (out / f"runtime_collection_cache_{args.artifact_version}.pkl").open("wb") as f:
        pickle.dump(cache, f)
    with (out / f"proposal_context_cache_{args.artifact_version}.pkl").open("wb") as f:
        pickle.dump(contexts, f)
    (out / f"event_records_{args.artifact_version}.json").write_text(json.dumps([compact_event(e) for e in event_records], indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"bundle_inventory_{args.artifact_version}.json").write_text(json.dumps(bundle_inventory(bundle_by_id), indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"e34r_passive_scores_{args.artifact_version}.json").write_text(json.dumps(passive, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "num_events": len(event_records),
        "proposal_detected_events": int(sum(1 for e in event_records if int(e["proposal_detected"]) == 1)),
        "num_bundles": len(bundle_by_id),
        "num_frames_processed": int(len(contexts)),
        "num_cached_proposals": int(sum(len(c["proposals"]) for c in contexts.values())),
        "num_cached_cues": int(sum(1 for e in event_records if e["cue"] is not None)),
        "num_focus_events": int(sum(1 for e in event_records if e["event_id"] in {"M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"})),
        "cache_build_time_sec": float(time.time() - t0),
        "cache_version": args.artifact_version,
    }
    (out / f"cache_summary_{args.artifact_version}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
