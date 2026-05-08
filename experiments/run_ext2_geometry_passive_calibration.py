from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ext1_utils import (
    Box,
    box_descriptor,
    box_iou,
    center_distance,
    collect_frames,
    dataset_inventory,
    frame_size,
    frames_by_index,
    l2,
    normalize_distance,
    object_history,
    read_csv,
    size_similarity,
    state_at,
    trajectory_prediction,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXT-2 geometry-aware passive memory calibration.")
    p.add_argument("--ext1-dir", default="results/ext1")
    p.add_argument("--output-dir", default="results/ext2")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def split_for_sequence(sequence_id: str) -> str:
    digest = hashlib.sha1(sequence_id.encode("utf-8")).hexdigest()
    return "dev" if int(digest[:8], 16) % 2 == 0 else "test"


def candidate_components(
    query_box: Box,
    states_all: list[tuple[int, Box, Any]],
    width: float,
    height: float,
    reappear_frame: int,
) -> dict[str, float]:
    states = [st for st in states_all if st[0] < reappear_frame]
    if not states:
        return {}
    last_frame, last_box, _ = states[-1]
    pred_box = trajectory_prediction(states[-8:]) or last_box
    q_desc = box_descriptor(query_box, width, height)
    last_desc = box_descriptor(last_box, width, height)
    pred_desc = box_descriptor(pred_box, width, height)
    candidate_age = max(0, int(reappear_frame - last_frame))
    dist_last = normalize_distance(center_distance(query_box, last_box), width, height)
    dist_pred = normalize_distance(center_distance(query_box, pred_box), width, height)
    shape_last = -l2(q_desc[2:], last_desc[2:])
    shape_pred = -l2(q_desc[2:], pred_desc[2:])
    return {
        "candidate_age": float(candidate_age),
        "last_iou": box_iou(query_box, last_box),
        "pred_iou": box_iou(query_box, pred_box),
        "last_distance": dist_last,
        "pred_distance": dist_pred,
        "last_size_similarity": size_similarity(query_box, last_box),
        "pred_size_similarity": size_similarity(query_box, pred_box),
        "shape_last": shape_last,
        "shape_pred": shape_pred,
        "recency_score": -min(1.0, candidate_age / 200.0),
        "trajectory_score": 0.8 * box_iou(query_box, pred_box) + 0.4 * size_similarity(query_box, pred_box) - dist_pred,
        "last_geometry_score": 1.2 * box_iou(query_box, last_box) + 0.5 * size_similarity(query_box, last_box) - dist_last,
        "support_baseline_score": 0.8 * box_iou(query_box, pred_box) + 0.4 * size_similarity(query_box, pred_box) - dist_pred,
    }


def score_variant(variant: str, c: dict[str, float], candidate_count: int) -> float:
    traj = c["trajectory_score"]
    shape = c["shape_last"]
    recency = c["recency_score"]
    last_geom = c["last_geometry_score"]
    age = c["candidate_age"]
    if variant == "A0_nops_current":
        return 0.45 * (0.8 * c["pred_iou"] - c["pred_distance"]) + 0.35 * shape + 0.20 * recency
    if variant == "A1_nops_no_recency":
        return 0.60 * traj + 0.40 * shape
    if variant == "A2_trajectory_heavy":
        return 0.85 * traj + 0.15 * shape
    if variant == "A3_support_trajectory_reference":
        return c["support_baseline_score"]
    if variant == "A4_last_geometry_reference":
        return last_geom
    if variant == "A5_gap_adaptive_no_recency":
        # Long-gaps should rely less on immediate recency and more on predicted support trajectory.
        if age >= 10:
            return 0.80 * traj + 0.20 * shape
        return 0.45 * last_geom + 0.35 * traj + 0.20 * shape
    if variant == "A6_competition_trajectory_boost":
        boost = 0.10 if candidate_count >= 3 else 0.0
        return (0.75 + boost) * traj + (0.25 - min(boost, 0.20)) * shape
    if variant == "A7_geometry_calibrated":
        # Conservative calibrated mix: no recency penalty, trajectory-dominant,
        # small last-geometry term for short gaps.
        short_gap_bonus = 0.15 * last_geom if age < 5 else 0.0
        return 0.75 * traj + 0.20 * shape + short_gap_bonus
    raise ValueError(variant)


VARIANTS = [
    "A0_nops_current",
    "A1_nops_no_recency",
    "A2_trajectory_heavy",
    "A3_support_trajectory_reference",
    "A4_last_geometry_reference",
    "A5_gap_adaptive_no_recency",
    "A6_competition_trajectory_boost",
    "A7_geometry_calibrated",
]


def evaluate_variants(ledger_rows: list[dict[str, str]], adapters: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    frame_cache: dict[tuple[str, str], tuple[dict[str, list[tuple[int, Box, Any]]], dict[int, Any]]] = {}
    for row in ledger_rows:
        if str(row.get("event_usable")) != "1":
            continue
        dataset_name = row["dataset_name"]
        if dataset_name not in adapters:
            continue
        key = (dataset_name, row["sequence_id"])
        if key not in frame_cache:
            frames = collect_frames(adapters[dataset_name], row["sequence_id"])
            frame_cache[key] = (object_history(frames), frames_by_index(frames))
        hist, frame_lookup = frame_cache[key]
        target_id = row["instance_id"]
        reappear = int(row["reappear_frame"])
        target_state = state_at(hist, target_id, reappear)
        frame = frame_lookup.get(reappear)
        if target_state is None:
            continue
        all_boxes = [b for states in hist.values() for _, b, _ in states[:1]]
        width, height = frame_size(frame, all_boxes)
        query_box = target_state[1]
        candidates = {iid: states for iid, states in hist.items() if any(st[0] < reappear for st in states)}
        comps_by_id: dict[str, dict[str, float]] = {}
        for iid, states in candidates.items():
            comps = candidate_components(query_box, states, width, height, reappear)
            if comps:
                comps_by_id[iid] = comps
        if target_id not in comps_by_id:
            continue
        split = split_for_sequence(row["sequence_id"])
        for variant in VARIANTS:
            scored = [
                (iid, score_variant(variant, comps, len(comps_by_id)))
                for iid, comps in comps_by_id.items()
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            ranked_ids = [iid for iid, _ in scored]
            top1 = ranked_ids[0] if ranked_ids else ""
            top1_hit = int(top1 == target_id)
            top3_hit = int(target_id in ranked_ids[:3])
            top5_hit = int(target_id in ranked_ids[:5])
            event_rows.append({
                "dataset_name": dataset_name,
                "sequence_id": row["sequence_id"],
                "event_id": row["event_id"],
                "split": split,
                "variant": variant,
                "target_instance_id_eval_only": target_id,
                "predicted_memory_id": top1,
                "top1": top1_hit,
                "top3": top3_hit,
                "top5": top5_hit,
                "false_retrieval": int(not top1_hit),
                "target_in_top5_but_lost_top1": int(top5_hit and not top1_hit),
                "candidate_count": len(comps_by_id),
                "gap_length": row.get("gap_length", ""),
                "num_similar_distractors": row.get("num_similar_distractors", ""),
            })
        # Component trace for current NOPS top1 competitor.
        current_scored = [
            (iid, score_variant("A0_nops_current", comps, len(comps_by_id)))
            for iid, comps in comps_by_id.items()
        ]
        current_scored.sort(key=lambda x: x[1], reverse=True)
        wrong_id = current_scored[0][0] if current_scored and current_scored[0][0] != target_id else (current_scored[1][0] if len(current_scored) > 1 else "")
        target_comps = comps_by_id[target_id]
        wrong_comps = comps_by_id.get(wrong_id, {})
        if wrong_comps:
            component_rows.append({
                "dataset_name": dataset_name,
                "sequence_id": row["sequence_id"],
                "event_id": row["event_id"],
                "split": split,
                "target_instance_id_eval_only": target_id,
                "wrong_top_competitor_id_eval_only": wrong_id,
                "target_candidate_age": target_comps["candidate_age"],
                "wrong_candidate_age": wrong_comps["candidate_age"],
                "target_recency_score": target_comps["recency_score"],
                "wrong_recency_score": wrong_comps["recency_score"],
                "target_trajectory_score": target_comps["trajectory_score"],
                "wrong_trajectory_score": wrong_comps["trajectory_score"],
                "target_shape_last": target_comps["shape_last"],
                "wrong_shape_last": wrong_comps["shape_last"],
                "target_a0_score": score_variant("A0_nops_current", target_comps, len(comps_by_id)),
                "wrong_a0_score": score_variant("A0_nops_current", wrong_comps, len(comps_by_id)),
                "target_a7_score": score_variant("A7_geometry_calibrated", target_comps, len(comps_by_id)),
                "wrong_a7_score": score_variant("A7_geometry_calibrated", wrong_comps, len(comps_by_id)),
                "recency_favors_wrong": int(wrong_comps["recency_score"] > target_comps["recency_score"]),
                "trajectory_favors_target": int(target_comps["trajectory_score"] > wrong_comps["trajectory_score"]),
                "a0_wrong_but_a7_target": int(
                    score_variant("A0_nops_current", wrong_comps, len(comps_by_id)) > score_variant("A0_nops_current", target_comps, len(comps_by_id))
                    and score_variant("A7_geometry_calibrated", target_comps, len(comps_by_id)) > score_variant("A7_geometry_calibrated", wrong_comps, len(comps_by_id))
                ),
            })
    return event_rows, component_rows


def summarize(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        groups[(row["variant"], "all")].append(row)
        groups[(row["variant"], row["split"])].append(row)
    out: list[dict[str, Any]] = []
    for variant in VARIANTS:
        all_rows = groups[(variant, "all")]
        dev_rows = groups[(variant, "dev")]
        test_rows = groups[(variant, "test")]
        def rate(rows: list[dict[str, Any]], key: str) -> float:
            return sum(int(r[key]) for r in rows) / max(len(rows), 1)
        out.append({
            "variant": variant,
            "event_count": len(all_rows),
            "global_top1": rate(all_rows, "top1"),
            "global_top3": rate(all_rows, "top3"),
            "global_top5": rate(all_rows, "top5"),
            "false_retrieval_rate": rate(all_rows, "false_retrieval"),
            "top5_but_lost_top1_count": sum(int(r["target_in_top5_but_lost_top1"]) for r in all_rows),
            "dev_top1": rate(dev_rows, "top1"),
            "test_top1": rate(test_rows, "top1"),
            "dev_event_count": len(dev_rows),
            "test_event_count": len(test_rows),
            "selected_as_best_by_dev": 0,
        })
    best = max(out, key=lambda r: (r["dev_top1"], r["test_top1"]))
    best["selected_as_best_by_dev"] = 1
    a0 = next(r for r in out if r["variant"] == "A0_nops_current")
    for row in out:
        row["global_delta_vs_a0"] = row["global_top1"] - a0["global_top1"]
        row["test_delta_vs_a0"] = row["test_top1"] - a0["test_top1"]
    return out


def failure_taxonomy(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in event_rows:
        if int(row["top1"]):
            reason = "success"
        elif int(row["top5"]):
            reason = "target_in_top5_but_wrong_top1"
        else:
            reason = "target_not_in_top5"
        if int(row.get("num_similar_distractors") or 0) > 0 and reason == "target_in_top5_but_wrong_top1":
            reason = "similar_distractor_top1_confusion"
        out.append({
            "variant": row["variant"],
            "event_id": row["event_id"],
            "split": row["split"],
            "failure_reason": reason,
        })
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inventory, adapters = dataset_inventory()
    ledger = read_csv(Path(args.ext1_dir) / "stage_EXT1_external_event_ledger_v1.csv")
    event_rows, component_rows = evaluate_variants(ledger, adapters)
    summary = summarize(event_rows)
    taxonomy = failure_taxonomy(event_rows)
    write_csv(out / f"stage_EXT2_ablation_summary_{args.artifact_version}.csv", summary)
    write_csv(out / f"stage_EXT2_event_delta_{args.artifact_version}.csv", event_rows)
    write_csv(out / f"stage_EXT2_component_score_trace_{args.artifact_version}.csv", component_rows)
    write_csv(out / f"stage_EXT2_failure_taxonomy_{args.artifact_version}.csv", taxonomy)
    write_csv(out / f"stage_EXT2_dataset_inventory_{args.artifact_version}.csv", inventory)
    selected = next(r for r in summary if int(r["selected_as_best_by_dev"]) == 1)
    nops_calibrated_candidates = [
        r for r in summary
        if r["variant"] not in {"A0_nops_current", "A3_support_trajectory_reference", "A4_last_geometry_reference"}
    ]
    best_calibrated = max(nops_calibrated_candidates, key=lambda r: (r["dev_top1"], r["test_top1"]))
    a0 = next(r for r in summary if r["variant"] == "A0_nops_current")
    support = next(r for r in summary if r["variant"] == "A3_support_trajectory_reference")
    recency_hurts = sum(
        1 for r in component_rows
        if int(r["recency_favors_wrong"]) and int(r["trajectory_favors_target"])
    )
    a0_wrong_a7_target = sum(int(r["a0_wrong_but_a7_target"]) for r in component_rows)
    failure_counts = Counter(r["failure_reason"] for r in taxonomy if r["variant"] == selected["variant"])
    compact = {
        "stage": "EXT-2",
        "event_count": int(selected["event_count"]),
        "a0_nops_current_top1": a0["global_top1"],
        "support_trajectory_reference_top1": support["global_top1"],
        "best_variant": selected["variant"],
        "best_global_top1": selected["global_top1"],
        "best_dev_top1": selected["dev_top1"],
        "best_test_top1": selected["test_top1"],
        "best_delta_vs_a0": selected["global_delta_vs_a0"],
        "best_nops_calibrated_variant": best_calibrated["variant"],
        "best_nops_calibrated_global_top1": best_calibrated["global_top1"],
        "best_nops_calibrated_test_top1": best_calibrated["test_top1"],
        "best_nops_calibrated_delta_vs_a0": best_calibrated["global_delta_vs_a0"],
        "remaining_gap_to_support_reference": support["global_top1"] - best_calibrated["global_top1"],
        "recency_favors_wrong_when_trajectory_favors_target_count": recency_hurts,
        "a0_wrong_but_calibrated_pairwise_target_count": a0_wrong_a7_target,
        "best_failure_counts": dict(failure_counts.most_common(5)),
        "primary_diagnosis": "current NOPS passive over-penalizes long-gap candidates through recency/weak trajectory weighting; trajectory-heavy calibrated scoring improves but still trails the pure support-trajectory reference",
        "next_recommendation": "validate calibrated geometry scoring on held-out/full-pixel data before changing main NOPS; LaSOT pixels are needed for appearance/full-pipeline claims, not for this geometry diagnosis",
    }
    report = "\n".join([
        "# Stage EXT-2 Geometry Passive Calibration",
        "",
        "## Scope",
        "",
        "Oracle-proposal / geometry-only external calibration on LaGOT annotations. No target ID is used in scoring; target identity is evaluation-only.",
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
    (out / f"stage_EXT2_compact_for_gpt_{args.artifact_version}.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"stage_EXT2_report_{args.artifact_version}.md").write_text(report, encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
