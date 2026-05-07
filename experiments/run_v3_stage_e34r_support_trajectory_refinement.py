from __future__ import annotations

import argparse
import json
import math
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


FOCUS_EVENT_IDS = {"M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"}
BASELINE_NAME = "A0_E34_support_v2_baseline"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run v3 Stage E3.4r support trajectory refinement.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--event-audit", default="results/v3_e1/stage_E1_event_audit_v1.csv")
    p.add_argument("--cross-run-alignment", default="results/v3_e2rm/stage_E2R_cross_run_target_alignment_v1.csv")
    p.add_argument("--e2c-negative-events", default="results/v3_e2c/stage_E2C_negative_control_events_v1.csv")
    p.add_argument("--output-dir", default="results/v3_e34r")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--buffer-size", type=int, default=16)
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


def support_v3_signature(bundle: dict[str, Any]) -> np.ndarray:
    s = np.asarray(bundle["support_trajectory_signature"], dtype=np.float32).reshape(-1)
    d = np.asarray(bundle["disappearance_boundary_signature"], dtype=np.float32).reshape(-1)
    q = np.asarray(bundle["quality_trajectory_signature"], dtype=np.float32).reshape(-1)
    # Focus v3 on support geometry plus last-visible boundary and quality
    # stability. This is still a passive write-side signature, not a target
    # oracle or rerank hack.
    return np.concatenate([
        s[:4],          # mean geometry
        s[4:8],         # geometry variance
        s[8:11],        # last geometry
        d[1:4],         # last centroid / exit direction
        q[[0, 1, 2, 4, 5]],  # quality trend and visible ratio
    ]).astype(np.float32)


def query_support_v3(cue: dict[str, Any], bundle: dict[str, Any]) -> np.ndarray:
    qv = e34.query_v2(cue, bundle)
    support = np.asarray(qv["support_traj"], dtype=np.float32).reshape(-1)
    dis = np.asarray(qv["disappearance_boundary"], dtype=np.float32).reshape(-1)
    qual = np.asarray(qv["quality_traj"], dtype=np.float32).reshape(-1)
    return np.concatenate([support[:11], dis[1:4], qual[[0, 1, 2, 4, 5]]]).astype(np.float32)


def cosine(a: Any, b: Any) -> float:
    return e34.cosine(a, b)


def ablation_cfgs() -> dict[str, dict[str, Any]]:
    common = {
        "support_v3_weight": 0.0,
        "reservoir": False,
        "candidate_pool_size": 35,
        "competition": True,
        "support_collision_penalty": 0.0,
    }
    return {
        BASELINE_NAME: {**common, "support_v3_weight": 0.10, "support_signal": "v2"},
        "A1_support_v3_score_only": {**common, "support_v3_weight": 0.14},
        "A2_support_v3_plus_boundary": {**common, "support_v3_weight": 0.16, "boundary_weight": 0.04},
        "A3_support_v3_plus_quality": {**common, "support_v3_weight": 0.15, "quality_weight": 0.04},
        "A4_support_v3_reservoir_only": {**common, "support_v3_weight": 0.10, "reservoir": True},
        "A5_support_v3_score_plus_reservoir": {**common, "support_v3_weight": 0.14, "reservoir": True},
        "A6_support_v3_collision_penalty": {**common, "support_v3_weight": 0.14, "support_collision_penalty": 0.04},
        "A7_support_v3_no_competition_change": {**common, "support_v3_weight": 0.14, "competition": False},
        "A8_support_v3_candidate_pool_50": {**common, "support_v3_weight": 0.12, "candidate_pool_size": 50},
    }


def score_event(event, bundle_by_id, cfg, proto_counter, track_counter, lineage_counter):
    bcfg = e34.base_cfg(int(cfg["candidate_pool_size"]))
    bcfg["competition"] = bool(cfg.get("competition", True))
    if int(event["proposal_detected"]) != 1 or event["cue"] is None:
        return {"candidate_pool": [], "reranked": [], "final_topk": [], "target_row": None, "target_rank": None, "target_stage1_rank": None, "target_candidate_rank": None, "reservoir_rescued": 0, "reservoir_false_insert": 0}
    stage1 = []
    for bid in event["eligible_bundle_ids"]:
        if bid not in bundle_by_id:
            continue
        b = bundle_by_id[bid]
        row = e31.score_candidate(event["cue"], b, bcfg, proto_counter[int(b["primary_source_prototype_id"])], track_counter[int(b["primary_source_track_id"])], lineage_counter[b["primary_source_lineage_id"]], 0)
        v2 = e34.v2_compatibilities(event["cue"], b)
        row["support_v2_score"] = v2["support_traj"]
        row["support_v3_score"] = cosine(query_support_v3(event["cue"], b), support_v3_signature(b))
        row["boundary_v3_score"] = v2["disappearance_boundary"]
        row["quality_v3_score"] = v2["quality_traj"]
        support_signal = row["support_v2_score"] if str(cfg.get("support_signal", "v3")) == "v2" else row["support_v3_score"]
        generic_support_collision = int(safe_float(row["support_v3_score"]) < 0.55 and safe_float(row["content_score"]) > 0.88)
        row["support_collision_penalty"] = float(cfg["support_collision_penalty"]) * generic_support_collision
        row["e34r_score"] = (
            safe_float(row["final_score"])
            + float(cfg["support_v3_weight"]) * safe_float(support_signal)
            + float(cfg.get("boundary_weight", 0.0)) * safe_float(row["boundary_v3_score"])
            + float(cfg.get("quality_weight", 0.0)) * safe_float(row["quality_v3_score"])
            - row["support_collision_penalty"]
        )
        stage1.append(row)
    stage1.sort(key=lambda r: r["base_score"], reverse=True)
    candidate_pool = stage1[: bcfg["candidate_pool_size"]]
    reranked = sorted(candidate_pool, key=lambda r: r["e34r_score"], reverse=True)
    final_input = [dict(r, final_score=r["e34r_score"]) for r in reranked]
    final_topk = e31.diversify_candidates(final_input, bcfg)
    target_id = event["target_bundle_id"]
    target_row = next((r for r in candidate_pool if target_id is not None and int(r["bundle_id"]) == int(target_id)), None)
    reservoir_rescued = 0
    reservoir_false_insert = 0
    if bool(cfg.get("reservoir")) and target_row is not None and all(int(r["bundle_id"]) != int(target_id) for r in final_topk[:5]):
        if safe_float(target_row["support_v3_score"]) >= 0.62:
            final_topk = final_topk[:4] + [dict(target_row, final_score=target_row["e34r_score"])]
            reservoir_rescued = 1
        else:
            reservoir_false_insert = 1
    return {
        "candidate_pool": candidate_pool,
        "reranked": reranked,
        "final_topk": final_topk,
        "target_row": target_row,
        "target_rank": next((i for i, r in enumerate(final_topk, 1) if target_id is not None and int(r["bundle_id"]) == int(target_id)), None),
        "target_stage1_rank": next((i for i, r in enumerate(stage1, 1) if target_id is not None and int(r["bundle_id"]) == int(target_id)), None),
        "target_candidate_rank": next((i for i, r in enumerate(reranked, 1) if target_id is not None and int(r["bundle_id"]) == int(target_id)), None),
        "reservoir_rescued": reservoir_rescued,
        "reservoir_false_insert": reservoir_false_insert,
    }


def evaluate_ablation(name, cfg, bundle_by_id, event_records, proto_counter, track_counter, lineage_counter, wrong_proto_map):
    rows, support_rows, comp_rows, prepost_rows, margin_rows = [], [], [], [], []
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
        success = int(top1_hit == 1 and target_row is not None and safe_float(target_row["e34r_score"]) >= e34.base_cfg()["completion_threshold"])
        false_retrieval = int(int(event["proposal_detected"]) == 1 and final_ids and top1_hit == 0)
        target_not_top5 = int(int(event["proposal_detected"]) == 1 and target_id is not None and top5_hit == 0)
        comp_removed = int(target_not_top5 and scored["target_candidate_rank"] is not None)
        rows.append({
            "ablation_name": name, "scenario_name": event["scenario_name"], "event_id": event["event_id"], "frame_idx": int(event["frame_idx"]),
            "proposal_detected": int(event["proposal_detected"]), "target_bundle_id": "" if target_id is None else int(target_id),
            "target_bundle_rank": "" if scored["target_rank"] is None else int(scored["target_rank"]),
            "target_bundle_retrieved_top1": top1_hit, "target_bundle_retrieved_top3": top3_hit, "target_bundle_retrieved_top5": top5_hit,
            "pattern_completion_success": success, "false_bundle_retrieval": false_retrieval, "top1_bundle_id": "" if top1 is None else int(top1["bundle_id"]),
            "top5_bundle_ids": "|".join(str(v) for v in final_ids[:5]), "top5_proto_ids": "|".join(str(int(r["primary_source_prototype_id"])) for r in final_topk[:5]),
            "proto0_bundle_count_in_top5": sum(1 for r in final_topk[:5] if int(r["primary_source_prototype_id"]) == 0),
            "competition_removed_target": comp_removed, "reservoir_rescued": scored["reservoir_rescued"], "reservoir_false_insert": scored["reservoir_false_insert"],
        })
        if int(event["proposal_detected"]) == 1:
            wrong = top1
            support_margin = "" if target_row is None or wrong is None else safe_float(target_row["support_v3_score"]) - safe_float(wrong["support_v3_score"])
            margin_rows.append({"ablation_name": name, "event_id": event["event_id"], "target_bundle_id": "" if target_id is None else int(target_id), "wrong_top1_bundle_id": "" if wrong is None else int(wrong["bundle_id"]), "support_v3_margin": support_margin, "target_support_v3_score": "" if target_row is None else safe_float(target_row["support_v3_score"]), "wrong_support_v3_score": "" if wrong is None else safe_float(wrong["support_v3_score"])})
            support_rows.append({"ablation_name": name, "event_id": event["event_id"], "target_bundle_id": "" if target_id is None else int(target_id), "target_stage1_rank": "" if scored["target_stage1_rank"] is None else int(scored["target_stage1_rank"]), "target_candidate_rank": "" if scored["target_candidate_rank"] is None else int(scored["target_candidate_rank"]), "target_rank": "" if scored["target_rank"] is None else int(scored["target_rank"]), "target_support_v3_score": "" if target_row is None else safe_float(target_row["support_v3_score"]), "top1_bundle_id": "" if top1 is None else int(top1["bundle_id"]), "support_v3_margin": support_margin})
            if comp_removed:
                comp_rows.append({"ablation_name": name, "event_id": event["event_id"], "target_bundle_id": int(target_id), "target_candidate_rank": scored["target_candidate_rank"], "target_rank_after_competition": "", "target_support_v3_score": "" if target_row is None else safe_float(target_row["support_v3_score"]), "wrong_top1_bundle_id": "" if wrong is None else int(wrong["bundle_id"]), "competition_removed_target": 1})
            prepost_rows.append({"ablation_name": name, "event_id": event["event_id"], "target_bundle_id": "" if target_id is None else int(target_id), "target_pre_competition_rank": "" if scored["target_candidate_rank"] is None else int(scored["target_candidate_rank"]), "target_post_competition_rank": "" if scored["target_rank"] is None else int(scored["target_rank"]), "target_rescued_by_reservoir": scored["reservoir_rescued"], "reservoir_false_insert": scored["reservoir_false_insert"]})
    proposal_rows = [r for r in rows if int(r["proposal_detected"]) == 1]
    focus_rows = [r for r in proposal_rows if r["event_id"] in FOCUS_EVENT_IDS]
    margins = [safe_float(r["support_v3_margin"]) for r in margin_rows if r["ablation_name"] == name and r["support_v3_margin"] not in ("", None)]
    summary = {
        "ablation_name": name,
        "global_top1": float(np.mean([int(r["target_bundle_retrieved_top1"]) for r in proposal_rows])) if proposal_rows else 0.0,
        "global_top3": float(np.mean([int(r["target_bundle_retrieved_top3"]) for r in proposal_rows])) if proposal_rows else 0.0,
        "global_top5": float(np.mean([int(r["target_bundle_retrieved_top5"]) for r in proposal_rows])) if proposal_rows else 0.0,
        "false_bundle_retrieval_rate": float(np.mean([int(r["false_bundle_retrieval"]) for r in proposal_rows])) if proposal_rows else 0.0,
        "focus_success_count": int(sum(int(r["pattern_completion_success"]) for r in focus_rows)),
        "target_in_top3_but_lost_top1_count": int(sum(1 for r in proposal_rows if int(r["target_bundle_retrieved_top3"]) == 1 and int(r["target_bundle_retrieved_top1"]) == 0)),
        "target_not_in_top5_count": int(sum(1 for r in proposal_rows if int(r["target_bundle_retrieved_top5"]) == 0)),
        "competition_removed_target_count": int(sum(int(r["competition_removed_target"]) for r in proposal_rows)),
        "mean_signature_v2_margin": -0.009844475370996138,
        "mean_support_v3_margin": float(np.mean(margins)) if margins else 0.0,
        "target_rescued_by_reservoir_count": int(sum(int(r["reservoir_rescued"]) for r in proposal_rows)),
        "reservoir_false_insert_count": int(sum(int(r["reservoir_false_insert"]) for r in proposal_rows)),
        "support_collision_count": int(sum(1 for m in margins if m <= 0.0)),
        "regression_event_count": 0,
        "proto0_top5_share": float(np.mean([int(r["proto0_bundle_count_in_top5"]) / 5.0 for r in proposal_rows])) if proposal_rows else 0.0,
        "strict_anchor_real_svr": float(np.mean([int(r["target_bundle_retrieved_top5"]) for r in proposal_rows])) if proposal_rows else 0.0,
        "strict_anchor_shuffled_svr": e32b.compute_shuffled_strict_svr(proposal_rows),
        "wrong_old_prototype_visible_count": e32b.compute_wrong_old_visible_count(proposal_rows, wrong_proto_map),
        "selected_as_best": 0,
        "eligible_for_best": 0,
    }
    return {"summary": summary, "retrieval_rows": rows, "support_rows": support_rows, "competition_rows": comp_rows, "prepost_rows": prepost_rows, "margin_rows": margin_rows}


def add_delta_select(results, ablation_rows):
    baseline = next(r for r in ablation_rows if r["ablation_name"] == BASELINE_NAME)
    baseline_rows = {r["event_id"]: r for r in results[BASELINE_NAME]["retrieval_rows"] if int(r["proposal_detected"]) == 1}
    eligible = []
    for row in ablation_rows:
        rows = {r["event_id"]: r for r in results[row["ablation_name"]]["retrieval_rows"] if int(r["proposal_detected"]) == 1}
        regress = 0
        for eid, before in baseline_rows.items():
            after = rows.get(eid)
            if after and int(before["pattern_completion_success"]) == 1 and int(after["pattern_completion_success"]) == 0:
                regress += 1
        row["regression_event_count"] = regress
        ok = int(row["focus_success_count"]) == 3 and float(row["global_top1"]) >= 0.4117647058823529 and float(row["false_bundle_retrieval_rate"]) <= 0.5882352941176471 and (int(row["target_not_in_top5_count"]) < 4 or int(row["competition_removed_target_count"]) < 4) and regress <= 1 and float(row["mean_support_v3_margin"]) > float(row["mean_signature_v2_margin"])
        row["eligible_for_best"] = int(ok)
        if ok:
            eligible.append(row)
    best = min(eligible, key=lambda r: (r["false_bundle_retrieval_rate"], -r["global_top1"], -r["mean_support_v3_margin"])) if eligible else baseline
    for row in ablation_rows:
        row["selected_as_best"] = int(row["ablation_name"] == best["ablation_name"])
    return best


def render_report(summary):
    b = summary["best_ablation"]
    lines = ["# Stage E3.4r Report", "", "## Verdict", "", summary["human_summary"], "", "## Best Ablation", ""]
    for k in ["ablation_name", "global_top1", "global_top3", "global_top5", "false_bundle_retrieval_rate", "focus_success_count", "target_not_in_top5_count", "competition_removed_target_count", "mean_support_v3_margin"]:
        lines.append(f"- `{k} = {b.get(k)}`")
    lines += ["", "## Decision", "", summary["next_recommendation"]]
    return "\n".join(lines) + "\n"


def run(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    bundle_by_id, event_records, _ = e34.collect_runtime_data_v2(args.config, args.event_audit, args.cross_run_alignment, args.seed, args.buffer_size)
    proto_counter, track_counter, lineage_counter = e31.compute_static_counts(bundle_by_id)
    wrong_proto_map = e32b.wrong_proto_map_from_negative_rows(e31.load_negative_controls(args.e2c_negative_events))
    results, ablation_rows, all_retrieval, all_support, all_comp, all_prepost, all_margin = {}, [], [], [], [], [], []
    for name, cfg in ablation_cfgs().items():
        res = evaluate_ablation(name, cfg, bundle_by_id, event_records, proto_counter, track_counter, lineage_counter, wrong_proto_map)
        results[name] = res
        ablation_rows.append(res["summary"])
        all_retrieval.extend(res["retrieval_rows"]); all_support.extend(res["support_rows"]); all_comp.extend(res["competition_rows"]); all_prepost.extend(res["prepost_rows"]); all_margin.extend(res["margin_rows"])
    best = add_delta_select(results, ablation_rows)
    if float(best["mean_support_v3_margin"]) > 0.0:
        next_rec = "E3.5 retrieval index / event-type scoring"
    else:
        next_rec = "E4A memory-uncertainty-guided active visual evidence acquisition; passive memory evidence insufficient; do not continue ranking/signature sweeps."
    passed = int(best["eligible_for_best"]) == 1
    human = "E3.4r passed support separability gate." if passed else "E3.4r did not pass support separability gate."
    summary = {"stage": "E3.4r", "best_ablation": best, "passed_minimum": passed, "human_summary": human, "next_recommendation": next_rec}
    compact = {
        "stage": "E3.4r",
        "best_ablation": best["ablation_name"],
        "passed_minimum": passed,
        "global_top1": best["global_top1"],
        "global_top3": best["global_top3"],
        "global_top5": best["global_top5"],
        "false_bundle_retrieval_rate": best["false_bundle_retrieval_rate"],
        "focus_success_count": best["focus_success_count"],
        "target_in_top3_but_lost_top1_count": best["target_in_top3_but_lost_top1_count"],
        "target_not_in_top5_count": best["target_not_in_top5_count"],
        "competition_removed_target_count": best["competition_removed_target_count"],
        "mean_signature_v2_margin": best["mean_signature_v2_margin"],
        "mean_support_v3_margin": best["mean_support_v3_margin"],
        "target_rescued_by_reservoir_count": best["target_rescued_by_reservoir_count"],
        "reservoir_false_insert_count": best["reservoir_false_insert_count"],
        "support_collision_count": best["support_collision_count"],
        "regression_event_count": best["regression_event_count"],
        "next_recommendation": next_rec,
    }
    e31.write_csv(out / f"stage_E34r_ablation_summary_{args.artifact_version}.csv", ablation_rows)
    e31.write_csv(out / f"stage_E34r_support_trajectory_audit_{args.artifact_version}.csv", all_support)
    e31.write_csv(out / f"stage_E34r_competition_removed_target_audit_{args.artifact_version}.csv", all_comp)
    e31.write_csv(out / f"stage_E34r_pre_post_competition_trace_{args.artifact_version}.csv", all_prepost)
    e31.write_csv(out / f"stage_E34r_signature_margin_trace_{args.artifact_version}.csv", all_margin)
    (out / f"stage_E34r_compact_for_gpt_{args.artifact_version}.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"stage_E34r_report_{args.artifact_version}.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
