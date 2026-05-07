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


FOCUS_EVENT_IDS = {"M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"}
BASELINE_NAME = "A0_E32b_baseline"
DIMENSIONS = (
    "content",
    "support",
    "motion",
    "context",
    "temporal",
    "disappearance",
    "provenance",
    "separation",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run v3 Stage E3.3 cue signature repair.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--event-audit", default="results/v3_e1/stage_E1_event_audit_v1.csv")
    p.add_argument("--cross-run-alignment", default="results/v3_e2rm/stage_E2R_cross_run_target_alignment_v1.csv")
    p.add_argument("--e2c-negative-events", default="results/v3_e2c/stage_E2C_negative_control_events_v1.csv")
    p.add_argument("--output-dir", default="results/v3_e33")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--audit-only", action="store_true")
    return p.parse_args()


def base_cfg() -> dict[str, Any]:
    return {
        "candidate_pool_size": 35,
        "final_topk": 5,
        "competition": True,
        "max_per_proto": 1,
        "max_per_anchor": 1,
        "max_per_lineage": 3,
        "temporal_decay": 120.0,
        "completion_threshold": 0.50,
        "rarity_bonus": 0.14,
        "hub_alpha": 0.03,
        "hub_beta": 0.015,
        "hub_gamma": 0.015,
        "lineage_hub_gamma": 0.005,
        "weights": {
            "content": 0.14,
            "support": 0.14,
            "motion": 0.10,
            "context": 0.17,
            "temporal": 0.15,
            "disappearance": 0.13,
            "provenance": 0.09,
            "separation": 0.08,
        },
    }


def ablation_cfgs() -> dict[str, dict[str, Any]]:
    common = {
        "enabled_dims": [],
        "signature_consistency_weight": 0.0,
        "separation_bonus_weight": 0.0,
        "generic_collision_penalty": 0.0,
        "content_scale": 1.0,
        "provenance_scale": 1.0,
        "dim_bonus": {},
    }
    return {
        BASELINE_NAME: {**common},
        "A1_disappearance_signature_only": {
            **common,
            "enabled_dims": ["disappearance"],
            "dim_bonus": {"disappearance": 0.08},
            "generic_collision_penalty": 0.015,
        },
        "A2_temporal_signature_only": {
            **common,
            "enabled_dims": ["temporal"],
            "dim_bonus": {"temporal": 0.08},
            "generic_collision_penalty": 0.015,
        },
        "A3_context_signature_only": {
            **common,
            "enabled_dims": ["context"],
            "dim_bonus": {"context": 0.08},
            "generic_collision_penalty": 0.015,
        },
        "A4_motion_phase_signature_only": {
            **common,
            "enabled_dims": ["motion"],
            "dim_bonus": {"motion": 0.08},
            "generic_collision_penalty": 0.015,
        },
        "A5_support_shape_signature_only": {
            **common,
            "enabled_dims": ["support"],
            "dim_bonus": {"support": 0.08},
            "generic_collision_penalty": 0.015,
        },
        "A6_provenance_signature_only": {
            **common,
            "enabled_dims": ["provenance"],
            "dim_bonus": {"provenance": 0.08},
            "generic_collision_penalty": 0.015,
        },
        "A7_separation_signature_only": {
            **common,
            "enabled_dims": ["separation"],
            "dim_bonus": {"separation": 0.10},
            "separation_bonus_weight": 0.08,
            "generic_collision_penalty": 0.02,
        },
        "A8_disappearance_plus_temporal": {
            **common,
            "enabled_dims": ["disappearance", "temporal"],
            "signature_consistency_weight": 0.10,
            "dim_bonus": {"disappearance": 0.06, "temporal": 0.06},
            "generic_collision_penalty": 0.025,
        },
        "A9_context_plus_motion": {
            **common,
            "enabled_dims": ["context", "motion"],
            "signature_consistency_weight": 0.08,
            "dim_bonus": {"context": 0.06, "motion": 0.06},
            "generic_collision_penalty": 0.02,
        },
        "A10_full_E33_signature": {
            **common,
            "enabled_dims": ["temporal", "disappearance", "context", "motion", "provenance", "support"],
            "signature_consistency_weight": 0.12,
            "separation_bonus_weight": 0.06,
            "dim_bonus": {
                "temporal": 0.04,
                "disappearance": 0.05,
                "context": 0.04,
                "motion": 0.03,
                "support": 0.03,
                "provenance": 0.05,
                "separation": 0.04,
            },
            "generic_collision_penalty": 0.04,
            "content_scale": 0.65,
        },
        "A11_full_E33_no_content": {
            **common,
            "enabled_dims": ["temporal", "disappearance", "context", "motion", "provenance", "support"],
            "signature_consistency_weight": 0.12,
            "separation_bonus_weight": 0.06,
            "dim_bonus": {
                "temporal": 0.04,
                "disappearance": 0.05,
                "context": 0.04,
                "motion": 0.03,
                "support": 0.03,
                "provenance": 0.05,
                "separation": 0.04,
            },
            "generic_collision_penalty": 0.04,
            "content_scale": 0.0,
        },
        "A12_full_E33_no_provenance": {
            **common,
            "enabled_dims": ["temporal", "disappearance", "context", "motion", "support"],
            "signature_consistency_weight": 0.11,
            "separation_bonus_weight": 0.05,
            "dim_bonus": {
                "temporal": 0.04,
                "disappearance": 0.05,
                "context": 0.04,
                "motion": 0.03,
                "support": 0.03,
                "separation": 0.04,
            },
            "generic_collision_penalty": 0.04,
            "content_scale": 0.65,
            "provenance_scale": 0.0,
        },
    }


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


def dim_scores(row: dict[str, Any]) -> dict[str, float]:
    return {
        "content": safe_float(row.get("content_score")),
        "support": safe_float(row.get("support_score")),
        "motion": safe_float(row.get("motion_score")),
        "context": safe_float(row.get("context_score")),
        "temporal": safe_float(row.get("temporal_score")),
        "disappearance": safe_float(row.get("disappearance_score")),
        "provenance": safe_float(row.get("provenance_score")),
        "separation": safe_float(row.get("separation_score")),
    }


def non_content_mean(row: dict[str, Any]) -> float:
    d = dim_scores(row)
    return float(np.mean([d["support"], d["motion"], d["context"], d["temporal"], d["disappearance"], d["provenance"], d["separation"]]))


def generic_collision(row: dict[str, Any]) -> bool:
    d = dim_scores(row)
    return bool(d["content"] >= 0.90 and non_content_mean(row) <= 0.72)


def signature_score(row: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, float, float]:
    d = dim_scores(row)
    dims = [d[k] for k in cfg["enabled_dims"] if k in d]
    consistency = geomean(dims)
    dim_bonus = sum(float(cfg["dim_bonus"].get(k, 0.0)) * d[k] for k in d)
    separation_bonus = float(cfg["separation_bonus_weight"]) * d["separation"]
    generic_penalty = float(cfg["generic_collision_penalty"]) * int(generic_collision(row))
    content_adjust = (float(cfg["content_scale"]) - 1.0) * 0.14 * d["content"]
    provenance_adjust = (float(cfg["provenance_scale"]) - 1.0) * 0.09 * d["provenance"]
    score = (
        safe_float(row["final_score"])
        + float(cfg["signature_consistency_weight"]) * consistency
        + dim_bonus
        + separation_bonus
        + content_adjust
        + provenance_adjust
        - generic_penalty
    )
    return float(score), float(consistency), float(generic_penalty)


def score_event(
    event: dict[str, Any],
    bundle_by_id: dict[int, dict[str, Any]],
    bcfg: dict[str, Any],
    cfg: dict[str, Any],
    proto_counter: Counter[int],
    track_counter: Counter[int],
    lineage_counter: Counter[int | None],
    hist_topk: Counter[int],
) -> dict[str, Any]:
    if int(event["proposal_detected"]) != 1 or event["cue"] is None:
        return {
            "candidate_pool": [],
            "reranked": [],
            "final_topk": [],
            "target_row": None,
            "target_rank": None,
            "target_candidate_rank": None,
        }
    eligible = [bundle_by_id[b] for b in event["eligible_bundle_ids"] if b in bundle_by_id]
    stage1 = []
    for bundle in eligible:
        row = e31.score_candidate(
            event["cue"],
            bundle,
            bcfg,
            proto_counter[int(bundle["primary_source_prototype_id"])],
            track_counter[int(bundle["primary_source_track_id"])],
            lineage_counter[bundle["primary_source_lineage_id"]],
            hist_topk[int(bundle["bundle_id"])],
        )
        final, consistency, generic_penalty = signature_score(row, cfg)
        row["e33_score"] = final
        row["signature_consistency_score"] = consistency
        row["generic_collision_penalty"] = generic_penalty
        stage1.append(row)
    stage1.sort(key=lambda r: r["base_score"], reverse=True)
    candidate_pool = stage1[: bcfg["candidate_pool_size"]]
    reranked = sorted(candidate_pool, key=lambda r: r["e33_score"], reverse=True)
    topk_for_diversify = [dict(r, final_score=r["e33_score"]) for r in reranked]
    final_topk = e31.diversify_candidates(topk_for_diversify, bcfg)
    target_id = event["target_bundle_id"]
    target_row = next((r for r in candidate_pool if target_id is not None and int(r["bundle_id"]) == int(target_id)), None)
    target_rank = next((i for i, r in enumerate(final_topk, start=1) if target_id is not None and int(r["bundle_id"]) == int(target_id)), None)
    target_candidate_rank = next((i for i, r in enumerate(reranked, start=1) if target_id is not None and int(r["bundle_id"]) == int(target_id)), None)
    return {
        "candidate_pool": candidate_pool,
        "reranked": reranked,
        "final_topk": final_topk,
        "target_row": target_row,
        "target_rank": target_rank,
        "target_candidate_rank": target_candidate_rank,
    }


def collision_reason(target: dict[str, Any] | None, wrong: dict[str, Any] | None) -> str:
    if target is None or wrong is None:
        return "metric_mismatch"
    td, wd = dim_scores(target), dim_scores(wrong)
    gaps = {k: td[k] - wd[k] for k in DIMENSIONS}
    close = [k for k, v in gaps.items() if abs(v) <= 0.03]
    if len(close) >= 5:
        return "multi_cue_collision"
    if np.mean(list(td.values())) < 0.48:
        return "target_signature_underdefined"
    if generic_collision(wrong):
        return "wrong_bundle_overgeneric"
    worst = min(gaps, key=lambda k: gaps[k])
    if gaps[worst] < -0.02:
        return f"{worst}_collision"
    return "ambiguous_event"


def make_pairwise_row(event: dict[str, Any], target: dict[str, Any] | None, wrong: dict[str, Any], rank: int) -> dict[str, Any]:
    td = dim_scores(target or {})
    wd = dim_scores(wrong)
    row: dict[str, Any] = {
        "event_id": event["event_id"],
        "scenario_name": event["scenario_name"],
        "frame_idx": int(event["frame_idx"]),
        "target_bundle_id": "" if event["target_bundle_id"] is None else int(event["target_bundle_id"]),
        "wrong_bundle_id": int(wrong["bundle_id"]),
        "wrong_rank": int(rank),
        "dominant_collision_reason": collision_reason(target, wrong),
    }
    for dim in DIMENSIONS:
        row[f"target_{dim}_score"] = td.get(dim, "")
        row[f"wrong_{dim}_score"] = wd.get(dim, "")
        row[f"cue_gap_{dim}"] = "" if target is None else float(td[dim] - wd[dim])
    row["target_signature_specificity"] = "" if target is None else float(np.mean([td["temporal"], td["disappearance"], td["context"], td["provenance"], td["separation"]]))
    row["wrong_signature_specificity"] = float(np.mean([wd["temporal"], wd["disappearance"], wd["context"], wd["provenance"], wd["separation"]]))
    return row


def classify_failure(target_rank: int | None, top1_hit: int, top3_hit: int, top5_hit: int, target_row: dict[str, Any] | None, wrong_top1: dict[str, Any] | None) -> str:
    if target_rank is None:
        return "target_not_in_top5"
    if top1_hit:
        return ""
    if top3_hit:
        return collision_reason(target_row, wrong_top1)
    if top5_hit:
        return "target_not_in_top3"
    return "target_not_in_top5"


def evaluate_ablation(
    name: str,
    cfg: dict[str, Any],
    bundle_by_id: dict[int, dict[str, Any]],
    event_records: list[dict[str, Any]],
    proto_counter: Counter[int],
    track_counter: Counter[int],
    lineage_counter: Counter[int | None],
    wrong_proto_map: dict[str, int],
) -> dict[str, Any]:
    bcfg = base_cfg()
    # Keep E32b's authoritative A0 semantics: historical top-k is not updated
    # inside the event pass, otherwise A0 collapses back to the older E31 metric
    # family (0.2941 top1 instead of 0.3529).
    hist_topk: Counter[int] = Counter()
    rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    taxonomy_rows: list[dict[str, Any]] = []
    focus_rows: list[dict[str, Any]] = []

    ordered_events = sorted(event_records, key=lambda r: (r["scenario_name"], int(r["frame_idx"]), r["event_id"]))
    for event in ordered_events:
        scored = score_event(event, bundle_by_id, bcfg, cfg, proto_counter, track_counter, lineage_counter, hist_topk)
        final_topk = scored["final_topk"]
        final_ids = [int(r["bundle_id"]) for r in final_topk]
        target_id = event["target_bundle_id"]
        target_row = scored["target_row"]
        target_rank = scored["target_rank"]
        top1 = final_topk[0] if final_topk else None
        top1_hit = int(target_id is not None and len(final_ids) > 0 and int(final_ids[0]) == int(target_id))
        top3_hit = int(target_id is not None and int(target_id) in set(final_ids[:3]))
        top5_hit = int(target_id is not None and int(target_id) in set(final_ids[:5]))
        success = int(top1_hit == 1 and target_row is not None and safe_float(target_row.get("e33_score", target_row.get("final_score"))) >= bcfg["completion_threshold"])
        lost_reason = classify_failure(target_rank, top1_hit, top3_hit, top5_hit, target_row, top1)
        false_retrieval = int(int(event["proposal_detected"]) == 1 and len(final_ids) > 0 and top1_hit == 0)
        row = {
            "ablation_name": name,
            "scenario_name": event["scenario_name"],
            "event_id": event["event_id"],
            "frame_idx": int(event["frame_idx"]),
            "proposal_detected": int(event["proposal_detected"]),
            "target_bundle_id": "" if target_id is None else int(target_id),
            "target_bundle_rank": "" if target_rank is None else int(target_rank),
            "target_candidate_rank": "" if scored["target_candidate_rank"] is None else int(scored["target_candidate_rank"]),
            "target_bundle_retrieved_top1": top1_hit,
            "target_bundle_retrieved_top3": top3_hit,
            "target_bundle_retrieved_top5": top5_hit,
            "pattern_completion_success": success,
            "false_bundle_retrieval": false_retrieval,
            "top1_bundle_id": "" if top1 is None else int(top1["bundle_id"]),
            "top5_bundle_ids": "|".join(str(v) for v in final_ids[:5]),
            "top5_proto_ids": "|".join(str(int(r["primary_source_prototype_id"])) for r in final_topk[:5]),
            "proto0_bundle_count_in_top5": sum(1 for r in final_topk[:5] if int(r["primary_source_prototype_id"]) == 0),
            "target_lost_reason": lost_reason,
            "strict_anchor_visible_top5": top5_hit,
        }
        rows.append(row)
        if target_id is not None and int(event["proposal_detected"]) == 1:
            if top1 is not None and (not top1_hit or event["event_id"] in FOCUS_EVENT_IDS):
                for rank, wrong in enumerate(final_topk[:3], start=1):
                    if int(wrong["bundle_id"]) != int(target_id):
                        pair = make_pairwise_row(event, target_row, wrong, rank)
                        pair["ablation_name"] = name
                        pairwise_rows.append(pair)
            if lost_reason:
                taxonomy_rows.append({
                    "ablation_name": name,
                    "event_id": event["event_id"],
                    "target_bundle_id": int(target_id),
                    "target_rank": "" if target_rank is None else int(target_rank),
                    "target_in_top3": top3_hit,
                    "target_in_top5": top5_hit,
                    "top1_bundle_id": "" if top1 is None else int(top1["bundle_id"]),
                    "failure_reason": lost_reason,
                })
                collision_rows.append({
                    "ablation_name": name,
                    "event_id": event["event_id"],
                    "target_bundle_id": int(target_id),
                    "wrong_top1_bundle_id": "" if top1 is None else int(top1["bundle_id"]),
                    "dominant_collision_reason": lost_reason,
                    "target_signature_underdefined": int(lost_reason == "target_signature_underdefined"),
                    "wrong_bundle_overgeneric": int(lost_reason == "wrong_bundle_overgeneric"),
                })
        if event["event_id"] in FOCUS_EVENT_IDS:
            focus_rows.append({
                "ablation_name": name,
                "event_id": event["event_id"],
                "target_bundle_id": "" if target_id is None else int(target_id),
                "target_bundle_rank": "" if target_rank is None else int(target_rank),
                "target_bundle_retrieved_top1": top1_hit,
                "target_bundle_retrieved_top3": top3_hit,
                "target_bundle_retrieved_top5": top5_hit,
                "pattern_completion_success": success,
                "top1_bundle_id": "" if top1 is None else int(top1["bundle_id"]),
                "target_lost_reason": lost_reason,
            })

    proposal_rows = [r for r in rows if int(r["proposal_detected"]) == 1]
    focus_eval = [r for r in rows if r["event_id"] in FOCUS_EVENT_IDS and int(r["proposal_detected"]) == 1]
    summary = {
        "ablation_name": name,
        "global_top1": 0.0 if not proposal_rows else float(np.mean([int(r["target_bundle_retrieved_top1"]) for r in proposal_rows])),
        "global_top3": 0.0 if not proposal_rows else float(np.mean([int(r["target_bundle_retrieved_top3"]) for r in proposal_rows])),
        "global_top5": 0.0 if not proposal_rows else float(np.mean([int(r["target_bundle_retrieved_top5"]) for r in proposal_rows])),
        "false_bundle_retrieval_rate": 0.0 if not proposal_rows else float(np.mean([int(r["false_bundle_retrieval"]) for r in proposal_rows])),
        "focus_top1_count": int(sum(int(r["target_bundle_retrieved_top1"]) for r in focus_eval)),
        "focus_success_count": int(sum(int(r["pattern_completion_success"]) for r in focus_eval)),
        "regression_event_count": 0,
        "target_in_top3_but_lost_top1_count": int(sum(1 for r in proposal_rows if int(r["target_bundle_retrieved_top3"]) == 1 and int(r["target_bundle_retrieved_top1"]) == 0)),
        "target_not_in_top5_count": int(sum(1 for r in proposal_rows if int(r["target_bundle_retrieved_top5"]) == 0)),
        "signature_collision_count": int(sum(1 for r in collision_rows if str(r["dominant_collision_reason"]).endswith("_collision") or str(r["dominant_collision_reason"]) == "multi_cue_collision")),
        "wrong_bundle_overgeneric_count": int(sum(1 for r in collision_rows if int(r["wrong_bundle_overgeneric"]) == 1)),
        "mean_target_wrong_signature_margin": float(np.mean([safe_float(r.get("cue_gap_separation"), 0.0) for r in pairwise_rows])) if pairwise_rows else 0.0,
        "proto0_top5_share": 0.0 if not proposal_rows else float(np.mean([int(r["proto0_bundle_count_in_top5"]) / 5.0 for r in proposal_rows])),
        "bundle552_top1_count": int(sum(1 for r in proposal_rows if str(r["top1_bundle_id"]) not in ("", None) and int(r["top1_bundle_id"]) == 552)),
        "strict_anchor_real_svr": 0.0 if not proposal_rows else float(np.mean([int(r["strict_anchor_visible_top5"]) for r in proposal_rows])),
        "strict_anchor_shuffled_svr": e32b.compute_shuffled_strict_svr(proposal_rows),
        "wrong_old_prototype_visible_count": e32b.compute_wrong_old_visible_count(proposal_rows, wrong_proto_map),
        "selected_as_best": 0,
    }
    return {
        "summary": summary,
        "retrieval_rows": rows,
        "pairwise_rows": pairwise_rows,
        "collision_rows": collision_rows,
        "taxonomy_rows": taxonomy_rows,
        "focus_rows": focus_rows,
    }


def add_delta_counts(results: dict[str, dict[str, Any]], ablation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_rows = {r["event_id"]: r for r in results[BASELINE_NAME]["retrieval_rows"] if int(r["proposal_detected"]) == 1}
    delta_rows: list[dict[str, Any]] = []
    for summary in ablation_rows:
        name = str(summary["ablation_name"])
        rows = {r["event_id"]: r for r in results[name]["retrieval_rows"] if int(r["proposal_detected"]) == 1}
        improved = regressed = unchanged_failure = unchanged_success = 0
        for event_id, before in baseline_rows.items():
            after = rows.get(event_id)
            if after is None:
                continue
            bsucc = int(before["pattern_completion_success"])
            asucc = int(after["pattern_completion_success"])
            if bsucc == 0 and asucc == 1:
                improved += 1
                klass = "improved"
            elif bsucc == 1 and asucc == 0:
                regressed += 1
                klass = "regressed"
            elif bsucc == 1 and asucc == 1:
                unchanged_success += 1
                klass = "unchanged_success"
            else:
                unchanged_failure += 1
                klass = "unchanged_failure"
            delta_rows.append({
                "ablation_name": name,
                "event_id": event_id,
                "sequence_id": after["scenario_name"],
                "baseline_target_rank": before["target_bundle_rank"],
                "e33_target_rank": after["target_bundle_rank"],
                "baseline_top1_bundle": before["top1_bundle_id"],
                "e33_top1_bundle": after["top1_bundle_id"],
                "baseline_success": bsucc,
                "e33_success": asucc,
                "delta_class": klass,
            })
        summary["improved_event_count"] = improved
        summary["regression_event_count"] = regressed
        summary["unchanged_failure_count"] = unchanged_failure
        summary["unchanged_success_count"] = unchanged_success
    return delta_rows


def select_best(ablation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(r for r in ablation_rows if r["ablation_name"] == BASELINE_NAME)
    eligible = []
    for row in ablation_rows:
        ok = (
            int(row["focus_success_count"]) == 3
            and float(row["global_top1"]) >= float(baseline["global_top1"])
            and float(row["false_bundle_retrieval_rate"]) < float(baseline["false_bundle_retrieval_rate"])
            and int(row["target_in_top3_but_lost_top1_count"]) < int(baseline["target_in_top3_but_lost_top1_count"])
            and int(row["regression_event_count"]) <= 1
        )
        row["eligible_for_best"] = int(ok)
        eligible.append(row) if ok else None
    best = min(eligible, key=lambda r: (r["false_bundle_retrieval_rate"], -r["global_top1"], r["regression_event_count"])) if eligible else baseline
    for row in ablation_rows:
        row["selected_as_best"] = int(row["ablation_name"] == best["ablation_name"])
    return best


def render_report(summary: dict[str, Any]) -> str:
    best = summary["best_ablation"]
    lines = [
        "# Stage E3.3 Report",
        "",
        "## Verdict",
        "",
        summary["human_summary"],
        "",
        "## Best Ablation",
        "",
    ]
    for key in [
        "ablation_name",
        "global_top1",
        "global_top3",
        "global_top5",
        "false_bundle_retrieval_rate",
        "focus_success_count",
        "regression_event_count",
        "target_in_top3_but_lost_top1_count",
        "target_not_in_top5_count",
        "signature_collision_count",
        "wrong_bundle_overgeneric_count",
    ]:
        lines.append(f"- `{key} = {best.get(key)}`")
    lines += ["", "## Focus Events", ""]
    for row in summary["focus_events"]:
        lines.append(
            f"- `{row['event_id']}`: rank={row['target_bundle_rank']}, "
            f"top1={row['target_bundle_retrieved_top1']}, success={row['pattern_completion_success']}, "
            f"reason={row['target_lost_reason']}"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "E3.3 只修改情景 signature / cue consistency 层，不进入 attach、promotion 或最终 identity 决策。",
        "如果本轮仍未通过，下一步应继续修写入侧 signature 或做 event-type-conditioned signature scoring，而不是回到 top3 rerank。",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle_by_id, event_records, _ = e31.collect_runtime_data(args.config, args.event_audit, args.cross_run_alignment, args.seed)
    proto_counter, track_counter, lineage_counter = e31.compute_static_counts(bundle_by_id)
    wrong_proto_map = e32b.wrong_proto_map_from_negative_rows(e31.load_negative_controls(args.e2c_negative_events))

    cfgs = ablation_cfgs()
    if args.audit_only:
        cfgs = {BASELINE_NAME: cfgs[BASELINE_NAME]}

    results: dict[str, dict[str, Any]] = {}
    ablation_rows: list[dict[str, Any]] = []
    all_pairwise: list[dict[str, Any]] = []
    all_collision: list[dict[str, Any]] = []
    all_taxonomy: list[dict[str, Any]] = []
    all_focus: list[dict[str, Any]] = []
    all_retrieval: list[dict[str, Any]] = []

    for name, cfg in cfgs.items():
        res = evaluate_ablation(name, cfg, bundle_by_id, event_records, proto_counter, track_counter, lineage_counter, wrong_proto_map)
        results[name] = res
        ablation_rows.append(res["summary"])
        all_pairwise.extend(res["pairwise_rows"])
        all_collision.extend(res["collision_rows"])
        all_taxonomy.extend(res["taxonomy_rows"])
        all_focus.extend(res["focus_rows"])
        all_retrieval.extend(res["retrieval_rows"])

    delta_rows = add_delta_counts(results, ablation_rows)
    best = select_best(ablation_rows) if not args.audit_only else ablation_rows[0]
    best_focus = [r for r in all_focus if r["ablation_name"] == best["ablation_name"]]
    collision_counts = dict(Counter(str(r["dominant_collision_reason"]) for r in all_pairwise if r["ablation_name"] == BASELINE_NAME))

    passed_minimum = (
        int(best["focus_success_count"]) == 3
        and float(best["global_top1"]) >= 0.3529
        and float(best["false_bundle_retrieval_rate"]) < 0.6471
        and int(best["target_in_top3_but_lost_top1_count"]) < 5
        and int(best["regression_event_count"]) <= 1
    )
    if passed_minimum:
        human = "E3.3 最低通过：signature/cue 修复在保持 focus 3/3 的前提下降低了 false retrieval。"
    elif best["ablation_name"] == BASELINE_NAME:
        human = "E3.3 未通过：增强 signature 后没有任何消融安全超过 E32b A0 baseline，说明需要继续修写入侧 signature 或分事件类型评分。"
    else:
        human = "E3.3 未通过：最佳消融有局部改善，但没有同时满足 false retrieval、top3 lost 和 regression guard。"

    summary = {
        "scope": "track_a_bridge_and_track_c_long_horizon",
        "bundle_count": len(bundle_by_id),
        "proposal_detected_events": int(sum(1 for r in all_retrieval if r["ablation_name"] == BASELINE_NAME and int(r["proposal_detected"]) == 1)),
        "best_ablation": best,
        "passed_minimum": bool(passed_minimum),
        "focus_events": best_focus,
        "baseline_collision_counts": collision_counts,
        "human_summary": human,
    }
    separability_summary = {
        "baseline": next(r for r in ablation_rows if r["ablation_name"] == BASELINE_NAME),
        "collision_counts": collision_counts,
        "target_in_top3_but_lost_top1_events": [
            r for r in all_taxonomy if r["ablation_name"] == BASELINE_NAME and str(r["failure_reason"]) not in ("", "target_not_in_top5", "target_not_in_top3")
        ],
        "human_summary": "Signature audit compares target bundle against wrong top candidates across content/support/motion/context/temporal/disappearance/provenance/separation.",
    }
    compact = {
        "stage": "E3.3",
        "best_ablation": best,
        "passed_minimum": bool(passed_minimum),
        "baseline_collision_counts": collision_counts,
        "focus_events": best_focus,
        "next_recommendation": "E3.3 pass allows later E4 consideration; otherwise continue cue/signature repair, not top3 rerank.",
    }

    e31.write_csv(out / f"stage_E33_pairwise_target_vs_wrong_{args.artifact_version}.csv", all_pairwise)
    e31.write_csv(out / f"stage_E33_signature_collision_audit_{args.artifact_version}.csv", all_collision)
    e31.write_csv(out / f"stage_E33_cue_dimension_ablation_{args.artifact_version}.csv", [
        {
            "dimension": dim,
            "target_beats_wrong_count": sum(1 for r in all_pairwise if r["ablation_name"] == BASELINE_NAME and r.get(f"cue_gap_{dim}") not in ("", None) and safe_float(r.get(f"cue_gap_{dim}")) > 0),
            "mean_target_minus_wrong": float(np.mean([safe_float(r.get(f"cue_gap_{dim}")) for r in all_pairwise if r["ablation_name"] == BASELINE_NAME and r.get(f"cue_gap_{dim}") not in ("", None)])) if all_pairwise else 0.0,
        }
        for dim in DIMENSIONS
    ])
    e31.write_csv(out / f"stage_E33_event_failure_taxonomy_{args.artifact_version}.csv", all_taxonomy)
    e31.write_csv(out / f"stage_E33_ablation_summary_{args.artifact_version}.csv", ablation_rows)
    e31.write_csv(out / f"stage_E33_focus_event_summary_{args.artifact_version}.csv", best_focus)
    (out / f"stage_E33_signature_separability_summary_{args.artifact_version}.json").write_text(json.dumps(separability_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"stage_E33_signature_separability_report_{args.artifact_version}.md").write_text(render_report({"best_ablation": separability_summary["baseline"], "human_summary": separability_summary["human_summary"], "focus_events": [r for r in all_focus if r["ablation_name"] == BASELINE_NAME and r["event_id"] in FOCUS_EVENT_IDS]}), encoding="utf-8")
    (out / f"stage_E33_summary_{args.artifact_version}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"stage_E33_report_{args.artifact_version}.md").write_text(render_report(summary), encoding="utf-8")
    (out / f"stage_E33_compact_for_gpt_{args.artifact_version}.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
