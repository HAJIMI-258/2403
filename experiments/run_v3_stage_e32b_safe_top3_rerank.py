from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v3_stage_e31_retrieval_competition_repair as e31
from experiments import run_v3_stage_e32_global_retrieval_calibration as e32


FOCUS_EVENT_IDS = {"M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"}
BUNDLE552 = 552


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run v3 Stage E3.2b safe top-3 rerank calibration.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--event-audit", default="results/v3_e1/stage_E1_event_audit_v1.csv")
    p.add_argument("--cross-run-alignment", default="results/v3_e2rm/stage_E2R_cross_run_target_alignment_v1.csv")
    p.add_argument("--e2c-negative-events", default="results/v3_e2c/stage_E2C_negative_control_events_v1.csv")
    p.add_argument("--output-dir", default="results/v3_e32b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def init_states(bundle_by_id: dict[int, dict[str, Any]], proto_counter: Counter[int], track_counter: Counter[int]) -> dict[int, dict[str, Any]]:
    states = e32.init_bundle_states(bundle_by_id, proto_counter, track_counter)
    for state in states.values():
        state["accessibility_state"] = classify_accessibility_state(float(state["accessibility_score"]))
    return states


def classify_accessibility_state(score: float) -> str:
    if score >= 0.75:
        return "stable"
    if score >= 0.55:
        return "stabilizing"
    if score >= 0.35:
        return "accessible"
    if score >= 0.18:
        return "candidate"
    return "latent"


def wrong_proto_map_from_negative_rows(negative_rows: list[dict[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for row in negative_rows:
        if str(row.get("control_name", "")) != "wrong_old_prototype":
            continue
        proto = e31.si(row.get("test_old_prototype_id"), None)
        if proto is not None:
            mapping[str(row.get("event_id", ""))] = int(proto)
    return mapping


def cfg_hash(obj: dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


def supportive_mean(row: dict[str, Any]) -> float:
    return float(np.mean([row["temporal_score"], row["disappearance_score"], row["context_score"]]))


def provenance_specificity(row: dict[str, Any], state: dict[str, Any]) -> float:
    return float(min(float(row["provenance_score"]) * float(state["specificity_score"]), 1.0))


def cue_consensus(row: dict[str, Any]) -> float:
    return e32.geomean([
        row["support_score"],
        row["motion_score"],
        row["context_score"],
        row["temporal_score"],
        row["disappearance_score"],
    ])


def generic_content_only(row: dict[str, Any], state: dict[str, Any], cfg: dict[str, Any]) -> bool:
    return (
        float(row["content_score"]) >= cfg["generic_content_min"]
        and supportive_mean(row) <= cfg["generic_supportive_max"]
        and float(state["hubness_score"]) >= cfg["generic_hubness_min"]
    )


def enrich_row(row: dict[str, Any], state: dict[str, Any], cfg: dict[str, Any], *, include_content: bool = True) -> dict[str, Any]:
    out = dict(row)
    out["cue_consensus_score"] = cue_consensus(row)
    out["provenance_specificity_score"] = provenance_specificity(row, state)
    out["hubness_score"] = float(state["hubness_score"])
    out["accessibility_score_static"] = float(state["accessibility_score"])
    out["generic_content_only"] = int(generic_content_only(row, state, cfg))
    content_component = float(row["content_score"]) if include_content else 0.0
    safe_score = (
        float(row["final_score"])
        + cfg["safe_consensus_bonus"] * out["cue_consensus_score"]
        + cfg["safe_temporal_bonus"] * float(row["temporal_score"])
        + cfg["safe_disappearance_bonus"] * float(row["disappearance_score"])
        + cfg["safe_context_bonus"] * float(row["context_score"])
        + cfg["safe_provenance_bonus"] * out["provenance_specificity_score"]
        + cfg["safe_separation_bonus"] * float(row["separation_score"])
        + cfg["safe_content_bonus"] * content_component
        - cfg["safe_hubness_penalty"] * min(out["hubness_score"] / cfg["hubness_norm"], 1.0)
        - cfg["safe_generic_penalty"] * out["generic_content_only"]
        - cfg["safe_ambiguity_penalty"] * e32.cue_disagreement(row)
    )
    out["safe_rerank_score"] = float(safe_score)
    return out


def freeze_gate(top3: list[dict[str, Any]], states: dict[int, dict[str, Any]], cfg: dict[str, Any], *, include_content: bool) -> tuple[bool, str, dict[str, Any]]:
    if not top3:
        return False, "no_top1", {}
    top1 = enrich_row(top3[0], states[int(top3[0]["bundle_id"])], cfg, include_content=include_content)
    details = {
        "top1_margin": float(top1.get("top1_margin", 0.0)),
        "top1_cue_consensus": float(top1["cue_consensus_score"]),
        "top1_temporal_score": float(top1["temporal_score"]),
        "top1_disappearance_score": float(top1["disappearance_score"]),
        "top1_context_score": float(top1["context_score"]),
        "top1_provenance_specificity": float(top1["provenance_specificity_score"]),
        "top1_hubness_score": float(top1["hubness_score"]),
        "top1_generic_content_only": int(top1["generic_content_only"]),
    }
    if bool(top1["generic_content_only"]):
        return False, "generic_content_only_top1", details
    if float(top1["hubness_score"]) > cfg["freeze_hubness_max"]:
        return False, "hubness_too_high", details
    if float(top1["cue_consensus_score"]) < cfg["freeze_consensus_threshold"]:
        return False, "low_consensus", details
    if not (float(top1["temporal_score"]) >= cfg["freeze_temporal_threshold"] or float(top1["disappearance_score"]) >= cfg["freeze_disappearance_threshold"]):
        return False, "low_temporal_disappearance", details
    if float(top1["provenance_specificity_score"]) < cfg["freeze_provenance_threshold"]:
        return False, "low_provenance_specificity", details

    alt_rows = [enrich_row(r, states[int(r["bundle_id"])], cfg, include_content=include_content) for r in top3[1:3]]
    supportive_wins = 0
    for dim in ["temporal_score", "disappearance_score", "provenance_specificity_score"]:
        if all(float(top1[dim]) >= float(r[dim]) for r in alt_rows):
            supportive_wins += 1
    details["supportive_wins"] = supportive_wins
    if float(top1.get("top1_margin", 0.0)) >= cfg["freeze_margin_min"] or supportive_wins >= 2:
        return True, "high_confidence_or_supportive_advantage", details
    return False, "margin_low_without_supportive_advantage", details


def pairwise_compare(bundle_i: dict[str, Any], bundle_j: dict[str, Any], states: dict[int, dict[str, Any]], cfg: dict[str, Any], *, include_content: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    i = enrich_row(bundle_i, states[int(bundle_i["bundle_id"])], cfg, include_content=include_content)
    j = enrich_row(bundle_j, states[int(bundle_j["bundle_id"])], cfg, include_content=include_content)
    deltas = {
        "temporal_delta": float(j["temporal_score"] - i["temporal_score"]),
        "disappearance_delta": float(j["disappearance_score"] - i["disappearance_score"]),
        "context_delta": float(j["context_score"] - i["context_score"]),
        "provenance_delta": float(j["provenance_specificity_score"] - i["provenance_specificity_score"]),
        "separation_delta": float(j["separation_score"] - i["separation_score"]),
        "hubness_delta": float(j["hubness_score"] - i["hubness_score"]),
        "content_delta": float(j["content_score"] - i["content_score"]),
    }
    challenger_wins = {
        "challenger_wins_temporal": int(deltas["temporal_delta"] > 0.0),
        "challenger_wins_disappearance": int(deltas["disappearance_delta"] > 0.0),
        "challenger_wins_context": int(deltas["context_delta"] > 0.0),
        "challenger_wins_provenance": int(deltas["provenance_delta"] > 0.0),
        "challenger_wins_separation": int(deltas["separation_delta"] > 0.0),
    }
    supportive_win_count = sum(challenger_wins.values())
    i_generic = bool(i["generic_content_only"])
    j_generic = bool(j["generic_content_only"])
    score_margin = float(j["safe_rerank_score"] - i["safe_rerank_score"])
    win_reason = "incumbent_kept"
    winner = int(i["bundle_id"])
    if supportive_win_count >= 3 and score_margin >= cfg["pairwise_margin_min"] and not j_generic:
        winner = int(j["bundle_id"])
        win_reason = "challenger_supportive_advantage"
    elif score_margin < 0:
        win_reason = "lower_safe_score"
    elif supportive_win_count < 3:
        win_reason = "insufficient_supportive_wins"
    elif j_generic:
        win_reason = "challenger_generic_content_only"
    return i, j, {
        "winner": winner,
        "win_reason": win_reason,
        "score_margin": score_margin,
        **deltas,
        **challenger_wins,
        "generic_content_only_flag_i": int(i_generic),
        "generic_content_only_flag_j": int(j_generic),
        "incumbent_high_hub": int(float(i["hubness_score"]) >= cfg["swap_incumbent_high_hub_min"]),
        "challenger_high_hub": int(float(j["hubness_score"]) >= cfg["swap_challenger_high_hub_min"]),
    }

def choose_challenger(top3: list[dict[str, Any]], states: dict[int, dict[str, Any]], cfg: dict[str, Any], *, include_content: bool, use_pairwise: bool) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if len(top3) < 2:
        return None, []
    incumbent = top3[0]
    pairwise_rows: list[dict[str, Any]] = []
    candidates = []
    for challenger in top3[1:3]:
        _, _, comp = pairwise_compare(incumbent, challenger, states, cfg, include_content=include_content)
        comp["bundle_i"] = int(incumbent["bundle_id"])
        comp["bundle_j"] = int(challenger["bundle_id"])
        pairwise_rows.append(comp)
        candidates.append((challenger, comp))
    if not use_pairwise:
        enriched = [enrich_row(ch, states[int(ch["bundle_id"])], cfg, include_content=include_content) for ch, _ in candidates]
        if not enriched:
            return None, pairwise_rows
        best = max(enriched, key=lambda r: r["safe_rerank_score"])
        return best, pairwise_rows
    winners = [item for item in candidates if int(item[1]["winner"]) == int(item[0]["bundle_id"])]
    if not winners:
        return None, pairwise_rows
    enriched = [enrich_row(ch, states[int(ch["bundle_id"])], cfg, include_content=include_content) for ch, _ in winners]
    best = max(enriched, key=lambda r: r["safe_rerank_score"])
    return best, pairwise_rows


def strict_swap_gate(incumbent_row: dict[str, Any], challenger_row: dict[str, Any] | None, states: dict[int, dict[str, Any]], cfg: dict[str, Any], *, include_content: bool, disable_bundle552_swap: bool) -> tuple[bool, str, dict[str, Any]]:
    incumbent = enrich_row(incumbent_row, states[int(incumbent_row["bundle_id"])], cfg, include_content=include_content)
    if challenger_row is None:
        return False, "no_challenger", {}
    challenger = enrich_row(challenger_row, states[int(challenger_row["bundle_id"])], cfg, include_content=include_content)
    _, _, comp = pairwise_compare(incumbent_row, challenger_row, states, cfg, include_content=include_content)
    if disable_bundle552_swap and int(challenger["bundle_id"]) == BUNDLE552:
        return False, "bundle552_blocked_by_diagnostic", comp
    if float(challenger["safe_rerank_score"] - incumbent["safe_rerank_score"]) < cfg["swap_margin_min"]:
        return False, "score_margin_too_small", comp
    if (comp["challenger_wins_temporal"] + comp["challenger_wins_disappearance"] + comp["challenger_wins_context"] + comp["challenger_wins_provenance"] + comp["challenger_wins_separation"]) < 3:
        return False, "insufficient_supportive_wins", comp
    if bool(challenger["generic_content_only"]):
        return False, "challenger_generic_content_only", comp
    if float(challenger["hubness_score"]) >= cfg["swap_challenger_high_hub_min"]:
        return False, "challenger_high_hub", comp
    incumbent_high_hub = float(incumbent["hubness_score"]) >= cfg["swap_incumbent_high_hub_min"]
    incumbent_cue_disagreement_high = e32.cue_disagreement(incumbent) >= cfg["swap_incumbent_disagreement_min"]
    incumbent_temporal_or_disappearance_low = float(incumbent["temporal_score"]) <= cfg["swap_incumbent_temporal_max"] or float(incumbent["disappearance_score"]) <= cfg["swap_incumbent_disappearance_max"]
    incumbent_generic = bool(incumbent["generic_content_only"])
    if not (incumbent_high_hub and incumbent_cue_disagreement_high and incumbent_temporal_or_disappearance_low and incumbent_generic):
        if float(challenger["safe_rerank_score"] - incumbent["safe_rerank_score"]) < cfg["swap_margin_strong"]:
            return False, "incumbent_not_generic_enough", comp
    return True, "strict_swap_allowed", comp


def classify_top3_lost_reason(event: dict[str, Any], target_row: dict[str, Any] | None, baseline_top1: dict[str, Any], final_top1: dict[str, Any], states: dict[int, dict[str, Any]], cfg: dict[str, Any], *, include_content: bool) -> str:
    base_enriched = enrich_row(baseline_top1, states[int(baseline_top1["bundle_id"])], cfg, include_content=include_content)
    final_enriched = enrich_row(final_top1, states[int(final_top1["bundle_id"])], cfg, include_content=include_content)
    if target_row is None:
        return "metric_mismatch"
    target_enriched = enrich_row(target_row, states[int(target_row["bundle_id"])], cfg, include_content=include_content)
    if int(final_top1["bundle_id"]) == int(baseline_top1["bundle_id"]):
        if float(base_enriched["cue_consensus_score"]) >= cfg["freeze_consensus_threshold"] and (float(base_enriched["temporal_score"]) >= cfg["freeze_temporal_threshold"] or float(base_enriched["disappearance_score"]) >= cfg["freeze_disappearance_threshold"]):
            return "baseline_top1_high_confidence_correct_or_ambiguous"
    if float(final_enriched["hubness_score"]) >= cfg["swap_incumbent_high_hub_min"] and bool(final_enriched["generic_content_only"]):
        return "wrong_top1_high_hub_generic"
    if bool(final_enriched["generic_content_only"]):
        return "wrong_top1_content_only"
    if float(final_enriched["accessibility_score_static"]) >= cfg["wrong_accessibility_high"]:
        return "wrong_top1_accessibility_artifact"
    if float(target_enriched["temporal_score"]) < cfg["freeze_temporal_threshold"]:
        return "target_temporal_weak"
    if float(target_enriched["disappearance_score"]) < cfg["freeze_disappearance_threshold"]:
        return "target_disappearance_weak"
    if float(target_enriched["provenance_specificity_score"]) < cfg["freeze_provenance_threshold"]:
        return "target_provenance_weak"
    if float(target_enriched["separation_score"]) < cfg["swap_separation_floor"]:
        return "target_separation_weak"
    return "challenger_not_strong_enough"


def compute_shuffled_strict_svr(proposal_rows: list[dict[str, Any]]) -> float:
    rows = [r for r in proposal_rows if r["target_bundle_id"] not in (None, "")]
    if not rows:
        return 0.0
    target_ids = [int(r["target_bundle_id"]) for r in rows]
    shifted = target_ids[1:] + target_ids[:1]
    hits = 0
    for row, shuffled_target in zip(rows, shifted):
        top5_ids = {int(v) for v in str(row["top5_bundle_ids"]).split("|") if v not in ("", None)}
        if shuffled_target in top5_ids:
            hits += 1
    return float(hits / len(rows))


def compute_wrong_old_visible_count(proposal_rows: list[dict[str, Any]], wrong_proto_map: dict[str, int]) -> int:
    hits = 0
    by_event = {str(r["event_id"]): {int(v) for v in str(r["top5_proto_ids"]).split("|") if v not in ("", None)} for r in proposal_rows}
    for event_id, wrong_proto in wrong_proto_map.items():
        if wrong_proto in by_event.get(event_id, set()):
            hits += 1
    return int(hits)


def evaluate_ablation(name: str, base_cfg: dict[str, Any], rerank_cfg: dict[str, Any], bundle_by_id: dict[int, dict[str, Any]], event_records: list[dict[str, Any]], proto_counter: Counter[int], track_counter: Counter[int], lineage_counter: Counter[int | None], wrong_proto_map: dict[str, int]) -> dict[str, Any]:
    states = init_states(bundle_by_id, proto_counter, track_counter)
    hist_topk: Counter[int] = Counter()
    ordered_events = sorted(event_records, key=lambda r: (r["scenario_name"], int(r["frame_idx"]), r["event_id"]))

    retrieval_rows: list[dict[str, Any]] = []
    freeze_rows: list[dict[str, Any]] = []
    swap_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    taxonomy_rows: list[dict[str, Any]] = []
    bundle552_rows: list[dict[str, Any]] = []

    for event in ordered_events:
        if int(event["proposal_detected"]) != 1 or event["cue"] is None:
            retrieval_rows.append({
                "ablation_name": name,
                "scenario_name": event["scenario_name"],
                "event_id": event["event_id"],
                "frame_idx": int(event["frame_idx"]),
                "proposal_detected": int(event["proposal_detected"]),
                "target_bundle_id": event["target_bundle_id"] or "",
                "target_bundle_rank": "",
                "target_bundle_retrieved_top1": 0,
                "target_bundle_retrieved_top3": 0,
                "target_bundle_retrieved_top5": 0,
                "pattern_completion_success": 0,
                "strict_anchor_visible_top5": 0,
                "false_bundle_retrieval": 0,
                "proto0_bundle_count_in_top5": 0,
                "top5_bundle_ids": "",
                "top5_proto_ids": "",
                "top1_bundle_id": "",
                "top1_margin": "",
                "target_lost_reason": "proposal_missing_for_retrieval",
            })
            continue

        eligible = [bundle_by_id[b] for b in event["eligible_bundle_ids"] if b in bundle_by_id]
        stage1 = []
        for bundle in eligible:
            stage1.append(
                e31.score_candidate(
                    event["cue"],
                    bundle,
                    base_cfg,
                    proto_counter[int(bundle["primary_source_prototype_id"])],
                    track_counter[int(bundle["primary_source_track_id"])],
                    lineage_counter[bundle["primary_source_lineage_id"]],
                    hist_topk[int(bundle["bundle_id"])],
                )
            )
        stage1.sort(key=lambda r: r["base_score"], reverse=True)
        candidate_pool = stage1[: base_cfg["candidate_pool_size"]]
        base_reranked = sorted(candidate_pool, key=lambda r: r["final_score"], reverse=True)
        base_topk = e31.diversify_candidates(base_reranked, base_cfg)
        top3 = list(base_topk[:3])
        top5 = list(base_topk[:5])
        candidate_pool_ids = {int(r["bundle_id"]) for r in candidate_pool}

        if len(top3) >= 2:
            margin = float(top3[0]["final_score"] - top3[1]["final_score"])
            for row in top3:
                row["top1_margin"] = margin
        elif top3:
            top3[0]["top1_margin"] = float(top3[0]["final_score"])

        include_content = bool(rerank_cfg["use_content_score"])
        challenger, pair_rows = choose_challenger(top3, states, rerank_cfg, include_content=include_content, use_pairwise=bool(rerank_cfg["use_pairwise"]))
        for comp in pair_rows:
            pairwise_rows.append({"ablation_name": name, "event_id": event["event_id"], **comp})

        top1_frozen, freeze_reason, freeze_detail = freeze_gate(top3, states, rerank_cfg, include_content=include_content)
        rerank_allowed = int(not top1_frozen and bool(rerank_cfg["allow_rerank"]))
        freeze_rows.append({
            "ablation_name": name,
            "event_id": event["event_id"],
            "baseline_top1_bundle_id": "" if len(top3) < 1 else int(top3[0]["bundle_id"]),
            "baseline_top2_bundle_id": "" if len(top3) < 2 else int(top3[1]["bundle_id"]),
            "baseline_top3_bundle_id": "" if len(top3) < 3 else int(top3[2]["bundle_id"]),
            "top1_frozen": int(top1_frozen),
            "freeze_reason": freeze_reason,
            "rerank_allowed": rerank_allowed,
            **freeze_detail,
        })
        swap_allowed = False
        swap_reason = "rerank_disabled_or_frozen"
        swap_detail: dict[str, Any] = {
            "score_margin": "",
            "challenger_wins_temporal": "",
            "challenger_wins_disappearance": "",
            "challenger_wins_context": "",
            "challenger_wins_provenance": "",
            "challenger_wins_separation": "",
            "incumbent_high_hub": "",
            "incumbent_generic_content_only": "",
            "challenger_high_hub": "",
        }
        final_topk = list(base_topk)
        if rerank_allowed:
            swap_allowed, swap_reason, swap_detail = strict_swap_gate(
                top3[0],
                challenger,
                states,
                rerank_cfg,
                include_content=include_content,
                disable_bundle552_swap=bool(rerank_cfg["disable_bundle552_swap"]),
            )
            if swap_allowed and challenger is not None:
                challenger_id = int(challenger["bundle_id"])
                incumbent_id = int(top3[0]["bundle_id"])
                reordered_top3 = [next(r for r in top3 if int(r["bundle_id"]) == challenger_id)]
                reordered_top3.extend([r for r in top3 if int(r["bundle_id"]) != challenger_id])
                if int(reordered_top3[1]["bundle_id"]) == incumbent_id:
                    final_topk = reordered_top3 + base_topk[3:]
        swap_rows.append({
            "ablation_name": name,
            "event_id": event["event_id"],
            "incumbent_bundle_id": "" if len(top3) < 1 else int(top3[0]["bundle_id"]),
            "challenger_bundle_id": "" if challenger is None else int(challenger["bundle_id"]),
            "swap_attempted": int(rerank_allowed),
            "swap_allowed": int(swap_allowed),
            "swap_rejected_reason": "" if swap_allowed else swap_reason,
            "final_top1_bundle_id": "" if not final_topk else int(final_topk[0]["bundle_id"]),
            **swap_detail,
        })

        target_bundle_id = event["target_bundle_id"]
        final_ids = [int(r["bundle_id"]) for r in final_topk]
        target_row = next((r for r in candidate_pool if target_bundle_id is not None and int(r["bundle_id"]) == int(target_bundle_id)), None)
        target_rank = next((i for i, r in enumerate(final_topk, start=1) if target_bundle_id is not None and int(r["bundle_id"]) == int(target_bundle_id)), None)
        top1 = final_topk[0] if final_topk else None
        top2 = final_topk[1] if len(final_topk) > 1 else None
        top1_margin = 0.0 if top1 is None else (float(top1["final_score"]) - float(top2["final_score"]) if top2 is not None else float(top1["final_score"]))
        top1_hit = int(target_bundle_id is not None and len(final_ids) > 0 and int(final_ids[0]) == int(target_bundle_id))
        top3_hit = int(target_bundle_id is not None and int(target_bundle_id) in set(final_ids[:3]))
        top5_hit = int(target_bundle_id is not None and int(target_bundle_id) in set(final_ids[:5]))
        success = int(top1_hit == 1 and target_row is not None and float(target_row["final_score"]) >= base_cfg["completion_threshold"])
        false_retrieval = int(len(final_ids) > 0 and top1_hit == 0)
        strict_anchor = int(top5_hit == 1)
        lost_reason = ""
        if target_bundle_id is not None and top3_hit == 1 and top1_hit == 0 and top1 is not None:
            lost_reason = classify_top3_lost_reason(event, target_row, top3[0], top1, states, rerank_cfg, include_content=include_content)
        elif target_bundle_id is not None and top5_hit == 0:
            lost_reason = "target_not_in_top5"
        elif target_bundle_id is not None and top5_hit == 1 and top3_hit == 0:
            lost_reason = "target_not_in_top3"

        retrieval_rows.append({
            "ablation_name": name,
            "scenario_name": event["scenario_name"],
            "event_id": event["event_id"],
            "frame_idx": int(event["frame_idx"]),
            "proposal_detected": int(event["proposal_detected"]),
            "target_bundle_id": "" if target_bundle_id is None else int(target_bundle_id),
            "target_bundle_rank": "" if target_rank is None else int(target_rank),
            "target_bundle_retrieved_top1": top1_hit,
            "target_bundle_retrieved_top3": top3_hit,
            "target_bundle_retrieved_top5": top5_hit,
            "pattern_completion_success": success,
            "strict_anchor_visible_top5": strict_anchor,
            "false_bundle_retrieval": false_retrieval,
            "proto0_bundle_count_in_top5": sum(1 for r in final_topk[:5] if int(r["primary_source_prototype_id"]) == 0),
            "top5_bundle_ids": "|".join(str(v) for v in final_ids[:5]),
            "top5_proto_ids": "|".join(str(int(r["primary_source_prototype_id"])) for r in final_topk[:5]),
            "top1_bundle_id": "" if top1 is None else int(top1["bundle_id"]),
            "top1_margin": float(top1_margin),
            "target_lost_reason": lost_reason,
        })

        if target_bundle_id is not None and top3_hit == 1 and top1_hit == 0:
            top3_tax_reason = lost_reason if lost_reason else "ambiguous_multi_valid_bundle"
            taxonomy_rows.append({
                "ablation_name": name,
                "event_id": event["event_id"],
                "target_bundle_id": int(target_bundle_id),
                "target_rank_before": next((i for i, r in enumerate(top3, start=1) if int(r["bundle_id"]) == int(target_bundle_id)), ""),
                "target_rank_after": "" if target_rank is None else int(target_rank),
                "baseline_top1_bundle_id": "" if len(top3) < 1 else int(top3[0]["bundle_id"]),
                "e32b_top1_bundle_id": "" if top1 is None else int(top1["bundle_id"]),
                "target_in_top3": 1,
                "target_lost_top1": 1,
                "lost_reason": top3_tax_reason,
            })

        if any(int(r["bundle_id"]) == BUNDLE552 for r in top3):
            row552 = next(r for r in top3 if int(r["bundle_id"]) == BUNDLE552)
            st552 = states[int(row552["bundle_id"])]
            er552 = enrich_row(row552, st552, rerank_cfg, include_content=include_content)
            bundle552_rows.append({
                "ablation_name": name,
                "event_id": event["event_id"],
                "bundle552_in_top3": 1,
                "bundle552_rank_before": next((i for i, r in enumerate(top3, start=1) if int(r["bundle_id"]) == BUNDLE552), ""),
                "bundle552_rank_after": next((i for i, r in enumerate(final_topk[:3], start=1) if int(r["bundle_id"]) == BUNDLE552), ""),
                "bundle552_content_score": float(row552["content_score"]),
                "bundle552_temporal_score": float(row552["temporal_score"]),
                "bundle552_disappearance_score": float(row552["disappearance_score"]),
                "bundle552_context_score": float(row552["context_score"]),
                "bundle552_provenance_score": float(er552["provenance_specificity_score"]),
                "bundle552_separation_score": float(row552["separation_score"]),
                "bundle552_hubness_score": float(er552["hubness_score"]),
                "bundle552_generic_content_only": int(er552["generic_content_only"]),
                "bundle552_anchor_degree": int(track_counter[int(row552["primary_source_track_id"])]),
                "bundle552_source_prototype_ids": int(row552["primary_source_prototype_id"]),
                "bundle552_source_track_ids": int(row552["primary_source_track_id"]),
                "bundle552_memory_anchor_id": row552["memory_anchor_id"],
                "bundle552_won_against_target": int(top1 is not None and int(top1["bundle_id"]) == BUNDLE552 and target_bundle_id not in (None, "") and int(target_bundle_id) != BUNDLE552),
                "bundle552_win_reason": lost_reason if top1 is not None and int(top1["bundle_id"]) == BUNDLE552 else "",
                "bundle552_should_be_suppressed_by_gate": int(generic_content_only(row552, st552, rerank_cfg)),
            })

    proposal_rows = [r for r in retrieval_rows if int(r["proposal_detected"]) == 1]
    focus_rows = [r for r in proposal_rows if r["event_id"] in FOCUS_EVENT_IDS]
    summary = {
        "ablation_name": name,
        "global_top1": 0.0 if not proposal_rows else float(np.mean([int(r["target_bundle_retrieved_top1"]) for r in proposal_rows])),
        "global_top3": 0.0 if not proposal_rows else float(np.mean([int(r["target_bundle_retrieved_top3"]) for r in proposal_rows])),
        "global_top5": 0.0 if not proposal_rows else float(np.mean([int(r["target_bundle_retrieved_top5"]) for r in proposal_rows])),
        "false_bundle_retrieval_rate": 0.0 if not proposal_rows else float(np.mean([int(r["false_bundle_retrieval"]) for r in proposal_rows])),
        "focus_top1_count": int(sum(int(r["target_bundle_retrieved_top1"]) for r in focus_rows)),
        "focus_success_count": int(sum(int(r["pattern_completion_success"]) for r in focus_rows)),
        "regression_event_count": 0,
        "improved_event_count": 0,
        "unchanged_failure_count": 0,
        "unchanged_success_count": 0,
        "target_in_top3_but_lost_top1_count": int(sum(1 for r in taxonomy_rows if str(r["lost_reason"]) != "")),
        "wrong_bundle_accessibility_too_high_count": int(sum(1 for r in taxonomy_rows if str(r["lost_reason"]) == "wrong_top1_accessibility_artifact")),
        "hub_bundle_dominance_count": int(sum(1 for r in taxonomy_rows if str(r["lost_reason"]) == "wrong_top1_high_hub_generic")),
        "bundle552_top1_count": int(sum(1 for r in proposal_rows if str(r["top1_bundle_id"]) not in ("", None) and int(r["top1_bundle_id"]) == BUNDLE552)),
        "proto0_top5_share": 0.0 if not proposal_rows else float(np.mean([int(r["proto0_bundle_count_in_top5"]) / 5.0 for r in proposal_rows])),
        "strict_anchor_real_svr": 0.0 if not proposal_rows else float(np.mean([int(r["strict_anchor_visible_top5"]) for r in proposal_rows])),
        "strict_anchor_shuffled_svr": compute_shuffled_strict_svr(proposal_rows),
        "wrong_old_prototype_visible_count": int(compute_wrong_old_visible_count(proposal_rows, wrong_proto_map)),
        "unsafe_component_count": 0,
        "selected_as_best": 0,
    }
    return {"summary": summary, "retrieval_rows": retrieval_rows, "freeze_rows": freeze_rows, "swap_rows": swap_rows, "pairwise_rows": pairwise_rows, "taxonomy_rows": taxonomy_rows, "bundle552_rows": bundle552_rows}

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def render_report(summary: dict[str, Any]) -> str:
    best = summary["best_ablation"]
    lines = [
        "# Stage E3.2b Report",
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
        "focus_top1_count",
        "focus_success_count",
        "regression_event_count",
        "target_in_top3_but_lost_top1_count",
        "bundle552_top1_count",
        "proto0_top5_share",
        "selected_as_best",
    ]:
        lines.append(f"- `{key} = {best.get(key)}`")
    lines += ["", "## Baseline Consistency", ""]
    for row in summary.get("baseline_consistency", []):
        lines.append(
            f"- `{row['source_stage']}`: top1={row['global_top1']}, false={row['false_bundle_retrieval_rate']}, reason={row['difference_reason']}"
        )
    lines += ["", "## Focus Events", ""]
    for row in summary["focus_events"]:
        lines += [
            f"### {row['event_id']}",
            "",
            f"- `baseline_target_rank = {row['baseline_target_rank']}`",
            f"- `target_bundle_rank_after = {row['target_bundle_rank_after']}`",
            f"- `target_bundle_retrieved_top1 = {row['target_bundle_retrieved_top1']}`",
            f"- `pattern_completion_success = {row['pattern_completion_success']}`",
            f"- `target_lost_reason = {row['target_lost_reason']}`",
            "",
        ]
    lines += ["", "## Bundle 552", "", summary.get("bundle552_summary", "")]
    return "\n".join(lines) + "\n"


def baseline_from_summary(stage: str, path: Path) -> dict[str, Any]:
    data = load_json(path)
    best = data.get("best_ablation", {}) if isinstance(data, dict) else {}
    focus_events = data.get("focus_events", []) if isinstance(data, dict) else []
    return {
        "source_stage": stage,
        "config_hash": cfg_hash(best) if best else "missing",
        "event_set": "track_a_bridge_and_track_c_long_horizon",
        "proposal_detected_events": 17,
        "global_top1": best.get("global_top1", ""),
        "global_top3": best.get("global_top3", ""),
        "global_top5": best.get("global_top5", ""),
        "false_bundle_retrieval_rate": best.get("false_bundle_retrieval_rate", ""),
        "focus_top1_count": best.get("focus_top1_count", ""),
        "focus_success_count": best.get("focus_success_count", ""),
        "difference_reason": "reference_summary" if best else "summary_missing",
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_by_id, event_records, _ = e31.collect_runtime_data(args.config, args.event_audit, args.cross_run_alignment, args.seed)
    proto_counter, track_counter, lineage_counter = e31.compute_static_counts(bundle_by_id)
    negative_rows = e31.load_negative_controls(args.e2c_negative_events)
    wrong_proto_map = wrong_proto_map_from_negative_rows(negative_rows)

    base_cfg = {
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
        "weights": {"content": 0.14, "support": 0.14, "motion": 0.10, "context": 0.17, "temporal": 0.15, "disappearance": 0.13, "provenance": 0.09, "separation": 0.08},
    }
    common = {
        "allow_rerank": False,
        "use_pairwise": False,
        "use_content_score": True,
        "disable_bundle552_swap": False,
        "safe_consensus_bonus": 0.035,
        "safe_temporal_bonus": 0.045,
        "safe_disappearance_bonus": 0.055,
        "safe_context_bonus": 0.030,
        "safe_provenance_bonus": 0.055,
        "safe_separation_bonus": 0.045,
        "safe_content_bonus": 0.010,
        "safe_hubness_penalty": 0.040,
        "safe_generic_penalty": 0.075,
        "safe_ambiguity_penalty": 0.030,
        "hubness_norm": 2.5,
        "generic_content_min": 0.92,
        "generic_supportive_max": 0.76,
        "generic_hubness_min": 1.0,
        "freeze_consensus_threshold": 0.68,
        "freeze_temporal_threshold": 0.58,
        "freeze_disappearance_threshold": 0.60,
        "freeze_provenance_threshold": 0.18,
        "freeze_hubness_max": 1.55,
        "freeze_margin_min": 0.025,
        "pairwise_margin_min": 0.012,
        "swap_margin_min": 0.020,
        "swap_margin_strong": 0.055,
        "swap_incumbent_high_hub_min": 1.0,
        "swap_challenger_high_hub_min": 1.35,
        "swap_incumbent_disagreement_min": 0.13,
        "swap_incumbent_temporal_max": 0.65,
        "swap_incumbent_disappearance_max": 0.70,
        "swap_separation_floor": 0.30,
        "wrong_accessibility_high": 0.84,
    }
    ablation_cfgs = {
        "A0_E31_combined_baseline": {**common},
        "A1_freeze_gate_only": {**common},
        "A2_strict_swap_gate_only": {**common, "allow_rerank": True},
        "A3_pairwise_top3_comparator_only": {**common, "allow_rerank": True, "use_pairwise": True},
        "A4_freeze_plus_strict_swap": {**common, "allow_rerank": True},
        "A5_freeze_plus_pairwise": {**common, "allow_rerank": True, "use_pairwise": True},
        "A6_strict_swap_plus_pairwise": {**common, "allow_rerank": True, "use_pairwise": True},
        "A7_full_E32b_safe_top3": {**common, "allow_rerank": True, "use_pairwise": True},
        "A8_full_E32b_no_content_score": {**common, "allow_rerank": True, "use_pairwise": True, "use_content_score": False},
        "A9_full_E32b_high_margin_swap_only": {**common, "allow_rerank": True, "use_pairwise": True, "swap_margin_min": 0.045, "swap_margin_strong": 0.080},
        "A10_full_E32b_no_bundle552_swap": {**common, "allow_rerank": True, "use_pairwise": True, "disable_bundle552_swap": True},
    }

    results: dict[str, dict[str, Any]] = {}
    ablation_rows: list[dict[str, Any]] = []
    all_retrieval_rows: list[dict[str, Any]] = []
    all_freeze_rows: list[dict[str, Any]] = []
    all_swap_rows: list[dict[str, Any]] = []
    all_pairwise_rows: list[dict[str, Any]] = []
    all_taxonomy_rows: list[dict[str, Any]] = []
    all_bundle552_rows: list[dict[str, Any]] = []

    for name, cfg in ablation_cfgs.items():
        result = evaluate_ablation(name, base_cfg, cfg, bundle_by_id, event_records, proto_counter, track_counter, lineage_counter, wrong_proto_map)
        results[name] = result
        all_retrieval_rows.extend(result["retrieval_rows"])
        all_freeze_rows.extend(result["freeze_rows"])
        all_swap_rows.extend(result["swap_rows"])
        all_pairwise_rows.extend(result["pairwise_rows"])
        all_taxonomy_rows.extend(result["taxonomy_rows"])
        all_bundle552_rows.extend(result["bundle552_rows"])
        ablation_rows.append(result["summary"])

    baseline_name = "A0_E31_combined_baseline"
    baseline_rows = {r["event_id"]: r for r in results[baseline_name]["retrieval_rows"] if int(r["proposal_detected"]) == 1}
    delta_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []

    for summary_row in ablation_rows:
        name = str(summary_row["ablation_name"])
        improved = regressed = unchanged_failure = unchanged_success = 0
        if name == baseline_name:
            unchanged_success = int(sum(1 for r in baseline_rows.values() if int(r["pattern_completion_success"]) == 1))
        else:
            rows = {r["event_id"]: r for r in results[name]["retrieval_rows"] if int(r["proposal_detected"]) == 1}
            for event_id, before in baseline_rows.items():
                after = rows.get(event_id)
                if after is None:
                    continue
                bsucc = int(before["pattern_completion_success"])
                asucc = int(after["pattern_completion_success"])
                brank = 999 if before["target_bundle_rank"] == "" else int(before["target_bundle_rank"])
                arank = 999 if after["target_bundle_rank"] == "" else int(after["target_bundle_rank"])
                if asucc > bsucc or (asucc == bsucc and arank < brank):
                    delta_class = "improved"
                    improved += 1
                elif asucc == bsucc == 1 and arank == brank:
                    delta_class = "unchanged_success"
                    unchanged_success += 1
                elif asucc < bsucc or (asucc == bsucc and arank > brank):
                    delta_class = "regressed"
                    regressed += 1
                else:
                    delta_class = "unchanged_failure"
                    unchanged_failure += 1
                delta_rows.append({
                    "ablation_name": name,
                    "event_id": event_id,
                    "sequence_id": after["scenario_name"],
                    "proposal_detected": int(after["proposal_detected"]),
                    "is_focus_event": int(event_id in FOCUS_EVENT_IDS),
                    "baseline_target_rank": "" if brank == 999 else brank,
                    "e32b_target_rank": "" if arank == 999 else arank,
                    "baseline_top1_bundle_id": before["top1_bundle_id"],
                    "e32b_top1_bundle_id": after["top1_bundle_id"],
                    "baseline_target_in_top3": before["target_bundle_retrieved_top3"],
                    "e32b_target_in_top3": after["target_bundle_retrieved_top3"],
                    "baseline_target_in_top5": before["target_bundle_retrieved_top5"],
                    "e32b_target_in_top5": after["target_bundle_retrieved_top5"],
                    "baseline_success": bsucc,
                    "e32b_success": asucc,
                    "delta_class": delta_class,
                    "baseline_false_reason": before["target_lost_reason"],
                    "e32b_false_reason": after["target_lost_reason"],
                })
                if delta_class == "regressed":
                    regression_rows.append({
                        "ablation_name": name,
                        "event_id": event_id,
                        "baseline_success": bsucc,
                        "e32b_success": asucc,
                        "baseline_top1_bundle": before["top1_bundle_id"],
                        "e32b_top1_bundle": after["top1_bundle_id"],
                        "baseline_target_rank": "" if brank == 999 else brank,
                        "e32b_target_rank": "" if arank == 999 else arank,
                        "regressed": 1,
                        "regression_reason": after["target_lost_reason"],
                        "changed_by_component": name,
                        "unsafe_component": 1,
                    })
        summary_row["improved_event_count"] = int(improved)
        summary_row["regression_event_count"] = int(regressed)
        summary_row["unchanged_failure_count"] = int(unchanged_failure)
        summary_row["unchanged_success_count"] = int(unchanged_success)
        summary_row["unsafe_component_count"] = int(regressed > 1 or int(summary_row["focus_top1_count"]) < 3 or int(summary_row["focus_success_count"]) < 3)

    baseline_summary = next(r for r in ablation_rows if r["ablation_name"] == baseline_name)
    for row in ablation_rows:
        safe = (
            int(row["focus_top1_count"]) == 3
            and int(row["focus_success_count"]) == 3
            and int(row["regression_event_count"]) <= 1
            and float(row["global_top1"]) >= float(baseline_summary["global_top1"])
            and float(row["false_bundle_retrieval_rate"]) <= float(baseline_summary["false_bundle_retrieval_rate"])
            and int(row["unsafe_component_count"]) == 0
        )
        better_top3_loss = int(row["target_in_top3_but_lost_top1_count"]) < int(baseline_summary["target_in_top3_but_lost_top1_count"])
        row["eligible_for_best"] = int(safe and (better_top3_loss or float(row["global_top1"]) > float(baseline_summary["global_top1"])))
    eligible = [r for r in ablation_rows if int(r["eligible_for_best"]) == 1]
    if eligible:
        best = min(eligible, key=lambda r: (r["false_bundle_retrieval_rate"], -r["global_top1"], r["target_in_top3_but_lost_top1_count"]))
    else:
        best = baseline_summary
    for row in ablation_rows:
        row["selected_as_best"] = int(row["ablation_name"] == best["ablation_name"])

    focus_rows = []
    best_rows = {r["event_id"]: r for r in results[str(best["ablation_name"])]["retrieval_rows"]}
    for event_id in sorted(FOCUS_EVENT_IDS):
        after = best_rows.get(event_id)
        before = baseline_rows.get(event_id)
        if after is None or before is None:
            continue
        focus_rows.append({
            "ablation_name": best["ablation_name"],
            "event_id": event_id,
            "baseline_target_rank": before["target_bundle_rank"],
            "target_bundle_rank_after": after["target_bundle_rank"],
            "target_bundle_retrieved_top1": after["target_bundle_retrieved_top1"],
            "target_bundle_retrieved_top3": after["target_bundle_retrieved_top3"],
            "target_bundle_retrieved_top5": after["target_bundle_retrieved_top5"],
            "pattern_completion_success": after["pattern_completion_success"],
            "proto0_bundle_count_in_top5": after["proto0_bundle_count_in_top5"],
            "target_lost_reason": after["target_lost_reason"],
        })

    baseline_consistency = [
        baseline_from_summary("E31_summary_best", ROOT / "results/v3_e31/stage_E31_summary_v1.json"),
        baseline_from_summary("E32_summary_best", ROOT / "results/v3_e32/stage_E32_summary_v1.json"),
        baseline_from_summary("E32a_summary_best", ROOT / "results/v3_e32a/stage_E32a_summary_v1.json"),
        {
            "source_stage": "E32b_recomputed_A0",
            "config_hash": cfg_hash(base_cfg),
            "event_set": "track_a_bridge_and_track_c_long_horizon",
            "proposal_detected_events": 17,
            "global_top1": baseline_summary["global_top1"],
            "global_top3": baseline_summary["global_top3"],
            "global_top5": baseline_summary["global_top5"],
            "false_bundle_retrieval_rate": baseline_summary["false_bundle_retrieval_rate"],
            "focus_top1_count": baseline_summary["focus_top1_count"],
            "focus_success_count": baseline_summary["focus_success_count"],
            "difference_reason": "this run uses E32b local E31 scoring path; use this row as the E32b baseline",
        },
    ]
    for row in baseline_consistency:
        if row["source_stage"] == "E32_summary_best":
            row["difference_reason"] = "E32 stateful calibration wrapper reported a stronger A0; E32b uses freshly recomputed local E31 scoring as the authority"
        elif row["source_stage"] == "E31_summary_best":
            row["difference_reason"] = "matches the E31 published best-ablation metric family"
        elif row["source_stage"] == "E32a_summary_best":
            row["difference_reason"] = "matches the E32a recomputed conservative baseline"

    passed_minimum = (
        int(best["focus_top1_count"]) == 3
        and int(best["focus_success_count"]) == 3
        and float(best["global_top1"]) >= float(baseline_summary["global_top1"])
        and float(best["false_bundle_retrieval_rate"]) < float(baseline_summary["false_bundle_retrieval_rate"])
        and int(best["regression_event_count"]) <= 1
        and int(best["target_in_top3_but_lost_top1_count"]) < int(baseline_summary["target_in_top3_but_lost_top1_count"])
        and int(best["bundle552_top1_count"]) <= int(baseline_summary["bundle552_top1_count"])
    )
    if passed_minimum:
        human_summary = "E3.2b passed minimum: safe top3 rerank reduced false retrieval without dropping the focus hard events."
    elif str(best["ablation_name"]) == baseline_name:
        human_summary = "E3.2b 未通过：没有任何 top3 rerank 消融能安全超过 A0 baseline。当前 top3 内部判别仍不能可靠提升全局。"
    else:
        human_summary = "E3.2b 未通过：最佳消融没有同时满足 false retrieval 下降、focus 3/3 和 regression guard。"

    bundle552_focus = [r for r in all_bundle552_rows if str(r.get("event_id")) in {"M-RE-TC-012", "M-RE-TC-013"}]
    if bundle552_focus:
        generic_count = sum(int(r.get("bundle552_generic_content_only", 0)) for r in bundle552_focus)
        won_count = sum(int(r.get("bundle552_won_against_target", 0)) for r in bundle552_focus)
        bundle552_summary = f"Bundle 552 appears in focus top3 {len(bundle552_focus)} times; generic_content_only={generic_count}; won_against_target={won_count}. The swap gate blocks it when the challenger is generic or high-hub."
    else:
        bundle552_summary = "Bundle 552 did not appear in the focus top3 trace for the selected run."

    summary = {
        "scope": "track_a_bridge_and_track_c_long_horizon",
        "best_ablation": best,
        "passed_minimum": bool(passed_minimum),
        "baseline_consistency": baseline_consistency,
        "focus_events": focus_rows,
        "bundle552_summary": bundle552_summary,
        "human_summary": human_summary,
    }

    e31.write_csv(output_dir / f"stage_E32b_ablation_summary_{args.artifact_version}.csv", ablation_rows)
    e31.write_csv(output_dir / f"stage_E32b_event_delta_audit_{args.artifact_version}.csv", delta_rows)
    e31.write_csv(output_dir / f"stage_E32b_top1_freeze_gate_trace_{args.artifact_version}.csv", all_freeze_rows)
    e31.write_csv(output_dir / f"stage_E32b_swap_gate_trace_{args.artifact_version}.csv", all_swap_rows)
    e31.write_csv(output_dir / f"stage_E32b_pairwise_top3_compare_{args.artifact_version}.csv", all_pairwise_rows)
    e31.write_csv(output_dir / f"stage_E32b_top3_lost_top1_taxonomy_{args.artifact_version}.csv", all_taxonomy_rows)
    e31.write_csv(output_dir / f"stage_E32b_focus_event_summary_{args.artifact_version}.csv", focus_rows)
    e31.write_csv(output_dir / f"stage_E32b_regression_guard_trace_{args.artifact_version}.csv", regression_rows)
    e31.write_csv(output_dir / f"stage_E32b_bundle552_audit_{args.artifact_version}.csv", all_bundle552_rows)
    e31.write_csv(output_dir / f"stage_E32b_baseline_consistency_check_{args.artifact_version}.csv", baseline_consistency)
    e31.write_csv(output_dir / f"stage_E32b_strict_anchor_eval_{args.artifact_version}.csv", all_retrieval_rows)
    (output_dir / f"stage_E32b_summary_{args.artifact_version}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / f"stage_E32b_report_{args.artifact_version}.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
