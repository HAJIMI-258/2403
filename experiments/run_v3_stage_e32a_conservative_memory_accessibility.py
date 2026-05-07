from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v3_stage_e31_retrieval_competition_repair as e31
from experiments import run_v3_stage_e32_global_retrieval_calibration as e32


FOCUS_EVENT_IDS = {"M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run v3 Stage E3.2a conservative memory accessibility calibration.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--event-audit", default="results/v3_e1/stage_E1_event_audit_v1.csv")
    p.add_argument("--cross-run-alignment", default="results/v3_e2rm/stage_E2R_cross_run_target_alignment_v1.csv")
    p.add_argument("--e2c-negative-events", default="results/v3_e2c/stage_E2C_negative_control_events_v1.csv")
    p.add_argument("--output-dir", default="results/v3_e32a")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def conservative_score(row: dict[str, Any], state: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, dict[str, float]]:
    consensus = e32.geomean([
        row["support_score"],
        row["motion_score"],
        row["context_score"],
        row["temporal_score"],
        row["disappearance_score"],
    ])
    td_mean = float(np.mean([row["temporal_score"], row["disappearance_score"], row["context_score"]]))
    access_bonus = cfg["small_accessibility_bonus"] * min(float(state["accessibility_score"]), cfg["accessibility_bonus_clip"])
    prov_bonus = cfg["provenance_bonus"] * min(float(state["specificity_score"]) * float(row["provenance_score"]), 1.0)
    consensus_bonus = cfg["cue_consensus_bonus"] * consensus
    td_bonus = cfg["temporal_disappearance_bonus"] * td_mean
    hub_penalty = cfg["hubness_penalty"] * min(float(state["hubness_score"]) / cfg["hubness_norm"], 1.0)
    ambiguity_penalty = cfg["ambiguity_penalty"] * e32.cue_disagreement(row)
    score = float(row["final_score"] + access_bonus + prov_bonus + consensus_bonus + td_bonus - hub_penalty - ambiguity_penalty)
    return score, {
        "cue_consensus_score": consensus,
        "temporal_disappearance_score": td_mean,
        "small_accessibility_bonus_clipped": access_bonus,
        "provenance_specificity_bonus": prov_bonus,
        "cue_consensus_bonus": consensus_bonus,
        "hubness_penalty": hub_penalty,
        "ambiguity_penalty": ambiguity_penalty,
    }


def classify_state(state: dict[str, Any]) -> str:
    acc = float(state["accessibility_score"])
    sup = float(state["suppression_score"])
    if sup >= 0.40:
        return "suppressed"
    if acc >= 0.75:
        return "stable"
    if acc >= 0.55:
        return "stabilizing"
    if acc >= 0.35:
        return "accessible"
    if acc >= 0.18:
        return "candidate"
    return "latent"


def init_states(bundle_by_id: dict[int, dict[str, Any]], proto_counter: Counter[int], track_counter: Counter[int]) -> dict[int, dict[str, Any]]:
    states = e32.init_bundle_states(bundle_by_id, proto_counter, track_counter)
    for state in states.values():
        state["pending_updates_applied"] = 0
        state["pending_replay_evidence"] = 0.0
    return states


def wrong_proto_map_from_negative_rows(negative_rows: list[dict[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for row in negative_rows:
        if str(row.get("control_name", "")) != "wrong_old_prototype":
            continue
        proto = e31.si(row.get("test_old_prototype_id"), None)
        if proto is not None:
            mapping[str(row.get("event_id", ""))] = int(proto)
    return mapping


def positive_gate(top1: dict[str, Any], state: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, float, float, list[str], float]:
    consensus = e32.geomean([
        top1["support_score"],
        top1["motion_score"],
        top1["context_score"],
        top1["temporal_score"],
        top1["disappearance_score"],
    ])
    reasons = []
    if consensus < cfg["gate_cue_consensus"]:
        reasons.append("low_cue_consensus")
    if float(top1["temporal_score"]) < cfg["gate_temporal"]:
        reasons.append("low_temporal_score")
    if not (float(top1["disappearance_score"]) >= cfg["gate_disappearance"] or float(top1["context_score"]) >= cfg["gate_context"]):
        reasons.append("low_disappearance_or_context")
    if float(state["hubness_score"]) > cfg["gate_hubness_max"]:
        reasons.append("hubness_too_high")
    if float(top1["top1_margin"]) < cfg["gate_margin"]:
        reasons.append("margin_too_low")
    passed = len(reasons) == 0
    proposed = cfg["pos_delta_scale"] * consensus * min(float(state["specificity_score"]), 1.0) * min(float(top1["provenance_score"]), 1.0)
    clipped = min(proposed, cfg["max_positive_delta"]) if passed else 0.0
    return passed, float(proposed), float(clipped), reasons, float(consensus)


def suppression_gate(top1: dict[str, Any], top5: list[dict[str, Any]], state: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, float, float, list[str], dict[str, int | float | bool]]:
    supportive_mean = float(np.mean([top1["temporal_score"], top1["disappearance_score"], top1["context_score"]]))
    proto_family_monopoly = sum(1 for row in top5[:5] if int(row["primary_source_prototype_id"]) == int(top1["primary_source_prototype_id"])) >= cfg["supp_proto_monopoly_min"]
    generic_content_only = float(top1["content_score"]) >= cfg["supp_content_min"] and supportive_mean <= cfg["supp_supportive_mean_max"]
    high_hubness = float(state["hubness_score"]) >= cfg["supp_hubness_min"]
    cue_disagreement_high = e32.cue_disagreement(top1) >= cfg["supp_disagreement_min"]
    temporal_or_disappearance_low = float(top1["temporal_score"]) <= cfg["supp_temporal_max"] or float(top1["disappearance_score"]) <= cfg["supp_disappearance_max"]
    low_margin = float(top1["top1_margin"]) <= cfg["supp_margin_max"]
    reasons = []
    if not high_hubness:
        reasons.append("not_high_hubness")
    if not cue_disagreement_high:
        reasons.append("cue_disagreement_not_high")
    if not temporal_or_disappearance_low:
        reasons.append("temporal_disappearance_not_low")
    if not (generic_content_only or proto_family_monopoly):
        reasons.append("not_generic_or_monopoly")
    if not low_margin:
        reasons.append("margin_too_high")
    passed = len(reasons) == 0
    severity = min(1.0, (float(state["hubness_score"]) / max(cfg["supp_hubness_min"], 1e-6) + e32.cue_disagreement(top1)) / 2.0)
    proposed = cfg["neg_delta_scale"] * severity
    clipped = min(proposed, cfg["max_negative_delta"]) if passed else 0.0
    detail = {
        "high_hubness": high_hubness,
        "cue_disagreement_high": cue_disagreement_high,
        "temporal_or_disappearance_low": temporal_or_disappearance_low,
        "generic_content_only": generic_content_only,
        "prototype_family_monopoly": proto_family_monopoly,
        "low_margin": low_margin,
        "supportive_mean": supportive_mean,
    }
    return passed, float(proposed), float(clipped), reasons, detail


def should_rerank_top3(top3: list[dict[str, Any]], states: dict[int, dict[str, Any]], cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    if len(top3) < 2:
        return False, ["insufficient_top3"]
    top1 = top3[0]
    top1_state = states[int(top1["bundle_id"])]
    consensus = e32.geomean([
        top1["support_score"],
        top1["motion_score"],
        top1["context_score"],
        top1["temporal_score"],
        top1["disappearance_score"],
    ])
    reasons: list[str] = []
    if float(top1["top1_margin"]) <= cfg["rerank_margin_max"]:
        reasons.append("low_margin_top1")
    if consensus <= cfg["rerank_consensus_max"]:
        reasons.append("low_consensus_top1")
    if float(top1_state["hubness_score"]) >= cfg["rerank_hubness_min"] and e32.cue_disagreement(top1) >= cfg["rerank_disagreement_min"]:
        reasons.append("high_hub_generic_top1")
    top1_supportive = float(top1["temporal_score"] + top1["disappearance_score"] + top1["provenance_score"])
    alt_better = any(float(row["temporal_score"] + row["disappearance_score"] + row["provenance_score"]) > top1_supportive + cfg["rerank_alt_advantage"] for row in top3[1:])
    if alt_better:
        reasons.append("alt_temporal_disappearance_provenance_better")
    return len(reasons) > 0, reasons

def classify_false_reason(event: dict[str, Any], target_row: dict[str, Any] | None, top1_row: dict[str, Any] | None, top_ids: list[int], candidate_pool_ids: set[int], state_by_bundle: dict[int, dict[str, Any]], cfg: dict[str, Any]) -> str:
    target_bundle_id = event["target_bundle_id"]
    if target_bundle_id is None:
        return "metric_mismatch"
    if int(target_bundle_id) not in candidate_pool_ids:
        return "target_not_in_candidate_pool"
    if int(target_bundle_id) in set(top_ids[:3]) and (not top_ids or int(top_ids[0]) != int(target_bundle_id)):
        return "target_in_top3_but_lost_top1"
    if int(target_bundle_id) in set(top_ids[:5]) and int(target_bundle_id) not in set(top_ids[:3]):
        return "target_in_top5_but_lost_top1"
    if target_row is None or top1_row is None:
        return "metric_mismatch"
    top1_state = state_by_bundle[int(top1_row["bundle_id"])]
    if float(top1_state["accessibility_score"]) >= cfg["false_accessibility_high"] and float(top1_state["hubness_score"]) >= cfg["supp_hubness_min"]:
        return "wrong_bundle_accessibility_too_high"
    if float(top1_state["hubness_score"]) >= cfg["supp_hubness_min"] and e32.cue_disagreement(top1_row) >= cfg["supp_disagreement_min"]:
        return "hub_bundle_dominance"
    if float(top1_row["content_score"]) >= cfg["supp_content_min"] and float(np.mean([top1_row["temporal_score"], top1_row["disappearance_score"], top1_row["context_score"]])) <= cfg["supp_supportive_mean_max"]:
        return "generic_content_dominance"
    if float(target_row["temporal_score"]) < cfg["gate_temporal"]:
        return "temporal_gap_mismatch"
    if float(target_row["disappearance_score"]) < cfg["gate_disappearance"]:
        return "disappearance_signature_weak"
    if float(target_row["context_score"]) < cfg["gate_context"]:
        return "context_signature_collision"
    if float(target_row["provenance_score"]) < cfg["provenance_floor"]:
        return "provenance_too_weak"
    return "ambiguous_multi_valid_bundle"


def apply_pending_updates(event_idx: int, states: dict[int, dict[str, Any]], pending_updates: dict[int, list[dict[str, Any]]]) -> None:
    for row in pending_updates.pop(event_idx, []):
        state = states[int(row["bundle_id"])]
        old_acc = float(state["accessibility_score"])
        new_acc = float(np.clip(old_acc + float(row["clipped_delta"]), 0.0, 1.0))
        state["accessibility_score"] = new_acc
        if str(row["update_kind"]) == "suppression":
            state["suppression_score"] = float(np.clip(float(state["suppression_score"]) + abs(float(row["clipped_delta"])), 0.0, 1.0))
            state["suppression_count"] += 1
        elif str(row["update_kind"]) == "positive":
            state["reactivation_count"] += 1
        state["pending_updates_applied"] += 1
        state["accessibility_state"] = classify_state(state)
        row["applied"] = 1
        row["applied_event_idx"] = event_idx
        row["applied_old_accessibility"] = old_acc
        row["applied_new_accessibility"] = new_acc


def schedule_update(update_rows: list[dict[str, Any]], pending_updates: dict[int, list[dict[str, Any]]], *, ablation_name: str, event: dict[str, Any], event_idx: int, bundle_id: int, old_accessibility: float, proposed_delta: float, clipped_delta: float, update_kind: str, source_component: str, update_reason: str, gate_passed: int, blocked_reasons: list[str], detail: dict[str, Any], effective_from_event_idx: int) -> None:
    row = {
        "ablation_name": ablation_name,
        "event_id": event["event_id"],
        "frame_idx": int(event["frame_idx"]),
        "event_order_idx": int(event_idx),
        "bundle_id": int(bundle_id),
        "update_kind": update_kind,
        "source_component": source_component,
        "update_reason": update_reason,
        "gate_passed": int(gate_passed),
        "blocked_reasons": "|".join(blocked_reasons),
        "old_accessibility": float(old_accessibility),
        "proposed_delta": float(proposed_delta),
        "clipped_delta": float(clipped_delta),
        "new_accessibility": float(np.clip(old_accessibility + clipped_delta, 0.0, 1.0)),
        "effective_from_event_idx": int(effective_from_event_idx),
        "current_event_accessibility_mutation": 0,
        "same_event_top1_changed_by_reconsolidation": 0,
        "applied": 0,
        "applied_event_idx": "",
        "applied_old_accessibility": "",
        "applied_new_accessibility": "",
    }
    row.update(detail)
    update_rows.append(row)
    if clipped_delta != 0.0:
        pending_updates[effective_from_event_idx].append(row)


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


def summarize_false_counts(taxonomy_rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for row in taxonomy_rows:
        if int(row["false_bundle_retrieval"]) != 1:
            continue
        counter[str(row["false_reason"])] += 1
    return {
        "target_in_top3_but_lost_top1_count": int(counter["target_in_top3_but_lost_top1"]),
        "wrong_bundle_accessibility_too_high_count": int(counter["wrong_bundle_accessibility_too_high"]),
        "hub_bundle_dominance_count": int(counter["hub_bundle_dominance"]),
    }

def evaluate_ablation(name: str, base_cfg: dict[str, Any], dyn_cfg: dict[str, Any], bundle_by_id: dict[int, dict[str, Any]], event_records: list[dict[str, Any]], proto_counter: Counter[int], track_counter: Counter[int], lineage_counter: Counter[int | None], wrong_proto_map: dict[str, int]) -> dict[str, Any]:
    states = init_states(bundle_by_id, proto_counter, track_counter)
    ordered_events = sorted(event_records, key=lambda r: (r["scenario_name"], int(r["frame_idx"]), r["event_id"]))
    hist_topk: Counter[int] = Counter()
    pending_updates: dict[int, list[dict[str, Any]]] = defaultdict(list)

    retrieval_rows: list[dict[str, Any]] = []
    top3_trace_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    taxonomy_rows: list[dict[str, Any]] = []

    for event_idx, event in enumerate(ordered_events, start=1):
        apply_pending_updates(event_idx, states, pending_updates)

        if int(event["proposal_detected"]) != 1 or event["cue"] is None:
            retrieval_rows.append({
                "ablation_name": name,
                "scenario_name": event["scenario_name"],
                "event_id": event["event_id"],
                "frame_idx": int(event["frame_idx"]),
                "proposal_detected": int(event["proposal_detected"]),
                "target_bundle_exists": int(event["target_bundle_exists"]),
                "target_bundle_id": event["target_bundle_id"] or "",
                "target_bundle_rank": "",
                "target_bundle_score": "",
                "target_bundle_retrieved_top1": 0,
                "target_bundle_retrieved_top3": 0,
                "target_bundle_retrieved_top5": 0,
                "pattern_completion_success": 0,
                "strict_anchor_visible_top5": 0,
                "loose_anchor_visible_top5": 0,
                "false_bundle_retrieval": 0,
                "proto0_bundle_count_in_top5": 0,
                "top5_bundle_ids": "",
                "top5_proto_ids": "",
                "top1_bundle_id": "",
                "top1_margin": "",
                "target_lost_reason": "proposal_missing_for_retrieval",
                "alignment_classification": event["alignment_classification"],
            })
            continue

        eligible = [bundle_by_id[b] for b in event["eligible_bundle_ids"] if b in bundle_by_id]
        if not eligible:
            retrieval_rows.append({
                "ablation_name": name,
                "scenario_name": event["scenario_name"],
                "event_id": event["event_id"],
                "frame_idx": int(event["frame_idx"]),
                "proposal_detected": int(event["proposal_detected"]),
                "target_bundle_exists": int(event["target_bundle_exists"]),
                "target_bundle_id": event["target_bundle_id"] or "",
                "target_bundle_rank": "",
                "target_bundle_score": "",
                "target_bundle_retrieved_top1": 0,
                "target_bundle_retrieved_top3": 0,
                "target_bundle_retrieved_top5": 0,
                "pattern_completion_success": 0,
                "strict_anchor_visible_top5": 0,
                "loose_anchor_visible_top5": 0,
                "false_bundle_retrieval": 0,
                "proto0_bundle_count_in_top5": 0,
                "top5_bundle_ids": "",
                "top5_proto_ids": "",
                "top1_bundle_id": "",
                "top1_margin": "",
                "target_lost_reason": "target_not_in_candidate_pool",
                "alignment_classification": event["alignment_classification"],
            })
            continue

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
        candidate_pool_ids = {int(r["bundle_id"]) for r in candidate_pool}
        base_reranked = sorted(candidate_pool, key=lambda r: r["final_score"], reverse=True)
        base_final_topk = e31.diversify_candidates(base_reranked, base_cfg)

        for row in base_final_topk:
            row["top1_margin"] = 0.0
        if len(base_final_topk) >= 2:
            margin = float(base_final_topk[0]["final_score"] - base_final_topk[1]["final_score"])
            for row in base_final_topk[:3]:
                row["top1_margin"] = margin
        elif base_final_topk:
            base_final_topk[0]["top1_margin"] = float(base_final_topk[0]["final_score"])

        final_topk = list(base_final_topk)
        rerank_triggered = False
        rerank_reasons: list[str] = []
        rerank_score_rows = []
        if dyn_cfg["top3_rerank_enabled"]:
            rerank_triggered, rerank_reasons = should_rerank_top3(base_final_topk[:3], states, dyn_cfg)
            if rerank_triggered:
                rescored_top3 = []
                for row in base_final_topk[:3]:
                    state = states[int(row["bundle_id"])]
                    score, breakdown = conservative_score(row, state, dyn_cfg)
                    rescored = dict(row)
                    rescored["conservative_score"] = score
                    rescored.update(breakdown)
                    rescored_top3.append(rescored)
                    rerank_score_rows.append(rescored)
                rescored_top3.sort(key=lambda r: (r["conservative_score"], r["final_score"]), reverse=True)
                base_top1_bundle_id = int(base_final_topk[0]["bundle_id"]) if base_final_topk else None
                if base_top1_bundle_id is not None and int(rescored_top3[0]["bundle_id"]) != base_top1_bundle_id:
                    base_top1_row = next(r for r in rescored_top3 if int(r["bundle_id"]) == base_top1_bundle_id)
                    new_top1_row = rescored_top3[0]
                    base_guard = float(base_top1_row["temporal_disappearance_score"] + base_top1_row["provenance_specificity_bonus"] + base_top1_row["cue_consensus_score"])
                    new_guard = float(new_top1_row["temporal_disappearance_score"] + new_top1_row["provenance_specificity_bonus"] + new_top1_row["cue_consensus_score"])
                    if float(new_top1_row["conservative_score"]) < float(base_top1_row["conservative_score"]) + dyn_cfg["rerank_switch_margin"] or new_guard < base_guard + dyn_cfg["rerank_switch_supportive_margin"]:
                        rerank_reasons.append("switch_guard_blocked")
                        rescored_top3 = [base_top1_row] + [r for r in rescored_top3 if int(r["bundle_id"]) != base_top1_bundle_id]
                final_topk = rescored_top3 + base_final_topk[3:]
            else:
                for row in base_final_topk[:3]:
                    rescored = dict(row)
                    rescored["conservative_score"] = float(row["final_score"])
                    rescored.update({
                        "cue_consensus_score": e32.geomean([row["support_score"], row["motion_score"], row["context_score"], row["temporal_score"], row["disappearance_score"]]),
                        "temporal_disappearance_score": float(np.mean([row["temporal_score"], row["disappearance_score"], row["context_score"]])),
                        "small_accessibility_bonus_clipped": 0.0,
                        "provenance_specificity_bonus": 0.0,
                        "cue_consensus_bonus": 0.0,
                        "hubness_penalty": 0.0,
                        "ambiguity_penalty": 0.0,
                    })
                    rerank_score_rows.append(rescored)
        else:
            for row in base_final_topk[:3]:
                rescored = dict(row)
                rescored["conservative_score"] = float(row["final_score"])
                rerank_score_rows.append(rescored)

        effective_score_map = {int(r["bundle_id"]): float(r.get("conservative_score", r["final_score"])) for r in rerank_score_rows}
        for row in candidate_pool:
            effective_score_map.setdefault(int(row["bundle_id"]), float(row["final_score"]))

        top_ids = [int(r["bundle_id"]) for r in final_topk]
        for row in final_topk:
            hist_topk[int(row["bundle_id"])] += 1

        target_bundle_id = event["target_bundle_id"]
        target_row = next((r for r in candidate_pool if target_bundle_id is not None and int(r["bundle_id"]) == int(target_bundle_id)), None)
        target_rank = next((i for i, r in enumerate(final_topk, start=1) if target_bundle_id is not None and int(r["bundle_id"]) == int(target_bundle_id)), None)
        top1 = final_topk[0] if final_topk else None
        top2 = final_topk[1] if len(final_topk) > 1 else None
        top1_score = None if top1 is None else float(effective_score_map[int(top1["bundle_id"])])
        top2_score = None if top2 is None else float(effective_score_map[int(top2["bundle_id"])])
        top1_margin = 0.0 if top1_score is None else (top1_score - top2_score if top2_score is not None else top1_score)
        if top1 is not None:
            top1["top1_margin"] = top1_margin

        top1_hit = int(target_bundle_id is not None and len(top_ids) > 0 and int(top_ids[0]) == int(target_bundle_id))
        top3_hit = int(target_bundle_id is not None and int(target_bundle_id) in set(top_ids[:3]))
        top5_hit = int(target_bundle_id is not None and int(target_bundle_id) in set(top_ids[:5]))
        target_score = None if target_bundle_id is None else effective_score_map.get(int(target_bundle_id))
        success = int(top1_hit == 1 and target_score is not None and float(target_score) >= base_cfg["completion_threshold"])
        false_retrieval = int(len(top_ids) > 0 and top1_hit == 0)
        false_reason = "" if false_retrieval == 0 else classify_false_reason(event, target_row, top1, top_ids, candidate_pool_ids, states, dyn_cfg)
        strict_anchor = int(top5_hit == 1)
        loose_anchor = int(top5_hit == 1)

        top3_trace_rows.append({
            "ablation_name": name,
            "event_id": event["event_id"],
            "frame_idx": int(event["frame_idx"]),
            "target_bundle_id": "" if target_bundle_id is None else int(target_bundle_id),
            "base_top1_bundle_id": "" if not base_final_topk else int(base_final_topk[0]["bundle_id"]),
            "reranked_top1_bundle_id": "" if top1 is None else int(top1["bundle_id"]),
            "rerank_triggered": int(rerank_triggered),
            "rerank_reasons": "|".join(rerank_reasons),
            "base_top3_bundle_ids": "|".join(str(int(r["bundle_id"])) for r in base_final_topk[:3]),
            "reranked_top3_bundle_ids": "|".join(str(int(r["bundle_id"])) for r in final_topk[:3]),
            "base_top1_margin": float(base_final_topk[0].get("top1_margin", 0.0)) if base_final_topk else "",
            "reranked_top1_margin": float(top1_margin),
            "target_rank_before": next((i for i, r in enumerate(base_final_topk, start=1) if target_bundle_id is not None and int(r["bundle_id"]) == int(target_bundle_id)), ""),
            "target_rank_after": "" if target_rank is None else int(target_rank),
            "proto0_bundle_count_in_top5_before": sum(1 for r in base_final_topk[:5] if int(r["primary_source_prototype_id"]) == 0),
            "proto0_bundle_count_in_top5_after": sum(1 for r in final_topk[:5] if int(r["primary_source_prototype_id"]) == 0),
            "target_lost_reason_after": false_reason,
        })
        if top1 is not None:
            state = states[int(top1["bundle_id"])]
            passed, proposed, clipped, blocked_reasons, consensus = positive_gate(top1, state, dyn_cfg)
            gate_rows.append({
                "ablation_name": name,
                "event_id": event["event_id"],
                "frame_idx": int(event["frame_idx"]),
                "bundle_id": int(top1["bundle_id"]),
                "gate_type": "positive_accessibility",
                "gate_passed": int(passed and dyn_cfg["delayed_accessibility_enabled"]),
                "blocked_reasons": "|".join(blocked_reasons if dyn_cfg["delayed_accessibility_enabled"] else ["component_disabled"]),
                "cue_consensus_score": float(consensus),
                "temporal_score": float(top1["temporal_score"]),
                "disappearance_score": float(top1["disappearance_score"]),
                "context_score": float(top1["context_score"]),
                "hubness_score": float(state["hubness_score"]),
                "top1_margin": float(top1_margin),
                "generic_content_only": 0,
                "prototype_family_monopoly": 0,
                "effective_from_event_idx": int(event_idx + 1),
            })
            schedule_update(
                update_rows,
                pending_updates,
                ablation_name=name,
                event=event,
                event_idx=event_idx,
                bundle_id=int(top1["bundle_id"]),
                old_accessibility=float(state["accessibility_score"]),
                proposed_delta=proposed,
                clipped_delta=clipped if dyn_cfg["delayed_accessibility_enabled"] else 0.0,
                update_kind="positive",
                source_component="delayed_accessibility",
                update_reason="multi_cue_positive" if passed else "positive_gate_blocked",
                gate_passed=int(passed and dyn_cfg["delayed_accessibility_enabled"]),
                blocked_reasons=[] if (passed and dyn_cfg["delayed_accessibility_enabled"]) else (blocked_reasons if dyn_cfg["delayed_accessibility_enabled"] else ["component_disabled"]),
                detail={"cue_consensus_score": float(consensus)},
                effective_from_event_idx=event_idx + 1,
            )

            sup_passed, sup_proposed, sup_clipped, sup_reasons, sup_detail = suppression_gate(top1, final_topk, state, dyn_cfg)
            gate_rows.append({
                "ablation_name": name,
                "event_id": event["event_id"],
                "frame_idx": int(event["frame_idx"]),
                "bundle_id": int(top1["bundle_id"]),
                "gate_type": "strict_suppression",
                "gate_passed": int(sup_passed and dyn_cfg["strict_suppression_enabled"]),
                "blocked_reasons": "|".join(sup_reasons if dyn_cfg["strict_suppression_enabled"] else ["component_disabled"]),
                "cue_consensus_score": float(e32.geomean([top1["support_score"], top1["motion_score"], top1["context_score"], top1["temporal_score"], top1["disappearance_score"]])),
                "temporal_score": float(top1["temporal_score"]),
                "disappearance_score": float(top1["disappearance_score"]),
                "context_score": float(top1["context_score"]),
                "hubness_score": float(state["hubness_score"]),
                "top1_margin": float(top1_margin),
                "generic_content_only": int(sup_detail["generic_content_only"]),
                "prototype_family_monopoly": int(sup_detail["prototype_family_monopoly"]),
                "effective_from_event_idx": int(event_idx + 1),
            })
            schedule_update(
                update_rows,
                pending_updates,
                ablation_name=name,
                event=event,
                event_idx=event_idx,
                bundle_id=int(top1["bundle_id"]),
                old_accessibility=float(state["accessibility_score"]),
                proposed_delta=-sup_proposed,
                clipped_delta=-(sup_clipped if dyn_cfg["strict_suppression_enabled"] else 0.0),
                update_kind="suppression",
                source_component="strict_suppression",
                update_reason="generic_hub_suppression" if sup_passed else "suppression_gate_blocked",
                gate_passed=int(sup_passed and dyn_cfg["strict_suppression_enabled"]),
                blocked_reasons=[] if (sup_passed and dyn_cfg["strict_suppression_enabled"]) else (sup_reasons if dyn_cfg["strict_suppression_enabled"] else ["component_disabled"]),
                detail={k: int(v) if isinstance(v, bool) else v for k, v in sup_detail.items()},
                effective_from_event_idx=event_idx + 1,
            )

        retrieval_rows.append({
            "ablation_name": name,
            "scenario_name": event["scenario_name"],
            "event_id": event["event_id"],
            "frame_idx": int(event["frame_idx"]),
            "proposal_detected": int(event["proposal_detected"]),
            "target_bundle_exists": int(event["target_bundle_exists"]),
            "target_bundle_id": "" if target_bundle_id is None else int(target_bundle_id),
            "target_bundle_rank": "" if target_rank is None else int(target_rank),
            "target_bundle_score": "" if target_score is None else float(target_score),
            "target_bundle_retrieved_top1": top1_hit,
            "target_bundle_retrieved_top3": top3_hit,
            "target_bundle_retrieved_top5": top5_hit,
            "pattern_completion_success": success,
            "strict_anchor_visible_top5": strict_anchor,
            "loose_anchor_visible_top5": loose_anchor,
            "false_bundle_retrieval": false_retrieval,
            "proto0_bundle_count_in_top5": sum(1 for r in final_topk[:5] if int(r["primary_source_prototype_id"]) == 0),
            "top5_bundle_ids": "|".join(str(int(v)) for v in top_ids[:5]),
            "top5_proto_ids": "|".join(str(int(r["primary_source_prototype_id"])) for r in final_topk[:5]),
            "top1_bundle_id": "" if top1 is None else int(top1["bundle_id"]),
            "top1_margin": float(top1_margin),
            "target_lost_reason": false_reason,
            "alignment_classification": event["alignment_classification"],
        })

        wrong_top1 = top1 if false_retrieval == 1 else None
        taxonomy_rows.append({
            "ablation_name": name,
            "event_id": event["event_id"],
            "target_bundle_id": "" if target_bundle_id is None else int(target_bundle_id),
            "target_rank": "" if target_rank is None else int(target_rank),
            "target_in_candidate_pool": int(target_bundle_id is not None and int(target_bundle_id) in candidate_pool_ids),
            "target_in_top5": top5_hit,
            "target_in_top3": top3_hit,
            "target_in_top1": top1_hit,
            "wrong_top1_bundle_id": "" if wrong_top1 is None else int(wrong_top1["bundle_id"]),
            "wrong_top1_source_prototype": "" if wrong_top1 is None else int(wrong_top1["primary_source_prototype_id"]),
            "wrong_top1_memory_anchor": "" if wrong_top1 is None else wrong_top1["memory_anchor_id"],
            "wrong_top1_canonical_lineage": "" if wrong_top1 is None else wrong_top1["canonical_lineage_id"],
            "target_score": "" if target_score is None else float(target_score),
            "wrong_top1_score": "" if wrong_top1 is None else float(top1_score),
            "score_margin": "" if wrong_top1 is None or target_score is None else float(top1_score - target_score),
            "target_content_score": "" if target_row is None else float(target_row["content_score"]),
            "target_support_score": "" if target_row is None else float(target_row["support_score"]),
            "target_motion_score": "" if target_row is None else float(target_row["motion_score"]),
            "target_context_score": "" if target_row is None else float(target_row["context_score"]),
            "target_temporal_score": "" if target_row is None else float(target_row["temporal_score"]),
            "target_disappearance_score": "" if target_row is None else float(target_row["disappearance_score"]),
            "target_provenance_score": "" if target_row is None else float(target_row["provenance_score"]),
            "target_accessibility_score": "" if target_row is None else float(states[int(target_row["bundle_id"])] ["accessibility_score"]),
            "wrong_content_score": "" if wrong_top1 is None else float(wrong_top1["content_score"]),
            "wrong_support_score": "" if wrong_top1 is None else float(wrong_top1["support_score"]),
            "wrong_motion_score": "" if wrong_top1 is None else float(wrong_top1["motion_score"]),
            "wrong_context_score": "" if wrong_top1 is None else float(wrong_top1["context_score"]),
            "wrong_temporal_score": "" if wrong_top1 is None else float(wrong_top1["temporal_score"]),
            "wrong_disappearance_score": "" if wrong_top1 is None else float(wrong_top1["disappearance_score"]),
            "wrong_provenance_score": "" if wrong_top1 is None else float(wrong_top1["provenance_score"]),
            "wrong_accessibility_score": "" if wrong_top1 is None else float(states[int(wrong_top1["bundle_id"])] ["accessibility_score"]),
            "false_reason": false_reason,
            "false_bundle_retrieval": false_retrieval,
        })

    proposal_rows = [row for row in retrieval_rows if int(row["proposal_detected"]) == 1]
    focus_rows = [row for row in proposal_rows if row["event_id"] in FOCUS_EVENT_IDS]
    summary = {
        "ablation_name": name,
        "global_top1": 0.0 if not proposal_rows else float(np.mean([int(row["target_bundle_retrieved_top1"]) for row in proposal_rows])),
        "global_top3": 0.0 if not proposal_rows else float(np.mean([int(row["target_bundle_retrieved_top3"]) for row in proposal_rows])),
        "global_top5": 0.0 if not proposal_rows else float(np.mean([int(row["target_bundle_retrieved_top5"]) for row in proposal_rows])),
        "false_bundle_retrieval_rate": 0.0 if not proposal_rows else float(np.mean([int(row["false_bundle_retrieval"]) for row in proposal_rows])),
        "focus_top1_count": int(sum(int(row["target_bundle_retrieved_top1"]) for row in focus_rows)),
        "focus_success_count": int(sum(int(row["pattern_completion_success"]) for row in focus_rows)),
        "regression_event_count": 0,
        "improved_event_count": 0,
        "unchanged_failure_count": 0,
        "unchanged_success_count": 0,
        "proto0_top5_share": 0.0 if not proposal_rows else float(np.mean([int(row["proto0_bundle_count_in_top5"]) / 5.0 for row in proposal_rows])),
        "strict_anchor_real_svr": 0.0 if not proposal_rows else float(np.mean([int(row["strict_anchor_visible_top5"]) for row in proposal_rows])),
        "strict_anchor_shuffled_svr": compute_shuffled_strict_svr(proposal_rows),
        "wrong_old_prototype_visible_count": int(compute_wrong_old_visible_count(proposal_rows, wrong_proto_map)),
        "unsafe_component_count": 0,
        "selected_as_best": 0,
        **summarize_false_counts(taxonomy_rows),
    }
    return {
        "summary": summary,
        "retrieval_rows": retrieval_rows,
        "top3_trace_rows": top3_trace_rows,
        "update_rows": update_rows,
        "gate_rows": gate_rows,
        "taxonomy_rows": taxonomy_rows,
    }

def render_report(summary: dict[str, Any]) -> str:
    best = summary["best_ablation"]
    lines = [
        "# Stage E3.2a Report",
        "",
        "## 结论",
        "",
        f"- 最优消融：`{best['ablation_name']}`",
        f"- focus 保持情况：`top1={best['focus_top1_count']}/3, success={best['focus_success_count']}/3`",
        f"- 全局 top1：`{best['global_top1']}`",
        f"- 全局 top3：`{best['global_top3']}`",
        f"- 全局 false retrieval：`{best['false_bundle_retrieval_rate']}`",
        f"- regression_event_count：`{best['regression_event_count']}`",
        "",
        "## 人话判断",
        "",
        summary["human_summary"],
        "",
        "## Focus Events",
        "",
    ]
    for row in summary["focus_events"]:
        lines.extend([
            f"### {row['event_id']}",
            "",
            f"- `baseline_target_rank = {row['baseline_target_rank']}`",
            f"- `target_bundle_rank_after = {row['target_bundle_rank_after']}`",
            f"- `target_bundle_retrieved_top1 = {row['target_bundle_retrieved_top1']}`",
            f"- `pattern_completion_success = {row['pattern_completion_success']}`",
            f"- `target_lost_reason = {row['target_lost_reason']}`",
            "",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_by_id, event_records, _ = e31.collect_runtime_data(args.config, args.event_audit, args.cross_run_alignment, args.seed)
    proto_counter, track_counter, lineage_counter = e31.compute_static_counts(bundle_by_id)
    negative_rows = e31.load_negative_controls(args.e2c_negative_events)
    wrong_proto_map = wrong_proto_map_from_negative_rows(negative_rows)

    a0_base = {
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
        "delayed_accessibility_enabled": False,
        "top3_rerank_enabled": False,
        "strict_suppression_enabled": False,
        "replay_evidence_only": False,
        "small_accessibility_bonus": 0.04,
        "accessibility_bonus_clip": 0.35,
        "provenance_bonus": 0.18,
        "cue_consensus_bonus": 0.12,
        "temporal_disappearance_bonus": 0.28,
        "hubness_penalty": 0.05,
        "hubness_norm": 2.5,
        "ambiguity_penalty": 0.05,
        "gate_cue_consensus": 0.72,
        "gate_temporal": 0.62,
        "gate_disappearance": 0.62,
        "gate_context": 0.70,
        "gate_hubness_max": 1.20,
        "gate_margin": 0.05,
        "pos_delta_scale": 0.04,
        "neg_delta_scale": 0.06,
        "max_positive_delta": 0.03,
        "max_negative_delta": 0.05,
        "rerank_margin_max": 0.055,
        "rerank_consensus_max": 0.74,
        "rerank_hubness_min": 1.00,
        "rerank_disagreement_min": 0.12,
        "rerank_alt_advantage": 0.12,
        "supp_hubness_min": 1.00,
        "supp_disagreement_min": 0.14,
        "supp_temporal_max": 0.72,
        "supp_disappearance_max": 0.76,
        "supp_content_min": 0.93,
        "supp_supportive_mean_max": 0.78,
        "supp_proto_monopoly_min": 2,
        "supp_margin_max": 0.05,
        "false_accessibility_high": 0.84,
        "rerank_switch_margin": 0.012,
        "rerank_switch_supportive_margin": 0.10,
        "provenance_floor": 0.65,
    }
    ablations = {
        "A0_E31_combined_baseline": (a0_base, {**common}),
        "A1_delayed_accessibility_only": (a0_base, {**common, "delayed_accessibility_enabled": True}),
        "A2_conservative_top3_rerank_only": (a0_base, {**common, "top3_rerank_enabled": True}),
        "A3_strict_suppression_only": (a0_base, {**common, "strict_suppression_enabled": True}),
        "A4_replay_evidence_only": (a0_base, {**common, "replay_evidence_only": True}),
        "A5_top3_rerank_plus_strict_suppression": (a0_base, {**common, "top3_rerank_enabled": True, "strict_suppression_enabled": True}),
        "A6_delayed_accessibility_plus_top3_rerank": (a0_base, {**common, "delayed_accessibility_enabled": True, "top3_rerank_enabled": True}),
        "A7_full_E32a_conservative": (a0_base, {**common, "delayed_accessibility_enabled": True, "top3_rerank_enabled": True, "strict_suppression_enabled": True, "replay_evidence_only": True}),
        "A8_full_E32a_no_accessibility_bonus": (a0_base, {**common, "top3_rerank_enabled": True, "strict_suppression_enabled": True, "replay_evidence_only": True, "small_accessibility_bonus": 0.0, "delayed_accessibility_enabled": False}),
        "A9_full_E32a_no_suppression": (a0_base, {**common, "delayed_accessibility_enabled": True, "top3_rerank_enabled": True, "strict_suppression_enabled": False, "replay_evidence_only": True}),
    }

    results, ablation_rows = {}, []
    all_top3_trace, all_update_rows, all_gate_rows, all_tax_rows, all_retrieval_rows = [], [], [], [], []
    for name, (base_cfg, dyn_cfg) in ablations.items():
        result = evaluate_ablation(name, base_cfg, dyn_cfg, bundle_by_id, event_records, proto_counter, track_counter, lineage_counter, wrong_proto_map)
        results[name] = result
        ablation_rows.append(result["summary"])
        all_top3_trace.extend(result["top3_trace_rows"])
        all_update_rows.extend(result["update_rows"])
        all_gate_rows.extend(result["gate_rows"])
        all_tax_rows.extend(result["taxonomy_rows"])
        all_retrieval_rows.extend(result["retrieval_rows"])

    baseline_rows = {row["event_id"]: row for row in results["A0_E31_combined_baseline"]["retrieval_rows"]}
    delta_rows, regression_rows = [], []
    component_map = {
        "A0_E31_combined_baseline": "baseline",
        "A1_delayed_accessibility_only": "delayed_accessibility",
        "A2_conservative_top3_rerank_only": "conservative_top3_rerank",
        "A3_strict_suppression_only": "strict_suppression",
        "A4_replay_evidence_only": "replay_evidence_only",
        "A5_top3_rerank_plus_strict_suppression": "top3_rerank+strict_suppression",
        "A6_delayed_accessibility_plus_top3_rerank": "delayed_accessibility+top3_rerank",
        "A7_full_E32a_conservative": "full_conservative",
        "A8_full_E32a_no_accessibility_bonus": "full_no_accessibility_bonus",
        "A9_full_E32a_no_suppression": "full_no_suppression",
    }

    for row in ablation_rows:
        name = str(row["ablation_name"])
        if name == "A0_E31_combined_baseline":
            row["improved_event_count"] = 0
            row["unchanged_failure_count"] = 0
            row["unchanged_success_count"] = len([r for r in results[name]["retrieval_rows"] if int(r["proposal_detected"]) == 1 and int(r["pattern_completion_success"]) == 1])
            row["regression_event_count"] = 0
            row["unsafe_component_count"] = 0
            continue
        improved = regressed = unchanged_failure = unchanged_success = 0
        rows = {r["event_id"]: r for r in results[name]["retrieval_rows"]}
        for event_id, after in rows.items():
            before = baseline_rows.get(event_id)
            if before is None or int(after["proposal_detected"]) != 1:
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
                "e32a_target_rank": "" if arank == 999 else arank,
                "baseline_top1_bundle_id": before["top1_bundle_id"],
                "e32a_top1_bundle_id": after["top1_bundle_id"],
                "baseline_target_in_top3": before["target_bundle_retrieved_top3"],
                "e32a_target_in_top3": after["target_bundle_retrieved_top3"],
                "baseline_target_in_top5": before["target_bundle_retrieved_top5"],
                "e32a_target_in_top5": after["target_bundle_retrieved_top5"],
                "baseline_success": bsucc,
                "e32a_success": asucc,
                "delta_class": delta_class,
                "baseline_false_reason": before["target_lost_reason"],
                "e32a_false_reason": after["target_lost_reason"],
            })
            if delta_class == "regressed":
                regression_rows.append({
                    "ablation_name": name,
                    "event_id": event_id,
                    "baseline_top1_bundle": before["top1_bundle_id"],
                    "e32a_top1_bundle": after["top1_bundle_id"],
                    "baseline_target_rank": "" if brank == 999 else brank,
                    "e32a_target_rank": "" if arank == 999 else arank,
                    "regression_reason": after["target_lost_reason"],
                    "changed_by_component": component_map[name],
                    "unsafe_component": 1,
                })
        row["improved_event_count"] = improved
        row["unchanged_failure_count"] = unchanged_failure
        row["unchanged_success_count"] = unchanged_success
        row["regression_event_count"] = regressed
        row["unsafe_component_count"] = int(regressed > 1 or int(row["focus_top1_count"]) < 3 or int(row["focus_success_count"]) < 3)

    safe_rows = [row for row in ablation_rows if int(row["focus_top1_count"]) == 3 and int(row["focus_success_count"]) == 3 and int(row["unsafe_component_count"]) == 0]
    if safe_rows:
        best = min(safe_rows, key=lambda r: (r["false_bundle_retrieval_rate"], -r["global_top1"], -r["global_top3"], r["regression_event_count"]))
    else:
        best = max(ablation_rows, key=lambda r: (r["focus_success_count"], r["focus_top1_count"], -r["false_bundle_retrieval_rate"], r["global_top1"]))
    best_name = str(best["ablation_name"])
    for row in ablation_rows:
        row["selected_as_best"] = int(str(row["ablation_name"]) == best_name)

    best_rows = {r["event_id"]: r for r in results[best_name]["retrieval_rows"]}
    focus_summary_rows = []
    for event_id in sorted(FOCUS_EVENT_IDS):
        after = best_rows.get(event_id)
        before = baseline_rows.get(event_id)
        if after is None or before is None:
            continue
        focus_summary_rows.append({
            "ablation_name": best_name,
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

    strict_eval_rows = []
    for row in all_retrieval_rows:
        strict_eval_rows.append({
            "ablation_name": row["ablation_name"],
            "event_id": row["event_id"],
            "scenario_name": row["scenario_name"],
            "strict_anchor_visible_top5": row["strict_anchor_visible_top5"],
            "loose_anchor_visible_top5": row["loose_anchor_visible_top5"],
            "target_bundle_retrieved_top5": row["target_bundle_retrieved_top5"],
            "target_bundle_retrieved_top3": row["target_bundle_retrieved_top3"],
            "target_bundle_retrieved_top1": row["target_bundle_retrieved_top1"],
        })
    strict_eval_rows.extend(e31.classify_normal_reference_rows(negative_rows))

    best_metrics = next(row for row in ablation_rows if str(row["ablation_name"]) == best_name)
    passed_minimum = int(best_metrics["focus_top1_count"]) == 3 and int(best_metrics["focus_success_count"]) == 3 and float(best_metrics["global_top1"]) >= 0.3529 and float(best_metrics["false_bundle_retrieval_rate"]) < 0.6471 and int(best_metrics["regression_event_count"]) <= 1 and float(best_metrics["proto0_top5_share"]) <= 0.1765 and int(best_metrics["wrong_bundle_accessibility_too_high_count"]) < 6 and int(best_metrics["target_in_top3_but_lost_top1_count"]) < 5 and int(best_metrics["wrong_old_prototype_visible_count"]) <= 2

    if passed_minimum:
        human_summary = "E3.2a 通过最低门槛：focus 3/3 没掉，全局 false retrieval 相比 E32 的 A0 基线继续下降，而且没有用同事件内的 accessibility/reconsolidation 去硬改当前 top1。"
    else:
        human_summary = "E3.2a 这轮没有通过最低门槛。问题不在候选池，而在保守校准对全局 false retrieval 的改善还不够，或者引入了新的 regression。"
        if int(best_metrics["regression_event_count"]) > 0:
            human_summary += f" 当前最优消融 `{best_name}` 还有 `{best_metrics['regression_event_count']}` 个回退事件。"
        if int(best_metrics["target_in_top3_but_lost_top1_count"]) >= 5:
            human_summary += " 主要残留 failure 仍然是 target 已经在 top3 里，但没有拿下 top1。"
        if int(best_metrics["wrong_bundle_accessibility_too_high_count"]) >= 6:
            human_summary += " 另一个主要残留是错误 bundle 的 accessibility 仍然偏高。"

    summary = {"scope": "track_a_bridge_and_track_c_long_horizon", "best_ablation": best_metrics, "focus_events": focus_summary_rows, "passed_minimum": passed_minimum, "current_event_accessibility_mutation": 0, "same_event_top1_changed_by_reconsolidation": 0, "human_summary": human_summary}

    e31.write_csv(output_dir / f"stage_E32a_ablation_summary_{args.artifact_version}.csv", ablation_rows)
    e31.write_csv(output_dir / f"stage_E32a_event_delta_audit_{args.artifact_version}.csv", delta_rows)
    e31.write_csv(output_dir / f"stage_E32a_top3_rerank_trace_{args.artifact_version}.csv", all_top3_trace)
    e31.write_csv(output_dir / f"stage_E32a_conservative_update_trace_{args.artifact_version}.csv", all_update_rows)
    e31.write_csv(output_dir / f"stage_E32a_accessibility_gate_trace_{args.artifact_version}.csv", all_gate_rows)
    e31.write_csv(output_dir / f"stage_E32a_regression_guard_trace_{args.artifact_version}.csv", regression_rows)
    e31.write_csv(output_dir / f"stage_E32a_false_retrieval_taxonomy_{args.artifact_version}.csv", all_tax_rows)
    e31.write_csv(output_dir / f"stage_E32a_focus_event_summary_{args.artifact_version}.csv", focus_summary_rows)
    e31.write_csv(output_dir / f"stage_E32a_strict_anchor_eval_{args.artifact_version}.csv", strict_eval_rows)
    (output_dir / f"stage_E32a_summary_{args.artifact_version}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / f"stage_E32a_report_{args.artifact_version}.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()



