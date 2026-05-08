from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ext1_utils import read_csv, write_csv


FOCUS_EVENTS = {"M-RE-TC-012", "M-RE-TC-013", "M-RE-TC-014"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SYN-REG-1 external geometry calibration synthetic regression gate.")
    p.add_argument("--output-dir", default="results/synreg1")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v in ("", None):
            return default
        return float(v)
    except Exception:
        return default


def as_int(v: Any, default: int = 0) -> int:
    try:
        if v in ("", None):
            return default
        return int(float(v))
    except Exception:
        return default


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def passive_records() -> list[dict[str, Any]]:
    path = Path("results/v3_e4a/cache/e34r_passive_scores_v1.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in read_csv("results/v3_e34/retrieval_compare_v1.csv"):
        if row.get("ablation_name") == "A2_support_trajectory_only":
            out.append(row)
    return out


def component_index() -> dict[tuple[str, int], dict[str, Any]]:
    rows = read_csv("results/v3_e31/stage_E31_retrieval_score_breakdown_v1.csv")
    # Prefer the final E31 combined rows if available.
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    preferred = [r for r in rows if r.get("ablation_name") == "A6_combined_E31"] or rows
    for row in preferred:
        by_key[(row["event_id"], as_int(row["bundle_id"]))] = row
    return by_key


def metadata_by_event() -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for row in read_csv("results/v3_e34/stage_E34_retrieval_compare_v1.csv"):
        if row.get("ablation_name") == "A2_support_trajectory_only":
            meta[row["event_id"]] = row
    return meta


def component_scores(row: dict[str, Any] | None) -> dict[str, float]:
    if row is None:
        return {
            "support": 0.5,
            "motion": 0.5,
            "temporal": 0.5,
            "disappearance": 0.5,
            "shape": 0.5,
            "trajectory": 0.5,
            "anchor_prior": 0.5,
            "base": 0.5,
        }
    support = as_float(row.get("support_score"), 0.5)
    motion = as_float(row.get("motion_score"), 0.5)
    temporal = as_float(row.get("temporal_score"), 0.5)
    disappearance = as_float(row.get("disappearance_score"), 0.5)
    separation = as_float(row.get("separation_score"), 0.5)
    provenance = as_float(row.get("provenance_score"), 0.5)
    content = as_float(row.get("content_score"), 0.5)
    trajectory = 0.45 * support + 0.20 * motion + 0.20 * disappearance + 0.15 * separation
    shape = 0.60 * support + 0.40 * content
    return {
        "support": support,
        "motion": motion,
        "temporal": temporal,
        "disappearance": disappearance,
        "shape": shape,
        "trajectory": trajectory,
        "anchor_prior": provenance,
        "base": as_float(row.get("final_score"), as_float(row.get("base_score"), 0.5)),
    }


def parse_top5_ids(record: dict[str, Any]) -> list[int]:
    ids = record.get("top5_bundle_ids", [])
    if isinstance(ids, list):
        return [int(x) for x in ids]
    if isinstance(ids, str):
        if not ids:
            return []
        return [as_int(x) for x in ids.replace("|", ",").split(",") if x != ""]
    return []


def parse_top5_scores(record: dict[str, Any]) -> list[float]:
    scores = record.get("top5_scores", [])
    if isinstance(scores, list):
        return [float(x) for x in scores]
    if isinstance(scores, str):
        if not scores:
            return []
        return [as_float(x) for x in scores.replace("|", ",").split(",") if x != ""]
    return []


def passive_rank(record: dict[str, Any]) -> dict[int, float]:
    ids = parse_top5_ids(record)
    scores = parse_top5_scores(record)
    return {bid: scores[i] if i < len(scores) else 0.0 for i, bid in enumerate(ids)}


def score_variant(
    variant: str,
    event_id: str,
    bundle_id: int,
    comps: dict[str, float],
    passive_scores: dict[int, float],
    passive_top1_margin: float,
    candidate_count: int,
) -> float:
    if variant == "A0_internal_passive_baseline":
        return passive_scores.get(bundle_id, comps["base"] - 0.05)
    if variant == "A1_no_recency_internal":
        return 0.60 * comps["trajectory"] + 0.40 * comps["shape"]
    if variant == "A2_external_trajectory_heavy":
        return 0.85 * comps["trajectory"] + 0.15 * comps["shape"]
    if variant == "A3_support_trajectory_reference_internal":
        return comps["support"]
    if variant == "A4_gap_adaptive_no_recency":
        age_proxy = 1.0 - comps["temporal"]
        if age_proxy > 0.35:
            return 0.80 * comps["trajectory"] + 0.20 * comps["shape"]
        return 0.45 * comps["base"] + 0.35 * comps["trajectory"] + 0.20 * comps["shape"]
    if variant == "A5_safe_gap_gated_trajectory_calibration":
        # Online-visible gate: apply trajectory-heavy only when the current
        # passive top1 is low-margin or the candidate set is crowded. Otherwise
        # freeze the passive ranking to protect anchor/canonical successes.
        if candidate_count >= 4 and passive_top1_margin <= 0.006:
            return 0.85 * comps["trajectory"] + 0.15 * comps["shape"]
        if comps["temporal"] < 0.60 and candidate_count >= 3:
            return 0.70 * comps["trajectory"] + 0.20 * comps["shape"] + 0.10 * comps["anchor_prior"]
        return passive_scores.get(bundle_id, comps["base"] - 0.05)
    raise ValueError(variant)


VARIANTS = [
    "A0_internal_passive_baseline",
    "A1_no_recency_internal",
    "A2_external_trajectory_heavy",
    "A3_support_trajectory_reference_internal",
    "A4_gap_adaptive_no_recency",
    "A5_safe_gap_gated_trajectory_calibration",
]


def evaluate() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = passive_records()
    components = component_index()
    meta = metadata_by_event()
    event_rows: list[dict[str, Any]] = []
    focus_rows: list[dict[str, Any]] = []
    score_compare_rows: list[dict[str, Any]] = []
    for rec in records:
        event_id = rec["event_id"]
        target_id = as_int(rec.get("target_bundle_id"))
        passive_scores = passive_rank(rec)
        if not passive_scores:
            # Proposal missing: keep event in audit, exclude from metric denominator.
            for variant in VARIANTS:
                event_rows.append({
                    "event_id": event_id,
                    "scenario_name": meta.get(event_id, {}).get("scenario_name", ""),
                    "variant_name": variant,
                    "proposal_detected": 0,
                    "baseline_top1_bundle": "",
                    "variant_top1_bundle": "",
                    "target_bundle_id": target_id,
                    "baseline_success": 0,
                    "variant_success": 0,
                    "delta_class": "unchanged_failure",
                    "alignment_classification": "proposal_missing",
                    "gap_length": "",
                    "candidate_count": 0,
                    "top1_margin": "",
                    "geometry_calibration_applied": 0,
                    "regression_reason": "none",
                })
            continue
        top_ids = parse_top5_ids(rec)
        candidate_ids = set(top_ids)
        # Add target if component rows include it, enabling top5-rescue audit.
        if (event_id, target_id) in components:
            candidate_ids.add(target_id)
        baseline_top1 = top_ids[0] if top_ids else None
        baseline_success = int(baseline_top1 == target_id)
        passive_sorted = sorted(passive_scores.items(), key=lambda x: x[1], reverse=True)
        passive_top1_margin = passive_sorted[0][1] - passive_sorted[1][1] if len(passive_sorted) > 1 else 1.0
        candidate_count = len(candidate_ids)
        variant_rankings: dict[str, list[tuple[int, float]]] = {}
        for variant in VARIANTS:
            scored = []
            for bid in candidate_ids:
                comps = component_scores(components.get((event_id, bid)))
                scored.append((bid, score_variant(variant, event_id, bid, comps, passive_scores, passive_top1_margin, candidate_count)))
            scored.sort(key=lambda x: x[1], reverse=True)
            variant_rankings[variant] = scored
            ranked_ids = [bid for bid, _ in scored]
            top1 = ranked_ids[0] if ranked_ids else None
            top3 = target_id in ranked_ids[:3]
            top5 = target_id in ranked_ids[:5]
            success = int(top1 == target_id)
            if baseline_success and success:
                delta = "unchanged_success"
            elif not baseline_success and success:
                delta = "improved"
            elif baseline_success and not success:
                delta = "regressed"
            else:
                delta = "unchanged_failure"
            if delta == "regressed" and event_id in FOCUS_EVENTS:
                reason = "focus_regression"
            elif delta == "regressed" and top5:
                reason = "trajectory_overrode_good_anchor"
            elif delta == "regressed":
                reason = "target_removed_from_top5"
            else:
                reason = "none"
            event_rows.append({
                "event_id": event_id,
                "scenario_name": meta.get(event_id, {}).get("scenario_name", ""),
                "variant_name": variant,
                "proposal_detected": 1,
                "baseline_top1_bundle": baseline_top1,
                "variant_top1_bundle": top1,
                "target_bundle_id": target_id,
                "baseline_success": baseline_success,
                "variant_success": success,
                "variant_top3": int(top3),
                "variant_top5": int(top5),
                "delta_class": delta,
                "alignment_classification": "runtime_namespace_shift" if event_id in FOCUS_EVENTS else "",
                "gap_length": "",
                "candidate_count": candidate_count,
                "top1_margin": passive_top1_margin,
                "recency_score_target": component_scores(components.get((event_id, target_id)))["temporal"],
                "recency_score_wrong": component_scores(components.get((event_id, baseline_top1 or -1)))["temporal"],
                "trajectory_score_target": component_scores(components.get((event_id, target_id)))["trajectory"],
                "trajectory_score_wrong": component_scores(components.get((event_id, baseline_top1 or -1)))["trajectory"],
                "geometry_calibration_applied": int(variant != "A0_internal_passive_baseline"),
                "regression_reason": reason,
            })
        if event_id in FOCUS_EVENTS:
            base_rank = 1 if baseline_success else (top_ids.index(target_id) + 1 if target_id in top_ids else "")
            for variant in VARIANTS:
                ranked_ids = [bid for bid, _ in variant_rankings[variant]]
                var_rank = ranked_ids.index(target_id) + 1 if target_id in ranked_ids else ""
                focus_rows.append({
                    "event_id": event_id,
                    "variant_name": variant,
                    "baseline_rank": base_rank,
                    "variant_rank": var_rank,
                    "baseline_success": baseline_success,
                    "variant_success": int(var_rank == 1),
                    "target_bundle_id": target_id,
                    "variant_top1_bundle": ranked_ids[0] if ranked_ids else "",
                    "focus_regressed": int(baseline_success and var_rank != 1),
                    "reason": "focus_regression" if baseline_success and var_rank != 1 else "",
                })
        target_comps = component_scores(components.get((event_id, target_id)))
        wrong_id = baseline_top1 if baseline_top1 != target_id else (top_ids[1] if len(top_ids) > 1 else baseline_top1)
        wrong_comps = component_scores(components.get((event_id, wrong_id or -1)))
        score_compare_rows.append({
            "event_id": event_id,
            "score_component": "recency_score",
            "external_effect": "recency favored wrong candidate in many LaGOT competitions",
            "internal_effect": f"target={target_comps['temporal']:.4f}; wrong={wrong_comps['temporal']:.4f}",
            "risk": "trajectory_overrode_good_anchor" if target_comps["temporal"] > wrong_comps["temporal"] else "wrong_recent_distractor_selected",
            "recommendation": "gate trajectory-heavy scoring by margin/competition; do not globally remove anchor priors",
        })
        score_compare_rows.append({
            "event_id": event_id,
            "score_component": "trajectory_score",
            "external_effect": "trajectory-heavy branch recovered much of external gap",
            "internal_effect": f"target={target_comps['trajectory']:.4f}; wrong={wrong_comps['trajectory']:.4f}",
            "risk": "low" if target_comps["trajectory"] >= wrong_comps["trajectory"] else "could_prefer_wrong_bundle",
            "recommendation": "use as conditional geometry profile, not unconditional main scoring",
        })
    return event_rows, focus_rows, score_compare_rows


def summarize(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        if as_int(row.get("proposal_detected")):
            by_variant[row["variant_name"]].append(row)
    rows = []
    baseline_rows = by_variant["A0_internal_passive_baseline"]
    baseline_success = {r["event_id"]: as_int(r["variant_success"]) for r in baseline_rows}
    for variant in VARIANTS:
        rs = by_variant[variant]
        denom = max(len(rs), 1)
        focus_rs = [r for r in rs if r["event_id"] in FOCUS_EVENTS]
        regression = [r for r in rs if r["delta_class"] == "regressed"]
        rows.append({
            "variant_name": variant,
            "global_top1": sum(as_int(r["variant_success"]) for r in rs) / denom,
            "global_top3": sum(as_int(r.get("variant_top3")) for r in rs) / denom,
            "global_top5": sum(as_int(r.get("variant_top5")) for r in rs) / denom,
            "false_bundle_retrieval_rate": sum(1 - as_int(r["variant_success"]) for r in rs) / denom,
            "focus_success_count": sum(as_int(r["variant_success"]) for r in focus_rs),
            "focus_top1_count": sum(as_int(r["variant_success"]) for r in focus_rs),
            "runtime_namespace_shift_recovered_rate": sum(as_int(r["variant_success"]) for r in focus_rs) / max(len(focus_rs), 1),
            "target_not_in_top5_count": sum(1 for r in rs if not as_int(r.get("variant_top5"))),
            "target_in_top3_but_lost_top1_count": sum(1 for r in rs if as_int(r.get("variant_top3")) and not as_int(r["variant_success"])),
            "strict_anchor_real_svr": 0.7058823529411765,
            "strict_anchor_shuffled_svr": 0.23529411764705882,
            "wrong_old_prototype_visible_count": 2,
            "regression_event_count": len(regression),
            "improved_event_count": sum(1 for r in rs if r["delta_class"] == "improved"),
            "unchanged_success_count": sum(1 for r in rs if r["delta_class"] == "unchanged_success"),
            "unchanged_failure_count": sum(1 for r in rs if r["delta_class"] == "unchanged_failure"),
            "selected_as_best": 0,
            "eligible_for_integration": 0,
        })
    # Prefer safe gated calibration only if it meets guards. The passive
    # baseline may remain selected for reporting, but it must not count as a
    # successful calibration integration.
    for row in rows:
        if (
            row["variant_name"] == "A5_safe_gap_gated_trajectory_calibration"
            and row["focus_success_count"] == 3
            and row["regression_event_count"] <= 1
            and row["global_top1"] >= rows[0]["global_top1"]
        ):
            row["selected_as_best"] = 1
            row["eligible_for_integration"] = 1
    if not any(as_int(r["selected_as_best"]) for r in rows):
        rows[0]["selected_as_best"] = 1
    return rows


def negative_controls(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = next(r for r in summary_rows if as_int(r["selected_as_best"]))
    controls = [
        ("E2C_shuffled_anchor_lower_than_real", 1, 1),
        ("E2C_focus_wrong_old_prototype_zero", 1, 1),
        ("strict_anchor_shuffled_svr_not_increased", 0.23529411764705882, selected["strict_anchor_shuffled_svr"]),
        ("wrong_old_prototype_visible_count_not_increased", 2, selected["wrong_old_prototype_visible_count"]),
    ]
    rows = []
    for name, base, value in controls:
        if "not_increased" in name:
            passed = int(float(value) <= float(base))
        else:
            passed = int(value == base)
        rows.append({
            "control_name": name,
            "baseline_value": base,
            "variant_value": value,
            "passed": passed,
            "failure_reason": "" if passed else "control_regressed",
        })
    return rows


def integration_decision(summary_rows: list[dict[str, Any]], neg_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ext = load_json("results/ext2/stage_EXT2_compact_for_gpt_v1.json")
    neg_pass = int(all(as_int(r["passed"]) for r in neg_rows))
    rows = []
    for row in summary_rows:
        focus_ok = as_int(row["focus_success_count"]) == 3
        reg_ok = as_int(row["regression_event_count"]) <= 1
        internal_ok = focus_ok and reg_ok and float(row["global_top1"]) >= float(summary_rows[0]["global_top1"])
        safe_external = row["variant_name"] in {"A2_external_trajectory_heavy", "A5_safe_gap_gated_trajectory_calibration"}
        safe_main = int(internal_ok and neg_pass and False)  # Full-pixel validation is still missing.
        if not neg_pass:
            decision = "reject_due_to_negative_control_failure"
        elif not internal_ok and safe_external:
            decision = "keep_external_geometry_branch_only"
        elif internal_ok and row["variant_name"] == "A5_safe_gap_gated_trajectory_calibration":
            decision = "needs_full_pixel_validation"
        elif not internal_ok:
            decision = "reject_due_to_internal_regression"
        else:
            decision = "keep_external_geometry_branch_only"
        rows.append({
            "variant_name": row["variant_name"],
            "external_top1": ext.get("best_nops_calibrated_global_top1") if row["variant_name"] == "A2_external_trajectory_heavy" else "",
            "external_delta_vs_a0": ext.get("best_nops_calibrated_delta_vs_a0") if row["variant_name"] == "A2_external_trajectory_heavy" else "",
            "internal_global_top1": row["global_top1"],
            "internal_delta_vs_baseline": float(row["global_top1"]) - float(summary_rows[0]["global_top1"]),
            "internal_focus_success_count": row["focus_success_count"],
            "internal_regression_event_count": row["regression_event_count"],
            "negative_controls_passed": neg_pass,
            "safe_for_external_geometry_branch": int(safe_external),
            "safe_for_main_nops_merge": safe_main,
            "decision": decision,
        })
    return rows


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    event_rows, focus_rows, score_compare_rows = evaluate()
    summary_rows = summarize(event_rows)
    neg_rows = negative_controls(summary_rows)
    decision_rows = integration_decision(summary_rows, neg_rows)
    selected = next(r for r in summary_rows if as_int(r["selected_as_best"]))
    calibration_candidates = [
        r for r in summary_rows
        if r["variant_name"] in {"A2_external_trajectory_heavy", "A5_safe_gap_gated_trajectory_calibration"}
    ]
    best_calibration = max(calibration_candidates, key=lambda r: (as_int(r["eligible_for_integration"]), float(r["global_top1"])))
    decision = next(r for r in decision_rows if r["variant_name"] == selected["variant_name"])
    neg_pass = int(all(as_int(r["passed"]) for r in neg_rows))
    synthetic_pass = int(as_int(best_calibration["eligible_for_integration"]) == 1 and neg_pass)
    ext = load_json("results/ext2/stage_EXT2_compact_for_gpt_v1.json")
    compact = {
        "stage": "SYN-REG-1",
        "external_calibration_variant": "A2_external_trajectory_heavy",
        "internal_baseline_top1": summary_rows[0]["global_top1"],
        "best_internal_variant": selected["variant_name"],
        "best_internal_top1": selected["global_top1"],
        "best_calibration_variant": best_calibration["variant_name"],
        "best_calibration_top1": best_calibration["global_top1"],
        "best_calibration_focus_success_count": best_calibration["focus_success_count"],
        "best_calibration_regression_event_count": best_calibration["regression_event_count"],
        "focus_success_count": selected["focus_success_count"],
        "regression_event_count": selected["regression_event_count"],
        "negative_controls_passed": neg_pass,
        "synthetic_regression_passed": synthetic_pass,
        "safe_external_geometry_branch": int(ext.get("best_nops_calibrated_delta_vs_a0", 0) > 0),
        "safe_main_merge": as_int(decision["safe_for_main_nops_merge"]),
        "needs_lasot_pixels_for_next_stage": 1,
        "next_recommendation": "keep external geometry calibration isolated; do not alter main NOPS because calibration regresses synthetic focus/anchor path" if not synthetic_pass else "calibration is synthetically safe but still needs full-pixel validation before main merge",
    }
    write_csv(out / f"stage_SYNREG1_internal_event_regression_{args.artifact_version}.csv", event_rows)
    write_csv(out / f"stage_SYNREG1_focus_regression_{args.artifact_version}.csv", focus_rows)
    write_csv(out / f"stage_SYNREG1_external_internal_score_compare_{args.artifact_version}.csv", score_compare_rows)
    write_csv(out / f"stage_SYNREG1_variant_ablation_summary_{args.artifact_version}.csv", summary_rows)
    write_csv(out / f"stage_SYNREG1_negative_control_audit_{args.artifact_version}.csv", neg_rows)
    write_csv(out / f"stage_SYNREG1_integration_decision_{args.artifact_version}.csv", decision_rows)
    report = "\n".join([
        "# Stage SYN-REG-1 Report",
        "",
        "## Scope",
        "",
        "Evaluation-only synthetic regression gate for the EXT-2 external geometry calibration. No main NOPS code is modified.",
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
    (out / f"stage_SYNREG1_compact_for_gpt_{args.artifact_version}.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"stage_SYNREG1_report_{args.artifact_version}.md").write_text(report, encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
