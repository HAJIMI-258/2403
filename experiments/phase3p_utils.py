"""Shared helpers for Phase 3P prototype-head continuity stabilization."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.phase3l_utils import (
    default_phase3l_memory_override,
    default_phase3l_tracking_override,
    evaluate_phase3l_bundle,
)
from experiments.phase3r_utils import load_config_payload, write_csv
from experiments.phase3s_utils import TRACK_A_NAME, TRACK_C_NAME


def default_phase3p_tracking_override() -> dict[str, Any]:
    return default_phase3l_tracking_override()


def default_phase3p_memory_override() -> dict[str, Any]:
    override = default_phase3l_memory_override()
    override.update(
        {
            "enable_phase3p_keep_head_default": False,
            "enable_phase3p_grouped_gating": False,
            "enable_phase3p_birth_suppression": False,
            "enable_phase3p_full_stabilization": False,
            "keep_head_min": 0.42,
            "replace_margin": 0.08,
            "replace_consistency_window": 6,
            "archived_reactivate_min": 0.58,
            "birth_margin": 0.10,
            "post_recovery_birth_suppression_window": 12,
            "lineage_internal_cooldown_after_head_switch": 8,
            "head_continuity_bonus": 0.03,
            "archived_sibling_penalty": 0.06,
            "newborn_prototype_penalty": 0.10,
        }
    )
    return override


def evaluate_phase3p_bundle(
    config_path: str | Path,
    *,
    tracking_override: dict[str, Any] | None = None,
    memory_override: dict[str, Any] | None = None,
    seed: int = 42,
    scenario_names: list[str] | None = None,
) -> dict[str, Any]:
    merged_tracking = default_phase3p_tracking_override()
    if tracking_override:
        merged_tracking.update(tracking_override)
    merged_memory = default_phase3p_memory_override()
    if memory_override:
        merged_memory.update(memory_override)
    return evaluate_phase3l_bundle(
        config_path,
        tracking_override=merged_tracking,
        memory_override=merged_memory,
        seed=seed,
        scenario_names=scenario_names,
        frame_record_mode="full",
    )


def classify_phase3p_action_bucket(row: dict[str, Any]) -> str:
    if int(row.get("proposal_detected", 0)) == 0:
        return "G.proposal_missing_or_upstream_missing"
    same_lineage = int(
        row.get(
            "same_lineage",
            row.get("matched_same_lineage_prototype", row.get("same_lineage_id", 0)),
        )
    )
    if same_lineage == 1:
        action_type = str(row.get("action_type", "") or "")
        selected_state = str(row.get("selected_prototype_state", "") or "")
        if action_type == "keep_head" or selected_state == "head":
            return "A.matched_lineage_kept_current_head"
        if action_type == "replace_head" or selected_state == "active_sibling":
            return "B.matched_lineage_replaced_with_active_sibling"
        if action_type == "reactivate_archived" or selected_state == "archived_sibling":
            return "C.matched_lineage_reactivated_archived_sibling"
        if action_type == "birth_sibling":
            return "D.matched_lineage_created_new_sibling"
        if action_type == "birth_lineage":
            return "E.matched_lineage_created_new_lineage"
    if int(row.get("new_prototype_created", 0)) == 1:
        return "E.matched_lineage_created_new_lineage"
    return "F.no_valid_lineage_match"


def build_phase3p_event_audit_rows(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    for row in event_rows:
        audit_row = dict(row)
        audit_row["action_bucket"] = classify_phase3p_action_bucket(row)
        audit_row["selected_prototype_state"] = str(row.get("selected_prototype_state", "") or "")
        audit_row["action_type"] = str(row.get("action_type", "") or "")
        audit_row["same_track"] = int(row.get("matched_same_track", row.get("same_track", 0)))
        audit_row["same_prototype"] = int(row.get("matched_same_prototype", row.get("same_prototype_id", 0)))
        audit_row["same_lineage"] = int(row.get("matched_same_lineage_prototype", row.get("same_lineage_id", 0)))
        audit_row["pfr_delta_if_any"] = int(row.get("pfr_delta_if_any", 0))
        audit_row["idsw_delta_if_any"] = int(row.get("idsw_delta_if_any", row.get("idsw_after_reentry_window", 0) or 0))
        audit_rows.append(audit_row)
    return audit_rows


def build_phase3p_action_breakdown(audit_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    counts = Counter(str(row["action_bucket"]) for row in audit_rows)
    total = len(audit_rows)
    matched_lineage_rows = [row for row in audit_rows if int(row.get("same_lineage", 0)) == 1]
    concept_rows = [row for row in audit_rows if int(row.get("concept_recovered", 0)) == 1]
    breakdown_rows: list[dict[str, Any]] = []
    for bucket in sorted(counts):
        bucket_rows = [row for row in audit_rows if str(row["action_bucket"]) == bucket]
        breakdown_rows.append(
            {
                "action_bucket": bucket,
                "num_events": int(len(bucket_rows)),
                "ratio": float(len(bucket_rows) / total) if total else 0.0,
                "same_prototype_rate": _mean_int(bucket_rows, "same_prototype"),
                "same_track_rate": _mean_int(bucket_rows, "same_track"),
                "pfr_contribution": int(sum(int(row.get("pfr_delta_if_any", 0)) for row in bucket_rows)),
                "idsw_contribution": int(sum(int(row.get("idsw_delta_if_any", 0)) for row in bucket_rows)),
            }
        )
    summary = {
        "head_keep_rate_given_matched_lineage": (
            sum(int(str(row["action_bucket"]).startswith("A.")) for row in matched_lineage_rows) / len(matched_lineage_rows)
            if matched_lineage_rows
            else 0.0
        ),
        "head_replacement_rate_given_matched_lineage": (
            sum(int(str(row["action_bucket"]).startswith("B.")) for row in matched_lineage_rows) / len(matched_lineage_rows)
            if matched_lineage_rows
            else 0.0
        ),
        "active_sibling_win_rate": (
            sum(int(str(row["action_bucket"]).startswith("B.")) for row in matched_lineage_rows) / len(matched_lineage_rows)
            if matched_lineage_rows
            else 0.0
        ),
        "archived_sibling_reactivation_rate": (
            sum(int(str(row["action_bucket"]).startswith("C.")) for row in matched_lineage_rows) / len(matched_lineage_rows)
            if matched_lineage_rows
            else 0.0
        ),
        "new_sibling_birth_rate_given_concept_recovery": (
            sum(int(str(row["action_bucket"]).startswith("D.")) for row in concept_rows) / len(concept_rows)
            if concept_rows
            else 0.0
        ),
        "pfr_contribution_by_action_type_total": float(sum(int(row.get("pfr_delta_if_any", 0)) for row in audit_rows)),
        "idsw_contribution_by_action_type_total": float(sum(int(row.get("idsw_delta_if_any", 0)) for row in audit_rows)),
    }
    return breakdown_rows, summary


def build_phase3p_lineage_aggregate(
    audit_rows: list[dict[str, Any]],
    prototype_lineage_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_lineage: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        lineage_id = _maybe_int(row.get("old_lineage_id"))
        if lineage_id is None:
            continue
        by_lineage[int(lineage_id)].append(row)

    head_history: dict[int, list[tuple[int, int | None]]] = defaultdict(list)
    for row in prototype_lineage_rows:
        lineage_id = _maybe_int(row.get("lineage_id"))
        if lineage_id is None:
            continue
        head_history[int(lineage_id)].append(
            (int(row.get("frame_index", 0)), _maybe_int(row.get("head_prototype_id")))
        )

    aggregate_rows: list[dict[str, Any]] = []
    for lineage_id, rows in sorted(by_lineage.items()):
        history = sorted(head_history.get(int(lineage_id), []), key=lambda item: item[0])
        head_switches = 0
        last_head = None
        for _, head_id in history:
            if head_id is None:
                continue
            if last_head is not None and int(head_id) != int(last_head):
                head_switches += 1
            last_head = head_id
        aggregate_rows.append(
            {
                "lineage_id": int(lineage_id),
                "num_reentry_events": int(len(rows)),
                "head_keep_events": int(sum(int(str(row["action_bucket"]).startswith("A.")) for row in rows)),
                "head_replacement_events": int(sum(int(str(row["action_bucket"]).startswith("B.")) for row in rows)),
                "archived_sibling_reactivation_events": int(
                    sum(int(str(row["action_bucket"]).startswith("C.")) for row in rows)
                ),
                "new_sibling_birth_events": int(sum(int(str(row["action_bucket"]).startswith("D.")) for row in rows)),
                "new_lineage_birth_events": int(sum(int(str(row["action_bucket"]).startswith("E.")) for row in rows)),
                "prototype_head_churn_per_lineage": int(head_switches),
                "pfr_contribution_by_action_type": int(sum(int(row.get("pfr_delta_if_any", 0)) for row in rows)),
                "idsw_contribution_by_action_type": int(sum(int(row.get("idsw_delta_if_any", 0)) for row in rows)),
            }
        )
    return aggregate_rows


def build_phase3p_top_failure_cases_markdown(audit_rows: list[dict[str, Any]], *, top_k: int = 10) -> str:
    ranked = sorted(
        audit_rows,
        key=lambda row: (
            int(row.get("same_prototype", 0)) == 1,
            -int(row.get("pfr_delta_if_any", 0)),
            -int(row.get("idsw_delta_if_any", 0)),
            -int(row.get("gap_length", 0)),
        ),
    )
    lines = ["# Phase 3P Top Failure Cases", ""]
    for index, row in enumerate(ranked[:top_k], start=1):
        lines.append(
            f"{index}. event={row.get('event_id')} lineage={row.get('old_lineage_id')} "
            f"bucket={row.get('action_bucket')} action={row.get('action_type')} "
            f"state={row.get('selected_prototype_state')} same_prototype={row.get('same_prototype')} "
            f"same_track={row.get('same_track')} pfr_delta={row.get('pfr_delta_if_any')} "
            f"idsw_delta={row.get('idsw_delta_if_any')} gap={row.get('gap_length')} "
            f"head_score={_fmt(row.get('head_score'))} active_sibling={_fmt(row.get('best_active_sibling_score'))} "
            f"archived_sibling={_fmt(row.get('best_archived_sibling_score'))} margin_vs_head={_fmt(row.get('score_margin_vs_current_head'))}"
        )
    lines.append("")
    return "\n".join(lines)


def build_phase3p_audit_summary_markdown(
    *,
    breakdown_rows: list[dict[str, Any]],
    summary: dict[str, float],
    dominant_bucket: str,
) -> str:
    lines = [
        "# Phase 3P Audit Summary",
        "",
        f"- dominant_action_bucket: {dominant_bucket}",
        f"- head_keep_rate_given_matched_lineage: {summary['head_keep_rate_given_matched_lineage']:.4f}",
        f"- head_replacement_rate_given_matched_lineage: {summary['head_replacement_rate_given_matched_lineage']:.4f}",
        f"- active_sibling_win_rate: {summary['active_sibling_win_rate']:.4f}",
        f"- archived_sibling_reactivation_rate: {summary['archived_sibling_reactivation_rate']:.4f}",
        f"- new_sibling_birth_rate_given_concept_recovery: {summary['new_sibling_birth_rate_given_concept_recovery']:.4f}",
        "",
        "## Action Breakdown",
        "",
    ]
    for row in breakdown_rows:
        lines.append(
            f"- {row['action_bucket']}: events={row['num_events']}, ratio={row['ratio']:.4f}, "
            f"same_prototype={row['same_prototype_rate']:.4f}, same_track={row['same_track_rate']:.4f}, "
            f"pfr_contribution={row['pfr_contribution']}, idsw_contribution={row['idsw_contribution']}"
        )
    lines.append("")
    return "\n".join(lines)


def save_phase3p_stage_a_outputs(
    *,
    output_dir: str | Path,
    event_rows: list[dict[str, Any]],
    prototype_lineage_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    audit_rows = build_phase3p_event_audit_rows(event_rows)
    breakdown_rows, summary = build_phase3p_action_breakdown(audit_rows)
    lineage_rows = build_phase3p_lineage_aggregate(audit_rows, prototype_lineage_rows)
    dominant_bucket = breakdown_rows[0]["action_bucket"] if breakdown_rows else "none"
    if breakdown_rows:
        dominant_bucket = max(breakdown_rows, key=lambda row: int(row["num_events"]))["action_bucket"]

    write_csv(output_root / "phase3p_event_audit.csv", audit_rows)
    write_csv(output_root / "phase3p_lineage_aggregate.csv", lineage_rows)
    (output_root / "phase3p_audit_summary.md").write_text(
        build_phase3p_audit_summary_markdown(
            breakdown_rows=breakdown_rows,
            summary=summary,
            dominant_bucket=dominant_bucket,
        ),
        encoding="utf-8",
    )
    (output_root / "phase3p_top_failure_cases.md").write_text(
        build_phase3p_top_failure_cases_markdown(audit_rows),
        encoding="utf-8",
    )
    return {
        "audit_rows": audit_rows,
        "breakdown_rows": breakdown_rows,
        "lineage_rows": lineage_rows,
        "summary": summary,
        "dominant_bucket": dominant_bucket,
    }


def _mean_int(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(int(row.get(key, 0)) for row in rows) / len(rows)) if rows else 0.0


def _maybe_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    return int(value)


def _fmt(value: Any) -> str:
    if value in ("", None):
        return "NA"
    return f"{float(value):.4f}"
