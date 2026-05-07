"""Shared helpers for Phase 3L lineage-preserving prototype updates."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.phase3r2_utils import load_csv_rows
from experiments.phase3r_utils import (
    evaluate_phase3_scenarios,
    extract_reentry_events,
    summarize_reentry_events,
)
from experiments.phase3s_utils import PHASE3S_SCENARIOS, TRACK_A_NAME, TRACK_C_NAME
from experiments.phase3x_utils import (
    build_phase3x_event_trace,
    classify_phase3x_failure_stage,
    collect_phase3x_audit_rows,
    plot_prototype_lineage_timeline,
)


def default_phase3l_tracking_override() -> dict[str, Any]:
    return {
        "keepalive_frames": 8,
        "dormant_frames": 16,
        "ghost_frames": 80,
        "tau_g": 12,
        "tau_res_short": 0.56,
        "tau_res_long": 0.68,
        "tau_continuation": 0.62,
        "continuation_margin": 0.08,
        "enable_identity_slots": False,
    }


def default_phase3l_memory_override() -> dict[str, Any]:
    return {
        "protect_linked_prototypes": True,
        "enable_explicit_lineage": True,
        "preserve_lineage_on_archive": True,
        "preserve_lineage_on_replace": True,
        "preserve_lineage_on_merge": True,
        "allow_alias_lineage": True,
        "enable_continuation_bank": True,
        "bind_continuation_to": "lineage",
        "continuation_topk_per_proto": 4,
        "continuation_topk_per_lineage": 4,
        "min_track_age_for_continuation": 4,
        "min_hits_for_continuation": 3,
        "continuation_max_gap": 96,
        "continuation_decay": 0.01,
        "decay_patience": 24,
    }


def evaluate_phase3l_bundle(
    config_path: str | Path,
    *,
    tracking_override: dict[str, Any] | None = None,
    memory_override: dict[str, Any] | None = None,
    seed: int = 42,
    scenario_names: list[str] | None = None,
    frame_record_mode: str = "full",
) -> dict[str, Any]:
    merged_tracking = default_phase3l_tracking_override()
    if tracking_override:
        merged_tracking.update(tracking_override)
    merged_memory = default_phase3l_memory_override()
    if memory_override:
        merged_memory.update(memory_override)

    runs = evaluate_phase3_scenarios(
        config_path,
        tracking_override=merged_tracking,
        memory_override=merged_memory,
        scenario_names=scenario_names or PHASE3S_SCENARIOS,
        collect_frames=True,
        frame_record_mode=frame_record_mode,
        seed=seed,
    )
    rows: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_frame_logs: list[dict[str, Any]] = []
    for run in runs:
        events, frame_logs = extract_reentry_events(run["scenario_name"], run["sequence"], run["result"])
        reentry = summarize_reentry_events(events)
        concept_rows = [row for row in events if int(row.get("concept_recovered", 0)) == 1]
        lineage_rows = [row for row in events if int(row.get("matched_same_lineage_prototype", 0)) == 1]
        mismatch_base = [
            row
            for row in events
            if int(row.get("proposal_detected", 0)) == 1
            and row.get("matched_prototype_id") not in ("", None)
            and int(row.get("new_prototype_created", 0)) == 0
        ]
        continuation_access = [
            row
            for row in concept_rows
            if int(row.get("continuation_bank_size", 0)) > 0
            or int(row.get("lineage_matched_continuation_count", 0)) > 0
        ]
        row = {
            "scenario_name": run["scenario_name"],
            "u_recall": float(run["result"].summary.u_recall),
            "same_track_reentry_recovery": float(reentry["same_track_reentry_recovery"]),
            "same_prototype_reentry_recovery": float(reentry["same_prototype_reentry_recovery"]),
            "same_lineage_prototype_reentry_recovery": float(reentry["same_lineage_prototype_reentry_recovery"]),
            "same_track_after_concept_recovery": float(reentry["same_track_after_concept_recovery"]),
            "same_track_after_lineage_recovery": float(reentry["same_track_after_lineage_recovery"]),
            "concept_recovered_events": int(reentry["concept_recovered_events"]),
            "lineage_aware_concept_recovered_events": int(reentry["lineage_aware_concept_recovered_events"]),
            "concept_recovered_but_lineage_mismatch_rate": (
                sum(int(int(row["matched_same_lineage_prototype"]) == 0) for row in mismatch_base) / len(mismatch_base)
                if mismatch_base
                else 0.0
            ),
            "continuation_bank_access_rate_given_concept_recovery": (
                len(continuation_access) / len(concept_rows) if concept_rows else 0.0
            ),
            "new_track_with_old_lineage_rate": (
                sum(
                    int(
                        int(row.get("new_track_created", 0)) == 1
                        and int(row.get("matched_same_lineage_prototype", 0)) == 1
                    )
                    for row in events
                )
                / len(lineage_rows)
                if lineage_rows
                else 0.0
            ),
            "pfr": float(run["result"].summary.pfr),
            "track_idsw": int(run["result"].primary_monitoring["track_idsw"]),
            "memory_growth": float(run["result"].summary.memory_growth),
            "reentry_events": int(reentry["num_events"]),
        }
        rows.append(row)
        all_events.extend(events)
        all_frame_logs.extend(frame_logs)
    return {
        "tracking_override": merged_tracking,
        "memory_override": merged_memory,
        "rows": rows,
        "events": all_events,
        "frame_logs": all_frame_logs,
        "runs": runs,
    }


def pick_best_scan_row(rows: list[dict[str, Any]], before_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: _scan_sort_key(row, before_lookup))


def _scan_sort_key(row: dict[str, Any], before_lookup: dict[str, dict[str, Any]]) -> tuple[float, ...]:
    before_a = before_lookup.get(TRACK_A_NAME, {})
    track_a_drop = max(0.0, float(before_a.get("u_recall", 0.0)) - float(row.get("track_a_u_recall", 0.0)))
    track_a_same_proto_drop = max(
        0.0,
        float(before_a.get("same_prototype_reentry_recovery", 0.0))
        - float(row.get("track_a_same_prototype_reentry_recovery", 0.0)),
    )
    memory_growth_penalty = max(
        0.0,
        float(row.get("track_a_memory_growth", 0.0)) - float(before_a.get("memory_growth", 0.0)),
    )
    mismatch_gap = max(0.0, float(row["track_c_concept_recovered_but_lineage_mismatch_rate"]) - 0.25)
    access_gap = max(0.0, 0.70 - float(row["track_c_continuation_bank_access_rate_given_concept_recovery"]))
    same_lineage_gap = max(0.0, 0.50 - float(row["track_c_same_lineage_prototype_reentry_recovery"]))
    same_track_after_gap = max(0.0, 0.40 - float(row["track_c_same_track_after_concept_recovery"]))
    same_proto_gap = max(0.0, 0.80 - float(row["track_c_same_prototype_reentry_recovery"]))
    pfr_gap = max(0.0, float(row["track_c_pfr"]) - 2.5)
    return (
        -mismatch_gap,
        -access_gap,
        -same_lineage_gap,
        -same_track_after_gap,
        -same_proto_gap,
        -pfr_gap,
        -float(row["track_c_concept_recovered_but_lineage_mismatch_rate"]),
        float(row["track_c_continuation_bank_access_rate_given_concept_recovery"]),
        float(row["track_c_same_lineage_prototype_reentry_recovery"]),
        float(row["track_c_same_track_after_concept_recovery"]),
        float(row["track_c_same_prototype_reentry_recovery"]),
        -float(row["track_c_pfr"]),
        -float(row["track_c_track_idsw"]),
        -track_a_drop,
        -track_a_same_proto_drop,
        -memory_growth_penalty,
    )


def build_phase3l_eval_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        grouped[str(row["scenario_name"])].append(row)

    output_rows: list[dict[str, Any]] = []
    for scenario_name, rows in sorted(grouped.items()):
        strict_events = [row for row in rows if int(row.get("concept_recovered", 0)) == 1]
        lineage_events = [row for row in rows if int(row.get("same_lineage_id", 0)) == 1]
        binding_events = [
            row
            for row in rows
            if int(row.get("same_lineage_id", 0)) == 1
            and (
                int(row.get("continuation_bank_nonempty", 0)) == 1
                or int(row.get("alive_same_lineage_continuation_count", 0)) > 0
                or int(row.get("lineage_matched_continuation_count", 0)) > 0
            )
        ]
        output_rows.extend(
            [
                {
                    "scenario_name": scenario_name,
                    "eval_mode": "strict",
                    "concept_recovered_events": len(strict_events),
                    "same_prototype_recovery": _mean_int(rows, "same_prototype_id"),
                    "same_track_after_recovery": (
                        sum(int(row.get("same_track", 0)) for row in strict_events) / len(strict_events)
                        if strict_events
                        else 0.0
                    ),
                },
                {
                    "scenario_name": scenario_name,
                    "eval_mode": "lineage",
                    "concept_recovered_events": len(lineage_events),
                    "same_prototype_recovery": _mean_int(rows, "same_lineage_id"),
                    "same_track_after_recovery": (
                        sum(int(row.get("same_track", 0)) for row in lineage_events) / len(lineage_events)
                        if lineage_events
                        else 0.0
                    ),
                },
                {
                    "scenario_name": scenario_name,
                    "eval_mode": "binding",
                    "concept_recovered_events": len(binding_events),
                    "same_prototype_recovery": (
                        len(binding_events) / len(strict_events) if strict_events else 0.0
                    ),
                    "same_track_after_recovery": (
                        sum(int(row.get("same_track", 0)) for row in binding_events) / len(binding_events)
                        if binding_events
                        else 0.0
                    ),
                },
            ]
        )
    return output_rows


def load_phase3x_before_lookup() -> dict[str, dict[str, Any]]:
    before_lookup: dict[str, dict[str, Any]] = {}
    phase3s_rows = load_csv_rows("results/phase3s/phase3s_final_summary_v1.csv")
    for row in phase3s_rows:
        if row.get("method") == "phase3s_current":
            before_lookup[str(row["scenario_name"])] = dict(row)
    summary_path = Path("results/phase3x/phase3x_final_audit_summary_v1.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        track_c = summary.get("track_c", {})
        track_a = summary.get("track_a", {})
        if TRACK_C_NAME in before_lookup:
            before_lookup[TRACK_C_NAME]["same_lineage_prototype_reentry_recovery"] = float(
                summary.get("same_lineage_prototype_reentry_recovery", 0.0)
            )
            before_lookup[TRACK_C_NAME]["same_track_after_lineage_recovery"] = float(
                track_c.get("same_track_after_lineage_recovery", 0.0)
            )
            before_lookup[TRACK_C_NAME]["concept_recovered_but_lineage_mismatch_rate"] = float(
                summary.get("concept_recovered_but_lineage_mismatch_rate", 0.0)
            )
            before_lookup[TRACK_C_NAME]["continuation_bank_access_rate_given_concept_recovery"] = float(
                summary.get("continuation_bank_access_rate_given_concept_recovery", 0.0)
            )
        if TRACK_A_NAME in before_lookup:
            before_lookup[TRACK_A_NAME]["same_lineage_prototype_reentry_recovery"] = float(
                track_a.get("same_lineage_prototype_reentry_recovery", 0.0)
            )
            before_lookup[TRACK_A_NAME]["same_track_after_lineage_recovery"] = float(
                track_a.get("same_track_after_lineage_recovery", 0.0)
            )
    return before_lookup


def plot_lineage_preservation_timeline(
    prototype_rows: list[dict[str, Any]],
    path: str | Path,
    *,
    scenario_name: str = TRACK_C_NAME,
) -> None:
    plot_prototype_lineage_timeline(prototype_rows, path, scenario_name=scenario_name)


def plot_strict_vs_lineage_vs_binding_eval(
    eval_rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    by_scenario: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in eval_rows:
        by_scenario[str(row["scenario_name"])][str(row["eval_mode"])] = row
    scenarios = [TRACK_A_NAME, TRACK_C_NAME]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    metrics = [
        ("same_prototype_recovery", "Prototype / Lineage / Access"),
        ("same_track_after_recovery", "Track after Recovery"),
    ]
    for axis, (metric_key, title) in zip(axes, metrics):
        x = np.arange(len(scenarios))
        strict_vals = [float(by_scenario.get(name, {}).get("strict", {}).get(metric_key, 0.0)) for name in scenarios]
        lineage_vals = [float(by_scenario.get(name, {}).get("lineage", {}).get(metric_key, 0.0)) for name in scenarios]
        binding_vals = [float(by_scenario.get(name, {}).get("binding", {}).get(metric_key, 0.0)) for name in scenarios]
        axis.bar(x - 0.24, strict_vals, width=0.24, label="strict", color="#adb5bd")
        axis.bar(x, lineage_vals, width=0.24, label="lineage", color="#1d3557")
        axis.bar(x + 0.24, binding_vals, width=0.24, label="lineage+binding", color="#2a9d8f")
        axis.set_xticks(x)
        axis.set_xticklabels(scenarios, rotation=14, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(title)
    axes[0].legend(frameon=False)
    figure.suptitle("Phase 3L Strict vs Lineage vs Binding")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_failure_stage_comparison(
    *,
    before_counts: dict[str, int],
    after_trace_rows: list[dict[str, Any]],
    path: str | Path,
    scenario_name: str = TRACK_C_NAME,
) -> None:
    after_counts = Counter(
        str(row.get("failure_stage", "unknown"))
        for row in after_trace_rows
        if str(row.get("scenario_name")) == scenario_name
    )
    labels = sorted(set(before_counts) | set(after_counts))
    before_vals = [int(before_counts.get(label, 0)) for label in labels]
    after_vals = [int(after_counts.get(label, 0)) for label in labels]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(11.8, 4.8))
    axis.bar(x - 0.18, before_vals, width=0.36, label="Phase 3X", color="#adb5bd")
    axis.bar(x + 0.18, after_vals, width=0.36, label="Phase 3L", color="#1d3557")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=18, ha="right")
    axis.set_ylabel("events")
    axis.set_title("Track C Failure Stage Comparison")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_continuation_access_after_concept(
    trace_rows: list[dict[str, Any]],
    path: str | Path,
    *,
    scenario_name: str = TRACK_C_NAME,
) -> None:
    rows = [
        row
        for row in trace_rows
        if str(row.get("scenario_name")) == scenario_name and int(row.get("concept_recovered", 0)) == 1
    ]
    if not rows:
        return
    access = sum(
        int(
            int(row.get("continuation_bank_nonempty", 0)) == 1
            or int(row.get("alive_same_lineage_continuation_count", 0)) > 0
            or int(row.get("lineage_matched_continuation_count", 0)) > 0
        )
        for row in rows
    )
    no_access = len(rows) - access
    same_lineage = sum(int(row.get("same_lineage_id", 0)) for row in rows)
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    axes[0].bar(["access", "no_access"], [access, no_access], color=["#2a9d8f", "#e76f51"])
    axes[0].set_ylabel("events")
    axes[0].set_title("Continuation Access after Concept Recovery")
    axes[1].bar(["same_lineage", "lineage_mismatch"], [same_lineage, len(rows) - same_lineage], color=["#1d3557", "#adb5bd"])
    axes[1].set_title("Lineage Outcome after Concept Recovery")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_phase3l_summary_rows(trace_rows: list[dict[str, Any]], bundle_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_lookup = {str(row["scenario_name"]): dict(row) for row in bundle_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        grouped[str(row["scenario_name"])].append(row)
    output_rows: list[dict[str, Any]] = []
    for scenario_name, base_row in sorted(row_lookup.items()):
        rows = grouped.get(scenario_name, [])
        concept_rows = [row for row in rows if int(row.get("concept_recovered", 0)) == 1]
        lineage_rows = [row for row in rows if int(row.get("same_lineage_id", 0)) == 1]
        access_rows = [
            row
            for row in concept_rows
            if int(row.get("continuation_bank_nonempty", 0)) == 1
            or int(row.get("lineage_matched_continuation_count", 0)) > 0
            or int(row.get("alive_same_lineage_continuation_count", 0)) > 0
        ]
        base_row["continuation_bank_access_rate_given_concept_recovery"] = (
            len(access_rows) / len(concept_rows) if concept_rows else 0.0
        )
        base_row["new_track_with_old_lineage_rate"] = (
            sum(
                int(
                    int(row.get("new_track_created", 0)) == 1 and int(row.get("same_lineage_id", 0)) == 1
                )
                for row in rows
            )
            / len(lineage_rows)
            if lineage_rows
            else 0.0
        )
        output_rows.append(base_row)
    return output_rows


def _mean_int(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(int(row.get(key, 0)) for row in rows) / len(rows)) if rows else 0.0
