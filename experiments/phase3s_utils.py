"""Shared helpers for Phase 3S prototype-centric identity continuation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.phase3r2_utils import load_csv_rows
from experiments.phase3r_utils import (
    GAP_BUCKETS,
    evaluate_phase3_scenarios,
    extract_reentry_events,
    summarize_gap_buckets,
    summarize_reentry_events,
)

TRACK_A_NAME = "track_a_bridge"
TRACK_C_NAME = "track_c_long_horizon"
PHASE3S_SCENARIOS = [TRACK_A_NAME, TRACK_C_NAME]


def default_phase3s_tracking_override() -> dict[str, Any]:
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


def default_phase3s_memory_override() -> dict[str, Any]:
    return {
        "protect_linked_prototypes": True,
        "decay_patience": 24,
        "enable_continuation_bank": True,
        "continuation_topk_per_proto": 4,
        "min_track_age_for_continuation": 4,
        "min_hits_for_continuation": 3,
        "continuation_max_gap": 96,
        "continuation_decay": 0.01,
    }


def evaluate_phase3s_bundle(
    config_path: str | Path,
    *,
    tracking_override: dict[str, Any] | None = None,
    memory_override: dict[str, Any] | None = None,
    seed: int = 42,
    scenario_names: list[str] | None = None,
    frame_record_mode: str = "lite",
) -> dict[str, Any]:
    merged_tracking = default_phase3s_tracking_override()
    if tracking_override:
        merged_tracking.update(tracking_override)
    merged_memory = default_phase3s_memory_override()
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
        row = {
            "scenario_name": run["scenario_name"],
            "u_recall": float(run["result"].summary.u_recall),
            "same_track_reentry_recovery": float(reentry["same_track_reentry_recovery"]),
            "same_prototype_reentry_recovery": float(reentry["same_prototype_reentry_recovery"]),
            "same_track_after_concept_recovery": float(reentry["same_track_after_concept_recovery"]),
            "continuation_bank_nonempty_rate": float(reentry["continuation_bank_nonempty_rate"]),
            "candidate_pool_nonempty_rate": float(reentry["candidate_pool_nonempty_rate"]),
            "continuation_attempt_rate": float(reentry["continuation_attempt_rate"]),
            "continuation_success_rate": float(reentry["continuation_success_rate"]),
            "new_track_with_old_prototype_rate": float(reentry["new_track_with_old_prototype_rate"]),
            "prototype_gated_resurrection_attempt_rate": float(
                reentry["prototype_gated_resurrection_attempt_rate"]
            ),
            "resurrection_success_given_candidate_exists": float(
                reentry["resurrection_success_given_candidate_exists"]
            ),
            "candidate_exists_events": int(reentry["candidate_exists_events"]),
            "concept_recovered_events": int(reentry["concept_recovered_events"]),
            "mean_candidate_pool_size": float(reentry["mean_candidate_pool_size"]),
            "proposal_detect_rate": float(reentry["proposal_detect_rate"]),
            "pfr": float(run["result"].summary.pfr),
            "track_idsw": int(run["result"].primary_monitoring["track_idsw"]),
            "memory_growth": float(run["result"].summary.memory_growth),
            "reentry_events": int(reentry["num_events"]),
            "reactivation_successes": int(run["result"].primary_monitoring["reactivation_successes"]),
            "created_tracks": int(run["result"].primary_monitoring["created_tracks"]),
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
        "gap_rows": summarize_gap_buckets(all_events),
    }


def load_phase3r3_before_lookup(
    path: str | Path = "results/phase3r3/phase3r3_final_summary_v1.csv",
) -> dict[str, dict[str, Any]]:
    rows = load_csv_rows(path)
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("method") != "phase3r3_current":
            continue
        lookup[str(row["scenario_name"])] = row
    audit_path = Path("results/phase3r3/reentry_audit_summary_v3.json")
    if audit_path.exists():
        audit_rows = json.loads(audit_path.read_text(encoding="utf-8"))
        for audit_row in audit_rows:
            scenario_name = str(audit_row["scenario_name"])
            row = lookup.get(scenario_name)
            if row is None:
                continue
            row["candidate_pool_nonempty_rate"] = float(audit_row.get("candidate_pool_nonempty_rate", 0.0))
            row["mean_candidate_pool_size"] = float(audit_row.get("mean_candidate_pool_size", 0.0))
            row["candidate_exists_events"] = int(audit_row.get("candidate_exists_events", 0))
    return lookup


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
    cont_bank_gap = max(0.0, 0.60 - float(row["track_c_continuation_bank_nonempty_rate"]))
    candidate_pool_gap = max(0.0, 0.60 - float(row["track_c_candidate_pool_nonempty_rate"]))
    same_track_after_gap = max(0.0, 0.45 - float(row["track_c_same_track_after_concept_recovery"]))
    same_track_gap = max(0.0, 0.30 - float(row["track_c_same_track_reentry_recovery"]))
    same_proto_gap = max(0.0, 0.80 - float(row["track_c_same_prototype_reentry_recovery"]))
    pfr_gap = max(0.0, float(row["track_c_pfr"]) - 2.5)
    return (
        -cont_bank_gap,
        -candidate_pool_gap,
        -same_track_after_gap,
        -same_track_gap,
        -same_proto_gap,
        -pfr_gap,
        float(row["track_c_continuation_bank_nonempty_rate"]),
        float(row["track_c_candidate_pool_nonempty_rate"]),
        float(row["track_c_same_track_after_concept_recovery"]),
        float(row["track_c_same_track_reentry_recovery"]),
        float(row["track_c_same_prototype_reentry_recovery"]),
        -float(row["track_c_pfr"]),
        -float(row["track_c_track_idsw"]),
        -track_a_drop,
        -track_a_same_proto_drop,
        -memory_growth_penalty,
    )


def coerce_reentry_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    int_keys = {
        "event_id",
        "instance_id",
        "old_track_id",
        "old_prototype_id",
        "disappear_frame",
        "reappear_frame",
        "gap_length",
        "proposal_detected",
        "matched_same_track",
        "matched_same_prototype",
        "new_track_created",
        "new_prototype_created",
        "reactivation_attempted",
        "concept_only_recovery",
        "concept_recovered",
        "same_track_after_concept_recovery",
        "idsw_after_reentry_window",
        "candidate_pool_size",
        "live_candidate_pool_size",
        "continuation_bank_size",
        "candidate_pool_nonempty",
        "resurrection_attempted",
        "resurrection_success",
        "best_candidate_gap",
        "continuation_attempted",
        "continuation_success",
        "best_continuation_gap",
        "best_continuation_age",
        "resurrected_from_continuation",
    }
    float_keys = {
        "reactivation_cost",
        "prototype_similarity",
        "position_error",
        "objectness_at_reentry",
        "resurrection_cost_best",
        "best_continuation_cost",
    }
    coerced: list[dict[str, Any]] = []
    for row in rows:
        parsed = dict(row)
        for key in int_keys:
            value = parsed.get(key)
            parsed[key] = 0 if value in ("", None) else int(float(value))
        for key in float_keys:
            value = parsed.get(key)
            parsed[key] = None if value in ("", None) else float(value)
        coerced.append(parsed)
    return coerced


def plot_reentry_vs_gap_v4(
    gap_rows: list[dict[str, Any]],
    path: str | Path,
    *,
    scenario_name: str = TRACK_C_NAME,
) -> None:
    rows = [row for row in gap_rows if row["scenario_name"] == scenario_name]
    order = [name for name, _, _ in GAP_BUCKETS]
    row_map = {str(row["gap_bucket"]): row for row in rows}

    x = np.arange(len(order))
    same_track = [float(row_map.get(bucket, {}).get("same_track_recovery_rate", 0.0)) for bucket in order]
    same_proto = [float(row_map.get(bucket, {}).get("same_prototype_recovery_rate", 0.0)) for bucket in order]
    same_track_after = [
        float(row_map.get(bucket, {}).get("same_track_after_concept_recovery_rate", 0.0)) for bucket in order
    ]

    figure, axis = plt.subplots(figsize=(10.2, 4.8))
    axis.plot(x, same_track, marker="o", linewidth=1.8, label="same-track", color="#1d3557")
    axis.plot(x, same_proto, marker="o", linewidth=1.8, label="same-prototype", color="#2a9d8f")
    axis.plot(x, same_track_after, marker="o", linewidth=1.8, label="track-after-concept", color="#e76f51")
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=18, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("recovery rate")
    axis.set_title("Phase 3S Re-entry Recovery vs Gap Length")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_continuation_bank_nonempty(
    events: list[dict[str, Any]],
    path: str | Path,
    *,
    scenario_name: str = TRACK_C_NAME,
) -> None:
    rows = [row for row in events if row["scenario_name"] == scenario_name and int(row["concept_recovered"]) == 1]
    concept_events = max(1, len(rows))

    live_nonempty_rate = sum(int(int(row.get("live_candidate_pool_size", 0)) > 0) for row in rows) / concept_events
    continuation_nonempty_rate = sum(
        int(int(row.get("continuation_bank_size", 0)) > 0) for row in rows
    ) / concept_events
    combined_nonempty_rate = sum(int(row.get("candidate_pool_nonempty", 0)) for row in rows) / concept_events
    bank_sizes = [int(row.get("continuation_bank_size", 0)) for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    axes[0].bar(
        ["live pool", "continuation bank", "combined"],
        [live_nonempty_rate, continuation_nonempty_rate, combined_nonempty_rate],
        color=["#457b9d", "#2a9d8f", "#e76f51"],
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("nonempty rate")
    axes[0].set_title("Candidate Coverage in Concept-Recovered Events")

    if bank_sizes:
        max_size = max(bank_sizes)
        bins = np.arange(0, max_size + 2) - 0.5
        axes[1].hist(bank_sizes, bins=bins, color="#6c757d", edgecolor="white")
        axes[1].set_xticks(range(0, max_size + 1))
    axes[1].set_xlabel("continuation bank size")
    axes[1].set_ylabel("events")
    axes[1].set_title("Continuation Bank Size Distribution")

    figure.suptitle("Phase 3S Continuation Bank Coverage")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_continuation_timeline(
    frame_logs: list[dict[str, Any]],
    path: str | Path,
    *,
    scenario_name: str = TRACK_C_NAME,
) -> None:
    rows = [row for row in frame_logs if row["scenario_name"] == scenario_name]
    rows.sort(key=lambda row: int(row["frame_id"]))
    frames = [int(row["frame_id"]) for row in rows]
    active = [int(row["num_active_tracks"]) for row in rows]
    dormant = [int(row["num_dormant_tracks"]) for row in rows]
    ghost = [int(row.get("num_ghost_tracks", 0)) for row in rows]
    retired = [int(row["num_retired_tracks"]) for row in rows]
    continuations = [int(row.get("num_continuations", 0)) for row in rows]
    archived = [int(row.get("continuation_archive_events", 0)) for row in rows]
    attempts = [int(row.get("continuation_resurrection_attempts", 0)) for row in rows]
    successes = [int(row.get("continuation_resurrection_successes", 0)) for row in rows]

    figure, axes = plt.subplots(2, 1, figsize=(11.4, 6.6), sharex=True)
    axes[0].plot(frames, active, label="active", color="#1d3557", linewidth=1.5)
    axes[0].plot(frames, dormant, label="dormant", color="#2a9d8f", linewidth=1.5)
    axes[0].plot(frames, ghost, label="ghost", color="#e9c46a", linewidth=1.5)
    axes[0].plot(frames, retired, label="retired", color="#e76f51", linewidth=1.5)
    axes[0].plot(frames, continuations, label="continuations", color="#6f42c1", linewidth=1.5)
    axes[0].set_ylabel("count")
    axes[0].set_title("Track / Continuation State Timeline")
    axes[0].legend(frameon=False, ncol=5)

    axes[1].plot(frames, archived, label="continuation archive", color="#6c757d", linewidth=1.4)
    axes[1].plot(frames, attempts, label="continuation attempts", color="#0d6efd", linewidth=1.4)
    axes[1].plot(frames, successes, label="continuation successes", color="#198754", linewidth=1.4)
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("events")
    axes[1].set_title("Continuation Archive / Resurrection Flow")
    axes[1].legend(frameon=False)

    figure.suptitle("Phase 3S Continuation Timeline")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_track_c_before_after_v4(
    before: dict[str, Any],
    after: dict[str, Any],
    path: str | Path,
) -> None:
    labels = [
        "same-track",
        "same-prototype",
        "track-after-concept",
        "candidate-pool",
        "PFR",
        "IDSW",
    ]
    before_values = [
        float(before["same_track_reentry_recovery"]),
        float(before["same_prototype_reentry_recovery"]),
        float(before["same_track_after_concept_recovery"]),
        float(before.get("candidate_pool_nonempty_rate", 0.0)),
        float(before["pfr"]),
        float(before["track_idsw"]),
    ]
    after_values = [
        float(after["same_track_reentry_recovery"]),
        float(after["same_prototype_reentry_recovery"]),
        float(after["same_track_after_concept_recovery"]),
        float(after["candidate_pool_nonempty_rate"]),
        float(after["pfr"]),
        float(after["track_idsw"]),
    ]

    x = np.arange(len(labels))
    width = 0.38

    figure, axis = plt.subplots(figsize=(10.8, 4.8))
    axis.bar(x - width / 2, before_values, width, label="Phase 3R.3", color="#adb5bd")
    axis.bar(x + width / 2, after_values, width, label="Phase 3S", color="#1d3557")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=16, ha="right")
    axis.set_title("Track C Before / After: Phase 3R.3 vs Phase 3S")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
