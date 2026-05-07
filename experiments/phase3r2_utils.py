"""Shared helpers for Phase 3R.2 concept-gated old-track resurrection."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.phase3r_utils import (
    GAP_BUCKETS,
    evaluate_phase3_scenarios,
    extract_reentry_events,
    summarize_gap_buckets,
    summarize_reentry_events,
)

TRACK_A_NAME = "track_a_bridge"
TRACK_C_NAME = "track_c_long_horizon"
PHASE3R2_SCENARIOS = [TRACK_A_NAME, TRACK_C_NAME]


def default_phase3r2_tracking_override() -> dict[str, Any]:
    return {
        "keepalive_frames": 8,
        "dormant_frames": 24,
        "ghost_frames": 64,
        "tau_g": 12,
        "tau_res_short": 0.60,
        "tau_res_long": 0.72,
    }


def default_phase3r2_memory_override() -> dict[str, Any]:
    return {
        "protect_linked_prototypes": True,
        "decay_patience": 24,
    }


def evaluate_phase3r2_bundle(
    config_path: str | Path,
    *,
    tracking_override: dict[str, Any] | None = None,
    memory_override: dict[str, Any] | None = None,
    seed: int = 42,
    scenario_names: list[str] | None = None,
) -> dict[str, Any]:
    merged_tracking = default_phase3r2_tracking_override()
    if tracking_override:
        merged_tracking.update(tracking_override)
    merged_memory = default_phase3r2_memory_override()
    if memory_override:
        merged_memory.update(memory_override)

    runs = evaluate_phase3_scenarios(
        config_path,
        tracking_override=merged_tracking,
        memory_override=merged_memory,
        scenario_names=scenario_names or PHASE3R2_SCENARIOS,
        collect_frames=True,
        frame_record_mode="lite",
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
            "prototype_gated_resurrection_attempt_rate": float(reentry["prototype_gated_resurrection_attempt_rate"]),
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


def load_phase3r_before_lookup(path: str | Path = "results/phase3r/phase3r_final_summary_v1.csv") -> dict[str, dict[str, Any]]:
    rows = load_csv_rows(path)
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("method") != "phase3r_current":
            continue
        lookup[str(row["scenario_name"])] = row
    return lookup


def pick_best_scan_row(rows: list[dict[str, Any]], before_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: _scan_sort_key(row, before_lookup))


def _scan_sort_key(row: dict[str, Any], before_lookup: dict[str, dict[str, Any]]) -> tuple[float, ...]:
    before_a = before_lookup.get(TRACK_A_NAME, {})
    track_a_drop = max(0.0, float(before_a.get("u_recall", 0.0)) - float(row.get("track_a_u_recall", 0.0)))
    memory_growth_penalty = max(0.0, float(row.get("track_a_memory_growth", 0.0)) - float(before_a.get("memory_growth", 0.0)))
    same_proto_gap = max(0.0, 0.80 - float(row["track_c_same_prototype_reentry_recovery"]))
    pfr_gap = max(0.0, float(row["track_c_pfr"]) - 2.0)
    same_track_gap = max(0.0, 0.35 - float(row["track_c_same_track_reentry_recovery"]))
    same_track_after_gap = max(0.0, 0.50 - float(row["track_c_same_track_after_concept_recovery"]))
    return (
        -same_proto_gap,
        -same_track_after_gap,
        -same_track_gap,
        -pfr_gap,
        float(row["track_c_same_track_after_concept_recovery"]),
        float(row["track_c_same_track_reentry_recovery"]),
        float(row["track_c_same_prototype_reentry_recovery"]),
        -float(row["track_c_pfr"]),
        -float(row["track_c_track_idsw"]),
        -track_a_drop,
        -memory_growth_penalty,
    )


def load_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_reentry_vs_gap(gap_rows: list[dict[str, Any]], path: str | Path, *, scenario_name: str = TRACK_C_NAME) -> None:
    rows = [row for row in gap_rows if row["scenario_name"] == scenario_name]
    order = [name for name, _, _ in GAP_BUCKETS]
    row_map = {str(row["gap_bucket"]): row for row in rows}

    x = np.arange(len(order))
    same_track = [float(row_map.get(bucket, {}).get("same_track_recovery_rate", 0.0)) for bucket in order]
    same_proto = [float(row_map.get(bucket, {}).get("same_prototype_recovery_rate", 0.0)) for bucket in order]
    same_track_after = [
        float(row_map.get(bucket, {}).get("same_track_after_concept_recovery_rate", 0.0)) for bucket in order
    ]

    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(x, same_track, marker="o", linewidth=1.8, label="same-track", color="#1d3557")
    axis.plot(x, same_proto, marker="o", linewidth=1.8, label="same-prototype", color="#2a9d8f")
    axis.plot(x, same_track_after, marker="o", linewidth=1.8, label="track-after-concept", color="#e76f51")
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=18, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Re-entry Recovery vs Gap Length")
    axis.set_ylabel("recovery rate")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_candidate_pool(events: list[dict[str, Any]], path: str | Path, *, scenario_name: str = TRACK_C_NAME) -> None:
    rows = [row for row in events if row["scenario_name"] == scenario_name and int(row["concept_recovered"]) == 1]
    concept_pool_sizes = [int(row.get("candidate_pool_size", 0)) for row in rows]
    size_counts: dict[int, int] = {}
    for size in concept_pool_sizes:
        size_counts[size] = size_counts.get(size, 0) + 1

    state_counts: dict[str, int] = {}
    for row in rows:
        state = str(row.get("best_candidate_state") or "none")
        state_counts[state] = state_counts.get(state, 0) + 1

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    sizes = sorted(size_counts)
    axes[0].bar([str(size) for size in sizes], [size_counts[size] for size in sizes], color="#457b9d")
    axes[0].set_title("Candidate Pool Size")
    axes[0].set_xlabel("pool size")
    axes[0].set_ylabel("events")

    states = list(state_counts.keys())
    colors = ["#2a9d8f", "#e9c46a", "#e76f51", "#9d4edd"]
    axes[1].bar(states, [state_counts[state] for state in states], color=colors[: len(states)])
    axes[1].set_title("Best Candidate State")
    axes[1].set_xlabel("state")
    axes[1].set_ylabel("events")

    figure.suptitle("Prototype-Gated Resurrection Candidate Pool")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_track_state_timeline(
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

    figure, axis = plt.subplots(figsize=(11.2, 5.0))
    axis.plot(frames, active, label="active", color="#1d3557", linewidth=1.6)
    axis.plot(frames, dormant, label="dormant", color="#2a9d8f", linewidth=1.6)
    axis.plot(frames, ghost, label="ghost", color="#e9c46a", linewidth=1.6)
    axis.plot(frames, retired, label="retired", color="#e76f51", linewidth=1.6)
    axis.set_title("Track State Timeline")
    axis.set_xlabel("frame")
    axis.set_ylabel("track count")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_track_c_before_after(before_row: dict[str, Any], after_row: dict[str, Any], path: str | Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    recovery_metrics = [
        ("same_track_reentry_recovery", "same-track"),
        ("same_prototype_reentry_recovery", "same-prototype"),
        ("same_track_after_concept_recovery", "track-after-concept"),
    ]
    failure_metrics = [
        ("pfr", "PFR"),
        ("track_idsw", "IDSW"),
    ]

    x_left = np.arange(len(recovery_metrics))
    x_right = np.arange(len(failure_metrics))
    width = 0.34

    axes[0].bar(
        x_left - width / 2,
        [float(before_row[key]) for key, _ in recovery_metrics],
        width=width,
        color="#adb5bd",
        label="phase3r",
    )
    axes[0].bar(
        x_left + width / 2,
        [float(after_row[key]) for key, _ in recovery_metrics],
        width=width,
        color="#2a9d8f",
        label="phase3r2",
    )
    axes[0].set_xticks(x_left)
    axes[0].set_xticklabels([label for _, label in recovery_metrics], rotation=12, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Recovery")
    axes[0].legend(frameon=False)

    axes[1].bar(
        x_right - width / 2,
        [float(before_row[key]) for key, _ in failure_metrics],
        width=width,
        color="#adb5bd",
        label="phase3r",
    )
    axes[1].bar(
        x_right + width / 2,
        [float(after_row[key]) for key, _ in failure_metrics],
        width=width,
        color="#e76f51",
        label="phase3r2",
    )
    axes[1].set_xticks(x_right)
    axes[1].set_xticklabels([label for _, label in failure_metrics])
    axes[1].set_title("Failure Cost")

    figure.suptitle("Track C Before / After Phase 3R.2")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
