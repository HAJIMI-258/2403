"""Shared helpers for Phase 3X prototype-lineage audit."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.phase3r2_utils import load_csv_rows
from experiments.phase3r_utils import (
    GAP_BUCKETS,
    evaluate_phase3_scenarios,
    extract_reentry_events,
    summarize_reentry_events,
    write_csv,
)
from experiments.phase3s_utils import (
    PHASE3S_SCENARIOS,
    TRACK_A_NAME,
    TRACK_C_NAME,
    default_phase3s_memory_override,
    default_phase3s_tracking_override,
)


def default_phase3x_tracking_override() -> dict[str, Any]:
    return default_phase3s_tracking_override()


def default_phase3x_memory_override() -> dict[str, Any]:
    return default_phase3s_memory_override()


def evaluate_phase3x_runs(
    config_path: str | Path,
    *,
    tracking_override: dict[str, Any] | None = None,
    memory_override: dict[str, Any] | None = None,
    seed: int = 42,
    scenario_names: list[str] | None = None,
) -> dict[str, Any]:
    merged_tracking = default_phase3x_tracking_override()
    if tracking_override:
        merged_tracking.update(tracking_override)
    merged_memory = default_phase3x_memory_override()
    if memory_override:
        merged_memory.update(memory_override)

    runs = evaluate_phase3_scenarios(
        config_path,
        tracking_override=merged_tracking,
        memory_override=merged_memory,
        scenario_names=scenario_names or [TRACK_C_NAME],
        collect_frames=True,
        frame_record_mode="full",
        seed=seed,
    )
    return {
        "runs": runs,
        "tracking_override": merged_tracking,
        "memory_override": merged_memory,
    }


def collect_phase3x_audit_rows(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    continuation_write_rows: list[dict[str, Any]] = []
    continuation_lifecycle_rows: list[dict[str, Any]] = []
    prototype_lineage_rows: list[dict[str, Any]] = []

    for run in runs:
        scenario_name = str(run["scenario_name"])
        events, _ = extract_reentry_events(scenario_name, run["sequence"], run["result"])
        event_rows.extend(events)
        for frame_record in run["result"].frame_records:
            memory_output = frame_record.memory_output
            for row in getattr(memory_output, "continuation_write_rows", []):
                continuation_write_rows.append({"scenario_name": scenario_name, **dict(row)})
            for row in getattr(memory_output, "continuation_lifecycle_rows", []):
                continuation_lifecycle_rows.append({"scenario_name": scenario_name, **dict(row)})
            for row in getattr(memory_output, "prototype_lineage_rows", []):
                prototype_lineage_rows.append({"scenario_name": scenario_name, **dict(row)})
    return event_rows, continuation_write_rows, continuation_lifecycle_rows, prototype_lineage_rows


def build_phase3x_event_trace(
    event_rows: list[dict[str, Any]],
    continuation_write_rows: list[dict[str, Any]],
    continuation_lifecycle_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    writes_by_track: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    lifecycle_by_uid: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in continuation_write_rows:
        if not row.get("continuation_uid"):
            continue
        writes_by_track[(str(row["scenario_name"]), int(row["track_id"]))].append(row)
    for row in continuation_lifecycle_rows:
        lifecycle_by_uid[(str(row["scenario_name"]), str(row["continuation_uid"]))].append(row)
    for rows in lifecycle_by_uid.values():
        rows.sort(key=lambda item: int(item["frame_index"]))

    trace_rows: list[dict[str, Any]] = []
    for event in sorted(event_rows, key=lambda item: (str(item["scenario_name"]), int(item["event_id"]))):
        scenario_name = str(event["scenario_name"])
        old_track_id = int(event["old_track_id"]) if event.get("old_track_id") not in ("", None) else -1
        old_lineage_id = _maybe_int(event.get("old_lineage_id"))
        reappear_frame = int(event["reappear_frame"])

        write_candidates = [
            row
            for row in writes_by_track.get((scenario_name, old_track_id), [])
            if int(row.get("write_success", 0)) == 1 and int(row.get("write_frame", 0)) < reappear_frame
        ]
        continuation_written_before = bool(write_candidates)
        continuation_uids = [str(row["continuation_uid"]) for row in write_candidates if row.get("continuation_uid")]

        alive_rows: list[dict[str, Any]] = []
        for continuation_uid in continuation_uids:
            history = lifecycle_by_uid.get((scenario_name, continuation_uid), [])
            if not history:
                continue
            eligible = [row for row in history if int(row["frame_index"]) <= reappear_frame]
            if not eligible:
                continue
            latest = eligible[-1]
            if int(latest.get("is_alive", 0)) == 1:
                alive_rows.append(latest)

        continuation_alive_at_reentry = bool(alive_rows)
        continuation_owner_prototype_id = (
            int(min(alive_rows, key=lambda item: int(item["age_since_last_seen"]))["current_owner_prototype_id"])
            if alive_rows
            else None
        )
        continuation_owner_lineage_id = (
            int(min(alive_rows, key=lambda item: int(item["age_since_last_seen"]))["current_owner_lineage_id"])
            if alive_rows
            else None
        )
        best_alive_row = min(alive_rows, key=lambda item: int(item["age_since_last_seen"])) if alive_rows else None
        alive_same_lineage_count = sum(
            int(old_lineage_id is not None and _maybe_int(row.get("current_owner_lineage_id")) == old_lineage_id)
            for row in alive_rows
        )
        alive_same_prototype_count = sum(
            int(_maybe_int(row.get("current_owner_prototype_id")) == _maybe_int(event.get("matched_prototype_id")))
            for row in alive_rows
        )

        trace_row = {
            "event_id": int(event["event_id"]),
            "scenario_name": scenario_name,
            "instance_id": int(event["instance_id"]),
            "old_track_id": old_track_id,
            "old_prototype_id": _maybe_int(event.get("old_prototype_id")),
            "old_lineage_id": old_lineage_id,
            "reappear_frame": reappear_frame,
            "proposal_detected": int(event["proposal_detected"]),
            "concept_recovered": int(event["matched_same_prototype"]),
            "matched_prototype_id": _maybe_int(event.get("matched_prototype_id")),
            "matched_lineage_id": _maybe_int(event.get("matched_lineage_id")),
            "same_prototype_id": int(event.get("same_prototype_id", event["matched_same_prototype"])),
            "same_lineage_id": int(event.get("same_lineage_id", event.get("matched_same_lineage_prototype", 0))),
            "continuation_written_before": int(continuation_written_before),
            "continuation_alive_at_reentry": int(continuation_alive_at_reentry),
            "continuation_owner_prototype_id": continuation_owner_prototype_id,
            "continuation_owner_lineage_id": continuation_owner_lineage_id,
            "continuation_bank_nonempty": int(int(event.get("continuation_bank_size", 0)) > 0),
            "continuation_attempted": int(event.get("continuation_attempted", 0)),
            "continuation_success": int(event.get("continuation_success", 0)),
            "same_track": int(event["matched_same_track"]),
            "new_track_created": int(event["new_track_created"]),
            "new_prototype_created": int(event["new_prototype_created"]),
            "prototype_matched_continuation_count": int(event.get("prototype_matched_continuation_count", 0)),
            "lineage_matched_continuation_count": int(event.get("lineage_matched_continuation_count", 0)),
            "alive_same_lineage_continuation_count": int(alive_same_lineage_count),
            "alive_same_prototype_continuation_count": int(alive_same_prototype_count),
            "best_continuation_uid": None if best_alive_row is None else str(best_alive_row["continuation_uid"]),
            "best_continuation_age": None if best_alive_row is None else int(best_alive_row["age_since_last_seen"]),
        }
        trace_row["failure_stage"] = classify_phase3x_failure_stage(trace_row)
        trace_rows.append(trace_row)

    return trace_rows


def classify_phase3x_failure_stage(row: dict[str, Any]) -> str:
    if int(row.get("proposal_detected", 0)) == 0:
        return "proposal_missing"
    if int(row.get("same_track", 0)) == 1:
        return "track_restored"
    if int(row.get("same_lineage_id", 0)) == 1 and int(row.get("new_track_created", 0)) == 1:
        return "new_track_old_lineage"
    if int(row.get("new_track_created", 0)) == 1 and int(row.get("same_lineage_id", 0)) == 0:
        return "new_track_new_lineage"
    if _maybe_int(row.get("matched_prototype_id")) is None:
        return "concept_not_recovered"
    if int(row.get("same_lineage_id", 0)) == 0:
        return "lineage_mismatch"
    if int(row.get("continuation_written_before", 0)) == 0:
        return "continuation_missing"
    if int(row.get("continuation_alive_at_reentry", 0)) == 0:
        return "continuation_dead"
    if int(row.get("continuation_attempted", 0)) == 0:
        return "continuation_not_attempted"
    if int(row.get("continuation_success", 0)) == 0:
        return "continuation_failed"
    return "track_restored"


def build_lineage_eval_rows(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[str(row["scenario_name"])].append(row)

    output_rows: list[dict[str, Any]] = []
    for scenario_name, rows in sorted(grouped.items()):
        strict_concept_events = sum(int(row.get("concept_recovered", 0)) for row in rows)
        lineage_concept_events = sum(int(row.get("same_lineage_id", 0)) for row in rows)
        output_rows.append(
            {
                "scenario_name": scenario_name,
                "eval_mode": "strict",
                "reentry_events": len(rows),
                "concept_recovered_events": strict_concept_events,
                "same_track_reentry_recovery": _mean_int(rows, "same_track"),
                "same_prototype_recovery": _mean_int(rows, "same_prototype_id"),
                "same_track_after_recovery": (
                    sum(int(row.get("same_track", 0)) for row in rows if int(row.get("concept_recovered", 0)) == 1)
                    / strict_concept_events
                    if strict_concept_events
                    else 0.0
                ),
            }
        )
        output_rows.append(
            {
                "scenario_name": scenario_name,
                "eval_mode": "lineage",
                "reentry_events": len(rows),
                "concept_recovered_events": lineage_concept_events,
                "same_track_reentry_recovery": _mean_int(rows, "same_track"),
                "same_prototype_recovery": _mean_int(rows, "same_lineage_id"),
                "same_track_after_recovery": (
                    sum(int(row.get("same_track", 0)) for row in rows if int(row.get("same_lineage_id", 0)) == 1)
                    / lineage_concept_events
                    if lineage_concept_events
                    else 0.0
                ),
            }
        )
    return output_rows


def summarize_phase3x_audit(
    trace_rows: list[dict[str, Any]],
    continuation_write_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        by_scenario[str(row["scenario_name"])].append(row)

    unique_write_attempts: dict[tuple[str, int, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in continuation_write_rows:
        unique_write_attempts[
            (
                str(row["scenario_name"]),
                int(row["track_id"]),
                _maybe_int(row.get("source_prototype_id")),
            )
        ].append(row)
    eligible_attempt_groups = []
    for rows in unique_write_attempts.values():
        eligible = [row for row in rows if str(row.get("write_reason")) not in {"missing_prototype_ref", "owner_prototype_missing"}]
        if eligible:
            eligible_attempt_groups.append(eligible)

    continuation_write_success_rate = (
        sum(int(any(int(row.get("write_success", 0)) == 1 for row in rows)) for rows in eligible_attempt_groups)
        / len(eligible_attempt_groups)
        if eligible_attempt_groups
        else 0.0
    )

    summary: dict[str, Any] = {
        "continuation_write_success_rate": float(continuation_write_success_rate),
        "scenarios": {},
    }

    track_c_rows = by_scenario.get(TRACK_C_NAME, [])
    strict_concept_rows = [row for row in track_c_rows if int(row["concept_recovered"]) == 1]
    lineage_rows = [row for row in track_c_rows if int(row["same_lineage_id"]) == 1]
    written_rows = [row for row in track_c_rows if int(row["continuation_written_before"]) == 1]
    mismatch_base = [
        row
        for row in track_c_rows
        if int(row["proposal_detected"]) == 1
        and _maybe_int(row.get("matched_prototype_id")) is not None
        and int(row["new_prototype_created"]) == 0
    ]

    summary.update(
        {
            "continuation_survival_until_concept_recovery_rate": (
                sum(int(row["continuation_alive_at_reentry"]) for row in written_rows) / len(written_rows)
                if written_rows
                else 0.0
            ),
            "concept_recovered_but_lineage_mismatch_rate": (
                sum(int(int(row["same_lineage_id"]) == 0) for row in mismatch_base) / len(mismatch_base)
                if mismatch_base
                else 0.0
            ),
            "same_lineage_prototype_reentry_recovery": _mean_int(track_c_rows, "same_lineage_id"),
            "continuation_bank_access_rate_given_concept_recovery": (
                sum(int(row["continuation_bank_nonempty"]) for row in strict_concept_rows) / len(strict_concept_rows)
                if strict_concept_rows
                else 0.0
            ),
            "continuation_bank_access_rate_given_same_lineage": (
                sum(
                    int(
                        int(row["continuation_bank_nonempty"]) == 1
                        or int(row["alive_same_lineage_continuation_count"]) > 0
                    )
                    for row in lineage_rows
                )
                / len(lineage_rows)
                if lineage_rows
                else 0.0
            ),
        }
    )

    failure_stage_counts = Counter(str(row["failure_stage"]) for row in track_c_rows)
    summary["track_c_failure_stage_counts"] = dict(sorted(failure_stage_counts.items()))
    summary["track_c_failure_stage_ratio"] = {
        stage: count / len(track_c_rows) if track_c_rows else 0.0
        for stage, count in sorted(failure_stage_counts.items())
    }

    summary["track_c"] = _scenario_summary(track_c_rows)
    summary["track_a"] = _scenario_summary(by_scenario.get(TRACK_A_NAME, []))
    summary["dominant_failure_stage"] = (
        max(failure_stage_counts.items(), key=lambda item: item[1])[0]
        if failure_stage_counts
        else "unknown"
    )
    summary["primary_loss_stage"] = infer_primary_loss_stage(summary, track_c_rows)
    return summary


def infer_primary_loss_stage(summary: dict[str, Any], track_c_rows: list[dict[str, Any]]) -> str:
    same_lineage = float(summary.get("same_lineage_prototype_reentry_recovery", 0.0))
    strict_same_proto = float(summary.get("track_c", {}).get("same_prototype_reentry_recovery", 0.0))
    write_rate = float(summary.get("continuation_write_success_rate", 0.0))
    survival_rate = float(summary.get("continuation_survival_until_concept_recovery_rate", 0.0))
    access_same_lineage = float(summary.get("continuation_bank_access_rate_given_same_lineage", 0.0))

    if same_lineage - strict_same_proto > 0.15:
        return "lineage_or_evaluation_interface"
    if write_rate < 0.50:
        return "before_archive"
    if survival_rate < 0.50:
        return "after_archive"
    if access_same_lineage < 0.50:
        return "concept_to_continuation_binding"
    return "prototype_update_or_merge_path"


def load_phase3s_before_lookup(
    path: str | Path = "results/phase3s/phase3s_final_summary_v1.csv",
) -> dict[str, dict[str, Any]]:
    rows = load_csv_rows(path)
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("method") != "phase3s_current":
            continue
        lookup[str(row["scenario_name"])] = row
    return lookup


def write_phase3x_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_prototype_lineage_timeline(
    prototype_rows: list[dict[str, Any]],
    path: str | Path,
    *,
    scenario_name: str = TRACK_C_NAME,
) -> None:
    rows = [row for row in prototype_rows if row["scenario_name"] == scenario_name]
    rows.sort(key=lambda row: (int(row["frame_index"]), int(row["prototype_id"])))
    if not rows:
        return

    frames = np.array([int(row["frame_index"]) for row in rows], dtype=np.int32)
    proto_ids = np.array([int(row["prototype_id"]) for row in rows], dtype=np.int32)
    lineage_ids = np.array([int(row["lineage_id"]) for row in rows], dtype=np.int32)

    figure, axes = plt.subplots(2, 1, figsize=(11.4, 6.0), sharex=True)
    scatter_0 = axes[0].scatter(frames, proto_ids, c=lineage_ids, cmap="tab20", s=18, alpha=0.85)
    axes[0].set_ylabel("prototype id")
    axes[0].set_title("Prototype IDs over Time")
    scatter_1 = axes[1].scatter(frames, lineage_ids, c=proto_ids, cmap="viridis", s=18, alpha=0.85)
    axes[1].set_ylabel("lineage id")
    axes[1].set_xlabel("frame")
    axes[1].set_title("Lineage IDs over Time")
    figure.colorbar(scatter_0, ax=axes[0], label="lineage id")
    figure.colorbar(scatter_1, ax=axes[1], label="prototype id")
    figure.suptitle("Phase 3X Prototype / Lineage Timeline")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_continuation_lifecycle(
    lifecycle_rows: list[dict[str, Any]],
    path: str | Path,
    *,
    scenario_name: str = TRACK_C_NAME,
) -> None:
    rows = [row for row in lifecycle_rows if row["scenario_name"] == scenario_name]
    if not rows:
        return

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["frame_index"])].append(row)
    frames = sorted(grouped.keys())
    alive = [sum(int(item["is_alive"]) for item in grouped[frame]) for frame in frames]
    dropped = [sum(int(item["is_alive"]) == 0 for item in grouped[frame]) for frame in frames]
    drop_counter = Counter(str(item["drop_reason"]) for item in rows if int(item["is_alive"]) == 0)

    figure, axes = plt.subplots(2, 1, figsize=(11.4, 6.4))
    axes[0].plot(frames, alive, label="alive", color="#2a9d8f", linewidth=1.6)
    axes[0].plot(frames, dropped, label="dropped", color="#e76f51", linewidth=1.4)
    axes[0].set_ylabel("continuations")
    axes[0].set_title("Continuation Lifecycle Counts")
    axes[0].legend(frameon=False)

    reasons = list(drop_counter.keys()) or ["alive_only"]
    counts = [drop_counter.get(reason, 0) for reason in reasons]
    axes[1].bar(reasons, counts, color="#6c757d")
    axes[1].set_ylabel("rows")
    axes[1].set_title("Continuation Drop Reasons")
    axes[1].tick_params(axis="x", rotation=20)

    figure.suptitle("Phase 3X Continuation Lifecycle")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_track_c_failure_stage(
    trace_rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    rows = [row for row in trace_rows if row["scenario_name"] == TRACK_C_NAME]
    counts = Counter(str(row["failure_stage"]) for row in rows)
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    figure, axis = plt.subplots(figsize=(11.0, 4.6))
    axis.bar(labels, values, color="#1d3557")
    axis.set_ylabel("events")
    axis.set_title("Track C Re-entry Failure Stages")
    axis.tick_params(axis="x", rotation=18)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_strict_vs_lineage_eval(
    eval_rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    rows_by_scenario: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in eval_rows:
        rows_by_scenario[str(row["scenario_name"])][str(row["eval_mode"])] = row

    scenarios = [TRACK_A_NAME, TRACK_C_NAME]
    labels = ["prototype/lineage", "track-after-recovery"]
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    for axis, metric_key, title in zip(
        axes,
        ["same_prototype_recovery", "same_track_after_recovery"],
        ["Prototype vs Lineage Recovery", "Track after Recovery"],
    ):
        x = np.arange(len(scenarios))
        strict_vals = [
            float(rows_by_scenario.get(scenario, {}).get("strict", {}).get(metric_key, 0.0))
            for scenario in scenarios
        ]
        lineage_vals = [
            float(rows_by_scenario.get(scenario, {}).get("lineage", {}).get(metric_key, 0.0))
            for scenario in scenarios
        ]
        axis.bar(x - 0.18, strict_vals, width=0.36, label="strict", color="#adb5bd")
        axis.bar(x + 0.18, lineage_vals, width=0.36, label="lineage", color="#1d3557")
        axis.set_xticks(x)
        axis.set_xticklabels(scenarios, rotation=14, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(title)
    axes[0].legend(frameon=False)
    figure.suptitle("Phase 3X Strict vs Lineage-aware Evaluation")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _scenario_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_events": 0,
            "same_track_reentry_recovery": 0.0,
            "same_prototype_reentry_recovery": 0.0,
            "same_lineage_prototype_reentry_recovery": 0.0,
            "same_track_after_concept_recovery": 0.0,
            "same_track_after_lineage_recovery": 0.0,
            "candidate_pool_nonempty_rate": 0.0,
            "continuation_bank_nonempty_rate": 0.0,
        }
    concept_events = sum(int(row["concept_recovered"]) for row in rows)
    lineage_events = sum(int(row["same_lineage_id"]) for row in rows)
    return {
        "num_events": len(rows),
        "same_track_reentry_recovery": _mean_int(rows, "same_track"),
        "same_prototype_reentry_recovery": _mean_int(rows, "same_prototype_id"),
        "same_lineage_prototype_reentry_recovery": _mean_int(rows, "same_lineage_id"),
        "same_track_after_concept_recovery": (
            sum(int(row["same_track"]) for row in rows if int(row["concept_recovered"]) == 1) / concept_events
            if concept_events
            else 0.0
        ),
        "same_track_after_lineage_recovery": (
            sum(int(row["same_track"]) for row in rows if int(row["same_lineage_id"]) == 1) / lineage_events
            if lineage_events
            else 0.0
        ),
        "candidate_pool_nonempty_rate": (
            sum(int(int(row["continuation_bank_nonempty"]) == 1) for row in rows if int(row["concept_recovered"]) == 1) / concept_events
            if concept_events
            else 0.0
        ),
        "continuation_bank_nonempty_rate": (
            sum(int(int(row["continuation_bank_nonempty"]) == 1) for row in rows if int(row["same_lineage_id"]) == 1) / lineage_events
            if lineage_events
            else 0.0
        ),
    }


def _mean_int(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(int(row[key]) for row in rows) / len(rows)) if rows else 0.0


def _maybe_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    return int(value)
