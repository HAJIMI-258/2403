"""Run Phase 3D Stage A.2: attach-state consume trace."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator, load_synth_dataset_config
from experiments.phase3d_utils import (
    default_phase3d_stagea_memory_override,
    default_phase3d_stagea_tracking_override,
)
from experiments.scenario_presets import build_phase3_track_scenarios
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.memory import MinimalPrototypeMemory
from nops_owr.objectness import MinimalObjectnessField
from nops_owr.tracking import MinimalTemporalIdentityTracker


TRACK_C_NAME = "track_c_long_horizon"
FORCED_SEARCH_HORIZON = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3D Stage A.2 consume trace.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--stagea1-coverage", default="results/phase3d/phase3d_stagea1_branch_coverage.csv")
    parser.add_argument("--output-dir", default="results/phase3d")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_track_c_sequence(config_path: Path, *, seed: int):
    base_config = load_synth_dataset_config(config_path)
    scenario_map = {s["name"]: s["config"] for s in build_phase3_track_scenarios(base_config)}
    return SyntheticStreamGenerator(scenario_map[TRACK_C_NAME], seed=seed).generate_sequence(0)


def _select_target_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv_rows(path)
    targets = [
        row
        for row in rows
        if row.get("scenario_name") == TRACK_C_NAME and str(row.get("matched_lineage_established", "0")) == "1"
    ]
    targets.sort(key=lambda row: (row.get("run_label", ""), int(row.get("reappear_frame", "0"))))
    return targets


def _build_lineage_summary(
    memory_output,
    tracking_output,
    *,
    lineage_id: int,
) -> dict[str, Any]:
    lineage_rows = [
        row for row in getattr(memory_output, "prototype_lineage_rows", []) if int(row.get("lineage_id", -1)) == lineage_id
    ]
    temp_lookup = getattr(memory_output, "temp_attach_lookup", {})
    temp_slot = temp_lookup.get(int(lineage_id))
    state_counts: dict[str, int] = {"active": 0, "dormant": 0, "ghost": 0, "retired": 0}
    state_track_ids: dict[str, list[int]] = {"active": [], "dormant": [], "ghost": [], "retired": []}
    for state_name, tracks in (
        ("active", tracking_output.active_tracks),
        ("dormant", tracking_output.dormant_tracks),
        ("ghost", tracking_output.ghost_tracks),
        ("retired", tracking_output.retired_tracks),
    ):
        for track in tracks:
            if track.lineage_id != lineage_id:
                continue
            state_counts[state_name] += 1
            state_track_ids[state_name].append(int(track.track_id))

    return {
        "lineage_rows": lineage_rows,
        "lineage_head_prototype_id": None
        if not lineage_rows
        else next((row.get("head_prototype_id") for row in lineage_rows if row.get("head_prototype_id") is not None), None),
        "lineage_continuation_bank_size": 0
        if not lineage_rows
        else max(int(row.get("continuation_bank_size", 0)) for row in lineage_rows),
        "temp_attach_id": None if temp_slot is None else int(temp_slot.get("temp_attach_id", -1)),
        "temp_attach_expired": False if temp_slot is None else bool(temp_slot.get("expired", False)),
        "temp_attach_source_track_id": None
        if temp_slot is None or temp_slot.get("source_track_id") is None
        else int(temp_slot["source_track_id"]),
        "temp_attach_source_prototype_id": None
        if temp_slot is None or temp_slot.get("source_prototype_id") is None
        else int(temp_slot["source_prototype_id"]),
        "temp_attach_age_since_last_seen": None
        if temp_slot is None
        else int(temp_slot.get("age_since_last_seen", 0)),
        "state_counts": state_counts,
        "state_track_ids": state_track_ids,
    }


def _collect_relevant_assignment_rows(
    tracking_output,
    memory_output,
    *,
    frame_index: int,
    run_label: str,
    target: dict[str, Any],
    lineage_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    target_lineage_id = int(target["matched_lineage_id"])
    target_old_track_id = int(target["old_track_id"]) if target.get("old_track_id") not in ("", None) else None
    rows: list[dict[str, Any]] = []
    for tracking_assignment, prototype_assignment in zip(tracking_output.assignments, memory_output.assignments):
        matched_lineage_id = getattr(prototype_assignment, "matched_lineage_id", None)
        linked_lineage_id = tracking_assignment.linked_lineage_id
        prototype_lineage_id = getattr(prototype_assignment, "lineage_id", None)
        relevant = (
            matched_lineage_id == target_lineage_id
            or linked_lineage_id == target_lineage_id
            or prototype_lineage_id == target_lineage_id
            or bool(getattr(prototype_assignment, "attach_state_written", False))
            or bool(getattr(prototype_assignment, "temp_attach_used", False))
        )
        if not relevant:
            continue
        rows.append(
            {
                "run_label": run_label,
                "target_event_id": int(target["event_id"]),
                "target_reappear_frame": int(target["reappear_frame"]),
                "frame_index": int(frame_index),
                "target_old_track_id": target_old_track_id,
                "target_old_prototype_id": None
                if target.get("old_prototype_id") in ("", None)
                else int(target["old_prototype_id"]),
                "target_lineage_id": target_lineage_id,
                "assignment_track_id": int(tracking_assignment.track_id),
                "assignment_source": str(tracking_assignment.assignment_source),
                "assignment_linked_lineage_id": None
                if linked_lineage_id is None
                else int(linked_lineage_id),
                "assignment_linked_prototype_id": None
                if tracking_assignment.linked_prototype_id is None
                else int(tracking_assignment.linked_prototype_id),
                "concept_recovered": int(bool(tracking_assignment.concept_recovered)),
                "matched_lineage_id": None if matched_lineage_id is None else int(matched_lineage_id),
                "prototype_lineage_id": None if prototype_lineage_id is None else int(prototype_lineage_id),
                "prototype_id": int(prototype_assignment.prototype_id),
                "attach_target": str(getattr(prototype_assignment, "recovery_attach_target", "none")),
                "attach_written": int(bool(getattr(prototype_assignment, "attach_state_written", False))),
                "temp_attach_used": int(bool(getattr(prototype_assignment, "temp_attach_used", False))),
                "temp_attach_id": None
                if getattr(prototype_assignment, "temp_attach_id", None) is None
                else int(prototype_assignment.temp_attach_id),
                "candidate_pool_size": int(tracking_assignment.candidate_pool_size),
                "live_candidate_pool_size": int(tracking_assignment.live_candidate_pool_size),
                "continuation_bank_size": int(tracking_assignment.continuation_bank_size),
                "attach_state_consumed_by_tracker": int(bool(tracking_assignment.attach_state_consumed_by_tracker)),
                "attach_state_consumed_by_continuation": int(
                    bool(tracking_assignment.attach_state_consumed_by_continuation)
                ),
                "restore_attempted_from_attach": int(bool(tracking_assignment.restore_attempted_from_attach)),
                "continuation_attempted": int(bool(tracking_assignment.continuation_attempted)),
                "continuation_success": int(bool(tracking_assignment.continuation_success)),
                "best_candidate_state": tracking_assignment.best_candidate_state,
                "best_candidate_gap": tracking_assignment.best_candidate_gap,
                "lineage_head_prototype_id": lineage_summary["lineage_head_prototype_id"],
                "lineage_continuation_bank_size": int(lineage_summary["lineage_continuation_bank_size"]),
                "lineage_temp_attach_id": lineage_summary["temp_attach_id"],
                "lineage_temp_attach_expired": int(bool(lineage_summary["temp_attach_expired"])),
                "lineage_temp_attach_source_track_id": lineage_summary["temp_attach_source_track_id"],
                "lineage_temp_attach_source_prototype_id": lineage_summary["temp_attach_source_prototype_id"],
                "lineage_temp_attach_age_since_last_seen": lineage_summary["temp_attach_age_since_last_seen"],
                "lineage_active_track_count": int(lineage_summary["state_counts"]["active"]),
                "lineage_dormant_track_count": int(lineage_summary["state_counts"]["dormant"]),
                "lineage_ghost_track_count": int(lineage_summary["state_counts"]["ghost"]),
                "lineage_retired_track_count": int(lineage_summary["state_counts"]["retired"]),
                "lineage_active_track_ids": "|".join(map(str, lineage_summary["state_track_ids"]["active"])),
                "lineage_dormant_track_ids": "|".join(map(str, lineage_summary["state_track_ids"]["dormant"])),
                "lineage_ghost_track_ids": "|".join(map(str, lineage_summary["state_track_ids"]["ghost"])),
                "lineage_retired_track_ids": "|".join(map(str, lineage_summary["state_track_ids"]["retired"])),
                "temp_slot_present_not_consumed": int(
                    lineage_summary["temp_attach_id"] is not None
                    and not bool(lineage_summary["temp_attach_expired"])
                    and bool(getattr(prototype_assignment, "attach_state_written", False))
                    and not bool(tracking_assignment.attach_state_consumed_by_tracker)
                ),
                "temp_slot_carries_current_track": int(
                    lineage_summary["temp_attach_source_track_id"] is not None
                    and lineage_summary["temp_attach_source_track_id"] == int(tracking_assignment.track_id)
                ),
            }
        )
    return rows


def _run_direct_trace(
    *,
    config_path: Path,
    seed: int,
    target_rows: list[dict[str, Any]],
    run_label: str,
    tracking_patch: dict[str, Any] | None = None,
    memory_patch: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    sequence = _load_track_c_sequence(config_path, seed=seed)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    tracking_config = dict(payload["tracking"])
    tracking_config.update(default_phase3d_stagea_tracking_override())
    if tracking_patch:
        tracking_config.update(tracking_patch)

    memory_config = dict(payload["memory"])
    memory_config.update(default_phase3d_stagea_memory_override())
    if memory_patch:
        memory_config.update(memory_patch)

    encoder = MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    objectness = MinimalObjectnessField(**payload["field"])
    tracker = MinimalTemporalIdentityTracker(**tracking_config)
    memory = MinimalPrototypeMemory(**memory_config)

    targets_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in target_rows:
        targets_by_frame.setdefault(int(row["reappear_frame"]), []).append(row)
    if not targets_by_frame:
        return []

    fallback_rows_by_event: dict[int, list[dict[str, Any]]] = {}
    max_target_frame = max(targets_by_frame) + (FORCED_SEARCH_HORIZON if run_label == "forced_consume" else 0)
    output_rows: list[dict[str, Any]] = []
    for frame_offset in range(1, len(sequence.frames)):
        prev_frame = sequence.frames[frame_offset - 1]
        current_frame = sequence.frames[frame_offset]
        encoding = encoder.encode(prev_frame.frame, current_frame.frame)
        objectness_output = objectness.compute(encoding)
        tracking_output = tracker.update(
            proposals=objectness_output.proposals,
            encoding=encoding,
            heatmap=objectness_output.heatmap,
            current_frame=current_frame.frame,
            frame_index=current_frame.frame_index,
        )
        memory_output = memory.update(
            tracking_output.assignments,
            frame_index=current_frame.frame_index,
            track_states=(
                tracking_output.active_tracks
                + tracking_output.dormant_tracks
                + tracking_output.ghost_tracks
                + tracking_output.retired_tracks
            ),
        )
        tracker.apply_concept_gated_resurrection(
            tracking_output,
            memory_output,
            frame_index=current_frame.frame_index,
            frame_shape=objectness_output.heatmap.shape,
        )
        tracker.bind_prototypes(memory_output.assignments)

        if current_frame.frame_index in targets_by_frame:
            for target in targets_by_frame[current_frame.frame_index]:
                lineage_summary = _build_lineage_summary(
                    memory_output,
                    tracking_output,
                    lineage_id=int(target["matched_lineage_id"]),
                )
                rows = _collect_relevant_assignment_rows(
                    tracking_output,
                    memory_output,
                    frame_index=current_frame.frame_index,
                    run_label=run_label,
                    target=target,
                    lineage_summary=lineage_summary,
                )
                if rows:
                    output_rows.extend(rows)
                else:
                    output_rows.append(
                        {
                            "run_label": run_label,
                            "target_event_id": int(target["event_id"]),
                            "target_reappear_frame": int(target["reappear_frame"]),
                            "frame_index": int(current_frame.frame_index),
                            "target_old_track_id": None
                            if target.get("old_track_id") in ("", None)
                            else int(target["old_track_id"]),
                            "target_old_prototype_id": None
                            if target.get("old_prototype_id") in ("", None)
                            else int(target["old_prototype_id"]),
                            "target_lineage_id": int(target["matched_lineage_id"]),
                            "assignment_track_id": None,
                            "assignment_source": "none",
                            "assignment_linked_lineage_id": None,
                            "assignment_linked_prototype_id": None,
                            "concept_recovered": 0,
                            "matched_lineage_id": None,
                            "prototype_lineage_id": None,
                            "prototype_id": None,
                            "attach_target": "none",
                            "attach_written": 0,
                            "temp_attach_used": 0,
                            "temp_attach_id": None,
                            "candidate_pool_size": 0,
                            "live_candidate_pool_size": 0,
                            "continuation_bank_size": 0,
                            "attach_state_consumed_by_tracker": 0,
                            "attach_state_consumed_by_continuation": 0,
                            "restore_attempted_from_attach": 0,
                            "continuation_attempted": 0,
                            "continuation_success": 0,
                            "best_candidate_state": None,
                            "best_candidate_gap": None,
                            "lineage_head_prototype_id": lineage_summary["lineage_head_prototype_id"],
                            "lineage_continuation_bank_size": int(lineage_summary["lineage_continuation_bank_size"]),
                            "lineage_temp_attach_id": lineage_summary["temp_attach_id"],
                            "lineage_temp_attach_expired": int(bool(lineage_summary["temp_attach_expired"])),
                            "lineage_temp_attach_source_track_id": lineage_summary["temp_attach_source_track_id"],
                            "lineage_temp_attach_source_prototype_id": lineage_summary["temp_attach_source_prototype_id"],
                            "lineage_temp_attach_age_since_last_seen": lineage_summary["temp_attach_age_since_last_seen"],
                            "lineage_active_track_count": int(lineage_summary["state_counts"]["active"]),
                            "lineage_dormant_track_count": int(lineage_summary["state_counts"]["dormant"]),
                            "lineage_ghost_track_count": int(lineage_summary["state_counts"]["ghost"]),
                            "lineage_retired_track_count": int(lineage_summary["state_counts"]["retired"]),
                            "lineage_active_track_ids": "|".join(map(str, lineage_summary["state_track_ids"]["active"])),
                            "lineage_dormant_track_ids": "|".join(map(str, lineage_summary["state_track_ids"]["dormant"])),
                            "lineage_ghost_track_ids": "|".join(map(str, lineage_summary["state_track_ids"]["ghost"])),
                            "lineage_retired_track_ids": "|".join(map(str, lineage_summary["state_track_ids"]["retired"])),
                            "temp_slot_present_not_consumed": int(
                                lineage_summary["temp_attach_id"] is not None
                                and not bool(lineage_summary["temp_attach_expired"])
                            ),
                            "temp_slot_carries_current_track": 0,
                        }
                    )
        if run_label == "forced_consume":
            for target in target_rows:
                target_event_id = int(target["event_id"])
                if target_event_id in fallback_rows_by_event:
                    continue
                target_lineage_id = int(target["matched_lineage_id"])
                lineage_summary = _build_lineage_summary(
                    memory_output,
                    tracking_output,
                    lineage_id=target_lineage_id,
                )
                fallback_rows = [
                    row
                    for row in _collect_relevant_assignment_rows(
                        tracking_output,
                        memory_output,
                        frame_index=current_frame.frame_index,
                        run_label=run_label,
                        target=target,
                        lineage_summary=lineage_summary,
                    )
                    if row["target_lineage_id"] == target_lineage_id
                    and row["attach_target"] == "temporary_attach_slot"
                ]
                if fallback_rows:
                    fallback_rows_by_event[target_event_id] = fallback_rows
        if current_frame.frame_index >= max_target_frame:
            break
    if run_label == "forced_consume":
        existing_events = {int(row["target_event_id"]) for row in output_rows}
        for target in target_rows:
            target_event_id = int(target["event_id"])
            if target_event_id in existing_events:
                continue
            output_rows.extend(fallback_rows_by_event.get(target_event_id, []))
    return output_rows


def _build_summary(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_rows = [row for row in trace_rows if row["run_label"] == "baseline"]
    forced_rows = [row for row in trace_rows if row["run_label"] == "forced_consume"]
    baseline_target_rows = list(baseline_rows)
    forced_target_rows = list(forced_rows)
    baseline_target_lineage_rows = [
        row
        for row in baseline_target_rows
        if row["target_lineage_id"] in (row["matched_lineage_id"], row["assignment_linked_lineage_id"], row["prototype_lineage_id"])
    ]
    forced_target_lineage_rows = [
        row
        for row in forced_target_rows
        if row["target_lineage_id"] in (row["matched_lineage_id"], row["assignment_linked_lineage_id"], row["prototype_lineage_id"])
    ]

    def _rate(rows: list[dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return float(sum(int(row.get(key, 0)) for row in rows) / len(rows))

    return {
        "baseline_target_rows": len(baseline_target_rows),
        "forced_target_rows": len(forced_target_rows),
        "baseline_target_lineage_rows": len(baseline_target_lineage_rows),
        "forced_target_lineage_rows": len(forced_target_lineage_rows),
        "baseline_unrelated_attach_rows": sum(int(row["attach_written"]) for row in baseline_target_rows)
        - sum(int(row["attach_written"]) for row in baseline_target_lineage_rows),
        "baseline_attach_state_consumed_by_tracker_rate": _rate(
            baseline_target_rows, "attach_state_consumed_by_tracker"
        ),
        "forced_attach_state_consumed_by_tracker_rate": _rate(
            forced_target_rows, "attach_state_consumed_by_tracker"
        ),
        "baseline_restore_attempted_from_attach_rate": _rate(
            baseline_target_rows, "restore_attempted_from_attach"
        ),
        "forced_restore_attempted_from_attach_rate": _rate(
            forced_target_rows, "restore_attempted_from_attach"
        ),
        "baseline_continuation_bank_size_max": max(
            [int(row["lineage_continuation_bank_size"]) for row in baseline_target_rows],
            default=0,
        ),
        "forced_continuation_bank_size_max": max(
            [int(row["lineage_continuation_bank_size"]) for row in forced_target_rows],
            default=0,
        ),
        "baseline_temp_slot_present_not_consumed_rate": _rate(
            baseline_target_rows, "temp_slot_present_not_consumed"
        ),
        "forced_temp_slot_present_not_consumed_rate": _rate(
            forced_target_rows, "temp_slot_present_not_consumed"
        ),
        "forced_temp_slot_carries_current_track_rate": _rate(
            forced_target_rows, "temp_slot_carries_current_track"
        ),
    }


def _write_summary_md(path: Path, trace_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    baseline_rows = [row for row in trace_rows if row["run_label"] == "baseline"]
    forced_rows = [row for row in trace_rows if row["run_label"] == "forced_consume"]

    def _first(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        return rows[0] if rows else None

    baseline = _first(baseline_rows)
    forced = _first(forced_rows)
    lines = [
        "# Phase 3D Stage A.2 Consume Trace",
        "",
        "## Main Finding",
        "",
        "The Stage A.1 `attach_written=1` signal was not a clean proof of recovery attach. In the traced baseline target frame, the target lineage itself has no live or continuation candidates, while the visible `attach_written` rows belong to other lineages' active-match traffic. On top of that, `apply_concept_gated_resurrection()` only consumes dormant/ghost tracks and lineage continuation banks; temp-attach slots are not baseline recovery candidates, and the slot payload stores the current attached track rather than an old recoverable identity.",
        "",
        "## Baseline Target Frame",
        "",
        "The baseline target is taken from the Stage A.1 matched-lineage row.",
        "",
    ]
    if baseline is not None:
        lines.extend(
            [
                f"- `frame = {baseline['frame_index']}`",
                f"- `target_lineage = {baseline['target_lineage_id']}`",
                f"- `lineage_continuation_bank_size = {baseline['lineage_continuation_bank_size']}`",
                f"- `lineage_active/dormant/ghost/retired = {baseline['lineage_active_track_count']}/{baseline['lineage_dormant_track_count']}/{baseline['lineage_ghost_track_count']}/{baseline['lineage_retired_track_count']}`",
                f"- `lineage_temp_attach_id = {baseline['lineage_temp_attach_id']}`",
                f"- `lineage_temp_attach_expired = {baseline['lineage_temp_attach_expired']}`",
                f"- `attach_state_consumed_by_tracker = {baseline['attach_state_consumed_by_tracker']}`",
                f"- `restore_attempted_from_attach = {baseline['restore_attempted_from_attach']}`",
                f"- `target-lineage assignment rows at this frame = {summary['baseline_target_lineage_rows']}`",
                f"- `unrelated attach-written rows at this frame = {summary['baseline_unrelated_attach_rows']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Forced Consume Check",
            "",
            "This run forces temp-attach creation and enables a debug consume hook that marks temp-attach state as consumed when the normal pool is empty. It does not repair recovery logic; it only proves whether the consumer path can observe the slot.",
            "",
        ]
    )
    if forced is not None:
        lines.extend(
            [
                f"- `frame = {forced['frame_index']}`",
                f"- `attach_target = {forced['attach_target']}`",
                f"- `attach_state_consumed_by_tracker = {forced['attach_state_consumed_by_tracker']}`",
                f"- `restore_attempted_from_attach = {forced['restore_attempted_from_attach']}`",
                f"- `lineage_temp_attach_id = {forced['lineage_temp_attach_id']}`",
                f"- `lineage_temp_attach_source_track_id = {forced['lineage_temp_attach_source_track_id']}`",
                f"- `assignment_track_id = {forced['assignment_track_id']}`",
                f"- `temp_slot_carries_current_track = {forced['temp_slot_carries_current_track']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            "1. The baseline empty-pool condition is real: the target lineage reaches the traced frame with no dormant/ghost recovery candidates and an empty lineage continuation bank.",
            "2. The apparent Stage A.1 attach success is partially audit pollution: the attach-written rows visible at the traced baseline frame belong to non-target lineages, not the target lineage.",
            "3. Even when a temp-attach slot exists on a lineage, baseline resurrection does not read that slot; it only reads live candidates and continuation banks.",
            "4. The temp-attach slot currently stores the current attached track id, not an old recoverable track identity. That means a future consumer can observe it, but it still would not be a valid old-track restoration source by itself.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = _select_target_rows(Path(args.stagea1_coverage))
    baseline_targets = [row for row in targets if row.get("run_label") == "baseline"]
    forced_targets = [row for row in targets if row.get("run_label") == "forced_temp_attach"]

    baseline_trace = _run_direct_trace(
        config_path=config_path,
        seed=args.seed,
        target_rows=baseline_targets,
        run_label="baseline",
    )
    forced_trace = _run_direct_trace(
        config_path=config_path,
        seed=args.seed,
        target_rows=forced_targets,
        run_label="forced_consume",
        tracking_patch={"debug_force_attach_consume": True},
        memory_patch={"debug_force_temp_attach": True},
    )
    trace_rows = baseline_trace + forced_trace
    summary = _build_summary(trace_rows)

    _write_csv(output_dir / "phase3d_stagea2_frame_trace.csv", trace_rows)
    (output_dir / "phase3d_stagea2_forced_consume_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_summary_md(output_dir / "phase3d_stagea2_consume_trace.md", trace_rows, summary)


if __name__ == "__main__":
    main()
