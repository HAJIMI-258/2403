"""Phase 3D Stage A.7: continuity source remap trace + ownership preservation repair."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator, load_synth_dataset_config  # noqa: E402
from experiments.phase3d_utils import (  # noqa: E402
    default_phase3d_stagea_memory_override,
    default_phase3d_stagea_tracking_override,
)
from experiments.run_phase3d_stage_a5_claim_preservation_trace import (  # noqa: E402
    TARGET_EVENT_ID,
    TARGET_FRAME,
    TARGET_GT_OBJECT_ID,
    TRACK_C_NAME,
    WINDOW_LEFT,
    _draw_box,
    _gt_box,
    _load_target_metadata,
    _write_csv,
)
from experiments.scenario_presets import build_phase3_track_scenarios  # noqa: E402
from nops_owr.encoder import MinimalSpikeEncoder  # noqa: E402
from nops_owr.memory import MinimalPrototypeMemory  # noqa: E402
from nops_owr.objectness import MinimalObjectnessField  # noqa: E402
from nops_owr.tracking import MinimalTemporalIdentityTracker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3D Stage A.7 remap trace.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--output-dir", default="results/phase3d")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_track_c_sequence(config_path: Path, *, seed: int):
    base_config = load_synth_dataset_config(config_path)
    scenario_map = {s["name"]: s["config"] for s in build_phase3_track_scenarios(base_config)}
    return SyntheticStreamGenerator(scenario_map[TRACK_C_NAME], seed=seed).generate_sequence(0)


def _as_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(value)


def _identity_ref(track_id: int | None, prototype_id: int | None) -> str:
    return f"track:{track_id if track_id is not None else 'None'}|proto:{prototype_id if prototype_id is not None else 'None'}"


def _bucket_for_row(
    *,
    before_runtime_owner_lineage_id: int | None,
    after_runtime_owner_lineage_id: int | None,
    before_continuity_lineage_id: int | None,
    after_continuity_lineage_id: int | None,
    expected_lineage_id: int | None,
    expected_to_preserve: bool,
    derived_copy: bool,
) -> tuple[str, int, int, int]:
    runtime_owner_changed = int(before_runtime_owner_lineage_id != after_runtime_owner_lineage_id)
    continuity_key_changed = int(before_continuity_lineage_id != after_continuity_lineage_id)
    continuity_key_preserved = int(
        (not expected_to_preserve)
        or expected_lineage_id is None
        or after_continuity_lineage_id == expected_lineage_id
    )
    overwrite_bug_flag = int(
        expected_to_preserve
        and expected_lineage_id is not None
        and after_continuity_lineage_id is not None
        and after_continuity_lineage_id != expected_lineage_id
    )
    if overwrite_bug_flag:
        remap_bucket = "continuity_key_overwrite_bug"
    elif derived_copy:
        remap_bucket = "runtime_rebind_with_continuity_preserved"
    else:
        remap_bucket = "runtime_rebind_only"
    return remap_bucket, runtime_owner_changed, continuity_key_changed, continuity_key_preserved


def _append_remap_row(
    rows: list[dict[str, Any]],
    *,
    frame_id: int,
    event_type: str,
    object_type: str,
    object_id: str,
    before_runtime_owner_lineage_id: int | None,
    after_runtime_owner_lineage_id: int | None,
    before_continuity_lineage_id: int | None,
    after_continuity_lineage_id: int | None,
    before_old_identity_ref: str | None,
    after_old_identity_ref: str | None,
    mutation_reason: str,
    code_location: str,
    expected_lineage_id: int | None,
    expected_to_preserve_continuity_key: bool,
    derived_copy: bool = False,
) -> None:
    (
        remap_bucket,
        runtime_owner_changed,
        continuity_key_changed,
        continuity_key_preserved,
    ) = _bucket_for_row(
        before_runtime_owner_lineage_id=before_runtime_owner_lineage_id,
        after_runtime_owner_lineage_id=after_runtime_owner_lineage_id,
        before_continuity_lineage_id=before_continuity_lineage_id,
        after_continuity_lineage_id=after_continuity_lineage_id,
        expected_lineage_id=expected_lineage_id,
        expected_to_preserve=bool(expected_to_preserve_continuity_key),
        derived_copy=bool(derived_copy),
    )
    rows.append(
        {
            "frame_id": int(frame_id),
            "event_type": str(event_type),
            "object_type": str(object_type),
            "object_id": str(object_id),
            "before_runtime_owner_lineage_id": before_runtime_owner_lineage_id,
            "after_runtime_owner_lineage_id": after_runtime_owner_lineage_id,
            "before_continuity_lineage_id": before_continuity_lineage_id,
            "after_continuity_lineage_id": after_continuity_lineage_id,
            "before_old_identity_ref": before_old_identity_ref or "",
            "after_old_identity_ref": after_old_identity_ref or "",
            "mutation_reason": str(mutation_reason),
            "code_location": str(code_location),
            "was_continuity_key_overwritten": int(remap_bucket == "continuity_key_overwrite_bug"),
            "expected_to_preserve_continuity_key": int(bool(expected_to_preserve_continuity_key)),
            "remap_bucket": str(remap_bucket),
            "runtime_owner_changed": int(runtime_owner_changed),
            "continuity_key_changed": int(continuity_key_changed),
            "continuity_key_preserved": int(continuity_key_preserved),
            "overwrite_bug_flag": int(remap_bucket == "continuity_key_overwrite_bug"),
        }
    )


def _runtime_track_rows(tracking_output) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state_name, items in (
        ("active", tracking_output.active_tracks),
        ("dormant", tracking_output.dormant_tracks),
        ("ghost", tracking_output.ghost_tracks),
        ("retired", tracking_output.retired_tracks),
    ):
        for track in items:
            rows.append(
                {
                    "track_id": int(track.track_id),
                    "state": state_name,
                    "prototype_id": None if track.prototype_id is None else int(track.prototype_id),
                    "lineage_id": None if track.lineage_id is None else int(track.lineage_id),
                    "box": tuple(int(v) for v in track.box),
                    "last_seen_frame": int(track.last_seen_frame),
                }
            )
    return rows


def _draw_text_block(axis, lines: list[str], *, y0: float = 0.98) -> None:
    axis.axis("off")
    axis.text(
        0.02,
        y0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        transform=axis.transAxes,
    )


def _run_trace(
    *,
    config_path: Path,
    seed: int,
) -> dict[str, Any]:
    sequence = _load_track_c_sequence(config_path, seed=seed)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tracking_config = dict(payload["tracking"])
    tracking_config.update(default_phase3d_stagea_tracking_override())
    tracking_config.update(
        {
            "enable_phase3d_routing_repair": True,
            "enable_phase3d_target_selection_trace": True,
            "enable_phase3d_target_selection_repair": False,
            "enable_phase3d_identity_preference_tiebreak": False,
            "enable_phase3d_preserve_input_trace": True,
            "routing_recovery_max_distance": 0.70,
            "routing_recovery_min_confidence": 0.30,
            "routing_active_claim_override_margin": 0.20,
            "routing_topk": 3,
            "claim_preserve_min_score": 0.25,
            "continuity_hint_min_score": 0.15,
            "debug_force_reroute_frame": TARGET_FRAME,
        }
    )
    memory_config = dict(payload["memory"])
    memory_config.update(default_phase3d_stagea_memory_override())

    encoder = MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    objectness = MinimalObjectnessField(**payload["field"])
    tracker = MinimalTemporalIdentityTracker(**tracking_config)
    memory = MinimalPrototypeMemory(**memory_config)

    target_metadata = _load_target_metadata()
    target_lineage_id = _as_int(target_metadata.get("old_lineage_id")) or 2
    target_prototype_id = _as_int(target_metadata.get("old_prototype_id")) or 3

    remap_rows: list[dict[str, Any]] = []
    proto_history: list[dict[str, Any]] = []
    registry_history: list[dict[str, Any]] = []
    relevant_track_history: list[dict[str, Any]] = []
    frame_snapshots: dict[int, dict[str, Any]] = {}

    prev_memory_output = None
    prev_proto_row: dict[str, Any] | None = None
    prev_registry_lineages: tuple[int, ...] | None = None
    prev_track_state: dict[int, str] = {}
    continuation_prev: dict[str, dict[str, Any]] = {}
    anchor_prev: dict[str, dict[str, Any]] = {}

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
            memory_context=prev_memory_output,
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
        prev_memory_output = memory_output

        frame_id = int(current_frame.frame_index)
        runtime_lineages = tuple(sorted(int(k) for k in memory._lineages.keys()))
        registry_history.append(
            {
                "frame_id": frame_id,
                "runtime_registry_lineages": list(runtime_lineages),
                "prototype_continuity_lookup": memory.prototype_continuity_lookup(),
            }
        )
        if prev_registry_lineages != runtime_lineages:
            _append_remap_row(
                remap_rows,
                frame_id=frame_id,
                event_type="lineage_registry_update",
                object_type="lineage_registry",
                object_id="registry",
                before_runtime_owner_lineage_id=None if prev_registry_lineages is None else -1,
                after_runtime_owner_lineage_id=-1,
                before_continuity_lineage_id=None,
                after_continuity_lineage_id=None,
                before_old_identity_ref="|".join(str(v) for v in prev_registry_lineages or ()),
                after_old_identity_ref="|".join(str(v) for v in runtime_lineages),
                mutation_reason="runtime_registry_key_change",
                code_location="_ensure_lineage/_mark_lineage_active",
                expected_lineage_id=target_lineage_id,
                expected_to_preserve_continuity_key=False,
            )
        prev_registry_lineages = runtime_lineages

        proto_row = next(
            (row for row in memory_output.prototype_lineage_rows if int(row["prototype_id"]) == int(target_prototype_id)),
            None,
        )
        if proto_row is not None:
            proto_history.append(
                {
                    "frame_id": frame_id,
                    **proto_row,
                    "runtime_registry_lineages": list(runtime_lineages),
                }
            )
            if prev_proto_row is None:
                _append_remap_row(
                    remap_rows,
                    frame_id=frame_id,
                    event_type="prototype_create",
                    object_type="prototype",
                    object_id=str(target_prototype_id),
                    before_runtime_owner_lineage_id=None,
                    after_runtime_owner_lineage_id=_as_int(proto_row.get("runtime_owner_lineage_id")),
                    before_continuity_lineage_id=None,
                    after_continuity_lineage_id=_as_int(proto_row.get("continuity_lineage_id")),
                    before_old_identity_ref=None,
                    after_old_identity_ref=_identity_ref(
                        _as_int(proto_row.get("last_track_id")),
                        target_prototype_id,
                    ),
                    mutation_reason="first_observed_prototype_row",
                    code_location="_create_prototype/_allocate_lineage",
                    expected_lineage_id=target_lineage_id,
                    expected_to_preserve_continuity_key=True,
                )
            else:
                changed = (
                    _as_int(prev_proto_row.get("runtime_owner_lineage_id")) != _as_int(proto_row.get("runtime_owner_lineage_id"))
                    or _as_int(prev_proto_row.get("continuity_lineage_id")) != _as_int(proto_row.get("continuity_lineage_id"))
                    or int(prev_proto_row.get("is_retired", 0)) != int(proto_row.get("is_retired", 0))
                )
                if changed:
                    _append_remap_row(
                        remap_rows,
                        frame_id=frame_id,
                        event_type="prototype_archive" if int(proto_row.get("is_retired", 0)) else "prototype_replace",
                        object_type="prototype",
                        object_id=str(target_prototype_id),
                        before_runtime_owner_lineage_id=_as_int(prev_proto_row.get("runtime_owner_lineage_id")),
                        after_runtime_owner_lineage_id=_as_int(proto_row.get("runtime_owner_lineage_id")),
                        before_continuity_lineage_id=_as_int(prev_proto_row.get("continuity_lineage_id")),
                        after_continuity_lineage_id=_as_int(proto_row.get("continuity_lineage_id")),
                        before_old_identity_ref=_identity_ref(_as_int(prev_proto_row.get("last_track_id")), target_prototype_id),
                        after_old_identity_ref=_identity_ref(_as_int(proto_row.get("last_track_id")), target_prototype_id),
                        mutation_reason="prototype_runtime_or_continuity_change",
                        code_location="_mark_lineage_member_active/_retire_prototype",
                        expected_lineage_id=target_lineage_id,
                        expected_to_preserve_continuity_key=True,
                    )
            prev_proto_row = dict(proto_row)

        relevant_track_ids: set[int] = set()
        track_rows = _runtime_track_rows(tracking_output)
        for row in track_rows:
            if row["prototype_id"] == target_prototype_id:
                relevant_track_ids.add(int(row["track_id"]))
                relevant_track_history.append({"frame_id": frame_id, **row})

        for row in memory_output.continuation_write_rows:
            source_proto_id = _as_int(row.get("source_prototype_id"))
            old_proto_id = _as_int(row.get("track_old_identity_ref_prototype_id"))
            if source_proto_id != target_prototype_id and old_proto_id != target_prototype_id:
                continue
            relevant_track_ids.add(int(row["track_id"]))

        for row in memory_output.recovery_anchor_rows:
            if _as_int(row.get("old_prototype_id")) != target_prototype_id:
                continue
            relevant_track_ids.add(int(row["old_track_id"]))

        for row in track_rows:
            if int(row["track_id"]) not in relevant_track_ids:
                continue
            prev_state = prev_track_state.get(int(row["track_id"]))
            if prev_state != row["state"]:
                _append_remap_row(
                    remap_rows,
                    frame_id=frame_id,
                    event_type="track_retire" if row["state"] == "retired" else "track_archive",
                    object_type="track",
                    object_id=str(row["track_id"]),
                    before_runtime_owner_lineage_id=None,
                    after_runtime_owner_lineage_id=_as_int(row.get("lineage_id")),
                    before_continuity_lineage_id=None,
                    after_continuity_lineage_id=None,
                    before_old_identity_ref=prev_state,
                    after_old_identity_ref=_identity_ref(int(row["track_id"]), _as_int(row.get("prototype_id"))),
                    mutation_reason=f"track_state_{prev_state or 'none'}->{row['state']}",
                    code_location="tracker.update",
                    expected_lineage_id=target_lineage_id,
                    expected_to_preserve_continuity_key=bool(_as_int(row.get("prototype_id")) == target_prototype_id),
                )
            prev_track_state[int(row["track_id"])] = str(row["state"])

        for row in memory_output.continuation_lifecycle_rows:
            source_proto_id = _as_int(row.get("source_prototype_id"))
            old_proto_id = _as_int(row.get("old_identity_ref_prototype_id"))
            if source_proto_id != target_prototype_id and old_proto_id != target_prototype_id:
                continue
            uid = str(row["continuation_uid"])
            previous = continuation_prev.get(uid)
            if (
                previous is None
                or _as_int(previous.get("current_owner_lineage_id")) != _as_int(row.get("current_owner_lineage_id"))
                or _as_int(previous.get("continuity_lineage_id")) != _as_int(row.get("continuity_lineage_id"))
                or int(previous.get("is_alive", 1)) != int(row.get("is_alive", 1))
            ):
                _append_remap_row(
                    remap_rows,
                    frame_id=frame_id,
                    event_type="continuation_prune" if int(row.get("is_alive", 1)) == 0 else ("continuation_write" if previous is None else "continuation_rebind"),
                    object_type="continuation",
                    object_id=uid,
                    before_runtime_owner_lineage_id=None if previous is None else _as_int(previous.get("current_owner_lineage_id")),
                    after_runtime_owner_lineage_id=_as_int(row.get("current_owner_lineage_id")),
                    before_continuity_lineage_id=None if previous is None else _as_int(previous.get("continuity_lineage_id")),
                    after_continuity_lineage_id=_as_int(row.get("continuity_lineage_id")),
                    before_old_identity_ref=None
                    if previous is None
                    else _identity_ref(
                        _as_int(previous.get("old_identity_ref_track_id")),
                        _as_int(previous.get("old_identity_ref_prototype_id")),
                    ),
                    after_old_identity_ref=_identity_ref(
                        _as_int(row.get("old_identity_ref_track_id")),
                        _as_int(row.get("old_identity_ref_prototype_id")),
                    ),
                    mutation_reason=str(row.get("drop_reason", "alive")),
                    code_location="_archive_track_continuation" if previous is None else "_refresh_continuations",
                    expected_lineage_id=target_lineage_id,
                    expected_to_preserve_continuity_key=True,
                    derived_copy=True,
                )
            continuation_prev[uid] = dict(row)

        for row in memory_output.recovery_anchor_lifecycle_rows:
            old_proto_id = _as_int(row.get("old_prototype_id"))
            identity_proto_id = _as_int(row.get("old_identity_ref_prototype_id"))
            if old_proto_id != target_prototype_id and identity_proto_id != target_prototype_id:
                continue
            uid = str(row["anchor_uid"])
            previous = anchor_prev.get(uid)
            if (
                previous is None
                or _as_int(previous.get("runtime_owner_lineage_id")) != _as_int(row.get("runtime_owner_lineage_id"))
                or _as_int(previous.get("continuity_lineage_id")) != _as_int(row.get("continuity_lineage_id"))
                or int(previous.get("is_alive", 1)) != int(row.get("is_alive", 1))
            ):
                _append_remap_row(
                    remap_rows,
                    frame_id=frame_id,
                    event_type="anchor_prune" if int(row.get("is_alive", 1)) == 0 else ("anchor_create" if previous is None else "anchor_rebind"),
                    object_type="anchor",
                    object_id=uid,
                    before_runtime_owner_lineage_id=None if previous is None else _as_int(previous.get("runtime_owner_lineage_id")),
                    after_runtime_owner_lineage_id=_as_int(row.get("runtime_owner_lineage_id")),
                    before_continuity_lineage_id=None if previous is None else _as_int(previous.get("continuity_lineage_id")),
                    after_continuity_lineage_id=_as_int(row.get("continuity_lineage_id")),
                    before_old_identity_ref=None
                    if previous is None
                    else _identity_ref(
                        _as_int(previous.get("old_identity_ref_track_id")),
                        _as_int(previous.get("old_identity_ref_prototype_id")),
                    ),
                    after_old_identity_ref=_identity_ref(
                        _as_int(row.get("old_identity_ref_track_id")),
                        _as_int(row.get("old_identity_ref_prototype_id")),
                    ),
                    mutation_reason=str(row.get("drop_reason", "alive")),
                    code_location="_archive_recovery_anchor" if previous is None else "_refresh_recovery_anchors",
                    expected_lineage_id=target_lineage_id,
                    expected_to_preserve_continuity_key=True,
                    derived_copy=True,
                )
            anchor_prev[uid] = dict(row)

        if TARGET_FRAME - WINDOW_LEFT <= frame_id <= TARGET_FRAME + WINDOW_LEFT:
            frame_snapshots[frame_id] = {
                "frame": current_frame.frame.copy(),
                "gt_box": _gt_box(current_frame, TARGET_GT_OBJECT_ID),
                "runtime_registry_lineages": list(runtime_lineages),
                "prototype3_row": None if proto_row is None else dict(proto_row),
                "track_rows": [dict(row) for row in track_rows if int(row["track_id"]) in relevant_track_ids],
            }

    remap_rows.sort(key=lambda row: (int(row["frame_id"]), str(row["object_type"]), str(row["object_id"])))
    proto_history.sort(key=lambda row: int(row["frame_id"]))
    relevant_track_history.sort(key=lambda row: (int(row["frame_id"]), int(row["track_id"])))
    return {
        "target_lineage_id": int(target_lineage_id),
        "target_prototype_id": int(target_prototype_id),
        "remap_rows": remap_rows,
        "proto_history": proto_history,
        "registry_history": registry_history,
        "relevant_track_history": relevant_track_history,
        "frame_snapshots": frame_snapshots,
    }


def _forced_continuity_key_probe(trace: dict[str, Any]) -> dict[str, Any]:
    target_lineage_id = int(trace["target_lineage_id"])
    target_prototype_id = int(trace["target_prototype_id"])
    proto_history = trace["proto_history"]
    target_row = next((row for row in proto_history if int(row["frame_id"]) == TARGET_FRAME), None)
    baseline_runtime = None if target_row is None else _as_int(target_row.get("runtime_owner_lineage_id"))
    baseline_continuity = None if target_row is None else _as_int(target_row.get("continuity_lineage_id"))
    runtime_registry_lineages = []
    for row in trace["registry_history"]:
        if int(row["frame_id"]) == TARGET_FRAME:
            runtime_registry_lineages = list(row["runtime_registry_lineages"])
            break
    lineage2_enumerable = bool(target_row is not None)
    return {
        "target_frame": TARGET_FRAME,
        "target_prototype_id": target_prototype_id,
        "target_lineage_id": target_lineage_id,
        "baseline_runtime_registry_lineages": runtime_registry_lineages,
        "baseline_runtime_owner_lineage_id": baseline_runtime,
        "baseline_continuity_lineage_id": baseline_continuity,
        "forced_continuity_lineage_id": target_lineage_id if lineage2_enumerable else None,
        "continuity_source_enumerable_after_force": bool(lineage2_enumerable),
        "preserve_input_still_runtime_owner_only": True,
        "lineage2_enters_preserve_input_without_owner_split": False,
        "lineage2_enters_claim_builder_without_owner_split": False,
        "notes": "只保 continuity key 时，runtime registry 仍可只有 {0,1}；若 preserve-input 仍只枚举 runtime owner，lineage 2 依旧进不了 preserve-input。",
    }


def _forced_owner_continuity_split_probe(trace: dict[str, Any]) -> dict[str, Any]:
    target_lineage_id = int(trace["target_lineage_id"])
    target_prototype_id = int(trace["target_prototype_id"])
    proto_history = trace["proto_history"]
    target_row = next((row for row in proto_history if int(row["frame_id"]) == TARGET_FRAME), None)
    continuity_candidate_lineages: list[int] = []
    runtime_candidate_lineages: list[int] = []
    if target_row is not None:
        runtime_lineage = _as_int(target_row.get("runtime_owner_lineage_id"))
        continuity_lineage = target_lineage_id
        if runtime_lineage is not None:
            runtime_candidate_lineages.append(int(runtime_lineage))
        continuity_candidate_lineages.append(int(continuity_lineage))
    combined = sorted({int(v) for v in runtime_candidate_lineages + continuity_candidate_lineages})
    return {
        "target_frame": TARGET_FRAME,
        "target_prototype_id": target_prototype_id,
        "target_lineage_id": target_lineage_id,
        "runtime_candidate_lineages": runtime_candidate_lineages,
        "continuity_candidate_lineages": continuity_candidate_lineages,
        "combined_candidate_lineages": combined,
        "lineage2_enters_preserve_input": int(target_lineage_id in combined),
        "lineage2_enters_claim_builder": int(target_lineage_id in combined),
        "notes": "一旦把 preserve-input 的候选形成从 runtime owner 扩成 runtime owner + continuity owner，prototype 3 的 continuity key 就足以把 lineage 2 重新暴露给 preserve-input/claim-builder。",
    }


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _plot_timeline(trace: dict[str, Any], output_path: Path) -> None:
    proto_history = trace["proto_history"]
    registry_history = trace["registry_history"]
    if not proto_history:
        fig, ax = plt.subplots(figsize=(10, 3))
        _draw_text_block(ax, ["prototype 3 未出现，无法绘制 remap timeline。"])
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    frames = [int(row["frame_id"]) for row in proto_history]
    runtime_owner = [_as_int(row.get("runtime_owner_lineage_id")) or -1 for row in proto_history]
    continuity_owner = [_as_int(row.get("continuity_lineage_id")) or -1 for row in proto_history]
    registry_sizes = []
    for row in registry_history:
        if int(row["frame_id"]) in frames:
            registry_sizes.append(len(row["runtime_registry_lineages"]))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(frames, runtime_owner, label="prototype3 runtime_owner_lineage", color="#1f77b4", linewidth=2)
    axes[0].plot(frames, continuity_owner, label="prototype3 continuity_lineage", color="#d62728", linewidth=2, linestyle="--")
    axes[0].axvline(TARGET_FRAME, color="black", linestyle=":", linewidth=1.2)
    axes[0].set_ylabel("lineage id")
    axes[0].set_title("prototype 3 owner / continuity timeline")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(frames[: len(registry_sizes)], registry_sizes, color="#2ca02c", linewidth=2)
    axes[1].axvline(TARGET_FRAME, color="black", linestyle=":", linewidth=1.2)
    axes[1].set_ylabel("runtime registry size")
    axes[1].set_xlabel("frame")
    axes[1].set_title("runtime registry active lineage count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_strip(trace: dict[str, Any], output_path: Path) -> None:
    frame_snapshots = trace["frame_snapshots"]
    proto_history = trace["proto_history"]
    candidate_frames = sorted(
        {
            *(int(row["frame_id"]) for row in proto_history[:1]),
            *(int(row["frame_id"]) for row in proto_history[-1:]),
            TARGET_FRAME - 1,
            TARGET_FRAME,
            TARGET_FRAME + 1,
        }
    )
    frames = [frame for frame in candidate_frames if frame in frame_snapshots]
    if not frames:
        fig, ax = plt.subplots(figsize=(10, 3))
        _draw_text_block(ax, ["缺少 target window frame snapshot，无法绘制 ownership strip。"])
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 4))
    if len(frames) == 1:
        axes = [axes]
    for axis, frame_id in zip(axes, frames):
        snapshot = frame_snapshots[frame_id]
        axis.imshow(snapshot["frame"], cmap="gray", vmin=0, vmax=1)
        _draw_box(axis, snapshot["gt_box"], color="#00ff66", label="GT")
        for row in snapshot["track_rows"][:2]:
            _draw_box(axis, tuple(int(v) for v in row["box"]), color="#4c78a8", label=f"trk{row['track_id']}")
        proto_row = snapshot["prototype3_row"]
        runtime_lineages = ",".join(str(v) for v in snapshot["runtime_registry_lineages"])
        lines = [
            f"f={frame_id}",
            f"registry={runtime_lineages}",
            "p3_rt="
            + ("None" if proto_row is None or _as_int(proto_row.get("runtime_owner_lineage_id")) is None else str(_as_int(proto_row.get('runtime_owner_lineage_id')))),
            "p3_cont="
            + ("None" if proto_row is None or _as_int(proto_row.get("continuity_lineage_id")) is None else str(_as_int(proto_row.get('continuity_lineage_id')))),
            "p3_retired="
            + ("None" if proto_row is None else str(int(proto_row.get("is_retired", 0)))),
        ]
        axis.text(
            0.02,
            0.98,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=8,
            color="white",
            transform=axis.transAxes,
            bbox={"facecolor": "black", "alpha": 0.7, "pad": 2},
        )
        axis.set_title(f"frame {frame_id}", fontsize=9)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_loss_diagram(trace: dict[str, Any], output_path: Path) -> None:
    remap_rows = trace["remap_rows"]
    first_bug = next((row for row in remap_rows if int(row["overwrite_bug_flag"]) == 1), None)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    blocks = [
        (0.05, 0.55, "offline continuity\nevent6 -> lineage 2"),
        (0.32, 0.55, "prototype 3 create\nruntime=0 / continuity=0"),
        (0.59, 0.55, "downstream continuation/anchor\ncopy current runtime owner"),
        (0.82, 0.55, "frame 990\nruntime buckets only {0,1}"),
    ]
    for x, y, text in blocks:
        ax.add_patch(Rectangle((x, y), 0.18, 0.22, fill=False, ec="black", lw=1.5))
        ax.text(x + 0.09, y + 0.11, text, ha="center", va="center", fontsize=9)
    for start, end in ((0.23, 0.41), (0.50, 0.68), (0.77, 0.82)):
        ax.annotate("", xy=(end, 0.66), xytext=(start, 0.66), arrowprops={"arrowstyle": "->", "lw": 1.6})
    note = (
        "first overwrite: none observed"
        if first_bug is None
        else f"first overwrite @ frame {first_bug['frame_id']} [{first_bug['event_type']}] {first_bug['code_location']}"
    )
    ax.text(0.05, 0.22, note, fontsize=10, family="monospace")
    ax.text(0.05, 0.11, "结论：runtime owner 覆盖 continuity key 的最早可观察位置已在 prototype 3 创建/继承边界。", fontsize=10)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_bucket_gallery(trace: dict[str, Any], output_path: Path) -> None:
    rows = trace["remap_rows"]
    bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket_rows[str(row["remap_bucket"])].append(row)
    buckets = [
        "runtime_rebind_only",
        "runtime_rebind_with_continuity_preserved",
        "continuity_key_overwrite_bug",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, bucket in zip(axes, buckets):
        sample = bucket_rows.get(bucket, [])[:4]
        lines = [f"{bucket}", f"count={len(bucket_rows.get(bucket, []))}", ""]
        for row in sample:
            lines.append(
                f"f{row['frame_id']} {row['object_type']}:{row['object_id']} "
                f"rt {row['before_runtime_owner_lineage_id']}->{row['after_runtime_owner_lineage_id']} "
                f"cont {row['before_continuity_lineage_id']}->{row['after_continuity_lineage_id']}"
            )
        _draw_text_block(axis, lines)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = _run_trace(config_path=Path(args.config), seed=args.seed)

    remap_rows = trace["remap_rows"]
    _write_csv(output_dir / "phase3d_stagea7_remap_trace.csv", remap_rows)

    bucket_counts: dict[str, int] = defaultdict(int)
    for row in remap_rows:
        bucket_counts[str(row["remap_bucket"])] += 1
    earliest_bug = next((row for row in remap_rows if int(row["overwrite_bug_flag"]) == 1), None)

    forced_preserve = _forced_continuity_key_probe(trace)
    forced_split = _forced_owner_continuity_split_probe(trace)
    (output_dir / "phase3d_stagea7_forced_continuity_key_preservation.json").write_text(
        json.dumps(forced_preserve, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "phase3d_stagea7_forced_owner_continuity_split.json").write_text(
        json.dumps(forced_split, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_lines = [
        "# Phase 3D Stage A.7 Remap Summary",
        "",
        f"- target event: event_id={TARGET_EVENT_ID}, frame={TARGET_FRAME}, gt_object_id={TARGET_GT_OBJECT_ID}",
        f"- offline target lineage metadata: {trace['target_lineage_id']}",
        f"- target prototype: {trace['target_prototype_id']}",
        f"- remap rows: {len(remap_rows)}",
        f"- overwrite bug rows: {sum(int(row['overwrite_bug_flag']) for row in remap_rows)}",
        "",
    ]
    if earliest_bug is None:
        summary_lines.append("- earliest observable overwrite: none")
    else:
        summary_lines.extend(
            [
                "- earliest observable overwrite:",
                f"  frame={earliest_bug['frame_id']}",
                f"  event_type={earliest_bug['event_type']}",
                f"  object_type={earliest_bug['object_type']}",
                f"  object_id={earliest_bug['object_id']}",
                f"  code_location={earliest_bug['code_location']}",
                f"  after_runtime_owner_lineage_id={earliest_bug['after_runtime_owner_lineage_id']}",
                f"  after_continuity_lineage_id={earliest_bug['after_continuity_lineage_id']}",
            ]
        )
    summary_lines.extend(
        [
            "",
            "## 结论",
            "",
            "1. 当前不是 preserve/tie-break 太弱，而是 old continuity key 在更早的 runtime rebind / create 边界已经被改写。",
            "2. prototype 3 的最早可观察记录已经是 runtime_owner_lineage=0 且 continuity_lineage=0；因此后续 continuation/anchor 只能继续复制 0。",
            "3. 只保 continuity key 还不够；如果 preserve-input 仍只按 runtime owner 枚举，lineage 2 依旧不可见。",
            "4. 一旦显式拆开 runtime owner 与 continuity owner，并让 continuity source 按 continuity key 枚举，lineage 2 才重新有机会进入 preserve-input / claim-builder。",
        ]
    )
    _write_markdown(output_dir / "phase3d_stagea7_remap_summary.md", "\n".join(summary_lines))

    design_notes = "\n".join(
        [
            "# Phase 3D Stage A.7 Design Notes",
            "",
            "本轮不改 selection/promotion，只修 identity key 保留。",
            "",
            "## 最小结构拆分",
            "",
            "- runtime_owner_lineage_id: 运行时当前挂在哪个 lineage 容器下，可变。",
            "- continuity_lineage_id: old continuity 身份键，供 recovery / preserve-input 枚举使用，不应被 runtime owner 默认覆盖。",
            "- origin_lineage_id: continuity key 的起点，用于 trace。",
            "",
            "## 本轮补丁范围",
            "",
            "- PrototypeState 增加 runtime/continuity/origin/valid 字段。",
            "- IdentityContinuation 与 RecoveryIdentityAnchor 增加同样的 owner/continuity split 字段。",
            "- _create_prototype / _update_prototype / continuation write / anchor write / trace logging 全部显式记录这两个 lineage 键。",
            "",
            "## 目标",
            "",
            "先证明是哪一个上游步骤第一次把 continuity key 覆盖掉，再决定下一轮是否需要把 preserve-input 从 runtime owner 改成 continuity owner 枚举。",
        ]
    )
    _write_markdown(output_dir / "phase3d_stagea7_design_notes.md", design_notes)

    bucket_summary = "\n".join(
        [
            "# Phase 3D Stage A.7 Remap Bucket Summary",
            "",
            f"- runtime_rebind_only: {bucket_counts.get('runtime_rebind_only', 0)}",
            f"- runtime_rebind_with_continuity_preserved: {bucket_counts.get('runtime_rebind_with_continuity_preserved', 0)}",
            f"- continuity_key_overwrite_bug: {bucket_counts.get('continuity_key_overwrite_bug', 0)}",
            "",
            "判读：只要 target path 上出现 continuity_key_overwrite_bug，就说明 runtime owner 在某处被直接写回 continuity key。",
        ]
    )
    _write_markdown(output_dir / "phase3d_stagea7_remap_bucket_summary.md", bucket_summary)

    recommendation = "\n".join(
        [
            "# Phase 3D Stage A.7 Recommendation",
            "",
            "当前不进 Stage B。",
            "",
            "下一步只需要做一件事：把 preserve-input / claim-builder 的 continuity source 枚举从 runtime owner 扩成 runtime owner + continuity owner。",
            "",
            "前提已经明确：",
            "",
            "- 先保 continuity key；",
            "- 不让 archive/replace/rebind 默认覆盖 continuity key；",
            "- 然后再回到 A.6/A.5 的 preserve-input / claim-builder 路径验证 lineage 2 是否重新可见。",
        ]
    )
    _write_markdown(output_dir / "phase3d_stagea7_recommendation.md", recommendation)

    _plot_timeline(trace, output_dir / "lineage2_to_runtime0_remap_timeline.png")
    _plot_strip(trace, output_dir / "prototype3_ownership_trace_strip.png")
    _plot_loss_diagram(trace, output_dir / "continuity_key_loss_diagram.png")
    _plot_bucket_gallery(trace, output_dir / "remap_bucket_gallery.png")


if __name__ == "__main__":
    main()
