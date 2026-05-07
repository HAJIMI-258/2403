"""Phase 3D Stage A.6: preserve-input lineage formation trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

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
    WINDOW_RIGHT,
    _draw_box,
    _gt_box,
    _iou,
    _load_target_metadata,
    _parse_box,
    _write_csv,
)
from experiments.scenario_presets import build_phase3_track_scenarios  # noqa: E402
from nops_owr.encoder import MinimalSpikeEncoder  # noqa: E402
from nops_owr.memory import MinimalPrototypeMemory  # noqa: E402
from nops_owr.objectness import MinimalObjectnessField  # noqa: E402
from nops_owr.tracking import MinimalTemporalIdentityTracker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3D Stage A.6 preserve-input trace.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--output-dir", default="results/phase3d")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_track_c_sequence(config_path: Path, *, seed: int):
    base_config = load_synth_dataset_config(config_path)
    scenario_map = {s["name"]: s["config"] for s in build_phase3_track_scenarios(base_config)}
    return SyntheticStreamGenerator(scenario_map[TRACK_C_NAME], seed=seed).generate_sequence(0)


def _parse_lineage_list(value: Any) -> list[int]:
    if value in (None, "", "None"):
        return []
    if isinstance(value, list):
        return [int(item) for item in value if item not in (None, "", "None")]
    return [int(part) for part in str(value).split("|") if part not in ("", "None")]


def _target_selection(selection_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [
        row
        for row in selection_rows
        if int(row.get("frame_id", -1)) == TARGET_FRAME and int(row.get("is_target_proposal", 0)) == 1
    ]
    if not rows:
        return None
    return sorted(rows, key=lambda row: -float(row.get("proposal_iou_to_gt", 0.0)))[0]


def _target_preserve_rows(preserve_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in preserve_rows
        if int(row.get("frame_id", -1)) == TARGET_FRAME and int(row.get("is_target_proposal", 0)) == 1
    ]
    return sorted(rows, key=lambda row: int(row["candidate_lineage_id"]))


def _target_claim_rows(claim_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in claim_rows
        if int(row.get("frame_id", -1)) == TARGET_FRAME and int(row.get("is_target_proposal", 0)) == 1
    ]
    return sorted(
        rows,
        key=lambda row: (
            -int(row.get("final_claim_visible", 0)),
            -float(row.get("claim_score_total") or 0.0),
            int(row["candidate_lineage_id"]),
        ),
    )


def _target_lineage_row(rows: list[dict[str, Any]], target_lineage_id: int | None) -> dict[str, Any] | None:
    if target_lineage_id is None:
        return None
    for row in rows:
        if int(row["candidate_lineage_id"]) == int(target_lineage_id):
            return row
    return None


def _failure_bucket(
    preserve_row: dict[str, Any] | None,
    claim_row: dict[str, Any] | None,
    selection_row: dict[str, Any] | None,
    target_lineage_id: int | None,
) -> str:
    if target_lineage_id is None:
        return "missing_target_lineage"
    if preserve_row is None or int(preserve_row.get("entered_preserve_input", 0)) == 0:
        return "input_formation_failure"
    if claim_row is None or int(claim_row.get("target_lineage_claim_visible_final", 0)) == 0:
        return "claim_visibility_failure"
    selected_lineage = None if selection_row is None else selection_row.get("selected_lineage_id")
    if selected_lineage in (None, "", "None") or int(selected_lineage) != int(target_lineage_id):
        return "visible_but_underweighted_failure"
    return "target_lineage_selected"


def _run_trace(
    *,
    config_path: Path,
    seed: int,
    run_label: str,
    tracking_patch: dict[str, Any],
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
    tracking_config.update(tracking_patch)
    memory_config = dict(payload["memory"])
    memory_config.update(default_phase3d_stagea_memory_override())

    encoder = MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    objectness = MinimalObjectnessField(**payload["field"])
    tracker = MinimalTemporalIdentityTracker(**tracking_config)
    memory = MinimalPrototypeMemory(**memory_config)

    target_metadata = _load_target_metadata()
    preserve_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    frame_snapshots: dict[int, dict[str, Any]] = {}
    prev_memory_output = None
    window_start = TARGET_FRAME - WINDOW_LEFT
    window_end = TARGET_FRAME + WINDOW_RIGHT

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

        if current_frame.frame_index < window_start or current_frame.frame_index > window_end:
            continue

        gt_box = _gt_box(current_frame, TARGET_GT_OBJECT_ID)
        target_proposal_id = None
        best_iou = -1.0
        for row in tracking_output.recovery_selection_rows:
            proposal_box = _parse_box(row.get("proposal_box"))
            iou = _iou(proposal_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                target_proposal_id = int(row["proposal_id"])

        routing_map = {
            (int(row["frame_id"]), int(row["proposal_id"])): row for row in tracking_output.routing_debug_rows
        }

        for row in tracking_output.preserve_input_rows:
            if int(row["frame_id"]) != int(current_frame.frame_index):
                continue
            proposal_box = _parse_box(row.get("proposal_box"))
            routing_row = routing_map.get((int(row["frame_id"]), int(row["proposal_id"])), {})
            active_candidates = routing_row.get("active_candidates_topk", []) or []
            runtime_lineages = [
                int(item["lineage_id"])
                for item in (routing_row.get("proposal_lineage_hint_topk", []) or [])
                if item.get("lineage_id") is not None
            ]
            preserve_rows.append(
                {
                    **row,
                    "run_label": run_label,
                    "proposal_iou_to_gt": _iou(proposal_box, gt_box),
                    "is_target_proposal": int(target_proposal_id is not None and int(row["proposal_id"]) == int(target_proposal_id)),
                    "target_lineage_match": int(
                        target_metadata["old_lineage_id"] is not None
                        and int(row["candidate_lineage_id"]) == int(target_metadata["old_lineage_id"])
                    ),
                    "target_lineage_id": target_metadata["old_lineage_id"],
                    "target_lineage_in_before_prune": int(
                        target_metadata["old_lineage_id"] is not None
                        and int(target_metadata["old_lineage_id"])
                        in _parse_lineage_list(row.get("preserve_candidate_lineages_before_prune"))
                    ),
                    "target_lineage_in_after_prune": int(
                        target_metadata["old_lineage_id"] is not None
                        and int(target_metadata["old_lineage_id"])
                        in _parse_lineage_list(row.get("preserve_candidate_lineages_after_prune"))
                    ),
                    "proposal_lineage_hint_topk": "|".join(str(v) for v in runtime_lineages)
                    if runtime_lineages
                    else row.get("proposal_lineage_hint_topk", ""),
                    "preempting_active_prototype_id": None
                    if not active_candidates or active_candidates[0].get("prototype_id") is None
                    else int(active_candidates[0]["prototype_id"]),
                }
            )

        for row in tracking_output.lineage_claim_rows:
            if int(row["frame_id"]) != int(current_frame.frame_index):
                continue
            proposal_box = _parse_box(row.get("proposal_box"))
            claim_rows.append(
                {
                    **row,
                    "run_label": run_label,
                    "proposal_iou_to_gt": _iou(proposal_box, gt_box),
                    "is_target_proposal": int(target_proposal_id is not None and int(row["proposal_id"]) == int(target_proposal_id)),
                    "target_lineage_match": int(
                        target_metadata["old_lineage_id"] is not None
                        and int(row["candidate_lineage_id"]) == int(target_metadata["old_lineage_id"])
                    ),
                }
            )

        for row in tracking_output.recovery_selection_rows:
            if int(row["frame_id"]) != int(current_frame.frame_index):
                continue
            proposal_box = _parse_box(row.get("proposal_box"))
            selection_rows.append(
                {
                    **row,
                    "run_label": run_label,
                    "proposal_iou_to_gt": _iou(proposal_box, gt_box),
                    "is_target_proposal": int(target_proposal_id is not None and int(row["proposal_id"]) == int(target_proposal_id)),
                    "target_lineage_match": int(
                        target_metadata["old_lineage_id"] is not None
                        and row.get("selected_lineage_id") not in ("", "None", None)
                        and int(row["selected_lineage_id"]) == int(target_metadata["old_lineage_id"])
                    ),
                }
            )

        frame_snapshots[int(current_frame.frame_index)] = {
            "image": current_frame.frame.copy(),
            "gt_box": gt_box,
            "target_proposal_id": target_proposal_id,
            "target_proposal_box": None,
        }
        for row in tracking_output.recovery_selection_rows:
            if int(row["frame_id"]) != int(current_frame.frame_index):
                continue
            if target_proposal_id is not None and int(row["proposal_id"]) == int(target_proposal_id):
                frame_snapshots[int(current_frame.frame_index)]["target_proposal_box"] = _parse_box(row.get("proposal_box"))
                break

    return {
        "run_label": run_label,
        "preserve_rows": preserve_rows,
        "claim_rows": claim_rows,
        "selection_rows": selection_rows,
        "frame_snapshots": frame_snapshots,
        "target_metadata": target_metadata,
    }


def _write_summary(
    path: Path,
    *,
    baseline_row: dict[str, Any] | None,
    continuity_row: dict[str, Any] | None,
    three_source_row: dict[str, Any] | None,
    baseline_claim: dict[str, Any] | None,
    continuity_claim: dict[str, Any] | None,
    three_source_claim: dict[str, Any] | None,
    baseline_selection: dict[str, Any] | None,
    continuity_selection: dict[str, Any] | None,
    three_source_selection: dict[str, Any] | None,
    target_metadata: dict[str, Any],
) -> None:
    lines = [
        "# Phase 3D Stage A.6 Preserve-Input Summary",
        "",
        f"- target event: `{TARGET_EVENT_ID}`",
        f"- target frame: `{TARGET_FRAME}`",
        f"- target lineage: `{target_metadata['old_lineage_id']}`",
        "",
        "## Baseline",
        "",
        f"- entered preserve input: `{0 if baseline_row is None else baseline_row.get('entered_preserve_input', 0)}`",
        f"- entered claim-builder input: `{0 if baseline_row is None else baseline_row.get('entered_claim_builder_input', 0)}`",
        f"- claim visible: `{0 if baseline_claim is None else baseline_claim.get('target_lineage_claim_visible_final', 0)}`",
        f"- final selected lineage: `{None if baseline_selection is None else baseline_selection.get('selected_lineage_id')}`",
        "",
        "## Force Continuity-Lineage Exposure",
        "",
        f"- entered preserve input: `{0 if continuity_row is None else continuity_row.get('entered_preserve_input', 0)}`",
        f"- entered claim-builder input: `{0 if continuity_row is None else continuity_row.get('entered_claim_builder_input', 0)}`",
        f"- claim visible: `{0 if continuity_claim is None else continuity_claim.get('target_lineage_claim_visible_final', 0)}`",
        f"- final selected lineage: `{None if continuity_selection is None else continuity_selection.get('selected_lineage_id')}`",
        "",
        "## Force Runtime + Recovery + Continuity Input",
        "",
        f"- entered preserve input: `{0 if three_source_row is None else three_source_row.get('entered_preserve_input', 0)}`",
        f"- entered claim-builder input: `{0 if three_source_row is None else three_source_row.get('entered_claim_builder_input', 0)}`",
        f"- claim visible: `{0 if three_source_claim is None else three_source_claim.get('target_lineage_claim_visible_final', 0)}`",
        f"- final selected lineage: `{None if three_source_selection is None else three_source_selection.get('selected_lineage_id')}`",
        "",
        "## Direct Answers",
        "",
        "1. 先看 target lineage 是否进入 preserve input；如果这里还是 0，后面的 claim / tie-break 都没有解释价值。",
        "2. continuity-lineage candidate formation 的目标不是直接让它赢，而是先让合法 continuity lineage 进入 preserve 输入和 claim builder。",
        "3. 只有 target lineage 先进入 preserve 输入，后续的 claim visibility / selection 才值得继续修。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_design_notes(path: Path) -> None:
    lines = [
        "# Phase 3D Stage A.6 Design Notes",
        "",
        "1. preserve 输入候选不再只看 runtime lineage hints。",
        "2. 新增三路输入拆分：runtime hints / recovery surface / continuity evidence。",
        "3. continuity-signaled lineage 只要具备合法 recovery surface，就必须进入 preserve 输入候选。",
        "4. 本轮不改 final ranking，不改 identity tie-break，只修 preserve 输入形成。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_failure_bucket_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Phase 3D Stage A.6 Failure Bucket Summary", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['run_label']}",
                "",
                f"- `failure_bucket = {row['failure_bucket']}`",
                f"- `entered_preserve_input = {row['entered_preserve_input']}`",
                f"- `entered_claim_builder = {row['entered_claim_builder']}`",
                f"- `visible_in_claim_set = {row['visible_in_claim_set']}`",
                f"- `final_selected_lineage = {row['final_selected_lineage']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_preserve_strip(path: Path, traces: list[dict[str, Any]], summary_rows: list[dict[str, Any]], target_lineage_id: int | None) -> None:
    frames = list(range(TARGET_FRAME - 2, TARGET_FRAME + 3))
    fig, axes = plt.subplots(len(traces), len(frames), figsize=(3.8 * len(frames), 3.8 * len(traces)))
    if len(traces) == 1:
        axes = np.asarray([axes])
    summary_lookup = {row["run_label"]: row for row in summary_rows}
    for row_idx, trace in enumerate(traces):
        run_label = trace["run_label"]
        target_rows = _target_preserve_rows(trace["preserve_rows"])
        target_row = _target_lineage_row(target_rows, target_lineage_id)
        for col_idx, frame_id in enumerate(frames):
            axis = axes[row_idx, col_idx]
            snapshot = trace["frame_snapshots"].get(frame_id)
            if snapshot is None:
                axis.axis("off")
                continue
            axis.imshow(snapshot["image"])
            axis.axis("off")
            _draw_box(axis, snapshot.get("gt_box"), color="lime", label="GT")
            _draw_box(axis, snapshot.get("target_proposal_box"), color="cyan", label="proposal")
            title = f"f{frame_id}"
            if frame_id == TARGET_FRAME:
                title += (
                    f"\nin={0 if target_row is None else target_row.get('entered_preserve_input', 0)}"
                    f" claimIn={0 if target_row is None else target_row.get('entered_claim_builder_input', 0)}"
                    f"\n{summary_lookup[run_label]['failure_bucket']}"
                )
            axis.set_title(title, fontsize=8)
            if col_idx == 0:
                axis.text(2, 12, run_label, color="white", fontsize=9, bbox={"facecolor": "black", "alpha": 0.6, "pad": 2})
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_source_barplot(path: Path, rows: list[dict[str, Any]], target_lineage_id: int | None) -> None:
    lineages = sorted({int(row["candidate_lineage_id"]) for row in rows} | ({int(target_lineage_id)} if target_lineage_id is not None else set()))
    labels = []
    runtime = []
    surface = []
    continuity = []
    merged = []
    row_lookup = {int(row["candidate_lineage_id"]): row for row in rows}
    for lineage_id in lineages:
        row = row_lookup.get(int(lineage_id), {})
        labels.append(str(lineage_id))
        runtime.append(float(row.get("runtime_hint_match", 0)))
        surface.append(float(row.get("surface_hint_score", 0.0)))
        continuity.append(float(row.get("continuity_lineage_hint_score", 0.0)))
        merged.append(float(row.get("entered_claim_builder_input", 0)))
    x = np.arange(len(labels))
    width = 0.2
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(x - 1.5 * width, runtime, width=width, label="runtime")
    axis.bar(x - 0.5 * width, surface, width=width, label="surface")
    axis.bar(x + 0.5 * width, continuity, width=width, label="continuity")
    axis.bar(x + 1.5 * width, merged, width=width, label="preserve->claim")
    axis.set_xticks(x, labels)
    axis.set_xlabel("lineage id")
    axis.set_ylabel("score / flag")
    axis.set_title("Frame 990 preserve-input source scores")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_lineage_drop_diagram(path: Path, baseline_row: dict[str, Any] | None, continuity_row: dict[str, Any] | None, three_source_row: dict[str, Any] | None, target_lineage_id: int | None) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.6))
    axis.axis("off")
    lines = [
        f"target lineage = {target_lineage_id}",
        "",
        f"baseline: in={0 if baseline_row is None else baseline_row.get('entered_preserve_input', 0)}, claimIn={0 if baseline_row is None else baseline_row.get('entered_claim_builder_input', 0)}, reason={None if baseline_row is None else baseline_row.get('preserve_input_drop_reason')}",
        f"continuity exposure: in={0 if continuity_row is None else continuity_row.get('entered_preserve_input', 0)}, claimIn={0 if continuity_row is None else continuity_row.get('entered_claim_builder_input', 0)}, reason={None if continuity_row is None else continuity_row.get('preserve_input_drop_reason')}",
        f"three-source input: in={0 if three_source_row is None else three_source_row.get('entered_preserve_input', 0)}, claimIn={0 if three_source_row is None else three_source_row.get('entered_claim_builder_input', 0)}, reason={None if three_source_row is None else three_source_row.get('preserve_input_drop_reason')}",
    ]
    axis.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_failure_gallery(path: Path, traces: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, len(traces), figsize=(5.0 * len(traces), 4.5))
    if len(traces) == 1:
        axes = [axes]
    summary_lookup = {row["run_label"]: row for row in summary_rows}
    for axis, trace in zip(axes, traces):
        snapshot = trace["frame_snapshots"].get(TARGET_FRAME)
        if snapshot is None:
            axis.axis("off")
            continue
        axis.imshow(snapshot["image"])
        axis.axis("off")
        _draw_box(axis, snapshot.get("gt_box"), color="lime", label="GT")
        _draw_box(axis, snapshot.get("target_proposal_box"), color="cyan", label="proposal")
        row = summary_lookup[trace["run_label"]]
        axis.set_title(
            f"{trace['run_label']}\n{row['failure_bucket']}\nin={row['entered_preserve_input']} claimIn={row['entered_claim_builder']}",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_recommendation(path: Path, rows: list[dict[str, Any]]) -> None:
    latest = rows[-1] if rows else None
    lines = ["# Phase 3D Stage A.6 Recommendation", ""]
    if latest is None:
        lines.append("没有生成 Stage A.6 的 target preserve 输入记录。")
    else:
        lines.extend(
            [
                f"- latest run: `{latest['run_label']}`",
                f"- failure bucket: `{latest['failure_bucket']}`",
                f"- entered preserve input: `{latest['entered_preserve_input']}`",
                f"- entered claim builder: `{latest['entered_claim_builder']}`",
                f"- visible in claim set: `{latest['visible_in_claim_set']}`",
                "",
                "只要 target lineage 还进不了 preserve 输入，就不要进入 Stage B。",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)

    baseline = _run_trace(
        config_path=config_path,
        seed=args.seed,
        run_label="baseline_reroute",
        tracking_patch={
            "enable_phase3d_claim_preservation_repair": False,
            "enable_phase3d_continuity_lineage_repair": False,
            "enable_phase3d_three_source_preserve_input": False,
        },
    )
    continuity = _run_trace(
        config_path=config_path,
        seed=args.seed,
        run_label="forced_continuity_exposure",
        tracking_patch={
            "enable_phase3d_claim_preservation_repair": True,
            "enable_phase3d_continuity_lineage_repair": True,
            "enable_phase3d_three_source_preserve_input": False,
        },
    )
    three_source = _run_trace(
        config_path=config_path,
        seed=args.seed,
        run_label="forced_three_source_input",
        tracking_patch={
            "enable_phase3d_claim_preservation_repair": True,
            "enable_phase3d_continuity_lineage_repair": True,
            "enable_phase3d_three_source_preserve_input": True,
        },
    )

    all_preserve_rows = baseline["preserve_rows"] + continuity["preserve_rows"] + three_source["preserve_rows"]
    target_metadata = baseline["target_metadata"]
    target_lineage_id = target_metadata["old_lineage_id"]

    baseline_preserve = _target_lineage_row(_target_preserve_rows(baseline["preserve_rows"]), target_lineage_id)
    continuity_preserve = _target_lineage_row(_target_preserve_rows(continuity["preserve_rows"]), target_lineage_id)
    three_source_preserve = _target_lineage_row(_target_preserve_rows(three_source["preserve_rows"]), target_lineage_id)
    baseline_claim = _target_lineage_row(_target_claim_rows(baseline["claim_rows"]), target_lineage_id)
    continuity_claim = _target_lineage_row(_target_claim_rows(continuity["claim_rows"]), target_lineage_id)
    three_source_claim = _target_lineage_row(_target_claim_rows(three_source["claim_rows"]), target_lineage_id)
    baseline_selection = _target_selection(baseline["selection_rows"])
    continuity_selection = _target_selection(continuity["selection_rows"])
    three_source_selection = _target_selection(three_source["selection_rows"])

    _write_csv(output_dir / "phase3d_stagea6_preserve_input_trace.csv", all_preserve_rows)

    summary_rows = []
    for trace, preserve_row, claim_row, selection_row in (
        (baseline, baseline_preserve, baseline_claim, baseline_selection),
        (continuity, continuity_preserve, continuity_claim, continuity_selection),
        (three_source, three_source_preserve, three_source_claim, three_source_selection),
    ):
        summary_rows.append(
            {
                "run_label": trace["run_label"],
                "failure_bucket": _failure_bucket(preserve_row, claim_row, selection_row, target_lineage_id),
                "entered_preserve_input": 0 if preserve_row is None else int(preserve_row.get("entered_preserve_input", 0)),
                "entered_claim_builder": 0 if preserve_row is None else int(preserve_row.get("entered_claim_builder_input", 0)),
                "visible_in_claim_set": 0 if claim_row is None else int(claim_row.get("target_lineage_claim_visible_final", 0)),
                "final_selected_lineage": None if selection_row is None else selection_row.get("selected_lineage_id"),
            }
        )

    forced_continuity_exposure = {
        "target_event_id": TARGET_EVENT_ID,
        "target_frame": TARGET_FRAME,
        "target_lineage_id": target_lineage_id,
        "baseline_entered_preserve_input": 0 if baseline_preserve is None else int(baseline_preserve.get("entered_preserve_input", 0)),
        "baseline_entered_claim_builder": 0 if baseline_preserve is None else int(baseline_preserve.get("entered_claim_builder_input", 0)),
        "continuity_entered_preserve_input": 0 if continuity_preserve is None else int(continuity_preserve.get("entered_preserve_input", 0)),
        "continuity_entered_claim_builder": 0 if continuity_preserve is None else int(continuity_preserve.get("entered_claim_builder_input", 0)),
        "baseline_final_lineage": None if baseline_selection is None else baseline_selection.get("selected_lineage_id"),
        "continuity_final_lineage": None if continuity_selection is None else continuity_selection.get("selected_lineage_id"),
    }
    (output_dir / "phase3d_stagea6_forced_continuity_exposure.json").write_text(json.dumps(forced_continuity_exposure, indent=2), encoding="utf-8")

    forced_three_source_input = {
        "target_event_id": TARGET_EVENT_ID,
        "target_frame": TARGET_FRAME,
        "target_lineage_id": target_lineage_id,
        "continuity_entered_preserve_input": 0 if continuity_preserve is None else int(continuity_preserve.get("entered_preserve_input", 0)),
        "continuity_entered_claim_builder": 0 if continuity_preserve is None else int(continuity_preserve.get("entered_claim_builder_input", 0)),
        "three_source_entered_preserve_input": 0 if three_source_preserve is None else int(three_source_preserve.get("entered_preserve_input", 0)),
        "three_source_entered_claim_builder": 0 if three_source_preserve is None else int(three_source_preserve.get("entered_claim_builder_input", 0)),
        "continuity_final_lineage": None if continuity_selection is None else continuity_selection.get("selected_lineage_id"),
        "three_source_final_lineage": None if three_source_selection is None else three_source_selection.get("selected_lineage_id"),
    }
    (output_dir / "phase3d_stagea6_forced_three_source_input.json").write_text(json.dumps(forced_three_source_input, indent=2), encoding="utf-8")

    _write_summary(
        output_dir / "phase3d_stagea6_preserve_input_summary.md",
        baseline_row=baseline_preserve,
        continuity_row=continuity_preserve,
        three_source_row=three_source_preserve,
        baseline_claim=baseline_claim,
        continuity_claim=continuity_claim,
        three_source_claim=three_source_claim,
        baseline_selection=baseline_selection,
        continuity_selection=continuity_selection,
        three_source_selection=three_source_selection,
        target_metadata=target_metadata,
    )
    _write_design_notes(output_dir / "phase3d_stagea6_design_notes.md")
    _write_failure_bucket_summary(output_dir / "phase3d_stagea6_failure_bucket_summary.md", summary_rows)
    _save_preserve_strip(output_dir / "frame990_preserve_input_strip.png", [baseline, continuity, three_source], summary_rows, target_lineage_id)
    _save_source_barplot(output_dir / "preserve_input_source_barplot.png", _target_preserve_rows(three_source["preserve_rows"]), target_lineage_id)
    _save_lineage_drop_diagram(output_dir / "lineage2_drop_path_diagram.png", baseline_preserve, continuity_preserve, three_source_preserve, target_lineage_id)
    _save_failure_gallery(output_dir / "failure_bucket_gallery_stagea6.png", [baseline, continuity, three_source], summary_rows)
    _write_recommendation(output_dir / "phase3d_stagea6_recommendation.md", summary_rows)


if __name__ == "__main__":
    main()
