"""Phase 3D Stage A.5: target-lineage claim preservation + identity-aware tie-break."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
from experiments.scenario_presets import build_phase3_track_scenarios  # noqa: E402
from nops_owr.encoder import MinimalSpikeEncoder  # noqa: E402
from nops_owr.memory import MinimalPrototypeMemory  # noqa: E402
from nops_owr.objectness import MinimalObjectnessField  # noqa: E402
from nops_owr.tracking import MinimalTemporalIdentityTracker  # noqa: E402


TRACK_C_NAME = "track_c_long_horizon"
TARGET_EVENT_ID = 6
TARGET_GT_OBJECT_ID = 2
TARGET_FRAME = 990
WINDOW_LEFT = 12
WINDOW_RIGHT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3D Stage A.5 claim preservation trace.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--output-dir", default="results/phase3d")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def _gt_box(frame_sample, gt_object_id: int) -> tuple[int, int, int, int] | None:
    for instance_id, box in zip(frame_sample.instance_ids, frame_sample.boxes):
        if int(instance_id) == int(gt_object_id):
            return tuple(int(v) for v in box)
    return None


def _parse_box(value: Any) -> tuple[int, int, int, int] | None:
    if value in (None, "", "None"):
        return None
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    text = str(value).strip().strip("()")
    if not text:
        return None
    return tuple(int(part.strip()) for part in text.split(","))


def _iou(box_a: tuple[int, int, int, int] | None, box_b: tuple[int, int, int, int] | None) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _draw_box(axis, box: tuple[int, int, int, int] | None, *, color: str, label: str | None = None, lw: float = 1.8) -> None:
    if box is None:
        return
    x1, y1, x2, y2 = [int(v) for v in box]
    axis.add_patch(Rectangle((x1, y1), max(1, x2 - x1), max(1, y2 - y1), fill=False, ec=color, lw=lw))
    if label:
        axis.text(
            x1,
            max(0, y1 - 4),
            label,
            color=color,
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 1},
        )


def _load_target_metadata() -> dict[str, Any]:
    path = Path("results/phase3d/phase3d_event_audit.csv")
    metadata = {
        "event_id": TARGET_EVENT_ID,
        "gt_object_id": TARGET_GT_OBJECT_ID,
        "target_frame": TARGET_FRAME,
        "old_track_id": None,
        "old_lineage_id": None,
        "old_prototype_id": None,
    }
    if not path.exists():
        return metadata
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row.get("event_id", -1)) != TARGET_EVENT_ID:
                continue
            metadata["old_track_id"] = None if row.get("old_track_id") in ("", "None", None) else int(row["old_track_id"])
            metadata["old_lineage_id"] = None if row.get("old_lineage_id") in ("", "None", None) else int(row["old_lineage_id"])
            metadata["old_prototype_id"] = None if row.get("old_prototype_id") in ("", "None", None) else int(row["old_prototype_id"])
            break
    return metadata


def _target_selection(selection_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [
        row
        for row in selection_rows
        if int(row.get("frame_id", -1)) == TARGET_FRAME and int(row.get("is_target_proposal", 0)) == 1
    ]
    if not rows:
        return None
    return sorted(rows, key=lambda row: -float(row.get("proposal_iou_to_gt", 0.0)))[0]


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


def _target_candidate_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in candidate_rows
        if int(row.get("frame_id", -1)) == TARGET_FRAME and int(row.get("is_target_proposal", 0)) == 1
    ]
    return sorted(rows, key=lambda row: (-float(row["recovery_score_total"]), float(row["raw_cost"])))


def _target_lineage_claim_row(claim_rows: list[dict[str, Any]], target_lineage_id: int | None) -> dict[str, Any] | None:
    if target_lineage_id is None:
        return None
    for row in claim_rows:
        if int(row["candidate_lineage_id"]) == int(target_lineage_id):
            return row
    return None


def _failure_bucket(selection_row: dict[str, Any] | None, claim_rows: list[dict[str, Any]], target_lineage_id: int | None) -> str:
    if selection_row is None or target_lineage_id is None:
        return "missing_target_selection"
    target_claim = _target_lineage_claim_row(claim_rows, target_lineage_id)
    if target_claim is None or int(target_claim.get("target_lineage_claim_visible_final", 0)) == 0:
        return "claim_visibility_failure"
    if int(selection_row.get("selected_lineage_id", -1)) != int(target_lineage_id):
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
            "routing_recovery_max_distance": 0.70,
            "routing_recovery_min_confidence": 0.30,
            "routing_active_claim_override_margin": 0.20,
            "routing_topk": 3,
            "claim_preserve_min_score": 0.25,
            "identity_preference_margin": 0.08,
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
    candidate_rows: list[dict[str, Any]] = []
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

        frame_candidate_rows = [
            row for row in tracking_output.recovery_candidate_rows if int(row["frame_id"]) == int(current_frame.frame_index)
        ]
        frame_claim_rows = [
            row for row in tracking_output.lineage_claim_rows if int(row["frame_id"]) == int(current_frame.frame_index)
        ]
        frame_selection_rows = [
            row for row in tracking_output.recovery_selection_rows if int(row["frame_id"]) == int(current_frame.frame_index)
        ]

        for row in frame_candidate_rows:
            proposal_box = _parse_box(row.get("proposal_box"))
            candidate_rows.append(
                {
                    **row,
                    "run_label": run_label,
                    "proposal_iou_to_gt": _iou(proposal_box, gt_box),
                    "is_target_proposal": int(target_proposal_id is not None and int(row["proposal_id"]) == int(target_proposal_id)),
                    "target_lineage_match": int(
                        target_metadata["old_lineage_id"] is not None
                        and int(row["candidate_lineage_id"]) == int(target_metadata["old_lineage_id"])
                    ),
                    "target_same_track_hint": int(
                        target_metadata["old_track_id"] is not None
                        and int(row["candidate_track_id"]) == int(target_metadata["old_track_id"])
                    ),
                    "target_same_prototype_hint": int(
                        target_metadata["old_prototype_id"] is not None
                        and row.get("candidate_prototype_id") not in ("", "None", None)
                        and int(row["candidate_prototype_id"]) == int(target_metadata["old_prototype_id"])
                    ),
                }
            )

        for row in frame_claim_rows:
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

        for row in frame_selection_rows:
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
                    "target_same_track_hint": int(
                        target_metadata["old_track_id"] is not None
                        and row.get("selected_track_id") not in ("", "None", None)
                        and int(row["selected_track_id"]) == int(target_metadata["old_track_id"])
                    ),
                    "target_same_prototype_hint": int(
                        target_metadata["old_prototype_id"] is not None
                        and row.get("selected_prototype_id") not in ("", "None", None)
                        and int(row["selected_prototype_id"]) == int(target_metadata["old_prototype_id"])
                    ),
                }
            )

        frame_snapshots[int(current_frame.frame_index)] = {
            "image": current_frame.frame.copy(),
            "gt_box": gt_box,
            "target_proposal_id": target_proposal_id,
            "target_proposal_box": None,
            "claim_rows": frame_claim_rows,
            "selection_rows": frame_selection_rows,
        }
        for row in frame_selection_rows:
            if target_proposal_id is not None and int(row["proposal_id"]) == int(target_proposal_id):
                frame_snapshots[int(current_frame.frame_index)]["target_proposal_box"] = _parse_box(row.get("proposal_box"))
                frame_snapshots[int(current_frame.frame_index)]["target_selection"] = row
                break

    return {
        "run_label": run_label,
        "candidate_rows": candidate_rows,
        "claim_rows": claim_rows,
        "selection_rows": selection_rows,
        "frame_snapshots": frame_snapshots,
        "target_metadata": target_metadata,
    }


def _write_summary(
    path: Path,
    *,
    baseline_selection: dict[str, Any] | None,
    preserve_selection: dict[str, Any] | None,
    tiebreak_selection: dict[str, Any] | None,
    baseline_target_claim: dict[str, Any] | None,
    preserve_target_claim: dict[str, Any] | None,
    tiebreak_target_claim: dict[str, Any] | None,
    target_metadata: dict[str, Any],
) -> None:
    lines = [
        "# Phase 3D Stage A.5 Claim Preservation Summary",
        "",
        f"- target event: `{TARGET_EVENT_ID}`",
        f"- target frame: `{TARGET_FRAME}`",
        f"- target lineage: `{target_metadata['old_lineage_id']}`",
        "",
        "## Baseline",
        "",
        f"- target lineage visible: `{0 if baseline_target_claim is None else baseline_target_claim.get('target_lineage_claim_visible_final', 0)}`",
        f"- claim drop stage: `{None if baseline_target_claim is None else baseline_target_claim.get('claim_drop_stage')}`",
        f"- claim drop reason: `{None if baseline_target_claim is None else baseline_target_claim.get('claim_drop_reason')}`",
        f"- final selected lineage: `{None if baseline_selection is None else baseline_selection.get('selected_lineage_id')}`",
        "",
        "## Claim Preservation Only",
        "",
        f"- target lineage visible: `{0 if preserve_target_claim is None else preserve_target_claim.get('target_lineage_claim_visible_final', 0)}`",
        f"- target lineage rank: `{None if preserve_selection is None else preserve_selection.get('target_lineage_rank')}`",
        f"- final selected lineage: `{None if preserve_selection is None else preserve_selection.get('selected_lineage_id')}`",
        "",
        "## Claim Preservation + Identity Tie-Break",
        "",
        f"- target lineage visible: `{0 if tiebreak_target_claim is None else tiebreak_target_claim.get('target_lineage_claim_visible_final', 0)}`",
        f"- target lineage rank: `{None if tiebreak_selection is None else tiebreak_selection.get('target_lineage_rank')}`",
        f"- final selected lineage: `{None if tiebreak_selection is None else tiebreak_selection.get('selected_lineage_id')}`",
        f"- identity tiebreak applied: `{0 if tiebreak_selection is None else tiebreak_selection.get('identity_preference_tiebreak_applied', 0)}`",
        "",
        "## Direct Answers",
        "",
        "1. Baseline failure is split between claim visibility and claim weighting; visibility has to be fixed first.",
        "2. Minimal claim preservation should change `target_lineage_visible` before any claim weighting changes are interpreted.",
        "3. Identity-aware tie-break only matters after the correct lineage is already visible in the final claim set.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_design_notes(path: Path) -> None:
    lines = [
        "# Phase 3D Stage A.5 Design Notes",
        "",
        "1. Rerouted proposals now carry an explicit final claim set, not just raw grouped claims.",
        "2. Matched-lineage and recovery-hint lineages are preserved into the final claim set if they expose legal recovery evidence.",
        "3. Identity-aware tie-break is only used after visibility is established and only on rerouted proposals.",
        "4. Within-lineage old-identity selection stays separate from lineage-level claim visibility and ranking.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_failure_bucket_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Phase 3D Stage A.5 Failure Bucket Summary", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['run_label']}",
                "",
                f"- `failure_bucket = {row['failure_bucket']}`",
                f"- `target_lineage_visible = {row['target_lineage_visible']}`",
                f"- `target_lineage_rank = {row['target_lineage_rank']}`",
                f"- `target_lineage_claim_score = {row['target_lineage_claim_score']}`",
                f"- `winning_lineage_claim_score = {row['winning_lineage_claim_score']}`",
                f"- `identity_gap_vs_winner = {row['identity_gap_vs_winner']}`",
                f"- `geometry_gap_vs_winner = {row['geometry_gap_vs_winner']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_claim_preservation_strip(path: Path, traces: list[dict[str, Any]], target_lineage_id: int | None) -> None:
    frames = list(range(TARGET_FRAME - 2, TARGET_FRAME + 3))
    fig, axes = plt.subplots(len(traces), len(frames), figsize=(3.8 * len(frames), 3.8 * len(traces)))
    if len(traces) == 1:
        axes = np.asarray([axes])
    for row_idx, trace in enumerate(traces):
        run_label = trace["run_label"]
        target_claims = _target_claim_rows(trace["claim_rows"])
        target_claim = _target_lineage_claim_row(target_claims, target_lineage_id)
        winner = target_claims[0]["candidate_lineage_id"] if target_claims else None
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
                    f"\nvis={0 if target_claim is None else target_claim.get('target_lineage_claim_visible_final', 0)}"
                    f" drop={None if target_claim is None else target_claim.get('claim_drop_stage')}"
                    f"\nwinL={winner}"
                )
            axis.set_title(title, fontsize=8)
            if col_idx == 0:
                axis.text(2, 12, run_label, color="white", fontsize=9, bbox={"facecolor": "black", "alpha": 0.6, "pad": 2})
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_visibility_rank_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row["run_label"] for row in rows]
    visibility = [float(row["target_lineage_visible"]) for row in rows]
    rank_values = [np.nan if row["target_lineage_rank"] in (None, "", "None") else float(row["target_lineage_rank"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, visibility, color=["tab:red", "tab:orange", "tab:blue"])
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Target-lineage visibility")
    axes[0].set_ylabel("visible")
    axes[1].plot(labels, rank_values, marker="o", color="tab:purple")
    axes[1].set_title("Target-lineage rank")
    axes[1].set_ylabel("rank")
    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_identity_tiebreak_barplot(path: Path, target_claims: list[dict[str, Any]]) -> None:
    if not target_claims:
        return
    labels = [str(row["candidate_lineage_id"]) for row in target_claims]
    continuity = [float(row.get("continuity_priority_score") or 0.0) for row in target_claims]
    geometry = [float(row.get("geometry_priority_score") or 0.0) for row in target_claims]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(x - width / 2, continuity, width=width, label="continuity")
    axis.bar(x + width / 2, geometry, width=width, label="geometry")
    axis.set_xticks(x, labels)
    axis.set_xlabel("lineage id")
    axis.set_ylabel("score")
    axis.set_title("Frame 990 claim priorities")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_failure_gallery(path: Path, traces: list[dict[str, Any]], buckets: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, len(traces), figsize=(5.0 * len(traces), 4.5))
    if len(traces) == 1:
        axes = [axes]
    for axis, trace, bucket_row in zip(axes, traces, buckets):
        snapshot = trace["frame_snapshots"].get(TARGET_FRAME)
        if snapshot is None:
            axis.axis("off")
            continue
        axis.imshow(snapshot["image"])
        axis.axis("off")
        _draw_box(axis, snapshot.get("gt_box"), color="lime", label="GT")
        _draw_box(axis, snapshot.get("target_proposal_box"), color="cyan", label="proposal")
        axis.set_title(
            f"{trace['run_label']}\n{bucket_row['failure_bucket']}\n"
            f"winL={bucket_row['winning_lineage']} targetL={bucket_row['target_lineage']}",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_recommendation(path: Path, rows: list[dict[str, Any]]) -> None:
    latest = rows[-1] if rows else None
    lines = ["# Phase 3D Stage A.5 Recommendation", ""]
    if latest is None:
        lines.append("No Stage A.5 target-selection rows were produced.")
    else:
        lines.extend(
            [
                f"- latest run: `{latest['run_label']}`",
                f"- failure bucket: `{latest['failure_bucket']}`",
                f"- target lineage visible: `{latest['target_lineage_visible']}`",
                f"- final selected lineage: `{latest['winning_lineage']}`",
                "",
                "Do not enter Stage B until the matched-lineage claim stays visible and the final winner follows continuity-first tie-breaks.",
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
            "enable_phase3d_identity_preference_tiebreak": False,
        },
    )
    preserve = _run_trace(
        config_path=config_path,
        seed=args.seed,
        run_label="forced_claim_preservation",
        tracking_patch={
            "enable_phase3d_claim_preservation_repair": True,
            "enable_phase3d_identity_preference_tiebreak": False,
        },
    )
    tiebreak = _run_trace(
        config_path=config_path,
        seed=args.seed,
        run_label="forced_identity_tiebreak",
        tracking_patch={
            "enable_phase3d_claim_preservation_repair": True,
            "enable_phase3d_identity_preference_tiebreak": True,
        },
    )

    all_candidate_rows = baseline["candidate_rows"] + preserve["candidate_rows"] + tiebreak["candidate_rows"]
    all_claim_rows = baseline["claim_rows"] + preserve["claim_rows"] + tiebreak["claim_rows"]
    target_metadata = baseline["target_metadata"]
    target_lineage_id = target_metadata["old_lineage_id"]

    baseline_selection = _target_selection(baseline["selection_rows"])
    preserve_selection = _target_selection(preserve["selection_rows"])
    tiebreak_selection = _target_selection(tiebreak["selection_rows"])
    baseline_target_claims = _target_claim_rows(baseline["claim_rows"])
    preserve_target_claims = _target_claim_rows(preserve["claim_rows"])
    tiebreak_target_claims = _target_claim_rows(tiebreak["claim_rows"])
    baseline_target_claim = _target_lineage_claim_row(baseline_target_claims, target_lineage_id)
    preserve_target_claim = _target_lineage_claim_row(preserve_target_claims, target_lineage_id)
    tiebreak_target_claim = _target_lineage_claim_row(tiebreak_target_claims, target_lineage_id)

    _write_csv(output_dir / "phase3d_stagea5_claim_preservation_trace.csv", all_claim_rows)

    within_identity_rows: list[dict[str, Any]] = []
    for trace in (baseline, preserve, tiebreak):
        selection = _target_selection(trace["selection_rows"])
        selected_lineage = None if selection is None or selection.get("selected_lineage_id") in ("", "None", None) else int(selection["selected_lineage_id"])
        for row in _target_candidate_rows(trace["candidate_rows"]):
            if selected_lineage is None and target_lineage_id is None:
                continue
            if (
                selected_lineage is not None and int(row["candidate_lineage_id"]) == int(selected_lineage)
            ) or (
                target_lineage_id is not None and int(row["candidate_lineage_id"]) == int(target_lineage_id)
            ):
                within_identity_rows.append(
                    {
                        "run_label": trace["run_label"],
                        "frame_id": int(row["frame_id"]),
                        "proposal_id": int(row["proposal_id"]),
                        "candidate_lineage_id": int(row["candidate_lineage_id"]),
                        "selected_lineage_id": selected_lineage,
                        "source_type": str(row["source_type"]),
                        "candidate_track_id": int(row["candidate_track_id"]),
                        "candidate_prototype_id": None if row.get("candidate_prototype_id") in ("", "None", None) else int(row["candidate_prototype_id"]),
                        "recovery_score_total": float(row["recovery_score_total"]),
                        "raw_cost": float(row["raw_cost"]),
                        "same_prototype_hint": int(row["same_prototype_hint"]),
                        "same_track_hint": int(row["same_track_hint"]),
                        "old_identity_ref_valid": int(row["old_identity_ref_valid"]),
                        "is_selected_lineage": int(selected_lineage is not None and int(row["candidate_lineage_id"]) == int(selected_lineage)),
                        "is_target_lineage": int(target_lineage_id is not None and int(row["candidate_lineage_id"]) == int(target_lineage_id)),
                        "target_same_track_hint": int(row.get("target_same_track_hint", 0)),
                        "target_same_prototype_hint": int(row.get("target_same_prototype_hint", 0)),
                    }
                )
    _write_csv(output_dir / "phase3d_stagea5_within_lineage_identity_trace.csv", within_identity_rows)

    summary_rows: list[dict[str, Any]] = []
    for trace, selection, target_claims, target_claim in (
        (baseline, baseline_selection, baseline_target_claims, baseline_target_claim),
        (preserve, preserve_selection, preserve_target_claims, preserve_target_claim),
        (tiebreak, tiebreak_selection, tiebreak_target_claims, tiebreak_target_claim),
    ):
        target_rank = next(
            (
                int(index + 1)
                for index, row in enumerate(target_claims)
                if target_lineage_id is not None and int(row["candidate_lineage_id"]) == int(target_lineage_id)
            ),
            None,
        )
        winner_claim = target_claims[0] if target_claims else None
        winning_lineage = None if selection is None else selection.get("selected_lineage_id")
        target_claim_score = None if target_claim is None or target_claim.get("claim_score_total") is None else float(target_claim["claim_score_total"])
        winning_claim_score = None if winner_claim is None or winner_claim.get("claim_score_total") is None else float(winner_claim["claim_score_total"])
        identity_gap = None
        geometry_gap = None
        if target_claim is not None and winner_claim is not None:
            identity_gap = float(target_claim.get("continuity_priority_score") or 0.0) - float(winner_claim.get("continuity_priority_score") or 0.0)
            geometry_gap = float(target_claim.get("geometry_priority_score") or 0.0) - float(winner_claim.get("geometry_priority_score") or 0.0)
        summary_rows.append(
            {
                "run_label": trace["run_label"],
                "failure_bucket": _failure_bucket(selection, target_claims, target_lineage_id),
                "target_lineage": target_lineage_id,
                "winning_lineage": winning_lineage,
                "target_lineage_visible": 0 if target_claim is None else int(target_claim.get("target_lineage_claim_visible_final", 0)),
                "target_lineage_rank": target_rank,
                "target_lineage_claim_score": target_claim_score,
                "winning_lineage_claim_score": winning_claim_score,
                "identity_gap_vs_winner": identity_gap,
                "geometry_gap_vs_winner": geometry_gap,
            }
        )

    forced_claim_preservation = {
        "target_event_id": TARGET_EVENT_ID,
        "target_frame": TARGET_FRAME,
        "target_lineage_id": target_lineage_id,
        "baseline_target_lineage_visible": 0 if baseline_target_claim is None else int(baseline_target_claim.get("target_lineage_claim_visible_final", 0)),
        "preserved_target_lineage_visible": 0 if preserve_target_claim is None else int(preserve_target_claim.get("target_lineage_claim_visible_final", 0)),
        "baseline_claim_drop_stage": None if baseline_target_claim is None else baseline_target_claim.get("claim_drop_stage"),
        "preserved_claim_drop_stage": None if preserve_target_claim is None else preserve_target_claim.get("claim_drop_stage"),
        "baseline_final_lineage": None if baseline_selection is None else baseline_selection.get("selected_lineage_id"),
        "preserved_final_lineage": None if preserve_selection is None else preserve_selection.get("selected_lineage_id"),
        "preserved_target_rank": next(
            (
                int(index + 1)
                for index, row in enumerate(preserve_target_claims)
                if target_lineage_id is not None and int(row["candidate_lineage_id"]) == int(target_lineage_id)
            ),
            None,
        ),
    }
    (output_dir / "phase3d_stagea5_forced_claim_preservation.json").write_text(json.dumps(forced_claim_preservation, indent=2), encoding="utf-8")

    forced_identity_tiebreak = {
        "target_event_id": TARGET_EVENT_ID,
        "target_frame": TARGET_FRAME,
        "target_lineage_id": target_lineage_id,
        "preserve_final_lineage": None if preserve_selection is None else preserve_selection.get("selected_lineage_id"),
        "tiebreak_final_lineage": None if tiebreak_selection is None else tiebreak_selection.get("selected_lineage_id"),
        "preserve_target_visible": 0 if preserve_target_claim is None else int(preserve_target_claim.get("target_lineage_claim_visible_final", 0)),
        "tiebreak_target_visible": 0 if tiebreak_target_claim is None else int(tiebreak_target_claim.get("target_lineage_claim_visible_final", 0)),
        "preserve_target_rank": next(
            (
                int(index + 1)
                for index, row in enumerate(preserve_target_claims)
                if target_lineage_id is not None and int(row["candidate_lineage_id"]) == int(target_lineage_id)
            ),
            None,
        ),
        "tiebreak_target_rank": next(
            (
                int(index + 1)
                for index, row in enumerate(tiebreak_target_claims)
                if target_lineage_id is not None and int(row["candidate_lineage_id"]) == int(target_lineage_id)
            ),
            None,
        ),
        "tiebreak_applied": 0 if tiebreak_selection is None else int(tiebreak_selection.get("identity_preference_tiebreak_applied", 0)),
        "continuity_priority_score": None if tiebreak_selection is None else tiebreak_selection.get("continuity_priority_score"),
        "geometry_priority_score": None if tiebreak_selection is None else tiebreak_selection.get("geometry_priority_score"),
    }
    (output_dir / "phase3d_stagea5_forced_identity_tiebreak.json").write_text(json.dumps(forced_identity_tiebreak, indent=2), encoding="utf-8")

    _write_summary(
        output_dir / "phase3d_stagea5_claim_preservation_summary.md",
        baseline_selection=baseline_selection,
        preserve_selection=preserve_selection,
        tiebreak_selection=tiebreak_selection,
        baseline_target_claim=baseline_target_claim,
        preserve_target_claim=preserve_target_claim,
        tiebreak_target_claim=tiebreak_target_claim,
        target_metadata=target_metadata,
    )
    _write_design_notes(output_dir / "phase3d_stagea5_design_notes.md")
    _write_failure_bucket_summary(output_dir / "phase3d_stagea5_failure_bucket_summary.md", summary_rows)
    _save_claim_preservation_strip(output_dir / "frame990_claim_preservation_strip.png", [baseline, preserve, tiebreak], target_lineage_id)
    _save_visibility_rank_plot(output_dir / "lineage2_visibility_vs_rank.png", summary_rows)
    _save_identity_tiebreak_barplot(output_dir / "identity_tiebreak_barplot.png", tiebreak_target_claims)
    _save_failure_gallery(output_dir / "failure_bucket_gallery.png", [baseline, preserve, tiebreak], summary_rows)
    _write_recommendation(output_dir / "phase3d_stagea5_recommendation.md", summary_rows)


if __name__ == "__main__":
    main()
