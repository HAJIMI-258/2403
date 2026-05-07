"""Phase 3D Stage A.4: recovery-lineage target selection trace."""

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
from PIL import Image

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
    parser = argparse.ArgumentParser(description="Run Phase 3D Stage A.4 target selection trace.")
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


def _figure_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    array = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return array


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


def _load_track_c_sequence(config_path: Path, *, seed: int):
    base_config = load_synth_dataset_config(config_path)
    scenario_map = {s["name"]: s["config"] for s in build_phase3_track_scenarios(base_config)}
    return SyntheticStreamGenerator(scenario_map[TRACK_C_NAME], seed=seed).generate_sequence(0)


def _gt_box(frame_sample, gt_object_id: int) -> tuple[int, int, int, int] | None:
    for instance_id, box in zip(frame_sample.instance_ids, frame_sample.boxes):
        if int(instance_id) == int(gt_object_id):
            return tuple(int(v) for v in box)
    return None


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


def _parse_box(value: Any) -> tuple[int, int, int, int] | None:
    if value in (None, "", "None"):
        return None
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    text = str(value).strip()
    if not text:
        return None
    text = text.strip("()")
    return tuple(int(part.strip()) for part in text.split(","))


def _augment_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    run_label: str,
    gt_box: tuple[int, int, int, int] | None,
    target_metadata: dict[str, Any],
    target_proposal_id: int | None,
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for row in rows:
        proposal_box = _parse_box(row.get("proposal_box"))
        candidate_lineage_id = int(row["candidate_lineage_id"])
        candidate_track_id = int(row["candidate_track_id"])
        candidate_prototype_id = row.get("candidate_prototype_id")
        candidate_prototype_id = (
            None if candidate_prototype_id in ("", "None", None) else int(candidate_prototype_id)
        )
        augmented.append(
            {
                **row,
                "run_label": run_label,
                "proposal_iou_to_gt": _iou(proposal_box, gt_box),
                "is_target_proposal": int(target_proposal_id is not None and int(row["proposal_id"]) == int(target_proposal_id)),
                "target_lineage_match": int(
                    target_metadata["old_lineage_id"] is not None
                    and int(candidate_lineage_id) == int(target_metadata["old_lineage_id"])
                ),
                "target_same_track_hint": int(
                    target_metadata["old_track_id"] is not None
                    and int(candidate_track_id) == int(target_metadata["old_track_id"])
                ),
                "target_same_prototype_hint": int(
                    target_metadata["old_prototype_id"] is not None
                    and candidate_prototype_id is not None
                    and int(candidate_prototype_id) == int(target_metadata["old_prototype_id"])
                ),
            }
        )
    return augmented


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
            "routing_recovery_max_distance": 0.70,
            "routing_recovery_min_confidence": 0.30,
            "routing_active_claim_override_margin": 0.20,
            "routing_topk": 3,
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
        candidate_rows.extend(
            _augment_candidate_rows(
                frame_candidate_rows,
                run_label=run_label,
                gt_box=gt_box,
                target_metadata=target_metadata,
                target_proposal_id=target_proposal_id,
            )
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
            "selection_rows": frame_selection_rows,
            "claim_rows": frame_claim_rows,
            "candidate_rows": frame_candidate_rows,
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


def _target_selection(selection_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    target_rows = [
        row
        for row in selection_rows
        if int(row.get("frame_id", -1)) == TARGET_FRAME and int(row.get("is_target_proposal", 0)) == 1
    ]
    if not target_rows:
        return None
    return sorted(target_rows, key=lambda row: -float(row.get("proposal_iou_to_gt", 0.0)))[0]


def _target_claim_rows(claim_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in claim_rows
        if int(row.get("frame_id", -1)) == TARGET_FRAME and int(row.get("is_target_proposal", 0)) == 1
    ]
    return sorted(rows, key=lambda row: -float(row["claim_score_total"]))


def _target_candidate_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in candidate_rows
        if int(row.get("frame_id", -1)) == TARGET_FRAME and int(row.get("is_target_proposal", 0)) == 1
    ]
    return sorted(rows, key=lambda row: (-float(row["recovery_score_total"]), float(row["raw_cost"])))


def _write_markdown_summary(
    path: Path,
    *,
    baseline_selection: dict[str, Any] | None,
    baseline_claims: list[dict[str, Any]],
    forced_two_stage_selection: dict[str, Any] | None,
    target_metadata: dict[str, Any],
) -> None:
    target_lineage = target_metadata["old_lineage_id"]
    target_claim = next(
        (row for row in baseline_claims if target_lineage is not None and int(row["candidate_lineage_id"]) == int(target_lineage)),
        None,
    )
    winner = baseline_claims[0] if baseline_claims else None
    lines = [
        "# Phase 3D Stage A.4 Candidate Ranking Summary",
        "",
        f"- target event: `{TARGET_EVENT_ID}`",
        f"- target frame: `{TARGET_FRAME}`",
        f"- target lineage: `{target_lineage}`",
        "",
        "## Baseline Reroute",
        "",
        f"- final selected lineage: `{None if baseline_selection is None else baseline_selection.get('selected_lineage_id')}`",
        f"- final selected source: `{None if baseline_selection is None else baseline_selection.get('selected_source_type')}`",
        f"- target-lineage matched: `{0 if baseline_selection is None else baseline_selection.get('target_lineage_match')}`",
        "",
        "## Claim Comparison",
        "",
        f"- claim winner lineage: `{None if winner is None else winner.get('candidate_lineage_id')}`",
        f"- claim winner score: `{None if winner is None else winner.get('claim_score_total')}`",
        f"- target-lineage claim score: `{None if target_claim is None else target_claim.get('claim_score_total')}`",
        "",
        "## Forced Two-Stage Probe",
        "",
        f"- final selected lineage: `{None if forced_two_stage_selection is None else forced_two_stage_selection.get('selected_lineage_id')}`",
        f"- final selected source: `{None if forced_two_stage_selection is None else forced_two_stage_selection.get('selected_source_type')}`",
        f"- target-lineage matched: `{0 if forced_two_stage_selection is None else forced_two_stage_selection.get('target_lineage_match')}`",
        f"- target same-track hint: `{0 if forced_two_stage_selection is None else forced_two_stage_selection.get('target_same_track_hint')}`",
        f"- target same-prototype hint: `{0 if forced_two_stage_selection is None else forced_two_stage_selection.get('target_same_prototype_hint')}`",
        "",
        "## Direct Answers",
        "",
        f"1. Correct target lineage is {'visible' if target_claim is not None else 'not visible'} in the consumer claim set after reroute.",
        f"2. Baseline winner is lineage `{None if winner is None else winner.get('candidate_lineage_id')}` instead of target lineage `{target_lineage}` because its claim score is higher under the current flat-to-local handoff.",
        f"3. Two-stage selection {'moves final selection toward the target lineage' if forced_two_stage_selection is not None and int(forced_two_stage_selection.get('target_lineage_match', 0)) == 1 else 'still does not land on the target lineage'} in the forced probe.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_design_notes(path: Path) -> None:
    lines = [
        "# Phase 3D Stage A.4 Design Notes",
        "",
        "1. Resurrection consumer now emits all-lineage recovery candidate rows for rerouted proposals.",
        "2. Candidate comparison is split into lineage-level claims and within-lineage identity selection.",
        "3. The Stage A.4 repair path is gated to rerouted proposals and does not touch normal same-lineage active matches.",
        "4. Two-stage selection first selects a recovery lineage from aggregated claim evidence, then chooses an old identity inside that lineage.",
        "5. Identity-aware tie-breaks prefer richer continuity evidence over a single locally convenient candidate.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_frame990_strip(path: Path, baseline: dict[str, Any], forced: dict[str, Any]) -> None:
    frames = list(range(TARGET_FRAME - 2, TARGET_FRAME + 3))
    fig, axes = plt.subplots(2, len(frames), figsize=(3.5 * len(frames), 7.5))
    for row_idx, trace in enumerate((baseline, forced)):
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
            selection = snapshot.get("target_selection")
            title = f"f{frame_id}"
            if selection is not None:
                title += f"\nselL={selection.get('selected_lineage_id')}"
            axis.set_title(title, fontsize=9)
            if col_idx == 0:
                axis.text(
                    2,
                    12,
                    "baseline" if row_idx == 0 else "two-stage",
                    color="white",
                    fontsize=9,
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 2},
                )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_lineage_barplot(path: Path, claim_rows: list[dict[str, Any]], target_lineage_id: int | None) -> None:
    if not claim_rows:
        return
    lineages = [str(row["candidate_lineage_id"]) for row in claim_rows]
    scores = [float(row["claim_score_total"]) for row in claim_rows]
    colors = ["tab:red" if target_lineage_id is not None and int(row["candidate_lineage_id"]) == int(target_lineage_id) else "tab:blue" for row in claim_rows]
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar(lineages, scores, color=colors)
    axis.set_xlabel("lineage id")
    axis.set_ylabel("claim score")
    axis.set_title("Frame 990 target proposal lineage claim scores")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_within_lineage_gallery(
    path: Path,
    *,
    target_candidates: list[dict[str, Any]],
    selected_lineage_id: int | None,
    target_lineage_id: int | None,
) -> None:
    selected_rows = [
        row for row in target_candidates if selected_lineage_id is not None and int(row["candidate_lineage_id"]) == int(selected_lineage_id)
    ][:6]
    target_rows = [
        row for row in target_candidates if target_lineage_id is not None and int(row["candidate_lineage_id"]) == int(target_lineage_id)
    ][:6]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panels = [
        ("selected lineage", selected_rows),
        ("target lineage", target_rows),
    ]
    for axis, (title, rows) in zip(axes, panels):
        axis.axis("off")
        axis.set_title(title)
        if not rows:
            axis.text(0.02, 0.95, "no candidates", va="top", fontsize=10)
            continue
        lines = []
        for row in rows:
            lines.append(
                f"L{row['candidate_lineage_id']} {row['source_type']} "
                f"trk={row['candidate_track_id']} proto={row['candidate_prototype_id']} "
                f"score={float(row['recovery_score_total']):.3f} raw={float(row['raw_cost']):.3f}"
            )
        axis.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=9, family="monospace")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_failure_gallery(path: Path, trace: dict[str, Any], target_metadata: dict[str, Any]) -> None:
    candidate_rows = [
        row for row in trace["candidate_rows"]
        if int(row["is_target_proposal"]) == 1 and int(row["frame_id"]) in range(TARGET_FRAME - 2, TARGET_FRAME + 3)
    ]
    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (int(row["target_lineage_match"]) == 0, -float(row["recovery_score_total"])),
    )[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, row in zip(axes.flatten(), candidate_rows):
        frame_id = int(row["frame_id"])
        snapshot = trace["frame_snapshots"].get(frame_id)
        if snapshot is None:
            axis.axis("off")
            continue
        axis.imshow(snapshot["image"])
        axis.axis("off")
        _draw_box(axis, snapshot.get("gt_box"), color="lime", label="GT")
        _draw_box(axis, snapshot.get("target_proposal_box"), color="cyan", label="proposal")
        axis.set_title(
            f"f{frame_id} L{row['candidate_lineage_id']} {row['source_type']}\n"
            f"targetL={target_metadata['old_lineage_id']} score={float(row['recovery_score_total']):.3f}",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


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
            "enable_phase3d_target_selection_repair": False,
        },
    )
    forced = _run_trace(
        config_path=config_path,
        seed=args.seed,
        run_label="forced_two_stage_selection",
        tracking_patch={
            "enable_phase3d_target_selection_repair": True,
        },
    )

    candidate_rows = baseline["candidate_rows"] + forced["candidate_rows"]
    claim_rows = baseline["claim_rows"] + forced["claim_rows"]
    selection_rows = baseline["selection_rows"] + forced["selection_rows"]

    _write_csv(output_dir / "phase3d_stagea4_candidate_ranking_trace.csv", candidate_rows)

    baseline_target_selection = _target_selection(baseline["selection_rows"])
    forced_target_selection = _target_selection(forced["selection_rows"])
    baseline_target_claims = _target_claim_rows(baseline["claim_rows"])
    baseline_target_candidates = _target_candidate_rows(baseline["candidate_rows"])
    target_metadata = baseline["target_metadata"]

    forced_lineage_visibility = {
        "target_event_id": TARGET_EVENT_ID,
        "target_frame": TARGET_FRAME,
        "target_lineage_id": target_metadata["old_lineage_id"],
        "target_proposal_id": None if baseline_target_selection is None else int(baseline_target_selection["proposal_id"]),
        "lineage2_claim_visible": int(
            any(
                target_metadata["old_lineage_id"] is not None
                and int(row["candidate_lineage_id"]) == int(target_metadata["old_lineage_id"])
                for row in baseline_target_claims
            )
        ),
        "lineage2_claim_score": None
        if target_metadata["old_lineage_id"] is None
        else next(
            (
                float(row["claim_score_total"])
                for row in baseline_target_claims
                if int(row["candidate_lineage_id"]) == int(target_metadata["old_lineage_id"])
            ),
            None,
        ),
        "winning_lineage_id": None if not baseline_target_claims else int(baseline_target_claims[0]["candidate_lineage_id"]),
        "winning_claim_score": None if not baseline_target_claims else float(baseline_target_claims[0]["claim_score_total"]),
        "top_claims": [
            {
                "lineage_id": int(row["candidate_lineage_id"]),
                "claim_score_total": float(row["claim_score_total"]),
                "target_lineage_match": int(row["target_lineage_match"]),
            }
            for row in baseline_target_claims[:5]
        ],
    }
    (output_dir / "phase3d_stagea4_forced_lineage2_visibility.json").write_text(
        json.dumps(forced_lineage_visibility, indent=2),
        encoding="utf-8",
    )

    forced_two_stage = {
        "target_event_id": TARGET_EVENT_ID,
        "target_frame": TARGET_FRAME,
        "target_lineage_id": target_metadata["old_lineage_id"],
        "baseline_final_lineage": None if baseline_target_selection is None else baseline_target_selection.get("selected_lineage_id"),
        "baseline_final_source": None if baseline_target_selection is None else baseline_target_selection.get("selected_source_type"),
        "two_stage_final_lineage": None if forced_target_selection is None else forced_target_selection.get("selected_lineage_id"),
        "two_stage_final_source": None if forced_target_selection is None else forced_target_selection.get("selected_source_type"),
        "baseline_target_lineage_match": 0 if baseline_target_selection is None else int(baseline_target_selection.get("target_lineage_match", 0)),
        "two_stage_target_lineage_match": 0 if forced_target_selection is None else int(forced_target_selection.get("target_lineage_match", 0)),
        "baseline_target_same_track_hint": 0 if baseline_target_selection is None else int(baseline_target_selection.get("target_same_track_hint", 0)),
        "two_stage_target_same_track_hint": 0 if forced_target_selection is None else int(forced_target_selection.get("target_same_track_hint", 0)),
        "baseline_target_same_prototype_hint": 0 if baseline_target_selection is None else int(baseline_target_selection.get("target_same_prototype_hint", 0)),
        "two_stage_target_same_prototype_hint": 0 if forced_target_selection is None else int(forced_target_selection.get("target_same_prototype_hint", 0)),
    }
    (output_dir / "phase3d_stagea4_forced_two_stage_selection.json").write_text(
        json.dumps(forced_two_stage, indent=2),
        encoding="utf-8",
    )

    _write_markdown_summary(
        output_dir / "phase3d_stagea4_candidate_ranking_summary.md",
        baseline_selection=baseline_target_selection,
        baseline_claims=baseline_target_claims,
        forced_two_stage_selection=forced_target_selection,
        target_metadata=target_metadata,
    )
    _write_design_notes(output_dir / "phase3d_stagea4_design_notes.md")

    _save_frame990_strip(output_dir / "frame990_candidate_ranking_strip.png", baseline, forced)
    _save_lineage_barplot(
        output_dir / "lineage_claim_barplot.png",
        baseline_target_claims,
        target_metadata["old_lineage_id"],
    )
    _save_within_lineage_gallery(
        output_dir / "within_lineage_identity_gallery.png",
        target_candidates=baseline_target_candidates,
        selected_lineage_id=None if baseline_target_selection is None else baseline_target_selection.get("selected_lineage_id"),
        target_lineage_id=target_metadata["old_lineage_id"],
    )
    _save_failure_gallery(
        output_dir / "target_selection_failure_gallery.png",
        baseline,
        target_metadata,
    )

    recommendation_lines = [
        "# Phase 3D Stage A.4 Recommendation",
        "",
        "1. Do not enter Stage B yet.",
        "2. Stage A.4 should be judged on whether two-stage selection changes the recovery landing toward the target lineage, not on total metrics.",
        "3. If target lineage is visible but still loses after the two-stage probe, the next repair should stay in Stage A and strengthen lineage-claim aggregation or within-lineage identity tie-breaks.",
        "",
    ]
    (output_dir / "phase3d_stagea4_recommendation.md").write_text(
        "\n".join(recommendation_lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
