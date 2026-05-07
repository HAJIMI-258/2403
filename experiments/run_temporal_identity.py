"""Run Day 4 temporal identity tracking on a synthetic sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator, load_synth_dataset_config
from metrics.metrics_core import greedy_match_boxes, identity_switches, u_recall
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.objectness import MinimalObjectnessField
from nops_owr.tracking import MinimalTemporalIdentityTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal temporal identity tracking.")
    parser.add_argument("--config", default="configs/synth.yaml", help="Path to the config file.")
    parser.add_argument("--sequence-id", type=int, default=3, help="Sequence index to generate.")
    parser.add_argument("--output-dir", default="results/day4_tracking", help="Directory for output artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_payload = _load_config_payload(args.config)
    synth_config = load_synth_dataset_config(args.config)
    sequence = SyntheticStreamGenerator(synth_config, seed=args.seed).generate_sequence(args.sequence_id)

    encoder = MinimalSpikeEncoder(**config_payload["model"]["spike_encoder"])
    objectness = MinimalObjectnessField(**_get_field_config(config_payload))
    tracker = MinimalTemporalIdentityTracker(**config_payload["tracking"])

    tracked_history: list[dict[int, int | None]] = []
    naive_history: list[dict[int, str | None]] = []
    frame_records: list[dict[str, object]] = []
    dormant_track_counts: list[int] = []
    reactivation_count = 0

    for frame_index in range(1, len(sequence.frames)):
        prev_frame = sequence.frames[frame_index - 1]
        current_frame = sequence.frames[frame_index]

        encoding = encoder.encode(prev_frame.frame, current_frame.frame)
        objectness_output = objectness.compute(encoding)
        tracking_output = tracker.update(
            proposals=objectness_output.proposals,
            encoding=encoding,
            heatmap=objectness_output.heatmap,
            current_frame=current_frame.frame,
            frame_index=current_frame.frame_index,
        )

        tracked_boxes = [assignment.box for assignment in tracking_output.assignments]
        tracked_ids = [assignment.track_id for assignment in tracking_output.assignments]
        recall = u_recall(current_frame.boxes, tracked_boxes, iou_threshold=0.5)

        tracked_history.append(
            _build_assignment_history(
                gt_boxes=current_frame.boxes,
                gt_instance_ids=current_frame.instance_ids,
                predicted_boxes=tracked_boxes,
                predicted_ids=tracked_ids,
            )
        )

        naive_history.append(
            _build_assignment_history(
                gt_boxes=current_frame.boxes,
                gt_instance_ids=current_frame.instance_ids,
                predicted_boxes=tracked_boxes,
                predicted_ids=[f"f{current_frame.frame_index}_p{index}" for index in range(len(tracked_boxes))],
            )
        )

        frame_records.append(
            {
                "frame_index": current_frame.frame_index,
                "frame": current_frame.frame,
                "gt_boxes": current_frame.boxes,
                "gt_instance_ids": current_frame.instance_ids,
                "assignments": tracking_output.assignments,
                "u_recall": recall,
                "active_track_count": len(tracking_output.active_tracks),
                "new_track_ids": tracking_output.new_track_ids,
                "reactivated_track_ids": tracking_output.reactivated_track_ids,
                "lost_track_ids": tracking_output.lost_track_ids,
                "cost_matrix": tracking_output.cost_matrix,
                "track_order": tracking_output.track_order,
                "proposal_order": tracking_output.proposal_order,
                "cost_stats": {
                    "min": tracking_output.cost_stats.min_cost,
                    "mean": tracking_output.cost_stats.mean_cost,
                    "max": tracking_output.cost_stats.max_cost,
                    "candidate_pairs": tracking_output.cost_stats.candidate_pairs,
                },
                "unmatched_track_count": tracking_output.unmatched_track_count,
                "unmatched_proposal_count": tracking_output.unmatched_proposal_count,
                "dormant_track_count": tracking_output.dormant_track_count,
            }
        )
        dormant_track_counts.append(int(tracking_output.dormant_track_count))
        reactivation_count += len(tracking_output.reactivated_track_ids)

    tracked_idsw = identity_switches(tracked_history)
    naive_idsw = identity_switches(naive_history)
    reentry_recovery = _compute_reentry_recovery(sequence, tracked_history)
    mean_recall = (
        float(sum(float(frame_record["u_recall"]) for frame_record in frame_records) / len(frame_records))
        if frame_records
        else 0.0
    )
    unique_track_ids = sorted(
        {
            assignment.track_id
            for frame_record in frame_records
            for assignment in frame_record["assignments"]
        }
    )
    selected_records = _select_visualization_frames(frame_records)
    mean_cost = (
        float(sum(float(record["cost_stats"]["mean"]) for record in frame_records) / len(frame_records))
        if frame_records
        else 0.0
    )
    max_cost = max((float(record["cost_stats"]["max"]) for record in frame_records), default=0.0)
    min_cost = min((float(record["cost_stats"]["min"]) for record in frame_records), default=0.0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"tracking_seq_{args.sequence_id:03d}.png"
    summary_path = output_dir / f"tracking_seq_{args.sequence_id:03d}.json"

    _save_tracking_figure(selected_records, figure_path)

    summary = {
        "sequence_id": args.sequence_id,
        "num_frames_evaluated": len(frame_records),
        "mean_u_recall": mean_recall,
        "tracked_idsw": tracked_idsw,
        "naive_idsw_without_temporal_identity": naive_idsw,
        "idsw_reduction": float(naive_idsw - tracked_idsw),
        "unique_tracks_created": len(unique_track_ids),
        "max_active_tracks": max((int(record["active_track_count"]) for record in frame_records), default=0),
        "selected_frames": [
            {
                "frame_index": int(record["frame_index"]),
                "u_recall": float(record["u_recall"]),
                "active_track_count": int(record["active_track_count"]),
                "dormant_track_count": int(record["dormant_track_count"]),
                "cost_stats": record["cost_stats"],
            }
            for record in selected_records
        ],
        "cost_matrix_stats": {
            "min_cost": min_cost,
            "mean_cost": mean_cost,
            "max_cost": max_cost,
        },
        "dormant_track_stats": {
            "mean_dormant_tracks": float(sum(dormant_track_counts) / len(dormant_track_counts)) if dormant_track_counts else 0.0,
            "max_dormant_tracks": max(dormant_track_counts, default=0),
            "reactivation_successes": int(reactivation_count),
        },
        "reentry_recovery": reentry_recovery,
        "config": {
            "encoder": config_payload["model"]["spike_encoder"],
            "field": _get_field_config(config_payload),
            "tracking": config_payload["tracking"],
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved_figure={figure_path}")
    print(f"saved_summary={summary_path}")
    print(f"mean_u_recall={mean_recall:.4f}")
    print(f"tracked_idsw={tracked_idsw}")
    print(f"naive_idsw_without_temporal_identity={naive_idsw}")
    print(f"unique_tracks_created={len(unique_track_ids)}")


def _build_assignment_history(
    gt_boxes: list[tuple[int, int, int, int]],
    gt_instance_ids: list[int],
    predicted_boxes: list[tuple[int, int, int, int]],
    predicted_ids: list[object],
) -> dict[int, object | None]:
    history = {instance_id: None for instance_id in gt_instance_ids}
    matches = greedy_match_boxes(gt_boxes, predicted_boxes, iou_threshold=0.5)
    for gt_index, pred_index, _ in matches:
        history[gt_instance_ids[gt_index]] = predicted_ids[pred_index]
    return history


def _select_visualization_frames(frame_records: list[dict[str, object]]) -> list[dict[str, object]]:
    if not frame_records:
        return []
    desired = min(4, len(frame_records))
    min_gap = max(1, len(frame_records) // (desired * 2))
    ranked = sorted(
        frame_records,
        key=lambda record: (
            float(record["cost_stats"]["candidate_pairs"]) > 0.0,
            float(record["u_recall"]),
            min(len(record["assignments"]), len(record["gt_boxes"])),
            -abs(len(record["assignments"]) - len(record["gt_boxes"])),
            -float(record["cost_stats"]["mean"]),
            -int(record["active_track_count"]),
        ),
        reverse=True,
    )

    selected: list[dict[str, object]] = []
    for candidate in ranked:
        frame_index = int(candidate["frame_index"])
        if any(abs(frame_index - int(record["frame_index"])) < min_gap for record in selected):
            continue
        selected.append(candidate)
        if len(selected) == desired:
            break

    if len(selected) < desired:
        evenly_spaced = [frame_records[int(index)] for index in np.linspace(0, len(frame_records) - 1, desired)]
        for candidate in evenly_spaced:
            frame_index = int(candidate["frame_index"])
            if any(frame_index == int(record["frame_index"]) for record in selected):
                continue
            selected.append(candidate)
            if len(selected) == desired:
                break

    selected.sort(key=lambda record: int(record["frame_index"]))
    return selected


def _save_tracking_figure(frame_records: list[dict[str, object]], figure_path: Path) -> None:
    if not frame_records:
        raise RuntimeError("No tracking frames available for visualization.")

    figure, axes = plt.subplots(2, len(frame_records), figsize=(5.2 * len(frame_records), 9))
    if len(frame_records) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for column_index, frame_record in enumerate(frame_records):
        axis = axes[0, column_index]
        frame = frame_record["frame"]
        axis.imshow(frame)
        axis.set_title(
            f"frame {frame_record['frame_index']} | recall {float(frame_record['u_recall']):.2f}\n"
            f"active {int(frame_record['active_track_count'])} | dormant {int(frame_record['dormant_track_count'])}\n"
            f"unmatched {int(frame_record['unmatched_track_count'])} | reactivated {len(frame_record['reactivated_track_ids'])}"
        )
        axis.axis("off")

        for box, instance_id in zip(frame_record["gt_boxes"], frame_record["gt_instance_ids"]):
            x1, y1, x2, y2 = box
            axis.add_patch(
                Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.6, edgecolor="lime")
            )
            axis.text(x1, max(0, y1 - 5), f"gt:{instance_id}", color="lime", fontsize=8)

        for assignment in frame_record["assignments"]:
            x1, y1, x2, y2 = assignment.box
            axis.add_patch(
                Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.6, edgecolor="white")
            )
            axis.text(
                x1,
                min(frame.shape[0] - 5, y2 + 10),
                f"T{assignment.track_id} c={assignment.match_cost:.2f}",
                color="white",
                fontsize=8,
            )

        matrix_axis = axes[1, column_index]
        cost_matrix = frame_record["cost_matrix"]
        if cost_matrix.size == 0:
            matrix_axis.text(0.5, 0.5, "no cost matrix", ha="center", va="center", fontsize=10)
            matrix_axis.set_xticks([])
            matrix_axis.set_yticks([])
        else:
            matrix_axis.imshow(cost_matrix, cmap="viridis", aspect="auto")
            matrix_axis.set_xticks(range(len(frame_record["proposal_order"])))
            matrix_axis.set_yticks(range(len(frame_record["track_order"])))
            matrix_axis.set_xticklabels([f"P{idx}" for idx in frame_record["proposal_order"]], rotation=45, ha="right")
            matrix_axis.set_yticklabels([f"T{idx}" for idx in frame_record["track_order"]])
        matrix_axis.set_title(
            "Cost Matrix\n"
            f"min={float(frame_record['cost_stats']['min']):.2f} "
            f"mean={float(frame_record['cost_stats']['mean']):.2f} "
            f"max={float(frame_record['cost_stats']['max']):.2f}"
        )
        matrix_axis.set_xlabel("proposals")
        matrix_axis.set_ylabel("tracks")

    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _load_config_payload(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _compute_reentry_recovery(sequence, assignment_history: list[dict[int, int | None]]) -> dict[str, float | int]:
    visible_history: dict[int, list[bool]] = {}
    assigned_history: dict[int, list[int | None]] = {}
    for frame_index in range(1, len(sequence.frames)):
        frame = sequence.frames[frame_index]
        assignment_map = assignment_history[frame_index - 1]
        visible_ids = set(frame.instance_ids)
        all_ids = set(visible_ids) | set(visible_history.keys()) | set(assignment_map.keys())
        for instance_id in all_ids:
            visible_history.setdefault(instance_id, []).append(instance_id in visible_ids)
            assigned_history.setdefault(instance_id, []).append(assignment_map.get(instance_id))

    events = 0
    recovered = 0
    for instance_id, visibility in visible_history.items():
        ids = assigned_history[instance_id]
        seen_ids = set()
        if visibility and visibility[0] and ids[0] is not None:
            seen_ids.add(ids[0])
        for index in range(1, len(visibility)):
            if visibility[index] and not visibility[index - 1]:
                events += 1
                if ids[index] is not None and ids[index] in seen_ids:
                    recovered += 1
            if visibility[index] and ids[index] is not None:
                seen_ids.add(ids[index])
    return {"events": events, "recovered": recovered, "rate": recovered / events if events else 0.0}


def _get_field_config(config_payload: dict) -> dict:
    if "field" in config_payload:
        return config_payload["field"]
    return config_payload["model"]["objectness"]


if __name__ == "__main__":
    main()
