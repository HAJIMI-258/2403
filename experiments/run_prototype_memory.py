"""Run Day 5 prototype memory on top of tracking outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator, load_synth_dataset_config
from metrics.metrics_core import greedy_match_boxes, summarize_phase1_metrics
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.memory import MinimalPrototypeMemory
from nops_owr.objectness import MinimalObjectnessField
from nops_owr.tracking import MinimalTemporalIdentityTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal prototype memory.")
    parser.add_argument("--config", default="configs/synth.yaml", help="Path to the config file.")
    parser.add_argument("--sequence-id", type=int, default=3, help="Sequence index to generate.")
    parser.add_argument("--output-dir", default="results/day5_memory", help="Directory for output artifacts.")
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
    memory = MinimalPrototypeMemory(**config_payload["memory"])

    gt_boxes_per_frame: list[list[tuple[int, int, int, int]]] = []
    pred_boxes_per_frame: list[list[tuple[int, int, int, int]]] = []
    prototype_concept_assignments: list[tuple[int, int]] = []
    instance_assignment_history: list[dict[int, int | None]] = []
    active_prototype_sets: list[list[int]] = []
    memory_sizes: list[int] = []
    frame_records: list[dict[str, object]] = []
    instance_prototype_series: dict[int, list[int | None]] = defaultdict(list)
    action_counter: Counter[str] = Counter()
    budget_prune_count = 0
    distance_series: list[float] = []
    similarity_series: list[float] = []
    update_weight_series: list[float] = []

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
        memory_output = memory.update(tracking_output.assignments, frame_index=current_frame.frame_index)

        predicted_boxes = [assignment.box for assignment in memory_output.assignments]
        predicted_ids = [assignment.prototype_id for assignment in memory_output.assignments]

        gt_boxes_per_frame.append(list(current_frame.boxes))
        pred_boxes_per_frame.append(list(predicted_boxes))
        active_prototype_sets.append(list(memory_output.active_prototype_ids))
        memory_sizes.append(memory_output.total_prototypes)
        budget_prune_count += len(memory_output.budget_pruned_ids)

        history = {instance_id: None for instance_id in current_frame.instance_ids}
        matches = greedy_match_boxes(current_frame.boxes, predicted_boxes, iou_threshold=0.5)
        for gt_index, pred_index, _ in matches:
            instance_id = current_frame.instance_ids[gt_index]
            concept_id = current_frame.concept_ids[gt_index]
            prototype_id = predicted_ids[pred_index]
            history[instance_id] = prototype_id
            prototype_concept_assignments.append((prototype_id, concept_id))

        instance_assignment_history.append(history)

        for instance_id in current_frame.instance_ids:
            instance_prototype_series[instance_id].append(history.get(instance_id))

        for assignment in memory_output.assignments:
            action_counter[assignment.action] += 1
            distance_series.append(float(assignment.distance))
            similarity_series.append(float(assignment.soft_similarity))
            update_weight_series.append(float(assignment.update_weight))

        frame_records.append(
            {
                "frame_index": current_frame.frame_index,
                "frame": current_frame.frame,
                "gt_boxes": current_frame.boxes,
                "gt_instance_ids": current_frame.instance_ids,
                "memory_assignments": memory_output.assignments,
                "u_recall": len(matches) / len(current_frame.boxes) if current_frame.boxes else 1.0,
                "memory_size": memory_output.total_prototypes,
                "active_prototypes": list(memory_output.active_prototype_ids),
                "budget_pruned_ids": list(memory_output.budget_pruned_ids),
                "distance_values": [float(assignment.distance) for assignment in memory_output.assignments],
                "similarity_values": [float(assignment.soft_similarity) for assignment in memory_output.assignments],
                "update_weights": [float(assignment.update_weight) for assignment in memory_output.assignments],
            }
        )

    metric_summary = summarize_phase1_metrics(
        gt_boxes_per_frame=gt_boxes_per_frame,
        pred_boxes_per_frame=pred_boxes_per_frame,
        prototype_concept_assignments=prototype_concept_assignments,
        instance_assignment_history=instance_assignment_history,
        active_prototype_sets=active_prototype_sets,
        memory_sizes=memory_sizes,
        iou_threshold=config_payload["evaluation"]["match_iou_threshold"],
    )

    final_snapshot = memory.snapshot()
    reattachment = _compute_reentry_reconnect_rate(sequence, instance_assignment_history)
    representative_frame = _select_representative_frame(frame_records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"memory_seq_{args.sequence_id:03d}.png"
    summary_path = output_dir / f"memory_seq_{args.sequence_id:03d}.json"

    _save_memory_figure(
        representative_frame=representative_frame,
        frame_records=frame_records,
        final_snapshot=final_snapshot,
        memory_budget=config_payload["memory"]["memory_budget"],
        distance_series=distance_series,
        similarity_series=similarity_series,
        update_weight_series=update_weight_series,
        figure_path=figure_path,
    )

    summary = {
        "sequence_id": args.sequence_id,
        "num_frames_evaluated": len(frame_records),
        "metrics": {
            "u_recall": metric_summary.u_recall,
            "purity": metric_summary.purity,
            "pfr": metric_summary.pfr,
            "idsw": metric_summary.idsw,
            "churn": metric_summary.churn,
            "memory_growth": metric_summary.memory_growth,
        },
        "memory": {
            "final_prototype_count": len(final_snapshot),
            "max_prototype_count": max(memory_sizes) if memory_sizes else 0,
            "memory_budget": config_payload["memory"]["memory_budget"],
            "budget_prune_count": budget_prune_count,
            "prototype_count_curve": memory_sizes,
            "distance_stats": _series_stats(distance_series),
            "similarity_stats": _series_stats(similarity_series),
            "update_weight_stats": _series_stats(update_weight_series),
            "top_hits": [
                {
                    "prototype_id": prototype.prototype_id,
                    "hits": prototype.hits,
                    "strength": prototype.strength,
                    "active": prototype.active,
                }
                for prototype in sorted(final_snapshot, key=lambda item: (item.hits, item.strength), reverse=True)[:8]
            ],
            "actions": dict(action_counter),
        },
        "reentry_reconnect": reattachment,
        "representative_frame": {
            "frame_index": int(representative_frame["frame_index"]),
            "u_recall": float(representative_frame["u_recall"]),
            "memory_size": int(representative_frame["memory_size"]),
        },
        "config": {
            "encoder": config_payload["model"]["spike_encoder"],
            "field": _get_field_config(config_payload),
            "tracking": config_payload["tracking"],
            "memory": config_payload["memory"],
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved_figure={figure_path}")
    print(f"saved_summary={summary_path}")
    print(f"u_recall={metric_summary.u_recall:.4f}")
    print(f"purity={metric_summary.purity:.4f}")
    print(f"pfr={metric_summary.pfr:.4f}")
    print(f"idsw={metric_summary.idsw}")
    print(f"churn={metric_summary.churn:.4f}")
    print(f"memory_growth={metric_summary.memory_growth:.4f}")
    print(f"final_prototype_count={len(final_snapshot)}")
    print(f"reentry_reconnect_rate={reattachment['rate']:.4f}")


def _compute_reentry_reconnect_rate(sequence, instance_assignment_history: list[dict[int, int | None]]) -> dict[str, float | int]:
    visible_history: dict[int, list[bool]] = defaultdict(list)
    assignment_history: dict[int, list[int | None]] = defaultdict(list)

    for frame_index in range(1, len(sequence.frames)):
        frame = sequence.frames[frame_index]
        assignment_map = instance_assignment_history[frame_index - 1]
        visible_ids = set(frame.instance_ids)
        all_ids = set(visible_ids) | set(assignment_map.keys()) | set(visible_history.keys())
        for instance_id in all_ids:
            visible_history[instance_id].append(instance_id in visible_ids)
            assignment_history[instance_id].append(assignment_map.get(instance_id))

    total_reentries = 0
    reconnect_hits = 0

    for instance_id, visibility in visible_history.items():
        assignments = assignment_history[instance_id]
        was_visible = visibility[0] if visibility else False
        seen_prototypes: set[int] = set()
        if was_visible and assignments and assignments[0] is not None:
            seen_prototypes.add(int(assignments[0]))

        for index in range(1, len(visibility)):
            current_visible = visibility[index]
            current_assignment = assignments[index]
            if current_visible and not was_visible:
                total_reentries += 1
                if current_assignment is not None and int(current_assignment) in seen_prototypes:
                    reconnect_hits += 1
            if current_visible and current_assignment is not None:
                seen_prototypes.add(int(current_assignment))
            was_visible = current_visible

    rate = reconnect_hits / total_reentries if total_reentries else 0.0
    return {
        "events": total_reentries,
        "reconnected": reconnect_hits,
        "rate": rate,
    }


def _select_representative_frame(frame_records: list[dict[str, object]]) -> dict[str, object]:
    return max(
        frame_records,
        key=lambda record: (
            float(record["u_recall"]),
            min(len(record["memory_assignments"]), len(record["gt_boxes"])),
            -abs(len(record["memory_assignments"]) - len(record["gt_boxes"])),
            -int(record["memory_size"]),
            int(record["frame_index"]),
        ),
    )


def _save_memory_figure(
    representative_frame: dict[str, object],
    frame_records: list[dict[str, object]],
    final_snapshot,
    memory_budget: int,
    distance_series: list[float],
    similarity_series: list[float],
    update_weight_series: list[float],
    figure_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(22, 5))

    frame = representative_frame["frame"]
    axes[0].imshow(frame)
    axes[0].set_title(
        f"Representative Frame {representative_frame['frame_index']}\n"
        f"recall {float(representative_frame['u_recall']):.2f} | memory {int(representative_frame['memory_size'])}"
    )
    axes[0].axis("off")
    for box, instance_id in zip(representative_frame["gt_boxes"], representative_frame["gt_instance_ids"]):
        x1, y1, x2, y2 = box
        axes[0].add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.4, edgecolor="lime")
        )
        axes[0].text(x1, max(0, y1 - 4), f"gt:{instance_id}", color="lime", fontsize=8)
    for assignment in representative_frame["memory_assignments"]:
        x1, y1, x2, y2 = assignment.box
        axes[0].add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.5, edgecolor="white")
        )
        axes[0].text(x1, min(frame.shape[0] - 5, y2 + 10), f"P{assignment.prototype_id}", color="white", fontsize=8)

    frame_indices = [int(record["frame_index"]) for record in frame_records]
    memory_sizes = [int(record["memory_size"]) for record in frame_records]
    active_sizes = [len(record["active_prototypes"]) for record in frame_records]
    axes[1].plot(frame_indices, memory_sizes, label="total prototypes", linewidth=2.0)
    axes[1].plot(frame_indices, active_sizes, label="active prototypes", linewidth=1.6)
    axes[1].axhline(memory_budget, color="crimson", linestyle="--", linewidth=1.2, label="memory budget")
    axes[1].set_title("Prototype Count Over Time")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("count")
    axes[1].legend(frameon=False)

    axes[2].hist(distance_series, bins=14, alpha=0.75, color="tab:orange")
    axes[2].set_title("Prototype Distance d*")
    axes[2].set_xlabel("distance")
    axes[2].set_ylabel("count")

    if similarity_series:
        axes[3].hist(similarity_series, bins=14, alpha=0.65, color="tab:blue", label="s*")
    if update_weight_series:
        axes[3].hist(update_weight_series, bins=14, alpha=0.50, color="tab:green", label="lr*s*")
    axes[3].set_title("Similarity / Update Weight")
    axes[3].set_xlabel("value")
    axes[3].set_ylabel("count")
    axes[3].legend(frameon=False)

    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _load_config_payload(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _get_field_config(config_payload: dict) -> dict:
    if "field" in config_payload:
        return config_payload["field"]
    return config_payload["model"]["objectness"]


def _series_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "mean": float(sum(values) / len(values)),
        "max": float(max(values)),
    }


if __name__ == "__main__":
    main()
