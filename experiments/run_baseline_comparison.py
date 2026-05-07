"""Run Day 6 comparison between minimal NOPS-OWR and weak baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import EdgeClusterBaseline, FrameDiffConnectedComponentsBaseline
from datasets import SyntheticStreamGenerator, load_synth_dataset_config
from metrics.metrics_core import greedy_match_boxes, summarize_phase1_metrics
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.memory import MinimalPrototypeMemory
from nops_owr.objectness import MinimalObjectnessField
from nops_owr.tracking import MinimalTemporalIdentityTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare minimal NOPS-OWR with two weak baselines.")
    parser.add_argument("--config", default="configs/synth.yaml", help="Path to the config file.")
    parser.add_argument(
        "--sequence-ids",
        default="2,3",
        help="Comma-separated synthetic sequence ids to evaluate.",
    )
    parser.add_argument("--output-dir", default="results/day6_baselines", help="Directory for comparison artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_payload = _load_config_payload(args.config)
    synth_config = load_synth_dataset_config(args.config)
    sequence_ids = [int(item.strip()) for item in args.sequence_ids.split(",") if item.strip()]

    sequences = {
        sequence_id: SyntheticStreamGenerator(synth_config, seed=args.seed).generate_sequence(sequence_id)
        for sequence_id in sequence_ids
    }

    method_records: dict[str, list[dict[str, float | int]]] = {
        "minimal_nops_owr": [],
        "baseline_frame_diff_cc": [],
        "baseline_edge_cluster": [],
    }

    for sequence_id, sequence in sequences.items():
        method_records["minimal_nops_owr"].append(
            _evaluate_main_pipeline(sequence, config_payload, sequence_id)
        )
        method_records["baseline_frame_diff_cc"].append(
            _evaluate_frame_diff_baseline(sequence, sequence_id)
        )
        method_records["baseline_edge_cluster"].append(
            _evaluate_edge_cluster_baseline(sequence, sequence_id)
        )

    aggregated = {
        method: _aggregate_records(records)
        for method, records in method_records.items()
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "baseline_comparison.csv"
    json_path = output_dir / "baseline_comparison.json"
    figure_path = output_dir / "baseline_comparison.png"

    _write_csv(csv_path, method_records, aggregated)
    json_path.write_text(
        json.dumps(
            {
                "sequence_ids": sequence_ids,
                "per_sequence": method_records,
                "aggregated": aggregated,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _save_figure(aggregated, figure_path)

    print(f"saved_csv={csv_path}")
    print(f"saved_json={json_path}")
    print(f"saved_figure={figure_path}")
    for method, metrics in aggregated.items():
        print(
            f"{method}: "
            f"u_recall={metrics['u_recall']:.4f}, "
            f"pfr={metrics['pfr']:.4f}, "
            f"idsw={metrics['idsw']:.2f}, "
            f"churn={metrics['churn']:.4f}, "
            f"memory_growth={metrics['memory_growth']:.4f}"
        )


def _evaluate_main_pipeline(sequence, config_payload: dict, sequence_id: int) -> dict[str, float | int]:
    encoder = MinimalSpikeEncoder(**config_payload["model"]["spike_encoder"])
    objectness = MinimalObjectnessField(**_get_field_config(config_payload))
    tracker = MinimalTemporalIdentityTracker(**config_payload["tracking"])
    memory = MinimalPrototypeMemory(**config_payload["memory"])

    metrics = _collect_stream_metrics(
        sequence=sequence,
        step_fn=lambda prev_frame, current_frame: _main_step(
            prev_frame,
            current_frame,
            encoder,
            objectness,
            tracker,
            memory,
        ),
        match_iou_threshold=config_payload["evaluation"]["match_iou_threshold"],
    )
    metrics["sequence_id"] = sequence_id
    return metrics


def _evaluate_frame_diff_baseline(sequence, sequence_id: int) -> dict[str, float | int]:
    baseline = FrameDiffConnectedComponentsBaseline()
    metrics = _collect_stream_metrics(
        sequence=sequence,
        step_fn=lambda prev_frame, current_frame: baseline.update(prev_frame.frame, current_frame.frame),
        match_iou_threshold=0.5,
    )
    metrics["sequence_id"] = sequence_id
    return metrics


def _evaluate_edge_cluster_baseline(sequence, sequence_id: int) -> dict[str, float | int]:
    baseline = EdgeClusterBaseline()
    metrics = _collect_stream_metrics(
        sequence=sequence,
        step_fn=lambda prev_frame, current_frame: baseline.update(current_frame.frame),
        match_iou_threshold=0.5,
    )
    metrics["sequence_id"] = sequence_id
    return metrics


def _main_step(prev_frame, current_frame, encoder, objectness, tracker, memory):
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
    return {
        "boxes": [assignment.box for assignment in memory_output.assignments],
        "ids": [assignment.prototype_id for assignment in memory_output.assignments],
        "active_ids": list(memory_output.active_prototype_ids),
        "memory_size": int(memory_output.total_prototypes),
    }


def _collect_stream_metrics(sequence, step_fn, match_iou_threshold: float) -> dict[str, float | int]:
    gt_boxes_per_frame: list[list[tuple[int, int, int, int]]] = []
    pred_boxes_per_frame: list[list[tuple[int, int, int, int]]] = []
    prototype_concept_assignments: list[tuple[int, int]] = []
    instance_assignment_history: list[dict[int, int | None]] = []
    active_sets: list[list[int]] = []
    memory_sizes: list[int] = []

    for frame_index in range(1, len(sequence.frames)):
        prev_frame = sequence.frames[frame_index - 1]
        current_frame = sequence.frames[frame_index]
        output = _normalize_output(step_fn(prev_frame, current_frame))
        pred_boxes = list(output["boxes"])
        pred_ids = list(output["ids"])

        gt_boxes_per_frame.append(list(current_frame.boxes))
        pred_boxes_per_frame.append(pred_boxes)
        active_sets.append(list(output["active_ids"]))
        memory_sizes.append(int(output["memory_size"]))

        history = {instance_id: None for instance_id in current_frame.instance_ids}
        matches = greedy_match_boxes(current_frame.boxes, pred_boxes, iou_threshold=match_iou_threshold)
        for gt_index, pred_index, _ in matches:
            instance_id = current_frame.instance_ids[gt_index]
            concept_id = current_frame.concept_ids[gt_index]
            predicted_id = int(pred_ids[pred_index])
            history[instance_id] = predicted_id
            prototype_concept_assignments.append((predicted_id, concept_id))
        instance_assignment_history.append(history)

    summary = summarize_phase1_metrics(
        gt_boxes_per_frame=gt_boxes_per_frame,
        pred_boxes_per_frame=pred_boxes_per_frame,
        prototype_concept_assignments=prototype_concept_assignments,
        instance_assignment_history=instance_assignment_history,
        active_prototype_sets=active_sets,
        memory_sizes=memory_sizes,
        iou_threshold=match_iou_threshold,
    )
    return {
        "u_recall": float(summary.u_recall),
        "purity": float(summary.purity),
        "pfr": float(summary.pfr),
        "idsw": int(summary.idsw),
        "churn": float(summary.churn),
        "memory_growth": float(summary.memory_growth),
        "final_memory_size": int(memory_sizes[-1]) if memory_sizes else 0,
    }


def _normalize_output(output) -> dict[str, object]:
    if isinstance(output, dict):
        return output
    return {
        "boxes": getattr(output, "boxes"),
        "ids": getattr(output, "ids"),
        "active_ids": getattr(output, "active_ids"),
        "memory_size": getattr(output, "memory_size"),
    }


def _aggregate_records(records: list[dict[str, float | int]]) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    metric_keys = [key for key in records[0].keys() if key != "sequence_id"]
    for key in metric_keys:
        aggregated[key] = float(sum(float(record[key]) for record in records) / len(records))
    return aggregated


def _write_csv(
    csv_path: Path,
    method_records: dict[str, list[dict[str, float | int]]],
    aggregated: dict[str, dict[str, float]],
) -> None:
    fieldnames = [
        "method",
        "sequence_id",
        "u_recall",
        "purity",
        "pfr",
        "idsw",
        "churn",
        "memory_growth",
        "final_memory_size",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, records in method_records.items():
            for record in records:
                writer.writerow({"method": method, **record})
            writer.writerow({"method": f"{method}__mean", "sequence_id": "avg", **aggregated[method]})


def _save_figure(aggregated: dict[str, dict[str, float]], figure_path: Path) -> None:
    methods = list(aggregated.keys())
    metric_keys = ["u_recall", "pfr", "idsw", "churn", "memory_growth", "final_memory_size"]
    titles = {
        "u_recall": "U-Recall",
        "pfr": "PFR",
        "idsw": "IDSW",
        "churn": "Churn",
        "memory_growth": "Memory Growth",
        "final_memory_size": "Final Memory Size",
    }

    figure, axes = plt.subplots(2, 3, figsize=(16, 8))
    for axis, metric_key in zip(axes.flat, metric_keys):
        values = [aggregated[method][metric_key] for method in methods]
        axis.bar(methods, values, color=["tab:blue", "tab:orange", "tab:green"])
        axis.set_title(titles[metric_key])
        axis.tick_params(axis="x", rotation=20)
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


if __name__ == "__main__":
    main()
