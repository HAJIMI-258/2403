"""Run Day 7 scenario summary for the minimal NOPS-OWR pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator, load_synth_dataset_config
from experiments.scenario_presets import build_phase1_scenarios
from metrics.metrics_core import greedy_match_boxes, summarize_phase1_metrics
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.memory import MinimalPrototypeMemory
from nops_owr.objectness import MinimalObjectnessField
from nops_owr.tracking import MinimalTemporalIdentityTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main pipeline on three explicit synthetic scenarios.")
    parser.add_argument("--config", default="configs/synth.yaml", help="Path to the config file.")
    parser.add_argument("--output-dir", default="results/day7_scenarios", help="Directory for output artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_payload = _load_config_payload(args.config)
    base_config = load_synth_dataset_config(args.config)
    scenarios = build_phase1_scenarios(base_config)

    results: list[dict[str, float | int | str]] = []
    for index, scenario in enumerate(scenarios):
        metrics = _run_scenario(
            scenario_name=scenario["name"],
            scenario_config=scenario["config"],
            config_payload=config_payload,
            seed=args.seed + index * 11,
        )
        results.append(metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scenario_summary.csv"
    json_path = output_dir / "scenario_summary.json"
    figure_path = output_dir / "scenario_summary.png"

    _write_csv(csv_path, results)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _save_figure(results, figure_path)

    print(f"saved_csv={csv_path}")
    print(f"saved_json={json_path}")
    print(f"saved_figure={figure_path}")
    for result in results:
        print(
            f"{result['scenario']}: "
            f"u_recall={result['u_recall']:.4f}, "
            f"purity={result['purity']:.4f}, "
            f"pfr={result['pfr']:.4f}, "
            f"idsw={int(result['idsw'])}, "
            f"memory_growth={result['memory_growth']:.4f}, "
            f"final_prototypes={int(result['final_prototype_count'])}"
        )

def _run_scenario(scenario_name: str, scenario_config, config_payload: dict, seed: int) -> dict[str, float | int | str]:
    sequence = SyntheticStreamGenerator(scenario_config, seed=seed).generate_sequence(0)
    encoder = MinimalSpikeEncoder(**config_payload["model"]["spike_encoder"])
    objectness = MinimalObjectnessField(**_get_field_config(config_payload))
    tracker = MinimalTemporalIdentityTracker(**config_payload["tracking"])
    memory = MinimalPrototypeMemory(**config_payload["memory"])

    gt_boxes_per_frame = []
    pred_boxes_per_frame = []
    prototype_concept_assignments = []
    instance_assignment_history = []
    active_sets = []
    memory_sizes = []

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

        pred_boxes = [assignment.box for assignment in memory_output.assignments]
        pred_ids = [assignment.prototype_id for assignment in memory_output.assignments]

        gt_boxes_per_frame.append(list(current_frame.boxes))
        pred_boxes_per_frame.append(list(pred_boxes))
        active_sets.append(list(memory_output.active_prototype_ids))
        memory_sizes.append(memory_output.total_prototypes)

        history = {instance_id: None for instance_id in current_frame.instance_ids}
        matches = greedy_match_boxes(current_frame.boxes, pred_boxes, iou_threshold=0.5)
        for gt_index, pred_index, _ in matches:
            history[current_frame.instance_ids[gt_index]] = pred_ids[pred_index]
            prototype_concept_assignments.append((pred_ids[pred_index], current_frame.concept_ids[gt_index]))
        instance_assignment_history.append(history)

    summary = summarize_phase1_metrics(
        gt_boxes_per_frame=gt_boxes_per_frame,
        pred_boxes_per_frame=pred_boxes_per_frame,
        prototype_concept_assignments=prototype_concept_assignments,
        instance_assignment_history=instance_assignment_history,
        active_prototype_sets=active_sets,
        memory_sizes=memory_sizes,
        iou_threshold=0.5,
    )

    return {
        "scenario": scenario_name,
        "sequence_length": scenario_config.sequence_length,
        "u_recall": float(summary.u_recall),
        "purity": float(summary.purity),
        "pfr": float(summary.pfr),
        "idsw": int(summary.idsw),
        "churn": float(summary.churn),
        "memory_growth": float(summary.memory_growth),
        "final_prototype_count": int(memory_sizes[-1]) if memory_sizes else 0,
    }


def _write_csv(csv_path: Path, results: list[dict[str, float | int | str]]) -> None:
    fieldnames = [
        "scenario",
        "sequence_length",
        "u_recall",
        "purity",
        "pfr",
        "idsw",
        "churn",
        "memory_growth",
        "final_prototype_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def _save_figure(results: list[dict[str, float | int | str]], figure_path: Path) -> None:
    scenarios = [result["scenario"] for result in results]
    figure, axes = plt.subplots(2, 3, figsize=(16, 8))
    metric_keys = ["u_recall", "purity", "pfr", "idsw", "memory_growth", "final_prototype_count"]
    titles = {
        "u_recall": "U-Recall",
        "purity": "Purity",
        "pfr": "PFR",
        "idsw": "IDSW",
        "memory_growth": "Memory Growth",
        "final_prototype_count": "Final Prototype Count",
    }

    for axis, metric_key in zip(axes.flat, metric_keys):
        values = [result[metric_key] for result in results]
        axis.bar(scenarios, values, color=["tab:blue", "tab:orange", "tab:red"])
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
