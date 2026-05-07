"""Evaluate re-entry recovery before and after motion prediction + keepalive."""

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

from datasets import SyntheticStreamGenerator, load_synth_dataset_config
from experiments.scenario_presets import build_hard_drift_occlusion_config, build_phase1_scenarios
from metrics.metrics_core import greedy_match_boxes, identity_switches, u_recall
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.objectness import MinimalObjectnessField
from nops_owr.tracking import MinimalTemporalIdentityTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tracking re-entry recovery.")
    parser.add_argument("--config", default="configs/synth.yaml", help="Path to the config file.")
    parser.add_argument("--output-dir", default="results/phase2_tracking", help="Directory for artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_config_payload(args.config)
    base_config = load_synth_dataset_config(args.config)

    scenario_map = {
        "multi_object_reentry": next(
            scenario["config"] for scenario in build_phase1_scenarios(base_config) if scenario["name"] == "multi_object_reentry"
        ),
        "hard_drift_occlusion": build_hard_drift_occlusion_config(base_config),
    }

    rows: list[dict[str, object]] = []
    for scenario_index, (scenario_name, scenario_config) in enumerate(scenario_map.items()):
        sequence = SyntheticStreamGenerator(scenario_config, seed=args.seed + scenario_index * 17).generate_sequence(0)
        rows.append(
            _evaluate_mode(
                sequence,
                payload,
                scenario_name,
                mode="before",
                tracking_override={
                    "use_linear_prediction": False,
                    "keepalive_frames": 0,
                    "use_dormant_reactivation": False,
                    "dormant_frames": 0,
                },
            )
        )
        rows.append(
            _evaluate_mode(
                sequence,
                payload,
                scenario_name,
                mode="after",
                tracking_override=payload["tracking"],
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tracking_reentry_eval_v1.csv"
    json_path = output_dir / "tracking_reentry_eval_v1.json"
    figure_path = output_dir / "tracking_reentry_eval_v1.png"

    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _save_figure(rows, figure_path)

    print(f"saved_csv={csv_path}")
    print(f"saved_json={json_path}")
    print(f"saved_figure={figure_path}")
    for row in rows:
        print(
            f"{row['scenario']}::{row['mode']}: "
            f"u_recall={float(row['u_recall']):.4f}, "
            f"idsw={int(row['idsw'])}, "
            f"reentry_rate={float(row['reentry_recovery_rate']):.4f}, "
            f"created_tracks={int(row['created_tracks'])}, "
            f"reactivations={int(row['reactivation_successes'])}"
        )


def _evaluate_mode(sequence, payload: dict, scenario_name: str, mode: str, tracking_override: dict[str, object]):
    encoder = MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    objectness = MinimalObjectnessField(**_get_field_config(payload))
    tracking_cfg = dict(payload["tracking"])
    tracking_cfg.update(tracking_override)
    tracker = MinimalTemporalIdentityTracker(**tracking_cfg)

    assignment_history: list[dict[int, int | None]] = []
    recalls = []
    active_track_counts = []
    dormant_track_counts = []
    created_tracks = set()
    reactivation_successes = 0

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
        predicted_boxes = [assignment.box for assignment in tracking_output.assignments]
        predicted_ids = [assignment.track_id for assignment in tracking_output.assignments]
        recalls.append(u_recall(current_frame.boxes, predicted_boxes, iou_threshold=0.5))
        active_track_counts.append(len(tracking_output.active_tracks))
        dormant_track_counts.append(int(tracking_output.dormant_track_count))
        created_tracks.update(predicted_ids)
        reactivation_successes += len(tracking_output.reactivated_track_ids)

        history = {instance_id: None for instance_id in current_frame.instance_ids}
        matches = greedy_match_boxes(current_frame.boxes, predicted_boxes, iou_threshold=0.5)
        for gt_index, pred_index, _ in matches:
            history[current_frame.instance_ids[gt_index]] = predicted_ids[pred_index]
        assignment_history.append(history)

    reconnect = _compute_reentry_recovery(sequence, assignment_history)
    return {
        "scenario": scenario_name,
        "mode": mode,
        "u_recall": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "idsw": int(identity_switches(assignment_history)),
        "reentry_events": int(reconnect["events"]),
        "recovered_old_id": int(reconnect["recovered"]),
        "reentry_recovery_rate": float(reconnect["rate"]),
        "mean_active_tracks": float(sum(active_track_counts) / len(active_track_counts)) if active_track_counts else 0.0,
        "mean_dormant_tracks": float(sum(dormant_track_counts) / len(dormant_track_counts)) if dormant_track_counts else 0.0,
        "peak_dormant_tracks": int(max(dormant_track_counts)) if dormant_track_counts else 0,
        "created_tracks": int(len(created_tracks)),
        "reactivation_successes": int(reactivation_successes),
    }


def _compute_reentry_recovery(sequence, assignment_history: list[dict[int, int | None]]):
    visible_history = {}
    assigned_history = {}
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


def _write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_figure(rows: list[dict[str, object]], figure_path: Path) -> None:
    scenarios = sorted({str(row["scenario"]) for row in rows})
    metrics = [
        ("u_recall", "U-Recall"),
        ("idsw", "IDSW"),
        ("reentry_recovery_rate", "Re-entry Rate"),
        ("reactivation_successes", "Reactivations"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = np.asarray(axes)

    for axis, (metric_key, title) in zip(axes.flat, metrics):
        before_values = [
            float(next(row[metric_key] for row in rows if row["scenario"] == scenario and row["mode"] == "before"))
            for scenario in scenarios
        ]
        after_values = [
            float(next(row[metric_key] for row in rows if row["scenario"] == scenario and row["mode"] == "after"))
            for scenario in scenarios
        ]
        x = np.arange(len(scenarios))
        width = 0.34
        axis.bar(x - width / 2, before_values, width=width, label="before", color="tab:gray")
        axis.bar(x + width / 2, after_values, width=width, label="after", color="tab:blue")
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(scenarios, rotation=18, ha="right")
        axis.legend(frameon=False)

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
