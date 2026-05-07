"""Run Phase 3 Track A / Track C bridge-synthetic prototypes with full protocol artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator, load_synth_dataset_config
from experiments.scenario_presets import build_phase3_track_scenarios
from metrics.metrics_core import u_recall
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.evaluation import StreamingEpisodeEvaluator
from nops_owr.memory import MinimalPrototypeMemory
from nops_owr.objectness import MinimalObjectnessField
from nops_owr.tracking import MinimalTemporalIdentityTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 Track A / Track C prototype episodes.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml", help="Path to the bridge config.")
    parser.add_argument("--output-dir", default="results/phase3_track_proto", help="Directory for output artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_config_payload(args.config)
    base_config = load_synth_dataset_config(args.config)
    scenarios = build_phase3_track_scenarios(base_config)
    evaluator = StreamingEpisodeEvaluator(payload)

    config_snapshot_path = output_dir / "config_snapshot_bridge_synth_generic_v1.yaml"
    config_snapshot_path.write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    protocol_source_path = ROOT / "protocol" / "protocol_nops_bench_v2.md"
    protocol_snapshot_path = output_dir / "protocol_snapshot_nops_bench_v2.md"
    protocol_snapshot_path.write_text(protocol_source_path.read_text(encoding="utf-8"), encoding="utf-8")

    rows: list[dict[str, object]] = []
    scenario_artifacts: list[dict[str, object]] = []

    for index, scenario in enumerate(scenarios):
        sequence = SyntheticStreamGenerator(scenario["config"], seed=args.seed + index * 17).generate_sequence(0)
        result = evaluator.evaluate(sequence)
        diagnostics = _collect_scenario_diagnostics(sequence, payload)
        row = _build_summary_row(scenario["name"], scenario["config"], sequence, result)
        rows.append(row)
        scenario_artifacts.append(
            _write_scenario_artifacts(
                output_dir=output_dir,
                scenario_name=scenario["name"],
                sequence=sequence,
                row=row,
                result=result,
                diagnostics=diagnostics,
            )
        )

    csv_path = output_dir / "track_proto_summary_v1.csv"
    json_path = output_dir / "track_proto_summary_v1.json"
    manifest_path = output_dir / "run_manifest_v1.json"
    summary_md_path = output_dir / "phase3_proto_summary_v1.md"
    failure_notes_path = output_dir / "phase3_failure_notes_v1.md"

    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_snapshot": _repo_relative(config_snapshot_path),
                "protocol_snapshot": _repo_relative(protocol_snapshot_path),
                "protocol_source": "protocol/protocol_nops_bench_v2.md",
                "scenarios": scenario_artifacts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    hard_reference = _load_hard_reference_recall()
    summary_md_path.write_text(
        _build_phase3_summary_markdown(rows, scenario_artifacts, hard_reference),
        encoding="utf-8",
    )
    failure_notes_path.write_text(_build_failure_notes_markdown(rows), encoding="utf-8")

    print(f"saved_csv={csv_path}")
    print(f"saved_json={json_path}")
    print(f"saved_manifest={manifest_path}")
    print(f"saved_summary={summary_md_path}")
    print(f"saved_failure_notes={failure_notes_path}")
    for row in rows:
        print(
            f"{row['scenario']}: "
            f"track_idsw={int(row['track_idsw'])}, "
            f"reentry_rate={float(row['reentry_recovery_rate']):.4f}, "
            f"reactivations={int(row['reactivation_successes'])}, "
            f"u_recall={float(row['u_recall']):.4f}, "
            f"memory_growth={float(row['memory_growth']):.4f}"
        )


def _build_summary_row(scenario_name: str, scenario_config, sequence, result) -> dict[str, object]:
    return {
        "scenario": scenario_name,
        "sequence_length": scenario_config.sequence_length,
        "difficulty_preset": sequence.metadata.get("difficulty_preset", ""),
        "u_recall": float(result.summary.u_recall),
        "purity": float(result.summary.purity),
        "pfr": float(result.summary.pfr),
        "prototype_idsw": int(result.summary.idsw),
        "track_idsw": int(result.primary_monitoring["track_idsw"]),
        "reentry_recovery_rate": float(result.primary_monitoring["reentry_recovery_rate"]),
        "reentry_events": int(result.primary_monitoring["reentry_events"]),
        "reactivation_successes": int(result.primary_monitoring["reactivation_successes"]),
        "created_tracks": int(result.primary_monitoring["created_tracks"]),
        "mean_unmatched_tracks": float(result.primary_monitoring["mean_unmatched_tracks"]),
        "mean_dormant_tracks": float(result.primary_monitoring["mean_dormant_tracks"]),
        "max_dormant_tracks": int(result.primary_monitoring["max_dormant_tracks"]),
        "objectness_recall": float(result.secondary_monitoring["objectness_recall"]),
        "false_hot_area": float(result.secondary_monitoring["false_hot_area"]),
        "churn": float(result.summary.churn),
        "memory_growth": float(result.summary.memory_growth),
        "budget_violation_frames": int(result.budget_report.violation_frames),
        "peak_memory_size": int(result.budget_report.peak_memory_size),
        "peak_proposals": int(result.budget_report.peak_proposals),
        "final_prototype_count": int(result.audit.final_proto_count),
    }


def _write_scenario_artifacts(
    *,
    output_dir: Path,
    scenario_name: str,
    sequence,
    row: dict[str, object],
    result,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    budget_json_path = output_dir / f"{scenario_name}_budget_report_v1.json"
    budget_csv_path = output_dir / f"{scenario_name}_budget_report_v1.csv"
    scenario_json_path = output_dir / f"{scenario_name}_summary_v1.json"
    tracking_fig_path = output_dir / f"{scenario_name}_tracking_diag_v1.png"
    memory_fig_path = output_dir / f"{scenario_name}_memory_diag_v1.png"

    budget_rows = [asdict(record) for record in result.budget_report.frame_records]
    budget_json_path.write_text(
        json.dumps({"summary": asdict(result.budget_report), "frames": budget_rows}, indent=2),
        encoding="utf-8",
    )
    _write_csv(budget_csv_path, budget_rows)

    _save_tracking_figure(
        sequence=sequence,
        scenario_name=scenario_name,
        row=row,
        diagnostics=diagnostics,
        figure_path=tracking_fig_path,
    )
    _save_memory_figure(
        sequence=sequence,
        scenario_name=scenario_name,
        row=row,
        result=result,
        diagnostics=diagnostics,
        figure_path=memory_fig_path,
    )

    scenario_json_path.write_text(
        json.dumps(
            {
                "scenario": scenario_name,
                "summary_row": row,
                "metric_summary": asdict(result.summary),
                "metric_audit": asdict(result.audit),
                "primary_monitoring": result.primary_monitoring,
                "secondary_monitoring": result.secondary_monitoring,
                "budget_report": asdict(result.budget_report),
                "action_counter": result.action_counter,
                "diagnostics": diagnostics["summary"],
                "artifacts": {
                    "tracking_figure": _repo_relative(tracking_fig_path),
                    "memory_figure": _repo_relative(memory_fig_path),
                    "budget_csv": _repo_relative(budget_csv_path),
                    "budget_json": _repo_relative(budget_json_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "scenario": scenario_name,
        "summary_json": _repo_relative(scenario_json_path),
        "tracking_figure": _repo_relative(tracking_fig_path),
        "memory_figure": _repo_relative(memory_fig_path),
        "budget_csv": _repo_relative(budget_csv_path),
        "budget_json": _repo_relative(budget_json_path),
    }


def _collect_scenario_diagnostics(sequence, payload: dict) -> dict[str, object]:
    encoder = MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    objectness = MinimalObjectnessField(**_get_field_config(payload))
    tracker = MinimalTemporalIdentityTracker(**payload["tracking"])
    memory = MinimalPrototypeMemory(**payload["memory"])

    frame_indices: list[int] = []
    mean_costs: list[float] = []
    min_costs: list[float] = []
    max_costs: list[float] = []
    unmatched_tracks: list[int] = []
    dormant_tracks: list[int] = []
    new_tracks: list[int] = []
    reactivated_tracks: list[int] = []
    objectness_recalls: list[float] = []
    false_hot_areas: list[float] = []
    memory_sizes: list[int] = []
    active_prototypes: list[int] = []
    birth_counts: list[int] = []
    merge_counts: list[int] = []
    reuse_counts: list[int] = []
    similarities: list[float] = []
    distances: list[float] = []
    update_weights: list[float] = []
    reentry_markers: list[int] = []

    tracking_rep_score: tuple[float, ...] | None = None
    memory_rep_score: tuple[float, ...] | None = None
    tracking_rep: dict[str, object] | None = None
    memory_rep: dict[str, object] | None = None

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
        memory_output = memory.update(tracking_output.assignments, frame_index=current_frame.frame_index)

        track_boxes = [assignment.box for assignment in tracking_output.assignments]
        proto_boxes = [assignment.box for assignment in memory_output.assignments]
        track_recall = u_recall(current_frame.boxes, track_boxes, iou_threshold=0.5)
        proto_recall = u_recall(current_frame.boxes, proto_boxes, iou_threshold=0.5)
        false_hot = _false_hot_area(objectness_output.binary_mask, current_frame.masks)

        action_counts = {"birth": 0, "merge": 0, "reuse": 0}
        for assignment in memory_output.assignments:
            action_counts[assignment.action] = action_counts.get(assignment.action, 0) + 1
            similarities.append(float(assignment.soft_similarity))
            distances.append(float(assignment.distance))
            update_weights.append(float(assignment.update_weight))

        frame_indices.append(int(current_frame.frame_index))
        min_costs.append(float(tracking_output.cost_stats.min_cost))
        mean_costs.append(float(tracking_output.cost_stats.mean_cost))
        max_costs.append(float(tracking_output.cost_stats.max_cost))
        unmatched_tracks.append(int(tracking_output.unmatched_track_count))
        dormant_tracks.append(int(tracking_output.dormant_track_count))
        new_tracks.append(len(tracking_output.new_track_ids))
        reactivated_tracks.append(len(tracking_output.reactivated_track_ids))
        objectness_recalls.append(float(track_recall))
        false_hot_areas.append(float(false_hot))
        memory_sizes.append(int(memory_output.total_prototypes))
        active_prototypes.append(len(memory_output.active_prototype_ids))
        birth_counts.append(int(action_counts.get("birth", 0)))
        merge_counts.append(int(action_counts.get("merge", 0)))
        reuse_counts.append(int(action_counts.get("reuse", 0)))
        reentry_markers.append(int(bool(current_frame.reentry_event)))

        track_snapshot = {
            "frame_offset": frame_offset,
            "frame_index": int(current_frame.frame_index),
            "gt_boxes": list(current_frame.boxes),
            "gt_instance_ids": list(current_frame.instance_ids),
            "track_assignments": [
                {
                    "track_id": int(assignment.track_id),
                    "box": tuple(int(value) for value in assignment.box),
                    "cost": float(assignment.match_cost),
                }
                for assignment in tracking_output.assignments
            ],
            "new_track_ids": list(tracking_output.new_track_ids),
            "reactivated_track_ids": list(tracking_output.reactivated_track_ids),
            "mean_cost": float(tracking_output.cost_stats.mean_cost),
            "unmatched_tracks": int(tracking_output.unmatched_track_count),
            "reentry_event": bool(current_frame.reentry_event),
            "track_recall": float(track_recall),
        }
        track_score = (
            float(current_frame.reentry_event),
            float(len(tracking_output.reactivated_track_ids)),
            float(track_recall),
            -float(tracking_output.unmatched_track_count),
            -float(tracking_output.cost_stats.mean_cost),
        )
        if tracking_rep_score is None or track_score > tracking_rep_score:
            tracking_rep_score = track_score
            tracking_rep = track_snapshot

        memory_snapshot = {
            "frame_offset": frame_offset,
            "frame_index": int(current_frame.frame_index),
            "gt_boxes": list(current_frame.boxes),
            "gt_instance_ids": list(current_frame.instance_ids),
            "prototype_assignments": [
                {
                    "prototype_id": int(assignment.prototype_id),
                    "box": tuple(int(value) for value in assignment.box),
                    "action": str(assignment.action),
                    "distance": float(assignment.distance),
                    "similarity": float(assignment.soft_similarity),
                    "update_weight": float(assignment.update_weight),
                }
                for assignment in memory_output.assignments
            ],
            "memory_size": int(memory_output.total_prototypes),
            "active_prototypes": len(memory_output.active_prototype_ids),
            "proto_recall": float(proto_recall),
        }
        memory_score = (
            float(proto_recall),
            float(len(memory_output.active_prototype_ids)),
            -abs(len(memory_output.assignments) - len(current_frame.boxes)),
            -float(memory_output.total_prototypes),
        )
        if memory_rep_score is None or memory_score > memory_rep_score:
            memory_rep_score = memory_score
            memory_rep = memory_snapshot

    return {
        "series": {
            "frame_index": frame_indices,
            "mean_cost": mean_costs,
            "min_cost": min_costs,
            "max_cost": max_costs,
            "unmatched_tracks": unmatched_tracks,
            "dormant_tracks": dormant_tracks,
            "new_tracks": new_tracks,
            "reactivated_tracks": reactivated_tracks,
            "objectness_recall": objectness_recalls,
            "false_hot_area": false_hot_areas,
            "memory_size": memory_sizes,
            "active_prototypes": active_prototypes,
            "birth_count": birth_counts,
            "merge_count": merge_counts,
            "reuse_count": reuse_counts,
            "reentry_marker": reentry_markers,
        },
        "representative_tracking_frame": tracking_rep,
        "representative_memory_frame": memory_rep,
        "summary": {
            "mean_cost_stats": _series_stats(mean_costs),
            "unmatched_track_stats": _series_stats(unmatched_tracks),
            "dormant_track_stats": _series_stats(dormant_tracks),
            "new_track_stats": _series_stats(new_tracks),
            "reactivated_track_stats": _series_stats(reactivated_tracks),
            "memory_size_stats": _series_stats(memory_sizes),
            "prototype_similarity_stats": _series_stats(similarities),
            "prototype_distance_stats": _series_stats(distances),
            "prototype_update_weight_stats": _series_stats(update_weights),
            "reentry_frame_count": int(sum(reentry_markers)),
        },
        "histograms": {
            "similarities": similarities,
            "distances": distances,
            "update_weights": update_weights,
        },
    }


def _save_tracking_figure(
    *,
    sequence,
    scenario_name: str,
    row: dict[str, object],
    diagnostics: dict[str, object],
    figure_path: Path,
) -> None:
    frame = diagnostics["representative_tracking_frame"]
    if frame is None:
        return

    image = sequence.frames[int(frame["frame_offset"])].frame
    series = diagnostics["series"]
    frame_indices = np.asarray(series["frame_index"], dtype=np.int32)
    reentry_mask = np.asarray(series["reentry_marker"], dtype=np.int32) > 0

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title(
        f"{scenario_name} tracking frame {int(frame['frame_index'])}\n"
        f"reentry={bool(frame['reentry_event'])} | reactivated={len(frame['reactivated_track_ids'])}"
    )
    axes[0, 0].axis("off")
    for box, instance_id in zip(frame["gt_boxes"], frame["gt_instance_ids"]):
        x1, y1, x2, y2 = box
        axes[0, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=1.4))
        axes[0, 0].text(x1, max(0, y1 - 4), f"gt:{instance_id}", color="lime", fontsize=8)
    for assignment in frame["track_assignments"]:
        x1, y1, x2, y2 = assignment["box"]
        label_color = "cyan" if assignment["track_id"] in frame["reactivated_track_ids"] else "white"
        axes[0, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=label_color, linewidth=1.5))
        axes[0, 0].text(
            x1,
            min(image.shape[0] - 5, y2 + 10),
            f"T{assignment['track_id']} c={assignment['cost']:.2f}",
            color=label_color,
            fontsize=8,
        )

    axes[0, 1].plot(frame_indices, series["min_cost"], label="min", linewidth=1.2, color="tab:green")
    axes[0, 1].plot(frame_indices, series["mean_cost"], label="mean", linewidth=1.8, color="tab:blue")
    axes[0, 1].plot(frame_indices, series["max_cost"], label="max", linewidth=1.2, color="tab:red")
    _mark_reentry_events(axes[0, 1], frame_indices, reentry_mask)
    axes[0, 1].set_title("Tracking Cost Statistics")
    axes[0, 1].set_xlabel("frame")
    axes[0, 1].set_ylabel("cost")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(frame_indices, series["unmatched_tracks"], label="unmatched", linewidth=1.8, color="tab:orange")
    axes[1, 0].plot(frame_indices, series["dormant_tracks"], label="dormant", linewidth=1.6, color="tab:purple")
    axes[1, 0].plot(frame_indices, series["new_tracks"], label="new", linewidth=1.4, color="tab:brown")
    axes[1, 0].plot(frame_indices, series["reactivated_tracks"], label="reactivated", linewidth=1.4, color="tab:cyan")
    _mark_reentry_events(axes[1, 0], frame_indices, reentry_mask)
    axes[1, 0].set_title("Track Pressure / Re-entry Activity")
    axes[1, 0].set_xlabel("frame")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(frame_indices, series["objectness_recall"], label="track recall", linewidth=1.8, color="tab:blue")
    axes[1, 1].plot(frame_indices, series["false_hot_area"], label="false hot area", linewidth=1.4, color="tab:gray")
    _mark_reentry_events(axes[1, 1], frame_indices, reentry_mask)
    axes[1, 1].set_title(
        f"Recall / Noise | IDSW={int(row['track_idsw'])} | reentry={float(row['reentry_recovery_rate']):.2f}"
    )
    axes[1, 1].set_xlabel("frame")
    axes[1, 1].set_ylabel("value")
    axes[1, 1].legend(frameon=False)

    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _save_memory_figure(
    *,
    sequence,
    scenario_name: str,
    row: dict[str, object],
    result,
    diagnostics: dict[str, object],
    figure_path: Path,
) -> None:
    frame = diagnostics["representative_memory_frame"]
    if frame is None:
        return

    image = sequence.frames[int(frame["frame_offset"])].frame
    series = diagnostics["series"]
    hist = diagnostics["histograms"]
    frame_indices = np.asarray(series["frame_index"], dtype=np.int32)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title(
        f"{scenario_name} memory frame {int(frame['frame_index'])}\n"
        f"prototypes={int(frame['memory_size'])} | proto recall={float(frame['proto_recall']):.2f}"
    )
    axes[0, 0].axis("off")
    for box, instance_id in zip(frame["gt_boxes"], frame["gt_instance_ids"]):
        x1, y1, x2, y2 = box
        axes[0, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=1.4))
        axes[0, 0].text(x1, max(0, y1 - 4), f"gt:{instance_id}", color="lime", fontsize=8)
    for assignment in frame["prototype_assignments"]:
        x1, y1, x2, y2 = assignment["box"]
        axes[0, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=1.5))
        axes[0, 0].text(
            x1,
            min(image.shape[0] - 5, y2 + 10),
            f"P{assignment['prototype_id']} {assignment['action']}",
            color="white",
            fontsize=8,
        )

    axes[0, 1].plot(frame_indices, series["memory_size"], label="memory size", linewidth=2.0, color="tab:blue")
    axes[0, 1].plot(frame_indices, series["active_prototypes"], label="active prototypes", linewidth=1.6, color="tab:orange")
    axes[0, 1].axhline(
        int(result.budget_report.memory_budget),
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label="memory budget",
    )
    axes[0, 1].set_title("Prototype Count Curve")
    axes[0, 1].set_xlabel("frame")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].legend(frameon=False)

    if hist["distances"]:
        axes[1, 0].hist(hist["distances"], bins=16, alpha=0.70, color="tab:orange", label="distance")
    if hist["similarities"]:
        axes[1, 0].hist(hist["similarities"], bins=16, alpha=0.55, color="tab:blue", label="similarity")
    axes[1, 0].set_title("Prototype Distance / Similarity")
    axes[1, 0].set_xlabel("value")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(frame_indices, series["birth_count"], label="birth", linewidth=1.6, color="tab:red")
    axes[1, 1].plot(frame_indices, series["merge_count"], label="merge", linewidth=1.6, color="tab:green")
    axes[1, 1].plot(frame_indices, series["reuse_count"], label="reuse", linewidth=1.6, color="tab:blue")
    axes[1, 1].set_title(
        f"Prototype Actions | PFR={float(row['pfr']):.2f} | final={int(row['final_prototype_count'])}"
    )
    axes[1, 1].set_xlabel("frame")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].legend(frameon=False)

    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _build_phase3_summary_markdown(
    rows: list[dict[str, object]],
    scenario_artifacts: list[dict[str, object]],
    hard_reference: float | None,
) -> str:
    by_name = {str(row["scenario"]): row for row in rows}
    track_a = by_name.get("track_a_bridge")
    track_c = by_name.get("track_c_long_horizon")

    lines = [
        "# Phase 3 Prototype Summary v1",
        "",
        "## Scope",
        "",
        "This bundle records the first complete Phase 3 prototype pass for generic bridge synthetic under the NOPS-Bench v2 protocol.",
        "",
        "## Scenario Summary",
        "",
        "| scenario | U-Recall | track IDSW | reentry recovery | PFR | memory growth | budget violations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {float(row['u_recall']):.4f} | {int(row['track_idsw'])} | "
            f"{float(row['reentry_recovery_rate']):.4f} | {float(row['pfr']):.4f} | "
            f"{float(row['memory_growth']):.4f} | {int(row['budget_violation_frames'])} |"
        )

    lines.extend(
        [
            "",
            "## Protocol Check",
            "",
            f"- Hard-synthetic reference U-Recall: {hard_reference:.4f}" if hard_reference is not None else "- Hard-synthetic reference U-Recall: unavailable",
            f"- Track A bridge keeps unknown discovery above the hard-synthetic reference: {'yes' if track_a and (hard_reference is None or float(track_a['u_recall']) >= hard_reference) else 'no'}",
            f"- Track C long-horizon keeps unknown discovery near the hard-synthetic reference: {'yes' if track_c and (hard_reference is None or float(track_c['u_recall']) >= hard_reference - 0.08) else 'no'}",
            f"- Tracking / re-entry is now the dominant bottleneck: {'yes' if track_c and (float(track_c['reentry_recovery_rate']) < 0.20 or int(track_c['track_idsw']) > 100) else 'no'}",
            f"- Memory stays inside the declared budget: {'yes' if all(int(row['budget_violation_frames']) == 0 for row in rows) else 'no'}",
            f"- Prototype fragmentation remains controlled: {'yes' if track_c and float(track_c['pfr']) <= 1.50 else 'no'}",
            "",
            "## Go / No-Go",
            "",
        ]
    )

    if track_c and (float(track_c["reentry_recovery_rate"]) < 0.20 or float(track_c["pfr"]) > 1.50):
        lines.extend(
            [
                "- Go for continued Phase 3 tracker work on generic bridge synthetic.",
                "- No-Go for advancing to Track B, real-data claims, or broader benchmark expansion.",
                "- The main blocker is long-gap tracking / re-entry, with prototype fragmentation as a secondary effect.",
            ]
        )
    else:
        lines.extend(
            [
                "- Go for continued Phase 3 prototype expansion.",
                "- Tracking / re-entry is stable enough to start considering the next protocol slice.",
            ]
        )

    lines.extend(["", "## Artifacts", ""])
    for artifact in scenario_artifacts:
        lines.append(
            f"- {artifact['scenario']}: {artifact['summary_json']}, {artifact['tracking_figure']}, {artifact['memory_figure']}, {artifact['budget_csv']}"
        )
    lines.append("")
    return "\n".join(lines)


def _build_failure_notes_markdown(rows: list[dict[str, object]]) -> str:
    lines = ["# Phase 3 Failure Notes v1", ""]
    for row in rows:
        scenario = str(row["scenario"])
        lines.append(f"## {scenario}")
        if float(row["reentry_recovery_rate"]) < 0.20 and int(row["track_idsw"]) > 50:
            lines.append("- Primary failure: long-gap re-entry is not reconnecting to prior track IDs reliably.")
            lines.append("- Secondary effect: tracker instability is leaking into prototype fragmentation and PFR.")
        elif int(row["track_idsw"]) > 50:
            lines.append("- Primary failure: tracking drift remains high even when re-entry reconnects.")
        elif float(row["pfr"]) > 1.0:
            lines.append("- Primary failure: prototype fragmentation is rising faster than tracking can stabilize it.")
        else:
            lines.append("- No dominant failure mode. Remaining errors look like moderate residual noise rather than collapse.")

        if int(row["budget_violation_frames"]) == 0:
            lines.append("- Budget status: controlled. No budget-violation frames were observed.")
        else:
            lines.append("- Budget status: unstable. Budget violations need attention before expanding scope.")

        lines.append(
            f"- Headline metrics: U-Recall={float(row['u_recall']):.4f}, track IDSW={int(row['track_idsw'])}, "
            f"reentry={float(row['reentry_recovery_rate']):.4f}, PFR={float(row['pfr']):.4f}."
        )
        lines.append("")
    return "\n".join(lines)


def _mark_reentry_events(axis, frame_indices: np.ndarray, reentry_mask: np.ndarray) -> None:
    if frame_indices.size == 0:
        return
    for frame_index in frame_indices[reentry_mask]:
        axis.axvline(int(frame_index), color="black", alpha=0.08, linewidth=0.8)


def _false_hot_area(binary_mask: np.ndarray, masks: list[np.ndarray]) -> float:
    gt_mask = np.zeros_like(binary_mask, dtype=bool)
    for mask in masks:
        gt_mask |= mask.astype(bool)
    false_hot = binary_mask.astype(bool) & ~gt_mask
    return float(false_hot.sum() / max(false_hot.size, 1))


def _series_stats(values: list[int] | list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "mean": float(sum(values) / len(values)),
        "max": float(max(values)),
    }


def _write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_config_payload(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _get_field_config(config_payload: dict) -> dict:
    if "field" in config_payload:
        return dict(config_payload["field"])
    return dict(config_payload["model"]["objectness"])


def _load_hard_reference_recall() -> float | None:
    reference_path = ROOT / "results" / "phase2b_final" / "scenario_summary_v2.csv"
    if not reference_path.exists():
        return None

    with reference_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("scenario") == "hard_drift_occlusion":
                try:
                    return float(row["u_recall"])
                except (KeyError, TypeError, ValueError):
                    return None
    return None


def _repo_relative(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


if __name__ == "__main__":
    main()
