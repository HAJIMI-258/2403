"""Generate Phase 2B hard-case failure slices with stage labels."""

from __future__ import annotations

import argparse
import csv
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

from baselines import EdgeClusterBaseline
from datasets import SyntheticStreamGenerator, load_synth_dataset_config
from experiments.phase2b_utils import get_field_config
from experiments.scenario_presets import build_hard_drift_occlusion_config
from metrics.metrics_core import greedy_match_boxes, purity
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.memory import MinimalPrototypeMemory
from nops_owr.objectness import MinimalObjectnessField
from nops_owr.tracking import MinimalTemporalIdentityTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Phase 2B failure slices for hard drift + occlusion.")
    parser.add_argument("--config", default="configs/synth.yaml", help="Path to the config file.")
    parser.add_argument("--output-dir", default="results/phase2b_failure_slicing", help="Directory for artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_payload = _load_config_payload(args.config)
    base_config = load_synth_dataset_config(args.config)
    hard_config = build_hard_drift_occlusion_config(base_config)
    sequence = SyntheticStreamGenerator(hard_config, seed=args.seed).generate_sequence(0)

    encoder = MinimalSpikeEncoder(**config_payload["model"]["spike_encoder"])
    objectness = MinimalObjectnessField(**get_field_config(config_payload))
    tracker = MinimalTemporalIdentityTracker(**config_payload["tracking"])
    memory = MinimalPrototypeMemory(**config_payload["memory"])
    baseline2 = EdgeClusterBaseline()

    previous_gt_to_track: dict[int, int | None] = {}
    concept_to_prototypes_seen: dict[int, set[int]] = defaultdict(set)
    rows: list[dict[str, object]] = []
    frame_visuals: list[dict[str, object]] = []

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
        baseline_output = baseline2.update(current_frame.frame)

        tracking_boxes = [assignment.box for assignment in tracking_output.assignments]
        tracking_ids = [assignment.track_id for assignment in tracking_output.assignments]
        prototype_boxes = [assignment.box for assignment in memory_output.assignments]
        prototype_ids = [assignment.prototype_id for assignment in memory_output.assignments]
        baseline_boxes = list(baseline_output.boxes)

        tracking_matches = greedy_match_boxes(current_frame.boxes, tracking_boxes, iou_threshold=0.5)
        memory_matches = greedy_match_boxes(current_frame.boxes, prototype_boxes, iou_threshold=0.5)
        baseline_matches = greedy_match_boxes(current_frame.boxes, baseline_boxes, iou_threshold=0.5)

        gt_to_track = {instance_id: None for instance_id in current_frame.instance_ids}
        for gt_index, pred_index, _ in tracking_matches:
            gt_to_track[current_frame.instance_ids[gt_index]] = tracking_ids[pred_index]

        id_switch = 0
        for instance_id, track_id in gt_to_track.items():
            previous_track_id = previous_gt_to_track.get(instance_id)
            if previous_track_id is not None and track_id is not None and previous_track_id != track_id:
                id_switch = 1
                break
        previous_gt_to_track = gt_to_track

        frame_pairs = [
            (prototype_ids[pred_index], current_frame.concept_ids[gt_index])
            for gt_index, pred_index, _ in memory_matches
        ]
        frame_purity = purity(frame_pairs)

        fragmentation_delta = 0
        for prototype_id, concept_id in frame_pairs:
            seen = concept_to_prototypes_seen[concept_id]
            if seen and prototype_id not in seen:
                fragmentation_delta += 1
            seen.add(prototype_id)

        recall_hit = len(tracking_matches) / len(current_frame.boxes) if current_frame.boxes else 1.0
        false_hot_area = _false_hot_area(objectness_output.binary_mask, current_frame.masks)
        proposal_confidence = max((proposal.score for proposal in objectness_output.proposals), default=0.0)
        new_tracks = len(tracking_output.new_track_ids)
        unmatched_tracks = tracking_output.unmatched_track_count
        proto_count = memory_output.total_prototypes
        birth_count = sum(1 for assignment in memory_output.assignments if assignment.action == "birth")
        frame_assignment_conflict = 1 if frame_pairs and len({pair[0] for pair in frame_pairs}) < len({pair[1] for pair in frame_pairs}) else 0

        fail_flags = _classify_failures(
            num_gt=len(current_frame.boxes),
            num_prop=len(objectness_output.proposals),
            recall_hit=recall_hit,
            id_switch=id_switch,
            false_hot_area=false_hot_area,
            proposal_confidence=proposal_confidence,
            new_tracks=new_tracks,
            unmatched_tracks=unmatched_tracks,
            frame_purity=frame_purity,
            fragmentation_delta=fragmentation_delta,
            frame_assignment_conflict=frame_assignment_conflict,
            birth_count=birth_count,
            proto_count=proto_count,
        )

        row = {
            "frame_id": current_frame.frame_index,
            "scenario_name": "hard_drift_occlusion",
            "num_gt": len(current_frame.boxes),
            "num_prop": len(objectness_output.proposals),
            "recall_hit": recall_hit,
            "id_switch": id_switch,
            "occlusion_ratio": _occlusion_ratio(current_frame.boxes),
            "drift_strength": _background_drift_strength(prev_frame.frame, current_frame.frame, current_frame.masks),
            "false_hot_area": false_hot_area,
            "proto_count": proto_count,
            "new_tracks": new_tracks,
            "unmatched_tracks": unmatched_tracks,
            "fail_type_objectness": fail_flags["objectness"],
            "fail_type_tracking": fail_flags["tracking"],
            "fail_type_memory": fail_flags["memory"],
            "fail_type_mixed": fail_flags["mixed"],
            "frame_purity": frame_purity,
            "fragmentation_delta": fragmentation_delta,
            "proposal_confidence": proposal_confidence,
            "birth_count": birth_count,
            "baseline2_recall": len(baseline_matches) / len(current_frame.boxes) if current_frame.boxes else 1.0,
            "baseline2_advantage": (
                len(baseline_matches) / len(current_frame.boxes) if current_frame.boxes else 1.0
            )
            - recall_hit,
        }
        rows.append(row)
        frame_visuals.append(
            {
                "frame": current_frame.frame,
                "gt_boxes": current_frame.boxes,
                "prop_boxes": tracking_boxes,
                "heatmap": objectness_output.heatmap,
                **row,
            }
        )

    failure_rows = [row for row in rows if _is_failure_row(row)]
    summary = _summarize_failures(failure_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "failure_slices_v2.csv"
    json_path = output_dir / "failure_slices_v2.json"
    figure_path = output_dir / "failure_slices_v2.png"

    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    _save_failure_figure(frame_visuals, figure_path)

    print(f"saved_csv={csv_path}")
    print(f"saved_json={json_path}")
    print(f"saved_figure={figure_path}")
    print(
        "failure_ratios="
        f"objectness:{summary['ratio_objectness']:.4f},"
        f"tracking:{summary['ratio_tracking']:.4f},"
        f"memory:{summary['ratio_memory']:.4f},"
        f"mixed:{summary['ratio_mixed']:.4f}"
    )


def _classify_failures(
    *,
    num_gt: int,
    num_prop: int,
    recall_hit: float,
    id_switch: int,
    false_hot_area: float,
    proposal_confidence: float,
    new_tracks: int,
    unmatched_tracks: int,
    frame_purity: float,
    fragmentation_delta: int,
    frame_assignment_conflict: int,
    birth_count: int,
    proto_count: int,
) -> dict[str, int]:
    objectness = int(
        (num_gt > num_prop or recall_hit <= 0.0)
        and (false_hot_area >= 0.030 or proposal_confidence <= 0.58 or num_prop == 0)
    )
    tracking = int(
        recall_hit >= 0.66
        and (id_switch == 1 or unmatched_tracks >= 1 or new_tracks >= max(2, num_gt))
    )
    memory = int(
        recall_hit >= 0.66
        and id_switch == 0
        and unmatched_tracks == 0
        and (
            fragmentation_delta > 0
            or frame_assignment_conflict == 1
            or frame_purity < 0.75
            or (birth_count > 0 and proto_count > max(1, num_gt))
        )
    )

    is_failure = recall_hit < 1.0 or id_switch == 1 or fragmentation_delta > 0 or frame_assignment_conflict == 1
    if not is_failure:
        return {"objectness": 0, "tracking": 0, "memory": 0, "mixed": 0}

    score_sum = objectness + tracking + memory
    if score_sum != 1:
        return {"objectness": 0, "tracking": 0, "memory": 0, "mixed": 1}
    return {"objectness": objectness, "tracking": tracking, "memory": memory, "mixed": 0}


def _summarize_failures(rows: list[dict[str, object]]) -> dict[str, float | int]:
    counts = Counter()
    for row in rows:
        for key in ("objectness", "tracking", "memory", "mixed"):
            counts[key] += int(row[f"fail_type_{key}"])

    total = max(len(rows), 1)
    return {
        "total_failure_frames": len(rows),
        "count_objectness": counts["objectness"],
        "count_tracking": counts["tracking"],
        "count_memory": counts["memory"],
        "count_mixed": counts["mixed"],
        "ratio_objectness": counts["objectness"] / total,
        "ratio_tracking": counts["tracking"] / total,
        "ratio_memory": counts["memory"] / total,
        "ratio_mixed": counts["mixed"] / total,
    }


def _is_failure_row(row: dict[str, object]) -> bool:
    return (
        float(row["recall_hit"]) < 1.0
        or int(row["id_switch"]) == 1
        or int(row["fail_type_memory"]) == 1
        or int(row["fail_type_mixed"]) == 1
    )


def _occlusion_ratio(boxes: list[tuple[int, int, int, int]]) -> float:
    if len(boxes) < 2:
        return 0.0
    overlap = 0.0
    normalizer = 0.0
    for index, box_a in enumerate(boxes):
        ax1, ay1, ax2, ay2 = box_a
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        normalizer += area_a
        for box_b in boxes[index + 1 :]:
            bx1, by1, bx2, by2 = box_b
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            overlap += max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    return overlap / max(normalizer, 1.0)


def _background_drift_strength(
    prev_frame: np.ndarray,
    current_frame: np.ndarray,
    masks: list[np.ndarray],
) -> float:
    prev_gray = _to_grayscale(prev_frame)
    current_gray = _to_grayscale(current_frame)
    background_mask = np.ones_like(prev_gray, dtype=bool)
    for mask in masks:
        background_mask &= ~mask.astype(bool)
    if not np.any(background_mask):
        return 0.0
    return float(np.abs(current_gray[background_mask] - prev_gray[background_mask]).mean())


def _false_hot_area(binary_mask: np.ndarray, masks: list[np.ndarray]) -> float:
    gt_mask = np.zeros_like(binary_mask, dtype=bool)
    for mask in masks:
        gt_mask |= mask.astype(bool)
    false_hot = binary_mask.astype(bool) & ~gt_mask
    return float(false_hot.sum() / max(false_hot.size, 1))


def _save_failure_figure(frame_visuals: list[dict[str, object]], figure_path: Path) -> None:
    worst_frames = sorted(
        [row for row in frame_visuals if _is_failure_row(row)],
        key=lambda row: (
            float(row["recall_hit"]),
            -int(row["fail_type_mixed"]),
            -float(row["baseline2_advantage"]),
            -float(row["false_hot_area"]),
        ),
    )[:4]
    if not worst_frames:
        worst_frames = frame_visuals[:1]

    figure, axes = plt.subplots(len(worst_frames), 2, figsize=(10, 4 * len(worst_frames)))
    if len(worst_frames) == 1:
        axes = np.array([axes])

    for axis_row, row in zip(axes, worst_frames):
        frame_axis, heatmap_axis = axis_row
        frame_axis.imshow(row["frame"])
        frame_axis.set_title(
            f"frame {row['frame_id']} | recall {float(row['recall_hit']):.2f} | "
            f"obj/trk/mem/mix={int(row['fail_type_objectness'])}/"
            f"{int(row['fail_type_tracking'])}/"
            f"{int(row['fail_type_memory'])}/"
            f"{int(row['fail_type_mixed'])}"
        )
        frame_axis.axis("off")
        for box in row["gt_boxes"]:
            x1, y1, x2, y2 = box
            frame_axis.add_patch(
                Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.5, edgecolor="lime")
            )
        for box in row["prop_boxes"]:
            x1, y1, x2, y2 = box
            frame_axis.add_patch(
                Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.4, edgecolor="white")
            )

        heatmap_axis.imshow(row["heatmap"], cmap="inferno", vmin=0.0, vmax=1.0)
        heatmap_axis.set_title(
            f"drift {float(row['drift_strength']):.3f} | occ {float(row['occlusion_ratio']):.3f}\n"
            f"false_hot {float(row['false_hot_area']):.3f} | proto {int(row['proto_count'])}"
        )
        heatmap_axis.axis("off")

    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = [
        "frame_id",
        "scenario_name",
        "num_gt",
        "num_prop",
        "recall_hit",
        "id_switch",
        "occlusion_ratio",
        "drift_strength",
        "false_hot_area",
        "proto_count",
        "new_tracks",
        "unmatched_tracks",
        "fail_type_objectness",
        "fail_type_tracking",
        "fail_type_memory",
        "fail_type_mixed",
        "frame_purity",
        "fragmentation_delta",
        "proposal_confidence",
        "birth_count",
        "baseline2_recall",
        "baseline2_advantage",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32) / 255.0
    return 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]


def _load_config_payload(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


if __name__ == "__main__":
    main()
