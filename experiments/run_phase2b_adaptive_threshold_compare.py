"""Compare fixed versus adaptive proposal thresholding on the hard scenario."""

from __future__ import annotations

import argparse
import csv
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
from experiments.phase2b_utils import get_field_config
from experiments.scenario_presets import build_hard_drift_occlusion_config
from metrics.metrics_core import u_recall
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.objectness import MinimalObjectnessField


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fixed vs adaptive thresholding on hard drift.")
    parser.add_argument("--config", default="results/phase2b_param_scan/best_config_v1.yaml", help="Path to config.")
    parser.add_argument("--output-dir", default="results/phase2b_adaptive_threshold", help="Directory for artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_config_payload(args.config)
    base_config = load_synth_dataset_config(args.config)
    hard_config = build_hard_drift_occlusion_config(base_config)
    sequence = SyntheticStreamGenerator(hard_config, seed=args.seed).generate_sequence(0)

    encoder = MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    fixed_cfg = get_field_config(payload)
    fixed_cfg["threshold_mode"] = "fixed"
    adaptive_cfg = dict(fixed_cfg)
    adaptive_cfg["threshold_mode"] = "quantile_local"

    fixed_model = MinimalObjectnessField(**fixed_cfg)
    adaptive_model = MinimalObjectnessField(**adaptive_cfg)

    rows: list[dict[str, object]] = []
    best_record: dict[str, object] | None = None

    for frame_offset in range(1, len(sequence.frames)):
        prev_frame = sequence.frames[frame_offset - 1]
        current_frame = sequence.frames[frame_offset]
        encoding = encoder.encode(prev_frame.frame, current_frame.frame)

        fixed_out = fixed_model.compute(encoding)
        adaptive_out = adaptive_model.compute(encoding)
        fixed_recall = u_recall(current_frame.boxes, [proposal.box for proposal in fixed_out.proposals], iou_threshold=0.5)
        adaptive_recall = u_recall(
            current_frame.boxes,
            [proposal.box for proposal in adaptive_out.proposals],
            iou_threshold=0.5,
        )
        fixed_false_hot = _false_hot_area(fixed_out.binary_mask, current_frame.masks)
        adaptive_false_hot = _false_hot_area(adaptive_out.binary_mask, current_frame.masks)

        row = {
            "frame_id": current_frame.frame_index,
            "fixed_num_prop": len(fixed_out.proposals),
            "adaptive_num_prop": len(adaptive_out.proposals),
            "fixed_u_recall": fixed_recall,
            "adaptive_u_recall": adaptive_recall,
            "fixed_false_hot_area": fixed_false_hot,
            "adaptive_false_hot_area": adaptive_false_hot,
        }
        rows.append(row)

        rank_key = (
            adaptive_recall - fixed_recall,
            adaptive_recall,
            fixed_recall,
            fixed_false_hot - adaptive_false_hot,
        )
        if best_record is None or rank_key > best_record["rank_key"]:
            best_record = {
                "rank_key": rank_key,
                "frame": current_frame.frame,
                "gt_boxes": current_frame.boxes,
                "fixed_out": fixed_out,
                "adaptive_out": adaptive_out,
                "row": row,
            }

    if best_record is None:
        raise RuntimeError("No valid frame was generated for adaptive threshold comparison.")

    summary = {
        "fixed_u_recall_mean": float(np.mean([row["fixed_u_recall"] for row in rows])),
        "adaptive_u_recall_mean": float(np.mean([row["adaptive_u_recall"] for row in rows])),
        "fixed_false_hot_area_mean": float(np.mean([row["fixed_false_hot_area"] for row in rows])),
        "adaptive_false_hot_area_mean": float(np.mean([row["adaptive_false_hot_area"] for row in rows])),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "adaptive_threshold_compare.csv"
    json_path = output_dir / "adaptive_threshold_compare.json"
    figure_path = output_dir / "adaptive_threshold_compare.png"

    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps({"summary": summary, "best_frame": best_record["row"], "rows": rows}, indent=2),
        encoding="utf-8",
    )
    _save_figure(best_record, figure_path)

    print(f"saved_csv={csv_path}")
    print(f"saved_json={json_path}")
    print(f"saved_figure={figure_path}")
    print(f"fixed_u_recall_mean={summary['fixed_u_recall_mean']:.4f}")
    print(f"adaptive_u_recall_mean={summary['adaptive_u_recall_mean']:.4f}")
    print(f"fixed_false_hot_area_mean={summary['fixed_false_hot_area_mean']:.4f}")
    print(f"adaptive_false_hot_area_mean={summary['adaptive_false_hot_area_mean']:.4f}")


def _save_figure(best_record: dict[str, object], figure_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    frame = best_record["frame"]
    gt_boxes = best_record["gt_boxes"]
    fixed_out = best_record["fixed_out"]
    adaptive_out = best_record["adaptive_out"]

    axes[0, 0].imshow(frame)
    axes[0, 0].set_title("Hard Frame")
    axes[0, 0].axis("off")
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        axes[0, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.5, edgecolor="lime"))

    axes[0, 1].imshow(fixed_out.normalized_objectness, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title(f"Fixed Heatmap\nmean thr={fixed_out.threshold:.2f}")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(frame)
    axes[0, 2].set_title("Fixed Proposals")
    axes[0, 2].axis("off")
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        axes[0, 2].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.5, edgecolor="lime"))
    for proposal in fixed_out.proposals:
        x1, y1, x2, y2 = proposal.box
        axes[0, 2].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.4, edgecolor="white"))

    axes[1, 0].imshow(fixed_out.binary_mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Fixed Binary")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(adaptive_out.normalized_objectness, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title(f"Adaptive Heatmap\nmean thr={adaptive_out.threshold:.2f}")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(frame)
    axes[1, 2].set_title("Adaptive Proposals")
    axes[1, 2].axis("off")
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        axes[1, 2].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.5, edgecolor="lime"))
    for proposal in adaptive_out.proposals:
        x1, y1, x2, y2 = proposal.box
        axes[1, 2].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.4, edgecolor="white"))

    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _false_hot_area(binary_mask: np.ndarray, masks: list[np.ndarray]) -> float:
    gt_mask = np.zeros_like(binary_mask, dtype=bool)
    for mask in masks:
        gt_mask |= mask.astype(bool)
    false_hot = binary_mask.astype(bool) & ~gt_mask
    return float(false_hot.sum() / max(false_hot.size, 1))


def _write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_config_payload(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


if __name__ == "__main__":
    main()
