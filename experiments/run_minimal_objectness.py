"""Run the minimal Day 3 objectness pipeline on a synthetic sequence."""

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
from metrics.metrics_core import u_recall
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.objectness import MinimalObjectnessField


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal spike encoder + objectness preview.")
    parser.add_argument("--config", default="configs/synth.yaml", help="Path to the config file.")
    parser.add_argument("--sequence-id", type=int, default=0, help="Sequence index to generate.")
    parser.add_argument("--output-dir", default="results/day3_objectness", help="Directory for output artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_payload = json.loads(json.dumps(_load_config_payload(args.config)))
    synth_config = load_synth_dataset_config(args.config)
    sequence = SyntheticStreamGenerator(synth_config, seed=args.seed).generate_sequence(args.sequence_id)

    encoder_cfg = config_payload["model"]["spike_encoder"]
    objectness_cfg = _get_field_config(config_payload)
    encoder = MinimalSpikeEncoder(**encoder_cfg)
    objectness = MinimalObjectnessField(**objectness_cfg)

    frame_summaries: list[dict[str, object]] = []
    best_record: dict[str, object] | None = None

    for frame_index in range(1, len(sequence.frames)):
        prev_frame = sequence.frames[frame_index - 1]
        current_frame = sequence.frames[frame_index]
        encoding = encoder.encode(prev_frame.frame, current_frame.frame)
        objectness_output = objectness.compute(encoding)

        predicted_boxes = [proposal.box for proposal in objectness_output.proposals]
        recall = u_recall(current_frame.boxes, predicted_boxes, iou_threshold=0.5)
        frame_summary = {
            "frame_index": current_frame.frame_index,
            "num_gt_boxes": len(current_frame.boxes),
            "num_proposals": len(predicted_boxes),
            "u_recall": recall,
            "mean_heatmap": float(objectness_output.heatmap.mean()),
            "max_heatmap": float(objectness_output.heatmap.max()),
            "normalized_unique_values": _rounded_unique_count(objectness_output.normalized_objectness),
            "normalized_histogram": _histogram(objectness_output.normalized_objectness),
            "binary_ratio": float(objectness_output.binary_mask.mean()),
            "proposal_boxes": predicted_boxes,
        }
        frame_summaries.append(frame_summary)

        rank_key = (
            recall,
            min(len(predicted_boxes), len(current_frame.boxes)),
            float(objectness_output.heatmap.max()),
            -abs(len(predicted_boxes) - len(current_frame.boxes)),
        )
        if best_record is None or rank_key > best_record["rank_key"]:
            best_record = {
                "rank_key": rank_key,
                "prev_frame": prev_frame.frame,
                "current_frame": current_frame.frame,
                "encoding": encoding,
                "objectness_output": objectness_output,
                "frame_summary": frame_summary,
                "gt_boxes": current_frame.boxes,
            }

    if best_record is None:
        raise RuntimeError("Sequence did not produce any valid frame pairs.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"objectness_seq_{args.sequence_id:03d}.png"
    summary_path = output_dir / f"objectness_seq_{args.sequence_id:03d}.json"

    _save_figure(
        best_record["prev_frame"],
        best_record["current_frame"],
        best_record["encoding"],
        best_record["objectness_output"],
        best_record["gt_boxes"],
        figure_path,
    )

    mean_recall = (
        float(sum(float(frame["u_recall"]) for frame in frame_summaries) / len(frame_summaries))
        if frame_summaries
        else 0.0
    )
    summary = {
        "sequence_id": args.sequence_id,
        "num_frames_evaluated": len(frame_summaries),
        "mean_u_recall": mean_recall,
        "best_frame": best_record["frame_summary"],
        "config": {
            "encoder": encoder_cfg,
            "objectness": objectness_cfg,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved_figure={figure_path}")
    print(f"saved_summary={summary_path}")
    print(f"mean_u_recall={mean_recall:.4f}")
    print(f"best_frame_u_recall={best_record['frame_summary']['u_recall']:.4f}")
    print(f"best_frame_index={best_record['frame_summary']['frame_index']}")


def _save_figure(
    prev_frame: np.ndarray,
    current_frame: np.ndarray,
    encoding,
    objectness_output,
    gt_boxes,
    figure_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 6, figsize=(24, 4.8))

    axes[0].imshow(current_frame)
    axes[0].set_title("Current Frame")
    axes[0].axis("off")
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        axes[0].add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.8, edgecolor="lime")
        )

    axes[1].imshow(encoding.spike_response, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("Spike Response")
    axes[1].axis("off")

    axes[2].imshow(objectness_output.raw_objectness, cmap="inferno")
    axes[2].set_title("Raw Objectness")
    axes[2].axis("off")

    axes[3].imshow(objectness_output.normalized_objectness, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[3].set_title(
        "Normalized Objectness\n"
        f"unique={_rounded_unique_count(objectness_output.normalized_objectness)}"
    )
    axes[3].axis("off")

    axes[4].imshow(objectness_output.binary_mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[4].set_title(f"Thresholded Mask\nobj>{objectness_output.threshold:.2f}")
    axes[4].axis("off")

    axes[5].imshow(current_frame)
    axes[5].set_title("Proposals Overlay")
    axes[5].axis("off")
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        axes[5].add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.4, edgecolor="lime")
        )
    for proposal in objectness_output.proposals:
        x1, y1, x2, y2 = proposal.box
        axes[5].add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1.6, edgecolor="white")
        )
        axes[5].text(
            x1,
            max(0, y1 - 4),
            f"{proposal.score:.2f}",
            color="white",
            fontsize=8,
        )

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


def _rounded_unique_count(array: np.ndarray) -> int:
    return int(np.unique(np.round(array.astype(np.float32), 3)).size)


def _histogram(array: np.ndarray, bins: int = 12) -> list[int]:
    counts, _ = np.histogram(array.astype(np.float32), bins=bins, range=(0.0, 1.0))
    return [int(count) for count in counts.tolist()]


if __name__ == "__main__":
    main()
