"""Generate a small preview for the synthetic streaming dataset."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a preview for the synthetic stream dataset.")
    parser.add_argument("--config", default="configs/synth.yaml", help="Path to the synthetic config file.")
    parser.add_argument("--sequence-id", type=int, default=0, help="Sequence index to generate.")
    parser.add_argument(
        "--output-dir",
        default="results/synth_preview",
        help="Directory where preview artifacts will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--num-panels", type=int, default=4, help="How many frames to visualize.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_synth_dataset_config(args.config)
    generator = SyntheticStreamGenerator(config, seed=args.seed)
    sequence = generator.generate_sequence(args.sequence_id)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_count = max(1, min(args.num_panels, len(sequence.frames)))
    indices = sorted({int(i) for i in np.linspace(0, len(sequence.frames) - 1, panel_count)})
    selected_frames = [sequence.frames[index] for index in indices]

    figure, axes = plt.subplots(1, len(selected_frames), figsize=(4 * len(selected_frames), 4))
    if len(selected_frames) == 1:
        axes = [axes]

    for axis, frame_sample in zip(axes, selected_frames):
        axis.imshow(frame_sample.frame)
        axis.set_title(f"frame {frame_sample.frame_index}")
        axis.axis("off")
        for box, instance_id, concept_id in zip(
            frame_sample.boxes,
            frame_sample.instance_ids,
            frame_sample.concept_ids,
        ):
            x1, y1, x2, y2 = box
            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    linewidth=1.5,
                    edgecolor="white",
                )
            )
            axis.text(x1, max(0, y1 - 4), f"id={instance_id} c={concept_id}", color="white", fontsize=8)

    figure.tight_layout()
    figure_path = output_dir / f"synth_preview_seq_{args.sequence_id:03d}.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    summary = {
        "sequence_id": sequence.sequence_id,
        "num_frames": len(sequence.frames),
        "resolution": config.resolution,
        "preview_frame_indices": indices,
        "first_frame_annotations": {
            "boxes": sequence.frames[0].boxes,
            "instance_ids": sequence.frames[0].instance_ids,
            "concept_ids": sequence.frames[0].concept_ids,
        },
    }
    summary_path = output_dir / f"synth_preview_seq_{args.sequence_id:03d}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved_figure={figure_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
