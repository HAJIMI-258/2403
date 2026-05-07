"""Compare hard_drift_occlusion before and after habituation/background suppression."""

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
from experiments.scenario_presets import build_hard_drift_occlusion_config
from metrics.metrics_core import u_recall
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.objectness import MinimalObjectnessField


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hard drift before/after objectness comparison.")
    parser.add_argument("--config", default="configs/synth.yaml", help="Path to the config file.")
    parser.add_argument("--output-dir", default="results/phase2_objectness", help="Directory for artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_config_payload(args.config)
    base_config = load_synth_dataset_config(args.config)
    hard_config = build_hard_drift_occlusion_config(base_config)
    sequence = SyntheticStreamGenerator(hard_config, seed=args.seed).generate_sequence(0)

    encoder = MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    before_cfg = dict(_get_field_config(payload))
    before_cfg["wr"] = 0.0
    before_cfg["hab_rho"] = 0.0
    before_cfg["hab_lambda"] = 0.0
    before_model = MinimalObjectnessField(**before_cfg)
    after_model = MinimalObjectnessField(**_get_field_config(payload))

    rows: list[dict[str, object]] = []
    for frame_offset in range(1, len(sequence.frames)):
        prev_frame = sequence.frames[frame_offset - 1]
        current_frame = sequence.frames[frame_offset]
        encoding = encoder.encode(prev_frame.frame, current_frame.frame)

        before_out = before_model.compute(encoding)
        after_out = after_model.compute(encoding)

        before_recall = u_recall(current_frame.boxes, [proposal.box for proposal in before_out.proposals], iou_threshold=0.5)
        after_recall = u_recall(current_frame.boxes, [proposal.box for proposal in after_out.proposals], iou_threshold=0.5)

        rows.append(
            {
                "frame_id": current_frame.frame_index,
                "num_gt": len(current_frame.boxes),
                "before_num_prop": len(before_out.proposals),
                "after_num_prop": len(after_out.proposals),
                "before_u_recall": before_recall,
                "after_u_recall": after_recall,
                "before_false_hot_area": _false_hot_area(before_out.binary_mask, current_frame.masks),
                "after_false_hot_area": _false_hot_area(after_out.binary_mask, current_frame.masks),
                "drift_strength": _background_drift_strength(prev_frame.frame, current_frame.frame, current_frame.masks),
            }
        )

    summary = {
        "before_u_recall_mean": float(np.mean([row["before_u_recall"] for row in rows])),
        "after_u_recall_mean": float(np.mean([row["after_u_recall"] for row in rows])),
        "before_false_hot_area_mean": float(np.mean([row["before_false_hot_area"] for row in rows])),
        "after_false_hot_area_mean": float(np.mean([row["after_false_hot_area"] for row in rows])),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "hard_drift_occlusion_before_after.csv"
    json_path = output_dir / "hard_drift_occlusion_before_after.json"
    figure_path = output_dir / "hard_drift_occlusion_before_after.png"

    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    _save_figure(rows, figure_path)

    print(f"saved_csv={csv_path}")
    print(f"saved_json={json_path}")
    print(f"saved_figure={figure_path}")
    print(f"before_u_recall_mean={summary['before_u_recall_mean']:.4f}")
    print(f"after_u_recall_mean={summary['after_u_recall_mean']:.4f}")
    print(f"before_false_hot_area_mean={summary['before_false_hot_area_mean']:.4f}")
    print(f"after_false_hot_area_mean={summary['after_false_hot_area_mean']:.4f}")


def _false_hot_area(binary_mask: np.ndarray, masks: list[np.ndarray]) -> float:
    gt_mask = np.zeros_like(binary_mask, dtype=bool)
    for mask in masks:
        gt_mask |= mask.astype(bool)
    false_hot = binary_mask.astype(bool) & ~gt_mask
    return float(false_hot.sum() / max(false_hot.size, 1))


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


def _save_figure(rows: list[dict[str, object]], figure_path: Path) -> None:
    frame_ids = [int(row["frame_id"]) for row in rows]
    before_recall = [float(row["before_u_recall"]) for row in rows]
    after_recall = [float(row["after_u_recall"]) for row in rows]
    before_false_hot = [float(row["before_false_hot_area"]) for row in rows]
    after_false_hot = [float(row["after_false_hot_area"]) for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(frame_ids, before_recall, label="before", linewidth=1.5)
    axes[0].plot(frame_ids, after_recall, label="after", linewidth=1.5)
    axes[0].set_title("Hard Drift U-Recall")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("U-Recall")
    axes[0].legend(frameon=False)

    axes[1].plot(frame_ids, before_false_hot, label="before", linewidth=1.5)
    axes[1].plot(frame_ids, after_false_hot, label="after", linewidth=1.5)
    axes[1].set_title("False Hot Area")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("ratio")
    axes[1].legend(frameon=False)

    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
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


def _get_field_config(config_payload: dict) -> dict:
    if "field" in config_payload:
        return config_payload["field"]
    return config_payload["model"]["objectness"]


if __name__ == "__main__":
    main()
