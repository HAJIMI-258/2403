"""Audit synthetic long-gap re-entry event generation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.synth_stream import (  # noqa: E402
    AppearancePerturbationConfig,
    BackgroundDriftConfig,
    BridgeSyntheticConfig,
    SynthDatasetConfig,
    SyntheticStreamGenerator,
)


def run_audit(
    sequences: int = 5,
    max_frames: int = 100,
    min_gap: int = 8,
    force_reentry_events: bool = True,
    guaranteed_reentry_count: int = 2,
    reentry_visibility_mode: str = "hard_hide",
    output_dir: str | Path = "results/synthetic_reentry_protocol_audit",
    seed: int = 41,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config = _build_config(
        sequences=sequences,
        max_frames=max_frames,
        force_reentry_events=force_reentry_events,
        guaranteed_reentry_count=guaranteed_reentry_count,
        reentry_visibility_mode=reentry_visibility_mode,
    )
    generator = SyntheticStreamGenerator(config, seed=seed)
    planned_rows: list[dict[str, Any]] = []
    actual_rows: list[dict[str, Any]] = []
    visibility_rows: list[dict[str, Any]] = []
    valid_count = 0
    mismatch_total = 0
    per_sequence_counts: dict[str, int] = {}

    for sequence_id in range(sequences):
        sequence = generator.generate_sequence(sequence_id)
        planned = [dict(row, sequence_id=sequence_id) for row in sequence.metadata.get("planned_reentry_event_rows", [])]
        actual = [dict(row, sequence_id=sequence_id) for row in sequence.metadata.get("actual_reentry_events", [])]
        planned_rows.extend(planned)
        actual_rows.extend(actual)
        actual_long_gap = [row for row in actual if int(row.get("gap_length", 0)) >= min_gap]
        per_sequence_counts[str(sequence_id)] = len(actual_long_gap)
        if actual_long_gap and bool(sequence.metadata.get("benchmark_valid", False)):
            valid_count += 1
        mismatch_total += int(sequence.metadata.get("event_ledger_discovery_mismatch_count", 0))
        visibility_rows.extend(_visibility_timeline_rows(sequence_id, sequence.frames))

    gap_bucket_counts = Counter(_gap_bucket(int(row["gap_length"])) for row in actual_rows)
    actual_long_gap_count = sum(int(int(row.get("gap_length", 0)) >= min_gap) for row in actual_rows)
    summary = {
        "sequence_count": sequences,
        "planned_event_count": len(planned_rows),
        "actual_event_count": len(actual_rows),
        "actual_long_gap_event_count": int(actual_long_gap_count),
        "min_gap": int(min_gap),
        "benchmark_valid_rate": valid_count / sequences if sequences else 0.0,
        "event_ledger_discovery_mismatch_count": mismatch_total,
        "per_sequence_actual_event_counts": per_sequence_counts,
        "gap_bucket_counts": dict(gap_bucket_counts),
    }
    _write_csv(output_path / "planned_reentry_events.csv", planned_rows)
    _write_csv(output_path / "actual_reentry_events.csv", actual_rows)
    _write_csv(output_path / "visibility_timeline.csv", visibility_rows)
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_path / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _build_config(
    sequences: int,
    max_frames: int,
    force_reentry_events: bool,
    guaranteed_reentry_count: int,
    reentry_visibility_mode: str,
) -> SynthDatasetConfig:
    return SynthDatasetConfig(
        name="synthetic_reentry_protocol_audit",
        resolution=(128, 128),
        sequence_length=max_frames,
        num_sequences=sequences,
        num_objects_range=(max(2, guaranteed_reentry_count), max(2, guaranteed_reentry_count)),
        shapes=("circle", "square", "triangle"),
        object_scale_range=(9, 18),
        velocity_range=(1.0, 2.5),
        spawn_margin=18,
        occlusion_probability=0.30,
        reentry_probability=1.0,
        background_drift=BackgroundDriftConfig(True, 0.04, 0.01),
        appearance_perturbation=AppearancePerturbationConfig(True, 0.05, 0.05, 0.0),
        bridge_synthetic=BridgeSyntheticConfig(
            enabled=True,
            difficulty_preset="forced_reentry_protocol",
            background_repeat_density=0.10,
            background_texture_strength=0.05,
            illumination_drift_strength=0.05,
            local_noise_std=0.01,
            local_blur_probability=0.0,
            camera_jitter_std=0.10,
            occlusion_duration_range=(8, 12),
            reentry_gap_range=(8, 18),
            crossing_probability=0.10,
            target_deformation_strength=0.03,
            low_contrast_probability=0.0,
            force_reentry_events=force_reentry_events,
            guaranteed_reentry_count=guaranteed_reentry_count,
            reentry_visibility_mode=reentry_visibility_mode,
            min_pre_visible_frames=8,
            min_post_visible_frames=8,
            actual_reentry_gap_range=(8, 18),
        ),
        outputs=("frame", "boxes", "masks", "instance_id", "concept_id"),
    )


def _visibility_timeline_rows(sequence_id: int, frames: list[Any]) -> list[dict[str, Any]]:
    all_ids = sorted({int(instance_id) for frame in frames for instance_id in frame.instance_ids})
    rows: list[dict[str, Any]] = []
    for frame in frames:
        visible = {int(instance_id) for instance_id in frame.instance_ids}
        for instance_id in all_ids:
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "frame_idx": frame.frame_index,
                    "instance_id": instance_id,
                    "visible": int(instance_id in visible),
                    "lifecycle_events": json.dumps(frame.lifecycle_events, sort_keys=True),
                }
            )
    return rows


def _gap_bucket(gap_length: int) -> str:
    if gap_length <= 3:
        return "gap_1_3"
    if gap_length <= 7:
        return "gap_4_7"
    if gap_length <= 15:
        return "gap_8_15"
    return "gap_16_plus"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _report(summary: dict[str, Any]) -> str:
    return (
        "# Synthetic Re-entry Protocol Audit\n\n"
        f"- planned_event_count: {summary['planned_event_count']}\n"
        f"- actual_event_count: {summary['actual_event_count']}\n"
        f"- actual_long_gap_event_count: {summary['actual_long_gap_event_count']}\n"
        f"- benchmark_valid_rate: {summary['benchmark_valid_rate']:.4f}\n"
        f"- gap_bucket_counts: {summary['gap_bucket_counts']}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--force-reentry-events", type=int, default=1)
    parser.add_argument("--guaranteed-reentry-count", type=int, default=2)
    parser.add_argument("--reentry-visibility-mode", type=str, default="hard_hide")
    parser.add_argument("--output-dir", type=str, default="results/synthetic_reentry_protocol_audit")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    summary = run_audit(
        sequences=args.sequences,
        max_frames=args.max_frames,
        min_gap=args.min_gap,
        force_reentry_events=bool(args.force_reentry_events),
        guaranteed_reentry_count=args.guaranteed_reentry_count,
        reentry_visibility_mode=args.reentry_visibility_mode,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
