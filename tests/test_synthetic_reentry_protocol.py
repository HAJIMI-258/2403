from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.synth_stream import (  # noqa: E402
    AppearancePerturbationConfig,
    BackgroundDriftConfig,
    BridgeSyntheticConfig,
    SynthDatasetConfig,
    SyntheticStreamGenerator,
)
from experiments.run_synthetic_reentry_protocol_audit import run_audit  # noqa: E402


class SyntheticReentryProtocolTest(unittest.TestCase):
    def test_forced_hard_hide_generates_actual_long_gap_reentry(self) -> None:
        config = SynthDatasetConfig(
            name="forced_reentry_test",
            resolution=(96, 96),
            sequence_length=60,
            num_sequences=1,
            num_objects_range=(2, 2),
            shapes=("circle", "square"),
            object_scale_range=(8, 14),
            velocity_range=(1.0, 2.0),
            spawn_margin=16,
            occlusion_probability=0.0,
            reentry_probability=1.0,
            background_drift=BackgroundDriftConfig(True, 0.03, 0.01),
            appearance_perturbation=AppearancePerturbationConfig(True, 0.03, 0.03, 0.0),
            bridge_synthetic=BridgeSyntheticConfig(
                enabled=True,
                force_reentry_events=True,
                guaranteed_reentry_count=1,
                reentry_visibility_mode="hard_hide",
                actual_reentry_gap_range=(8, 8),
                min_pre_visible_frames=6,
                min_post_visible_frames=6,
            ),
            outputs=("frame", "boxes", "masks", "instance_id", "concept_id"),
        )
        sequence = SyntheticStreamGenerator(config, seed=5).generate_sequence(0)
        self.assertGreaterEqual(int(sequence.metadata["actual_long_gap_reentry_count"]), 1)
        actual_events = sequence.metadata["actual_reentry_events"]
        event = next(row for row in actual_events if int(row["gap_length"]) >= 8)
        instance_id = int(event["instance_id"])
        disappear_frame = int(event["disappear_frame"])
        reappear_frame = int(event["reappear_frame"])
        for frame_idx in range(disappear_frame + 1, reappear_frame):
            self.assertNotIn(instance_id, sequence.frames[frame_idx].instance_ids)
        self.assertIn(instance_id, sequence.frames[reappear_frame].instance_ids)

    def test_protocol_audit_reports_valid_long_gap_events(self) -> None:
        output_dir = Path(tempfile.mkdtemp(prefix="synthetic_reentry_protocol_"))
        try:
            summary = run_audit(
                sequences=2,
                max_frames=60,
                min_gap=8,
                force_reentry_events=True,
                guaranteed_reentry_count=1,
                reentry_visibility_mode="hard_hide",
                output_dir=output_dir,
                seed=7,
            )
            self.assertGreater(summary["actual_long_gap_event_count"], 0)
            self.assertGreater(summary["benchmark_valid_rate"], 0.0)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
