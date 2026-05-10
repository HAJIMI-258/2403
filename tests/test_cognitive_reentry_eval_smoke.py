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
from experiments.run_cognitive_reentry_eval import run_eval  # noqa: E402
from nops_owr.attention import AttentionGate  # noqa: E402
from nops_owr.cognition import PredictiveRecognizer  # noqa: E402
from nops_owr.cognition.visual_cognitive_loop import VisualCognitiveLoop  # noqa: E402
from nops_owr.encoder.spike_encoder import MinimalSpikeEncoder  # noqa: E402
from nops_owr.memory import EpisodicMemory, MinimalPrototypeMemory  # noqa: E402
from nops_owr.objectness.field import MinimalObjectnessField  # noqa: E402
from nops_owr.tracking.temporal_identity import MinimalTemporalIdentityTracker  # noqa: E402


class CognitiveReentryEvalSmokeTest(unittest.TestCase):
    def test_event_aware_loop_and_reentry_eval_summary(self) -> None:
        config = SynthDatasetConfig(
            name="cognitive_reentry_eval_smoke",
            resolution=(96, 96),
            sequence_length=24,
            num_sequences=1,
            num_objects_range=(2, 3),
            shapes=("circle", "square", "triangle"),
            object_scale_range=(8, 14),
            velocity_range=(1.0, 2.0),
            spawn_margin=16,
            occlusion_probability=0.35,
            reentry_probability=0.55,
            background_drift=BackgroundDriftConfig(
                enabled=True,
                brightness_amplitude=0.04,
                texture_noise_std=0.01,
            ),
            appearance_perturbation=AppearancePerturbationConfig(
                enabled=True,
                scale_jitter=0.05,
                intensity_jitter=0.05,
                edge_blur_probability=0.0,
            ),
            bridge_synthetic=BridgeSyntheticConfig(enabled=False),
            outputs=("frame", "boxes", "masks", "instance_id", "concept_id"),
        )
        sequence = SyntheticStreamGenerator(config, seed=11).generate_sequence(0)
        loop = VisualCognitiveLoop(
            encoder=MinimalSpikeEncoder(),
            objectness_field=MinimalObjectnessField(tau_obj=0.35, threshold_mode="fixed", min_area=12),
            tracker=MinimalTemporalIdentityTracker(),
            prototype_memory=MinimalPrototypeMemory(memory_budget=16),
            attention_gate=AttentionGate(max_attended_objects=3),
            episodic_memory=EpisodicMemory(memory_budget=32),
            recognizer=PredictiveRecognizer(),
        )

        saw_memory_context = False
        for frame_idx in range(1, 21):
            sample = sequence.frames[frame_idx]
            result = loop.step(
                sequence.frames[frame_idx - 1].frame,
                sample.frame,
                frame_idx,
                ground_truth={
                    "boxes": sample.boxes,
                    "instance_ids": sample.instance_ids,
                    "concept_ids": sample.concept_ids,
                },
            )
            saw_memory_context = saw_memory_context or result.memory_context_used
            self.assertIsInstance(result.cognitive_events, list)
            self.assertIsInstance(loop.active_episode_by_track, dict)

        self.assertIsNotNone(loop._prev_memory_output)
        self.assertTrue(saw_memory_context)

        episode_id = loop.episodic_memory.write_or_extend_episode(
            result.object_files[0],
            frame_index=21,
        ) if result.object_files else None
        if episode_id is not None:
            loop.episodic_memory.close_episode(episode_id, frame_index=22, close_reason="smoke")
            self.assertTrue(loop.episodic_memory.get_episode(episode_id).closed)

        output_dir = Path(tempfile.mkdtemp(prefix="cognitive_reentry_smoke_"))
        try:
            summary = run_eval(sequences=1, max_frames=30, output_dir=output_dir, seed=13)
            self.assertIn("reentry_event_count", summary)
            self.assertIn("false_resurrection_rate", summary)
            self.assertIn("false_resurrection_rate_at_reentry", summary)
            self.assertIn("long_gap_reentry_success_rate", summary)
            self.assertIn("memory_compression_ratio", summary)
            self.assertIn("failure_buckets", summary)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
