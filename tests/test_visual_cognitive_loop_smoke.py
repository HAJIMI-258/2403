from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.synth_stream import (
    AppearancePerturbationConfig,
    BackgroundDriftConfig,
    BridgeSyntheticConfig,
    SynthDatasetConfig,
    SyntheticStreamGenerator,
)
from nops_owr.attention import AttentionGate
from nops_owr.cognition import PredictiveRecognizer
from nops_owr.cognition.visual_cognitive_loop import VisualCognitiveLoop
from nops_owr.encoder.spike_encoder import MinimalSpikeEncoder
from nops_owr.memory import EpisodicMemory, MinimalPrototypeMemory
from nops_owr.objectness.field import MinimalObjectnessField
from nops_owr.tracking.temporal_identity import MinimalTemporalIdentityTracker


class VisualCognitiveLoopSmokeTest(unittest.TestCase):
    def test_visual_cognitive_loop_smoke_runs_on_synthetic_stream(self) -> None:
        config = SynthDatasetConfig(
            name="cognitive_loop_smoke",
            resolution=(96, 96),
            sequence_length=12,
            num_sequences=1,
            num_objects_range=(2, 3),
            shapes=("circle", "square", "triangle"),
            object_scale_range=(8, 14),
            velocity_range=(1.0, 2.0),
            spawn_margin=16,
            occlusion_probability=0.2,
            reentry_probability=0.2,
            background_drift=BackgroundDriftConfig(
                enabled=True,
                brightness_amplitude=0.05,
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
        sequence = SyntheticStreamGenerator(config, seed=7).generate_sequence(0)

        loop = VisualCognitiveLoop(
            encoder=MinimalSpikeEncoder(),
            objectness_field=MinimalObjectnessField(
                tau_obj=0.35,
                threshold_mode="fixed",
                min_area=12,
                max_proposals=4,
            ),
            tracker=MinimalTemporalIdentityTracker(),
            prototype_memory=MinimalPrototypeMemory(memory_budget=16),
            attention_gate=AttentionGate(max_attended_objects=2),
            episodic_memory=EpisodicMemory(memory_budget=32),
            recognizer=PredictiveRecognizer(),
        )

        results = []
        for frame_idx in range(1, 11):
            result = loop.step(
                sequence.frames[frame_idx - 1].frame,
                sequence.frames[frame_idx].frame,
                frame_idx,
            )
            results.append(result)
            self.assertEqual(result.frame_index, frame_idx)
            self.assertIsInstance(result.object_files, list)
            self.assertIsInstance(result.recognition_decisions, list)
            self.assertIsInstance(result.cognitive_events, list)
            self.assertLessEqual(len(result.attended_object_files), loop.attention_gate.max_attended_objects)
            self.assertIn("attended_object_ratio", result.metrics_snapshot)
            self.assertIn("memory_context_available", result.metrics_snapshot)
            if frame_idx >= 2:
                self.assertTrue(result.memory_context_used)
            self.assertLessEqual(len(loop.episodic_memory), loop.episodic_memory.memory_budget)

        self.assertEqual(len(results), 10)
        self.assertGreaterEqual(len(loop.episodic_memory), 0)

        # Old minimal modules should remain importable through the cognitive loop path.
        self.assertIsNotNone(MinimalSpikeEncoder)
        self.assertIsNotNone(MinimalObjectnessField)
        self.assertIsNotNone(MinimalTemporalIdentityTracker)
        self.assertIsNotNone(MinimalPrototypeMemory)


if __name__ == "__main__":
    unittest.main()
