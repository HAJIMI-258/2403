from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nops_owr.cognition.object_file import ObjectFile, SupportMaskSummary
from nops_owr.encoder.spike_encoder import SpikeEncoding
from nops_owr.memory import EpisodicMemory
from nops_owr.objectness.field import ObjectnessOutput
from nops_owr.objectness.memory_guided_proposals import MemoryGuidedProposalAugmenter, MemoryGuidedProposalConfig


class MemoryGuidedProposalsTest(unittest.TestCase):
    def test_closed_episode_adds_memory_guided_proposals(self) -> None:
        encoding = _encoding()
        memory = EpisodicMemory(memory_budget=16)
        object_file = _object_file(frame_index=1)
        episode_id = memory.begin_episode(object_file, frame_index=1, track_id=7, metadata={"gt_instance_id": 1})
        memory.close_episode(episode_id, frame_index=2, close_reason="test")
        output = _objectness_output(encoding)
        augmenter = MemoryGuidedProposalAugmenter(
            MemoryGuidedProposalConfig(max_memory_episodes=1, windows_per_episode=4, max_added_proposals=4)
        )
        proposals = augmenter.augment(
            objectness_output=output,
            encoding=encoding,
            current_frame=None,
            episodic_memory=memory,
            frame_index=20,
        )
        memory_proposals = [proposal for proposal in proposals if proposal.source == "memory_guided_window"]
        self.assertGreater(len(memory_proposals), 0)
        self.assertIn("source_episode_id", memory_proposals[0].metadata)
        self.assertEqual(memory_proposals[0].metadata["source_episode_id"], episode_id)


def _encoding() -> SpikeEncoding:
    heat = np.zeros((32, 32), dtype=np.float32)
    heat[12:20, 12:20] = 1.0
    return SpikeEncoding(
        prev_gray=np.zeros((32, 32), dtype=np.float32),
        current_gray=heat,
        frame_diff=heat,
        edge_map=heat,
        on_spikes=heat,
        off_spikes=np.zeros((32, 32), dtype=np.float32),
        spike_response=heat,
    )


def _objectness_output(encoding: SpikeEncoding) -> ObjectnessOutput:
    return ObjectnessOutput(
        activation_map=encoding.spike_response,
        boundary_term=encoding.edge_map,
        persistence_term=encoding.spike_response,
        temporal_term=encoding.spike_response,
        surprise_term=encoding.spike_response,
        habituation_map=np.zeros((32, 32), dtype=np.float32),
        habituation_term=np.zeros((32, 32), dtype=np.float32),
        habituation_response=np.zeros((32, 32), dtype=np.float32),
        residual_term=np.zeros((32, 32), dtype=np.float32),
        raw_objectness=encoding.spike_response,
        normalized_objectness=encoding.spike_response,
        heatmap=encoding.spike_response,
        binary_mask=encoding.spike_response > 0.5,
        threshold_map=np.zeros((32, 32), dtype=np.float32),
        threshold=0.0,
        proposals=[],
    )


def _object_file(frame_index: int) -> ObjectFile:
    return ObjectFile(
        object_file_id=f"of:{frame_index}:0",
        frame_index=frame_index,
        proposal_index=0,
        box=(10, 10, 20, 20),
        raw_box=(10, 10, 20, 20),
        support_box=(10, 10, 20, 20),
        centroid=(15.0, 15.0),
        area=100.0,
        score=0.8,
        quality_score=0.8,
        support_mask_summary=SupportMaskSummary(area=100.0, bbox=(10, 10, 20, 20), fill_ratio=1.0, compactness=1.0, boundary_smoothness=1.0),
        appearance_signature=np.ones(15, dtype=np.float32),
        shape_signature=np.asarray([10 / 32, 10 / 32, 1.0, 100 / 1024, 1.0, 1.0, 1.0], dtype=np.float32),
        context_signature=np.asarray([15 / 32, 15 / 32, 1.0, 0.0, 0.2, 0.2], dtype=np.float32),
        motion_signature=np.zeros(0, dtype=np.float32),
        confidence=0.8,
    )


if __name__ == "__main__":
    unittest.main()
