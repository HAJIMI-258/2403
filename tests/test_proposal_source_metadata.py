from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nops_owr.cognition.object_file import ObjectFileBuilder
from nops_owr.encoder.spike_encoder import SpikeEncoding
from nops_owr.objectness.field import ObjectnessOutput, Proposal


class ProposalSourceMetadataTest(unittest.TestCase):
    def test_proposal_source_reaches_object_file_metadata(self) -> None:
        proposal = Proposal(
            box=(2, 2, 10, 10),
            raw_box=(2, 2, 10, 10),
            support_box=(2, 2, 10, 10),
            area=16,
            raw_area=16,
            score=0.5,
            quality_score=0.6,
            centroid=(6.0, 6.0),
            support_mask=np.ones((8, 8), dtype=bool),
            fill_ratio=0.25,
            compactness=0.5,
            boundary_smoothness=1.0,
            near_boundary=0,
            source="memory_guided_window",
            source_score=0.7,
            metadata={"source_episode_id": 3},
        )
        encoding = SpikeEncoding(
            prev_gray=np.zeros((16, 16), dtype=np.float32),
            current_gray=np.ones((16, 16), dtype=np.float32),
            frame_diff=np.ones((16, 16), dtype=np.float32),
            edge_map=np.ones((16, 16), dtype=np.float32),
            on_spikes=np.ones((16, 16), dtype=np.float32),
            off_spikes=np.zeros((16, 16), dtype=np.float32),
            spike_response=np.ones((16, 16), dtype=np.float32),
        )
        output = ObjectnessOutput(
            activation_map=encoding.spike_response,
            boundary_term=encoding.edge_map,
            persistence_term=encoding.spike_response,
            temporal_term=encoding.spike_response,
            surprise_term=encoding.spike_response,
            habituation_map=np.zeros((16, 16), dtype=np.float32),
            habituation_term=np.zeros((16, 16), dtype=np.float32),
            habituation_response=np.zeros((16, 16), dtype=np.float32),
            residual_term=np.zeros((16, 16), dtype=np.float32),
            raw_objectness=encoding.spike_response,
            normalized_objectness=encoding.spike_response,
            heatmap=encoding.spike_response,
            binary_mask=np.ones((16, 16), dtype=bool),
            threshold_map=np.zeros((16, 16), dtype=np.float32),
            threshold=0.0,
            proposals=[proposal],
        )
        object_file = ObjectFileBuilder().build(output, encoding, 1)[0]
        self.assertEqual(object_file.proposal_source, "memory_guided_window")
        self.assertEqual(object_file.metadata["proposal_source"], "memory_guided_window")
        self.assertEqual(object_file.metadata["source_episode_id"], 3)


if __name__ == "__main__":
    unittest.main()
