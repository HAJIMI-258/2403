import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_spiking_morph_permanence_eval import _object_file, _object_spec, _render_observation
from nops_owr.descriptor.spiking_invariant_descriptor import SpikingInvariantDescriptorBuilder
from nops_owr.encoder.spike_encoder import MinimalSpikeEncoder


class SpikingInvariantDescriptorTest(unittest.TestCase):
    def test_descriptor_dimensions_density_and_hash_determinism(self) -> None:
        encoder = MinimalSpikeEncoder()
        builder = SpikingInvariantDescriptorBuilder(spike_dim=64, hash_bits=64, seed=3)
        spec = _object_spec(0, np.random.default_rng(7))
        prev, current, box = _render_observation(spec, scale=1.0, aspect=1.0, brightness=1.0, occlusion=0.0)
        encoding = encoder.encode(prev, current)
        object_file = _object_file(0, 1, box, encoding, source="test")

        left = builder.build(object_file, encoding)
        right = builder.build(object_file, encoding)

        self.assertEqual(left.spike_signature.shape, (64,))
        self.assertEqual(left.binary_hash.shape, (64,))
        self.assertLessEqual(left.spike_density, 0.20)
        self.assertGreater(left.spike_density, 0.0)
        np.testing.assert_array_equal(left.binary_hash, right.binary_hash)

    def test_moderate_scale_change_keeps_nonzero_similarity(self) -> None:
        encoder = MinimalSpikeEncoder()
        builder = SpikingInvariantDescriptorBuilder(spike_dim=64, hash_bits=64, seed=11)
        spec = _object_spec(1, np.random.default_rng(13))
        prev_a, current_a, box_a = _render_observation(spec, scale=1.0, aspect=1.0, brightness=1.0, occlusion=0.0)
        prev_b, current_b, box_b = _render_observation(spec, scale=1.25, aspect=1.1, brightness=1.05, occlusion=0.0)
        enc_a = encoder.encode(prev_a, current_a)
        enc_b = encoder.encode(prev_b, current_b)
        desc_a = builder.build(_object_file(1, 1, box_a, enc_a, source="a"), enc_a)
        desc_b = builder.build(_object_file(1, 2, box_b, enc_b, source="b"), enc_b)

        overlap = float(np.dot(desc_a.spike_signature, desc_b.spike_signature))
        denom = float(np.linalg.norm(desc_a.spike_signature) * np.linalg.norm(desc_b.spike_signature))
        similarity = 0.0 if denom <= 1e-6 else overlap / denom
        self.assertGreaterEqual(similarity, 0.10)


if __name__ == "__main__":
    unittest.main()
