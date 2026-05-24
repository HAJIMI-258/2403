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
from nops_owr.memory.spiking_object_memory import SpikingObjectMemoryBank


def _descriptor(object_id: int, frame_index: int, *, scale: float = 1.0, aspect: float = 1.0, seed: int = 5):
    encoder = MinimalSpikeEncoder()
    builder = SpikingInvariantDescriptorBuilder(spike_dim=64, hash_bits=64, seed=seed)
    spec = _object_spec(object_id, np.random.default_rng(100 + object_id))
    prev, current, box = _render_observation(spec, scale=scale, aspect=aspect, brightness=1.0, occlusion=0.0)
    encoding = encoder.encode(prev, current)
    return builder.build(_object_file(object_id, frame_index, box, encoding, source="test"), encoding)


class SpikingObjectMemoryTest(unittest.TestCase):
    def test_repeated_observations_update_same_capsule(self) -> None:
        bank = SpikingObjectMemoryBank(max_capsules=8, spike_dim=64, min_match_score=0.50)
        first = _descriptor(0, 1)
        capsule_id = bank.create_capsule(first, frame_index=1)

        second = _descriptor(0, 2, scale=1.15, aspect=1.1)
        matches = bank.match(second, frame_index=2)
        self.assertEqual(matches[0].capsule_id, capsule_id)
        bank.write_or_update(second, frame_index=2, confirmed_capsule_id=capsule_id)
        self.assertEqual(len(bank), 1)
        self.assertGreater(bank.capsules[capsule_id].observation_count, 1)

    def test_budget_and_memory_bytes_are_bounded(self) -> None:
        bank = SpikingObjectMemoryBank(max_capsules=2, spike_dim=64)
        for object_id in range(5):
            bank.create_capsule(_descriptor(object_id, object_id + 1), frame_index=object_id + 1)
        self.assertLessEqual(len(bank), 2)
        self.assertGreater(bank.memory_bytes(), 0)
        self.assertGreaterEqual(bank.mean_spike_density(), 0.0)
        self.assertLessEqual(bank.mean_spike_density(), 0.20)

    def test_distinct_object_has_lower_match_than_self(self) -> None:
        bank = SpikingObjectMemoryBank(max_capsules=8, spike_dim=64)
        capsule_id = bank.create_capsule(_descriptor(0, 1), frame_index=1)
        same = bank.match(_descriptor(0, 2, scale=1.1), frame_index=2)[0]
        other = bank.match(_descriptor(3, 3), frame_index=3)[0]
        self.assertEqual(same.capsule_id, capsule_id)
        self.assertGreaterEqual(same.score, other.score)


if __name__ == "__main__":
    unittest.main()
