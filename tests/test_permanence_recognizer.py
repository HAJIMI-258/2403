import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_spiking_morph_permanence_eval import _object_file, _object_spec, _render_observation
from nops_owr.cognition.permanence_recognizer import PermanenceRecognizer
from nops_owr.encoder.spike_encoder import MinimalSpikeEncoder
from nops_owr.memory.spiking_object_memory import SpikingMemoryMatch


def _dummy_object_file():
    spec = _object_spec(0, np.random.default_rng(17))
    prev, current, box = _render_observation(spec, scale=1.0, aspect=1.0, brightness=1.0, occlusion=0.0)
    encoding = MinimalSpikeEncoder().encode(prev, current)
    return _object_file(0, 1, box, encoding, source="test")


def _match(capsule_id: int, score: float, *, deformation: float = 0.8, spike: float = 0.8, conflict: float = 0.0) -> SpikingMemoryMatch:
    return SpikingMemoryMatch(
        capsule_id=capsule_id,
        score=score,
        rank=1,
        identity_score=score,
        deformation_score=deformation,
        spike_score=spike,
        hash_score=0.8,
        novelty_score=max(0.0, 1.0 - score),
        conflict_score=conflict,
        decision_hint="test",
        metadata={},
    )


class PermanenceRecognizerTest(unittest.TestCase):
    def test_no_matches_creates_new_object_decision(self) -> None:
        decision = PermanenceRecognizer().decide(_dummy_object_file(), [])
        self.assertEqual(decision.decision_type, "new_object")
        self.assertIsNone(decision.capsule_id)

    def test_high_score_and_margin_is_same_object(self) -> None:
        recognizer = PermanenceRecognizer()
        decision = recognizer.decide(_dummy_object_file(), [_match(1, 0.92), _match(2, 0.55)])
        self.assertEqual(decision.decision_type, "same_object")
        self.assertEqual(decision.capsule_id, 1)

    def test_high_spike_low_deformation_is_familiar_deformed(self) -> None:
        recognizer = PermanenceRecognizer(same_object_threshold=0.90, uncertain_threshold=0.55)
        decision = recognizer.decide(_dummy_object_file(), [_match(1, 0.66, deformation=0.25, spike=0.82), _match(2, 0.40)])
        self.assertEqual(decision.decision_type, "familiar_but_deformed")

    def test_ambiguous_top_matches_are_not_same_object(self) -> None:
        recognizer = PermanenceRecognizer()
        decision = recognizer.decide(_dummy_object_file(), [_match(1, 0.78, conflict=0.6), _match(2, 0.76)])
        self.assertIn(decision.decision_type, {"false_resurrection_risk", "uncertain_hold"})


if __name__ == "__main__":
    unittest.main()
