import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_spiking_morph_permanence_eval import run_eval


class SpikingMorphPermanenceEvalTest(unittest.TestCase):
    def test_tiny_eval_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_eval(
                output_dir=tmp,
                seed=3,
                object_count=2,
                events_per_object=1,
                max_capsules=4,
                spike_dim=64,
                max_frames=40,
            )
            out = Path(tmp)
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "events.csv").exists())
            self.assertTrue((out / "report.md").exists())
            loaded = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            for key in (
                "same_instance_reentry_recall",
                "false_resurrection_rate",
                "bytes_per_capsule",
                "mean_spike_density",
            ):
                self.assertIn(key, summary)
                self.assertIn(key, loaded)


if __name__ == "__main__":
    unittest.main()
