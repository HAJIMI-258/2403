import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_spiking_permanence_calibration_sweep import run_sweep


class SpikingPermanenceCalibrationSweepTest(unittest.TestCase):
    def test_tiny_sweep_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_sweep(
                output_dir=tmp,
                seed=5,
                object_count=2,
                events_per_object=1,
                max_capsules=4,
                spike_dim=64,
            )
            out = Path(tmp)
            self.assertTrue((out / "sweep_summary.csv").exists())
            self.assertTrue((out / "sweep_summary.json").exists())
            self.assertTrue((out / "sweep_report.md").exists())
            loaded = json.loads((out / "sweep_summary.json").read_text(encoding="utf-8"))
            self.assertIn("best_safe_config", summary)
            self.assertIn("best_safe_config", loaded)
            self.assertGreater(loaded["config_count"], 0)
            csv_header = (out / "sweep_summary.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("match_profile", csv_header)


if __name__ == "__main__":
    unittest.main()
