import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_spiking_permanence_consolidation_sweep import run_sweep


class SpikingPermanenceConsolidationSweepTest(unittest.TestCase):
    def test_tiny_consolidation_sweep_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_sweep(
                output_dir=tmp,
                seed=11,
                object_count=3,
                events_per_object=1,
                max_capsules=8,
                spike_dim=64,
            )
            out = Path(tmp)
            self.assertTrue((out / "consolidation_summary.csv").exists())
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "report.md").exists())
            loaded = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            for key in ("best_safe_config", "baseline_context_1", "main_diagnosis"):
                self.assertIn(key, summary)
                self.assertIn(key, loaded)
            self.assertGreater(loaded["config_count"], 0)


if __name__ == "__main__":
    unittest.main()
