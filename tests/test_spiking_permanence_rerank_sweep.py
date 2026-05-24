import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_spiking_permanence_rerank_sweep import run_sweep


class SpikingPermanenceRerankSweepTest(unittest.TestCase):
    def test_tiny_rerank_sweep_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rerank"
            summary = run_sweep(
                matches_csv=Path(tmp) / "missing_matches.csv",
                output_dir=out,
                rerun_eval=True,
                seed=9,
                object_count=3,
                events_per_object=2,
                max_capsules=8,
                spike_dim=64,
            )
            self.assertTrue((out / "rerank_profile_summary.csv").exists())
            self.assertTrue((out / "rerank_event_details.csv").exists())
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "report.md").exists())
            loaded = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            for key in ("profile_count", "best_profile", "current_profile", "main_diagnosis"):
                self.assertIn(key, summary)
                self.assertIn(key, loaded)
            self.assertGreater(summary["profile_count"], 0)


if __name__ == "__main__":
    unittest.main()
