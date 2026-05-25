from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from experiments.run_lasot_spiking_permanence_eval import run_eval
from lasot_test_utils import create_pixel_mini_lasot


class LaSOTSpikingPermanenceEvalTest(unittest.TestCase):
    def test_spiking_permanence_eval_runs_on_mini_lasot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = create_pixel_mini_lasot(Path(tmp))
            out = Path(tmp) / "spiking"
            summary = run_eval(root=root, output_dir=out, max_events=1, pre_context=5, post_context=3)
            self.assertEqual(summary["evaluated_event_count"], 1)
            self.assertIn("target_capsule_presence_rate", summary)
            self.assertIn("same_instance_recall_at_reentry", summary)
            self.assertIn("false_resurrection_rate_at_reentry", summary)
            self.assertTrue((out / "spiking_reentry_events.csv").exists())
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
