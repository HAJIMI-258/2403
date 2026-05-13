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

from experiments.run_lasot_objectness_profile_sweep import run_sweep
from lasot_test_utils import create_pixel_mini_lasot


class LaSOTObjectnessProfileSweepTest(unittest.TestCase):
    def test_profile_sweep_outputs_best_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = create_pixel_mini_lasot(Path(tmp))
            summary = run_sweep(root=root, output_dir=Path(tmp) / "sweep", max_events=1, pre_context=5, post_context=3)
            self.assertEqual(summary["evaluated_event_count"], 1)
            self.assertIn("best_profile", summary)
            self.assertTrue((Path(tmp) / "sweep" / "profile_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
