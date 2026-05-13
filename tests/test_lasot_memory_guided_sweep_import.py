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

from experiments.run_lasot_memory_guided_proposal_sweep import run_sweep
from lasot_test_utils import create_pixel_mini_lasot


class LaSOTMemoryGuidedSweepImportTest(unittest.TestCase):
    def test_memory_guided_sweep_runs_on_mini_lasot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = create_pixel_mini_lasot(Path(tmp))
            summary = run_sweep(
                root=root,
                output_dir=Path(tmp) / "sweep",
                max_events=1,
                pre_context=5,
                post_context=3,
                frame_stride=1,
                profile_filter="M4_closed_episode_template_windows_k16",
            )
            self.assertEqual(summary["evaluated_event_count"], 1)
            self.assertEqual(summary["profile_count"], 1)
            self.assertIn("best_profile", summary)
            self.assertTrue((Path(tmp) / "sweep" / "profile_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
