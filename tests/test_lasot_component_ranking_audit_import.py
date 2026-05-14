from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from experiments.run_lasot_component_ranking_audit import run_audit
from lasot_test_utils import create_pixel_mini_lasot


class LasotComponentRankingAuditSmokeTest(unittest.TestCase):
    def test_mini_lasot_component_ranking_audit_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = create_pixel_mini_lasot(Path(tmp) / "lasot")
            out = Path(tmp) / "out"
            summary = run_audit(
                root=root,
                output_dir=out,
                max_events=1,
                min_gap=8,
                pre_context=8,
                post_context=4,
                objectness_profile="A8_quantile_q050_component_props48",
                max_image_side=64,
            )

            self.assertIn("dominant_failure_mode", summary)
            self.assertGreaterEqual(summary["evaluated_event_count"], 1)
            self.assertTrue((out / "component_ranking_frames.csv").exists())


if __name__ == "__main__":
    unittest.main()
