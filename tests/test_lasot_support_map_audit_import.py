from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from experiments.run_lasot_support_map_audit import run_audit
from lasot_test_utils import create_pixel_mini_lasot


class LasotSupportMapAuditSmokeTest(unittest.TestCase):
    def test_support_map_audit_runs_on_mini_lasot(self) -> None:
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

            self.assertIn("dominant_support_failure", summary)
            self.assertIn("support_failure_counts", summary)
            self.assertTrue((out / "support_map_frames.csv").exists())


if __name__ == "__main__":
    unittest.main()
