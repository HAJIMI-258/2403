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

from experiments.run_lasot_component_capacity_audit import run_capacity_audit
from lasot_test_utils import create_pixel_mini_lasot


class LasotComponentCapacityAuditSmokeTest(unittest.TestCase):
    def test_capacity_audit_runs_on_mini_lasot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = create_pixel_mini_lasot(Path(tmp) / "lasot")
            out = Path(tmp) / "out"
            summary = run_capacity_audit(
                root=root,
                output_dir=out,
                max_events=1,
                min_gap=8,
                pre_context=8,
                post_context=4,
                profiles="A8_quantile_q050_component_props48,A10_quantile_q050_component_props96",
                max_image_side=64,
            )

            self.assertIn("capacity_improves_recall", summary)
            self.assertTrue((out / "capacity_summary.csv").exists())
            self.assertEqual(len(summary["profiles"]), 2)


if __name__ == "__main__":
    unittest.main()
