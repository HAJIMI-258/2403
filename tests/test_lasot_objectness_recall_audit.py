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

from experiments.run_lasot_objectness_recall_audit import run_audit
from lasot_test_utils import create_pixel_mini_lasot


class LaSOTObjectnessRecallAuditTest(unittest.TestCase):
    def test_objectness_recall_audit_outputs_summary_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = create_pixel_mini_lasot(Path(tmp))
            summary = run_audit(root=root, output_dir=Path(tmp) / "recall", max_events=1, pre_context=5, post_context=3)
            self.assertIn("proposal_recall_iou_025", summary)
            self.assertIn("proposal_recall_iou_050", summary)
            self.assertIn("mean_best_iou", summary)


if __name__ == "__main__":
    unittest.main()
