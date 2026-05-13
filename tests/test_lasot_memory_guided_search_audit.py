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

from experiments.run_lasot_memory_guided_search_audit import run_audit
from lasot_test_utils import create_pixel_mini_lasot


class LaSOTMemoryGuidedSearchAuditTest(unittest.TestCase):
    def test_memory_guided_search_audit_runs_on_mini_lasot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = create_pixel_mini_lasot(Path(tmp))
            summary = run_audit(
                root=root,
                output_dir=Path(tmp) / "memsearch",
                max_events=1,
                pre_context=5,
                post_context=3,
                frame_stride=1,
                max_search_candidates=8,
            )
            self.assertEqual(summary["evaluated_event_count"], 1)
            self.assertIn("normal_proposal_recall", summary)
            self.assertIn("memory_search_recall", summary)
            self.assertIn("failure_buckets", summary)
            self.assertTrue((Path(tmp) / "memsearch" / "memory_guided_search_events.csv").exists())


if __name__ == "__main__":
    unittest.main()
