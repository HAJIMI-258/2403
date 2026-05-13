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

from datasets.external.lasot_adapter import LaSOTAdapter
from experiments.run_lasot_event_window_eval import run_eval
from lasot_test_utils import create_pixel_mini_lasot
from nops_owr.evaluation.external_event_windows import collect_lasot_reentry_events, make_event_window


class LaSOTEventWindowEvalTest(unittest.TestCase):
    def test_event_window_eval_runs_on_mini_lasot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = create_pixel_mini_lasot(Path(tmp))
            adapter = LaSOTAdapter(root)
            events = collect_lasot_reentry_events(adapter, min_gap=8, max_events=5)
            self.assertEqual(len(events), 1)
            frames = list(adapter.iter_frames("bicycle-1"))
            window = make_event_window(frames, events[0], pre_context=5, post_context=3)
            self.assertIsNotNone(window)
            self.assertLessEqual(window.window_start_frame, events[0].disappear_frame)
            self.assertGreaterEqual(window.window_end_frame, events[0].reappear_frame)

            summary = run_eval(root=root, output_dir=Path(tmp) / "eval", max_events=1, pre_context=5, post_context=3)
            self.assertEqual(summary["evaluated_event_count"], 1)
            self.assertIn("failure_buckets", summary)
            self.assertIn("proposal_recall_at_reentry", summary)

if __name__ == "__main__":
    unittest.main()
