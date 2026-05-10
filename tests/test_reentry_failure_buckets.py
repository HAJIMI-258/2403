from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_cognitive_reentry_eval import REENTRY_FIELDNAMES, run_eval  # noqa: E402


class ReentryFailureBucketsTest(unittest.TestCase):
    def test_summary_and_csv_include_failure_bucket_diagnostics(self) -> None:
        output_dir = Path(tempfile.mkdtemp(prefix="reentry_failure_buckets_"))
        try:
            summary = run_eval(sequences=1, max_frames=35, output_dir=output_dir, min_gap=8, seed=17)
            self.assertIn("failure_buckets", summary)
            self.assertIn("target_episode_presence_rate", summary)
            self.assertIn("target_episode_top5_rate", summary)
            self.assertIn("same_instance_precision_at_reentry", summary)
            self.assertIn("false_resurrection_rate_at_reentry", summary)

            with (output_dir / "reentry_events.csv").open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertIsNotNone(reader.fieldnames)
                for field in ("failure_bucket", "target_episode_rank", "top1_margin", "rejection_reason"):
                    self.assertIn(field, reader.fieldnames)
                for field in REENTRY_FIELDNAMES:
                    self.assertIn(field, reader.fieldnames)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
