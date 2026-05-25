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

from experiments.run_lasot_spiking_oracle_ablation import run_ablation
from lasot_test_utils import create_pixel_mini_lasot


class LaSOTSpikingOracleAblationTest(unittest.TestCase):
    def test_spiking_oracle_ablation_outputs_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = create_pixel_mini_lasot(Path(tmp))
            out = Path(tmp) / "spiking_oracle"
            result = run_ablation(root=root, output_dir=out, max_events=1, pre_context=5, post_context=3)
            comparison = result["comparison"]
            self.assertIn("normal_target_capsule_top5_rate", comparison)
            self.assertIn("oracle_history_and_reappear_target_capsule_top5_rate", comparison)
            self.assertIn("normal_same_instance_recall_at_reentry", comparison)
            self.assertTrue((out / "oracle_history_and_reappear_summary.json").exists())
            self.assertTrue((out / "comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
