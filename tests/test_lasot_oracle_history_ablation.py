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

from experiments.run_lasot_oracle_history_ablation import run_ablation
from lasot_test_utils import create_pixel_mini_lasot


class LaSOTOracleHistoryAblationTest(unittest.TestCase):
    def test_oracle_history_ablation_outputs_three_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = create_pixel_mini_lasot(Path(tmp))
            result = run_ablation(root=root, output_dir=Path(tmp) / "oracle", max_events=1, pre_context=5, post_context=3)
            summaries = result["summaries"]
            self.assertIn("normal", summaries)
            self.assertIn("oracle_reappear_only", summaries)
            self.assertIn("oracle_history_and_reappear", summaries)
            comparison = result["comparison"]
            self.assertIn("normal_target_episode_top5_rate", comparison)
            self.assertIn("oracle_history_and_reappear_target_episode_top5_rate", comparison)


if __name__ == "__main__":
    unittest.main()
