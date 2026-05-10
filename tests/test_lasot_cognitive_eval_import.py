from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class LaSOTCognitiveEvalImportTest(unittest.TestCase):
    def test_script_imports(self) -> None:
        import experiments.run_lasot_cognitive_reentry_eval as module

        self.assertTrue(hasattr(module, "run_eval"))


if __name__ == "__main__":
    unittest.main()
