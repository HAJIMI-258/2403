import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_spiking_permanence_evidence_audit import run_audit


class SpikingPermanenceEvidenceAuditTest(unittest.TestCase):
    def test_tiny_audit_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit"
            summary = run_audit(
                events_csv=Path(tmp) / "missing_events.csv",
                output_dir=out,
                rerun_eval=True,
                seed=5,
                object_count=3,
                events_per_object=2,
                max_capsules=8,
                spike_dim=64,
            )
            self.assertTrue((out / "evidence_audit.csv").exists())
            self.assertTrue((out / "evidence_distribution_summary.csv").exists())
            self.assertTrue((out / "predicate_candidate_summary.csv").exists())
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "report.md").exists())
            loaded = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            for key in (
                "reentry_event_count",
                "top1_true_rate",
                "true_capsule_top5_rate",
                "best_safe_predicate",
                "evidence_separability",
                "main_diagnosis",
            ):
                self.assertIn(key, summary)
                self.assertIn(key, loaded)
            self.assertGreater(summary["predicate_candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
