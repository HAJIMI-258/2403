from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_lasot_protocol_audit import run_audit


class LaSOTProtocolAuditSmokeTest(unittest.TestCase):
    def test_protocol_audit_finds_long_gap_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_mini_lasot(Path(tmp), gap=8)
            output_dir = Path(tmp) / "results"
            summary = run_audit(root=root, output_dir=output_dir, max_sequences=5, min_gap=8)
            self.assertGreaterEqual(summary["total_long_gap_reentry_event_count"], 1)
            self.assertTrue(summary["benchmark_valid"])
            self.assertTrue((output_dir / "lasot_sequence_inventory.csv").exists())
            self.assertTrue((output_dir / "lasot_reentry_events.csv").exists())


def _create_mini_lasot(tmp: Path, gap: int = 8) -> Path:
    seq_dir = tmp / "bicycle" / "bicycle-1"
    img_dir = seq_dir / "img"
    img_dir.mkdir(parents=True)
    frame_count = gap + 4
    for index in range(frame_count):
        (img_dir / f"{index + 1:08d}.jpg").write_bytes(b"placeholder")
    (seq_dir / "groundtruth.txt").write_text(
        "\n".join("1,1,10,10" for _ in range(frame_count)),
        encoding="utf-8",
    )
    full_occ = [0, 0, 0] + [1] * gap + [0]
    out_view = [0] * frame_count
    (seq_dir / "full_occlusion.txt").write_text(",".join(str(v) for v in full_occ), encoding="utf-8")
    (seq_dir / "out_of_view.txt").write_text(",".join(str(v) for v in out_view), encoding="utf-8")
    return tmp


if __name__ == "__main__":
    unittest.main()
