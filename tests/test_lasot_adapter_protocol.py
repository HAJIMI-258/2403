from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.external.lasot_adapter import LaSOTAdapter


class LaSOTAdapterProtocolTest(unittest.TestCase):
    def test_mini_lasot_sequence_derives_long_gap_reentry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _create_mini_lasot(Path(tmp), gap=8)
            adapter = LaSOTAdapter(root)
            sequences = list(adapter.iter_sequences())
            self.assertIn("bicycle-1", sequences)

            frames = list(adapter.iter_frames("bicycle-1"))
            self.assertEqual(len(frames), 12)
            self.assertTrue(frames[0].boxes)
            self.assertFalse(frames[3].boxes)
            self.assertTrue(frames[11].boxes)

            events = adapter.derive_events("bicycle-1")
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].disappear_frame, 2)
            self.assertEqual(events[0].reappear_frame, 11)
            self.assertEqual(events[0].gap_length, 8)


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
