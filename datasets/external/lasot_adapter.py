from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .base_video_memory_dataset import ExternalEvent, FrameSampleExternal


class LaSOTAdapter:
    dataset_name = "lasot"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def iter_sequences(self) -> Iterable[str]:
        if not self.root.exists():
            return iter(())
        seqs = [p.name for p in self.root.iterdir() if p.is_dir()]
        return iter(sorted(seqs))

    def iter_frames(self, sequence_id: str, *, limit: int | None = None) -> Iterable[FrameSampleExternal]:
        seq = self.root / sequence_id
        imgs = sorted((seq / "img").glob("*.jpg"))
        gt = seq / "groundtruth.txt"
        boxes = []
        if gt.exists():
            for line in gt.read_text(encoding="utf-8", errors="ignore").splitlines():
                vals = [float(v) for v in line.replace(",", " ").split()[:4]]
                x, y, w, h = vals
                boxes.append((x, y, x + w, y + h))
        for idx, img in enumerate(imgs[:limit]):
            box = boxes[idx] if idx < len(boxes) else None
            yield FrameSampleExternal(sequence_id=sequence_id, frame_idx=idx, frame_path=img, boxes=[] if box is None else [box], instance_ids=["target"] if box else [], visibility=[True] if box else [])

    def derive_events(self, sequence_id: str) -> list[ExternalEvent]:
        return []
