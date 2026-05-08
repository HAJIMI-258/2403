from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .base_video_memory_dataset import ExternalEvent, FrameSampleExternal


class TAOAdapter:
    dataset_name = "tao"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.annotation_path = self.root / "annotations" / "train.json"

    def _load(self):
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"TAO annotation file not found: {self.annotation_path}")
        return json.loads(self.annotation_path.read_text(encoding="utf-8"))

    def iter_sequences(self) -> Iterable[str]:
        if not self.annotation_path.exists():
            return iter(())
        data = self._load()
        videos = data.get("videos", [])
        return iter(sorted(str(v.get("id", v.get("name"))) for v in videos))

    def iter_frames(self, sequence_id: str, *, limit: int | None = None) -> Iterable[FrameSampleExternal]:
        data = self._load()
        images = [im for im in data.get("images", []) if str(im.get("video_id")) == str(sequence_id)]
        anns_by_img: dict[int, list[dict]] = {}
        for ann in data.get("annotations", []):
            anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)
        for idx, im in enumerate(sorted(images, key=lambda x: x.get("frame_index", x.get("id")))[:limit]):
            anns = anns_by_img.get(int(im["id"]), [])
            boxes = []
            ids = []
            cats = []
            for ann in anns:
                x, y, w, h = [float(v) for v in ann["bbox"]]
                boxes.append((x, y, x + w, y + h))
                ids.append(str(ann.get("track_id", ann.get("id"))))
                cats.append(ann.get("category_id"))
            yield FrameSampleExternal(sequence_id=sequence_id, frame_idx=int(im.get("frame_index", idx)), frame_path=self.root / str(im.get("file_name", "")), boxes=boxes, instance_ids=ids, category_ids=cats, visibility=[True] * len(boxes))

    def derive_events(self, sequence_id: str) -> list[ExternalEvent]:
        return []
