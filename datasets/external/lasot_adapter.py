from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .base_video_memory_dataset import ExternalEvent, FrameSampleExternal


class LaSOTAdapter:
    dataset_name = "lasot"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _sequence_dir(self, sequence_id: str) -> Path | None:
        direct = self.root / sequence_id
        if (direct / "img").exists():
            return direct
        category = sequence_id.split("-")[0]
        nested = self.root / category / sequence_id
        if (nested / "img").exists():
            return nested
        if not self.root.exists():
            return None
        for img_dir in self.root.rglob("img"):
            if img_dir.is_dir() and img_dir.parent.name == sequence_id:
                return img_dir.parent
        return None

    def iter_sequences(self) -> Iterable[str]:
        if not self.root.exists():
            return iter(())
        seqs = []
        for img_dir in self.root.rglob("img"):
            if img_dir.is_dir() and (img_dir.parent / "groundtruth.txt").exists():
                seqs.append(img_dir.parent.name)
        return iter(sorted(seqs))

    def iter_frames(self, sequence_id: str, *, limit: int | None = None) -> Iterable[FrameSampleExternal]:
        seq = self._sequence_dir(sequence_id)
        if seq is None:
            return
        imgs = sorted((seq / "img").glob("*.jpg"))
        gt = seq / "groundtruth.txt"
        boxes = []
        if gt.exists():
            for line in gt.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.replace(",", " ").split()[:4]
                if len(parts) < 4:
                    continue
                vals = [float(v) for v in parts]
                x, y, w, h = vals
                boxes.append((x, y, x + w, y + h))
        visibility = self._visibility_flags(seq, len(imgs))
        for idx, img in enumerate(imgs[:limit]):
            box = boxes[idx] if idx < len(boxes) else None
            visible = visibility[idx] if idx < len(visibility) else True
            yield FrameSampleExternal(
                sequence_id=sequence_id,
                frame_idx=idx,
                frame_path=img,
                boxes=[] if box is None or not visible else [box],
                instance_ids=["target"] if box is not None and visible else [],
                category_ids=[sequence_id.split("-")[0]] if box is not None and visible else [],
                visibility=[visible] if box is not None else [],
                metadata={"sequence_dir": str(seq)},
            )

    @staticmethod
    def _read_flag_file(path: Path, n: int) -> list[int]:
        if not path.exists():
            return [0] * n
        vals: list[int] = []
        text = path.read_text(encoding="utf-8", errors="ignore")
        for raw in text.replace(",", " ").split():
            try:
                vals.append(1 if int(float(raw)) > 0 else 0)
            except Exception:
                vals.append(0)
        if len(vals) < n:
            vals.extend([0] * (n - len(vals)))
        return vals[:n]

    def _visibility_flags(self, seq: Path, n: int) -> list[bool]:
        full_occ = self._read_flag_file(seq / "full_occlusion.txt", n)
        out_view = self._read_flag_file(seq / "out_of_view.txt", n)
        return [not bool(a or b) for a, b in zip(full_occ, out_view)]

    def derive_events(self, sequence_id: str) -> list[ExternalEvent]:
        seq = self._sequence_dir(sequence_id)
        if seq is None:
            return []
        n = len(list((seq / "img").glob("*.jpg")))
        if n <= 0:
            return []
        visible = self._visibility_flags(seq, n)
        events: list[ExternalEvent] = []
        last_visible: int | None = None
        missing_start: int | None = None
        for idx, is_visible in enumerate(visible):
            if is_visible:
                if missing_start is not None and last_visible is not None:
                    gap = idx - last_visible - 1
                    if gap > 0:
                        events.append(ExternalEvent(
                            dataset_name=self.dataset_name,
                            sequence_id=sequence_id,
                            instance_id=f"{sequence_id}:target",
                            disappear_frame=last_visible,
                            reappear_frame=idx,
                            gap_length=gap,
                            event_type="visibility_reentry",
                            metadata={"category_id": sequence_id.split("-")[0]},
                        ))
                    missing_start = None
                last_visible = idx
            elif last_visible is not None and missing_start is None:
                missing_start = idx
        return events
