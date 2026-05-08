from __future__ import annotations

import configparser
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .base_video_memory_dataset import ExternalEvent, FrameSampleExternal


class LaGOTAdapter:
    """Adapter for LaGOT MOTChallenge-format annotations.

    LaGOT is built on top of LaSOT validation videos. The public repository
    provides annotations and tracker outputs. Raw pixels are not bundled, so this
    adapter supports oracle-proposal / geometry-only memory evaluation.
    """

    dataset_name = "lagot_annotations"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.zip_path = self.root / "data" / "lagot_motchallenge_format.zip"
        self._gt_files: dict[str, str] | None = None
        self._seqinfo_files: dict[str, str] | None = None

    def _ensure_index(self) -> None:
        if self._gt_files is not None and self._seqinfo_files is not None:
            return
        if not self.zip_path.exists():
            raise FileNotFoundError(f"LaGOT MOTChallenge zip not found: {self.zip_path}")
        gt_files: dict[str, str] = {}
        seqinfo_files: dict[str, str] = {}
        with zipfile.ZipFile(self.zip_path) as zf:
            for name in zf.namelist():
                parts = name.split("/")
                if name.endswith("/gt/gt.txt") and len(parts) >= 7:
                    seq = parts[-3]
                    gt_files[seq] = name
                elif name.endswith("/seqinfo.ini") and len(parts) >= 6:
                    seq = parts[-2]
                    seqinfo_files[seq] = name
        self._gt_files = gt_files
        self._seqinfo_files = seqinfo_files

    def iter_sequences(self) -> Iterable[str]:
        self._ensure_index()
        assert self._gt_files is not None
        for seq in sorted(self._gt_files):
            yield seq

    def _read_seqinfo(self, sequence_id: str) -> dict[str, Any]:
        self._ensure_index()
        assert self._seqinfo_files is not None
        name = self._seqinfo_files.get(sequence_id)
        if not name:
            return {}
        with zipfile.ZipFile(self.zip_path) as zf:
            text = zf.read(name).decode("utf-8")
        parser = configparser.ConfigParser()
        parser.read_string(text)
        sec = parser["Sequence"]
        return {
            "seq_length": int(sec.get("seqLength", 0)),
            "width": int(sec.get("imWidth", 0)),
            "height": int(sec.get("imHeight", 0)),
            "frame_rate": int(sec.get("frameRate", 0)),
        }

    def _read_gt_rows(self, sequence_id: str) -> list[dict[str, Any]]:
        self._ensure_index()
        assert self._gt_files is not None
        gt_name = self._gt_files.get(sequence_id)
        if not gt_name:
            raise KeyError(sequence_id)
        rows: list[dict[str, Any]] = []
        with zipfile.ZipFile(self.zip_path) as zf:
            text = zf.read(gt_name).decode("utf-8")
        for line in text.splitlines():
            if not line.strip():
                continue
            p = line.split(",")
            frame_idx = int(float(p[0]))
            instance_id = str(int(float(p[1])))
            x, y, w, h = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
            conf = float(p[6]) if len(p) > 6 else 1.0
            visibility = float(p[8]) if len(p) > 8 and p[8] != "-1" else 1.0
            rows.append({
                "frame_idx": frame_idx,
                "instance_id": instance_id,
                "box": (x, y, x + w, y + h),
                "conf": conf,
                "visibility": visibility,
            })
        return rows

    @staticmethod
    def category_from_sequence(sequence_id: str) -> str:
        name = sequence_id
        if name.startswith("lagot_"):
            name = name[len("lagot_"):]
        return name.split("-")[0]

    def iter_frames(self, sequence_id: str, *, limit: int | None = None) -> Iterable[FrameSampleExternal]:
        seqinfo = self._read_seqinfo(sequence_id)
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self._read_gt_rows(sequence_id):
            by_frame[int(row["frame_idx"])].append(row)
        count = 0
        category = self.category_from_sequence(sequence_id)
        for frame_idx in sorted(by_frame):
            rows = by_frame[frame_idx]
            yield FrameSampleExternal(
                sequence_id=sequence_id,
                frame_idx=frame_idx,
                frame_path=None,
                boxes=[r["box"] for r in rows],
                instance_ids=[f"{sequence_id}:{r['instance_id']}" for r in rows],
                category_ids=[category for _ in rows],
                visibility=[bool(float(r["conf"]) > 0 and float(r["visibility"]) > 0) for r in rows],
                metadata={**seqinfo, "source": "lagot_motchallenge_annotations_only"},
            )
            count += 1
            if limit is not None and count >= limit:
                return

    def derive_events(self, sequence_id: str) -> list[ExternalEvent]:
        by_obj: dict[str, list[int]] = defaultdict(list)
        for row in self._read_gt_rows(sequence_id):
            by_obj[f"{sequence_id}:{row['instance_id']}"].append(int(row["frame_idx"]))
        events: list[ExternalEvent] = []
        category = self.category_from_sequence(sequence_id)
        for iid, frames in by_obj.items():
            unique_frames = sorted(set(frames))
            for prev_frame, next_frame in zip(unique_frames, unique_frames[1:]):
                gap = int(next_frame - prev_frame - 1)
                if gap <= 0:
                    continue
                events.append(ExternalEvent(
                    dataset_name=self.dataset_name,
                    sequence_id=sequence_id,
                    instance_id=iid,
                    disappear_frame=int(prev_frame),
                    reappear_frame=int(next_frame),
                    gap_length=gap,
                    event_type="visibility_reentry",
                    metadata={"category_id": category},
                ))
        return events

