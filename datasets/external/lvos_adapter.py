from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .base_video_memory_dataset import ExternalEvent, FrameSampleExternal


class LVOSAdapter:
    """Adapter for LVOS-style data.

    Supported smoke-test input:
    - HuggingFace `allenai/molmo2-single-object-track/lvosv1` parquet export,
      stored as `train-*.parquet`. This sample contains point trajectories rather
      than full frames/masks, so boxes are small point-centered proxies.

    Full LVOS image/mask layout can be added under the same interface later.
    """

    dataset_name = "lvos"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.parquet_files = sorted(self.root.glob("*.parquet"))
        self._df = None

    def _load_df(self):
        if self._df is None:
            if not self.parquet_files:
                raise FileNotFoundError(f"No LVOS parquet files found under {self.root}")
            import pandas as pd

            self._df = pd.concat([pd.read_parquet(p) for p in self.parquet_files], ignore_index=True)
        return self._df

    def iter_sequences(self) -> Iterable[str]:
        df = self._load_df()
        for seq in sorted(str(v) for v in df["video"].dropna().unique()):
            yield seq

    def _rows_for_sequence(self, sequence_id: str):
        df = self._load_df()
        return df[df["video"].astype(str) == str(sequence_id)]

    @staticmethod
    def _point_box(point: Any, radius: float = 8.0) -> tuple[float, float, float, float]:
        arr = np.asarray(point, dtype=float).reshape(-1)
        x, y = float(arr[0]), float(arr[1])
        return (x - radius, y - radius, x + radius, y + radius)

    def iter_frames(self, sequence_id: str, *, limit: int | None = None) -> Iterable[FrameSampleExternal]:
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        width = height = None
        for _, row in self._rows_for_sequence(sequence_id).iterrows():
            width, height = int(row["width"]), int(row["height"])
            obj_id = str(row["obj_id"][0] if isinstance(row["obj_id"], (list, tuple, np.ndarray)) else row["obj_id"])
            for traj in row["frame_trajectories"]:
                frame_idx = int(traj["frame"])
                for point in traj["points"]:
                    by_frame[frame_idx].append({
                        "box": self._point_box(point["point"]),
                        "instance_id": f"{row['id']}:{obj_id}:{point['id']}",
                        "visibility": not bool(point.get("occluded", False)),
                    })
        count = 0
        for frame_idx in sorted(by_frame):
            items = by_frame[frame_idx]
            yield FrameSampleExternal(
                sequence_id=sequence_id,
                frame_idx=frame_idx,
                frame_path=None,
                boxes=[item["box"] for item in items],
                instance_ids=[item["instance_id"] for item in items],
                visibility=[item["visibility"] for item in items],
                metadata={"width": width, "height": height, "source": "hf_point_track_parquet"},
            )
            count += 1
            if limit is not None and count >= limit:
                return

    def derive_events(self, sequence_id: str) -> list[ExternalEvent]:
        frames = list(self.iter_frames(sequence_id))
        by_obj: dict[str, list[tuple[int, bool]]] = defaultdict(list)
        for fr in frames:
            for iid, vis in zip(fr.instance_ids, fr.visibility):
                by_obj[str(iid)].append((fr.frame_idx, bool(vis)))
        events: list[ExternalEvent] = []
        for iid, states in by_obj.items():
            states.sort()
            was_hidden = False
            hidden_start = None
            last_hidden = None
            for frame_idx, visible in states:
                if not visible and not was_hidden:
                    was_hidden = True
                    hidden_start = frame_idx
                if not visible:
                    last_hidden = frame_idx
                if visible and was_hidden and hidden_start is not None and last_hidden is not None:
                    events.append(ExternalEvent(
                        dataset_name=self.dataset_name,
                        sequence_id=sequence_id,
                        instance_id=iid,
                        disappear_frame=int(hidden_start),
                        reappear_frame=int(frame_idx),
                        gap_length=int(frame_idx - hidden_start),
                        event_type="visibility_reentry",
                        metadata={"last_hidden_frame": int(last_hidden)},
                    ))
                    was_hidden = False
        return events
