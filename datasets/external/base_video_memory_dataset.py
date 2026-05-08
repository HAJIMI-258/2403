from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameSampleExternal:
    sequence_id: str
    frame_idx: int
    frame_path: Path | None
    boxes: list[Box]
    masks: list[Path | None] = field(default_factory=list)
    instance_ids: list[str] = field(default_factory=list)
    category_ids: list[str | int | None] = field(default_factory=list)
    visibility: list[float | bool] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalEvent:
    dataset_name: str
    sequence_id: str
    instance_id: str
    disappear_frame: int
    reappear_frame: int
    gap_length: int
    event_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVideoMemoryDataset(Protocol):
    dataset_name: str
    root: Path

    def iter_sequences(self) -> Iterable[str]:
        ...

    def iter_frames(self, sequence_id: str, *, limit: int | None = None) -> Iterable[FrameSampleExternal]:
        ...

    def derive_events(self, sequence_id: str) -> list[ExternalEvent]:
        ...
