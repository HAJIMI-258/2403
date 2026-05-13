"""External-video event window helpers.

The helpers in this module are deliberately dataset-light. They use adapter
outputs and visibility/event annotations to create local windows around a
re-entry event without changing online model behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from datasets.external.base_video_memory_dataset import ExternalEvent, FrameSampleExternal


@dataclass(slots=True)
class ExternalEventWindow:
    event: ExternalEvent
    frames: list[FrameSampleExternal]
    window_start_frame: int
    window_end_frame: int
    pre_visible_frame_count: int
    invisible_gap_frame_count: int
    post_visible_frame_count: int


def collect_lasot_reentry_events(
    adapter: Any,
    *,
    min_gap: int,
    category_filter: str = "",
    sequence_filter: str = "",
    max_events: int | None = None,
) -> list[ExternalEvent]:
    categories = {item.strip() for item in category_filter.split(",") if item.strip()}
    seq_filter = sequence_filter.strip()
    events: list[ExternalEvent] = []
    for sequence_id in sorted(adapter.iter_sequences()):
        category = sequence_category(sequence_id)
        if categories and category not in categories:
            continue
        if seq_filter and seq_filter not in sequence_id:
            continue
        for event in adapter.derive_events(sequence_id):
            if int(event.gap_length) < int(min_gap):
                continue
            events.append(event)
            if max_events is not None and len(events) >= int(max_events):
                return events
    return events


def make_event_window(
    frames: list[FrameSampleExternal],
    event: ExternalEvent,
    *,
    pre_context: int,
    post_context: int,
    frame_stride: int = 1,
) -> ExternalEventWindow | None:
    if not frames:
        return None
    start = max(0, int(event.disappear_frame) - int(pre_context))
    end = int(event.reappear_frame) + int(post_context)
    stride = max(1, int(frame_stride))
    window_frames = [
        frame
        for frame in frames
        if start <= int(frame.frame_idx) <= end and (int(frame.frame_idx) - start) % stride == 0
    ]
    required_indices = {int(event.disappear_frame), int(event.reappear_frame)}
    present = {int(frame.frame_idx) for frame in window_frames}
    for frame in frames:
        if int(frame.frame_idx) in required_indices and start <= int(frame.frame_idx) <= end and int(frame.frame_idx) not in present:
            window_frames.append(frame)
            present.add(int(frame.frame_idx))
    window_frames.sort(key=lambda row: int(row.frame_idx))
    if not any(int(frame.frame_idx) == int(event.reappear_frame) for frame in window_frames):
        return None
    if len(window_frames) < 2:
        return None
    return ExternalEventWindow(
        event=event,
        frames=window_frames,
        window_start_frame=start,
        window_end_frame=end,
        pre_visible_frame_count=sum(
            int(frame_is_visible(frame) and int(frame.frame_idx) <= int(event.disappear_frame))
            for frame in window_frames
        ),
        invisible_gap_frame_count=sum(
            int(not frame_is_visible(frame) and int(event.disappear_frame) < int(frame.frame_idx) < int(event.reappear_frame))
            for frame in window_frames
        ),
        post_visible_frame_count=sum(
            int(frame_is_visible(frame) and int(frame.frame_idx) >= int(event.reappear_frame))
            for frame in window_frames
        ),
    )


def frame_phase(frame: FrameSampleExternal, event: ExternalEvent) -> str:
    idx = int(frame.frame_idx)
    if idx < int(event.disappear_frame):
        return "pre_visible" if frame_is_visible(frame) else "pre_invisible"
    if idx == int(event.disappear_frame):
        return "pre_visible"
    if idx < int(event.reappear_frame):
        return "invisible"
    if idx == int(event.reappear_frame):
        return "reappear"
    return "post_visible" if frame_is_visible(frame) else "post_invisible"


def frame_is_visible(frame: FrameSampleExternal) -> bool:
    if frame.visibility:
        return bool(frame.visibility[0])
    return bool(frame.boxes)


def frame_gt_box(frame: FrameSampleExternal) -> tuple[float, float, float, float] | None:
    return None if not frame.boxes else tuple(float(v) for v in frame.boxes[0])


def sequence_category(sequence_id: str) -> str:
    return str(sequence_id).split("-")[0]


def load_rgb_frame(path: Path | None, *, max_image_side: int = 160) -> tuple[np.ndarray, float, float]:
    if path is None:
        raise FileNotFoundError("Frame path is missing.")
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - local optional package.
        raise RuntimeError("PIL/Pillow required for pixel eval") from exc

    image = Image.open(path).convert("RGB")
    original_w, original_h = image.size
    max_side = max(original_w, original_h)
    if int(max_image_side) > 0 and max_side > int(max_image_side):
        scale = float(max_image_side) / float(max_side)
        new_size = (max(1, int(round(original_w * scale))), max(1, int(round(original_h * scale))))
        image = image.resize(new_size)
    array = np.asarray(image, dtype=np.uint8)
    scaled_h, scaled_w = array.shape[:2]
    return array, scaled_w / max(1.0, float(original_w)), scaled_h / max(1.0, float(original_h))


def scale_box(
    box: tuple[float, float, float, float] | None,
    scale_x: float,
    scale_y: float,
) -> tuple[float, float, float, float] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
