"""Baseline 1: frame difference + connected components + IoU tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics.metrics_core import bbox_iou

Box = tuple[int, int, int, int]


@dataclass(slots=True)
class BaselineTrack:
    track_id: int
    box: Box
    missed_frames: int
    active: bool


@dataclass(slots=True)
class BaselineFrameOutput:
    boxes: list[Box]
    ids: list[int]
    active_ids: list[int]
    memory_size: int


class FrameDiffConnectedComponentsBaseline:
    """Simple dynamic foreground baseline."""

    def __init__(
        self,
        diff_threshold: float = 0.12,
        min_area: int = 96,
        iou_threshold: float = 0.30,
        max_missed_frames: int = 8,
    ) -> None:
        self.diff_threshold = float(diff_threshold)
        self.min_area = int(min_area)
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)

        self._tracks: dict[int, BaselineTrack] = {}
        self._archived_track_ids: set[int] = set()
        self._next_track_id = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._archived_track_ids.clear()
        self._next_track_id = 0

    def update(self, prev_frame: np.ndarray, current_frame: np.ndarray) -> BaselineFrameOutput:
        boxes = _extract_diff_boxes(prev_frame, current_frame, self.diff_threshold, self.min_area)
        matches = _match_tracks(self._tracks, boxes, self.iou_threshold)
        matched_track_ids = {track_id for track_id, _ in matches}
        matched_box_indices = {box_index for _, box_index in matches}

        for track in self._tracks.values():
            if not track.active:
                continue
            if track.track_id in matched_track_ids:
                continue
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                track.active = False

        assignments: list[tuple[int, Box]] = []

        for track_id, box_index in matches:
            track = self._tracks[track_id]
            track.box = boxes[box_index]
            track.missed_frames = 0
            track.active = True
            assignments.append((track_id, boxes[box_index]))

        for box_index, box in enumerate(boxes):
            if box_index in matched_box_indices:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = BaselineTrack(track_id=track_id, box=box, missed_frames=0, active=True)
            self._archived_track_ids.add(track_id)
            assignments.append((track_id, box))

        assignments.sort(key=lambda item: item[0])
        active_ids = sorted(track.track_id for track in self._tracks.values() if track.active)
        return BaselineFrameOutput(
            boxes=[box for _, box in assignments],
            ids=[track_id for track_id, _ in assignments],
            active_ids=active_ids,
            memory_size=max(len(self._archived_track_ids), len(active_ids)),
        )


def _extract_diff_boxes(
    prev_frame: np.ndarray,
    current_frame: np.ndarray,
    diff_threshold: float,
    min_area: int,
) -> list[Box]:
    prev_gray = _to_grayscale(prev_frame)
    current_gray = _to_grayscale(current_frame)
    diff = np.abs(current_gray - prev_gray)
    threshold = max(diff_threshold, float(diff.mean() + 1.2 * diff.std()))
    binary = diff > threshold
    binary = _box_blur(binary.astype(np.float32), 3) >= 0.30
    return _extract_components(binary, min_area)


def _match_tracks(
    tracks: dict[int, BaselineTrack],
    boxes: list[Box],
    iou_threshold: float,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for track in tracks.values():
        if not track.active:
            continue
        for box_index, box in enumerate(boxes):
            iou = bbox_iou(track.box, box)
            if iou < iou_threshold:
                continue
            candidates.append((1.0 - iou, track.track_id, box_index))

    candidates.sort()
    matched_tracks: set[int] = set()
    matched_boxes: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, track_id, box_index in candidates:
        if track_id in matched_tracks or box_index in matched_boxes:
            continue
        matched_tracks.add(track_id)
        matched_boxes.add(box_index)
        matches.append((track_id, box_index))
    return matches


def _extract_components(binary: np.ndarray, min_area: int) -> list[Box]:
    visited = np.zeros_like(binary, dtype=bool)
    height, width = binary.shape
    boxes: list[Box] = []

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not binary[y, x]:
                continue
            queue = [(y, x)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.pop()
                pixels.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if visited[ny, nx] or not binary[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(pixels) < min_area:
                continue
            ys = np.array([py for py, _ in pixels], dtype=np.int32)
            xs = np.array([px for _, px in pixels], dtype=np.int32)
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

    return boxes


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32) / 255.0
    return 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]


def _box_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = np.full((kernel_size, kernel_size), 1.0 / (kernel_size * kernel_size), dtype=np.float32)
    pad = kernel_size // 2
    padded = np.pad(image.astype(np.float32), ((pad, pad), (pad, pad)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)
    for iy in range(kernel_size):
        for ix in range(kernel_size):
            output += kernel[iy, ix] * padded[iy : iy + image.shape[0], ix : ix + image.shape[1]]
    return output

