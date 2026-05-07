"""Baseline 2: edge saliency proposals + online clustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Box = tuple[int, int, int, int]


@dataclass(slots=True)
class ClusterState:
    cluster_id: int
    signature: np.ndarray
    hits: int


@dataclass(slots=True)
class ClusterFrameOutput:
    boxes: list[Box]
    ids: list[int]
    active_ids: list[int]
    memory_size: int


class EdgeClusterBaseline:
    """Single-frame saliency proposals with online feature clustering."""

    def __init__(
        self,
        edge_threshold: float = 0.28,
        min_area: int = 96,
        cluster_threshold: float = 0.92,
        momentum: float = 0.80,
    ) -> None:
        self.edge_threshold = float(edge_threshold)
        self.min_area = int(min_area)
        self.cluster_threshold = float(cluster_threshold)
        self.momentum = float(momentum)

        self._clusters: dict[int, ClusterState] = {}
        self._next_cluster_id = 0

    def reset(self) -> None:
        self._clusters.clear()
        self._next_cluster_id = 0

    def update(self, current_frame: np.ndarray) -> ClusterFrameOutput:
        boxes = _extract_edge_boxes(current_frame, self.edge_threshold, self.min_area)
        assignments: list[tuple[int, Box]] = []

        for box in boxes:
            signature = _proposal_feature(current_frame, box)
            cluster_id = self._assign_cluster(signature)
            assignments.append((cluster_id, box))

        assignments.sort(key=lambda item: item[0])
        active_ids = sorted({cluster_id for cluster_id, _ in assignments})
        return ClusterFrameOutput(
            boxes=[box for _, box in assignments],
            ids=[cluster_id for cluster_id, _ in assignments],
            active_ids=active_ids,
            memory_size=len(self._clusters),
        )

    def _assign_cluster(self, signature: np.ndarray) -> int:
        if not self._clusters:
            return self._create_cluster(signature)

        similarities = [
            (_cosine_similarity(cluster.signature, signature), cluster.cluster_id)
            for cluster in self._clusters.values()
        ]
        similarities.sort(reverse=True)
        best_similarity, best_cluster_id = similarities[0]
        if best_similarity >= self.cluster_threshold:
            cluster = self._clusters[best_cluster_id]
            cluster.signature = _normalize_signature(
                self.momentum * cluster.signature + (1.0 - self.momentum) * signature
            )
            cluster.hits += 1
            return best_cluster_id
        return self._create_cluster(signature)

    def _create_cluster(self, signature: np.ndarray) -> int:
        cluster_id = self._next_cluster_id
        self._next_cluster_id += 1
        self._clusters[cluster_id] = ClusterState(cluster_id=cluster_id, signature=signature.copy(), hits=1)
        return cluster_id


def _extract_edge_boxes(current_frame: np.ndarray, edge_threshold: float, min_area: int) -> list[Box]:
    gray = _to_grayscale(current_frame)
    edge = _sobel_edges(gray)
    threshold = max(edge_threshold, float(edge.mean() + 1.1 * edge.std()))
    binary = edge > threshold
    binary = _box_blur(binary.astype(np.float32), 5) >= 0.22
    return _extract_components(binary, min_area)


def _proposal_feature(frame: np.ndarray, box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    patch = frame[y1:y2, x1:x2].astype(np.float32) / 255.0
    if patch.size == 0:
        return np.zeros(6, dtype=np.float32)

    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    area_ratio = (box_width * box_height) / float(frame.shape[0] * frame.shape[1])
    aspect_ratio = min(4.0, box_width / max(1.0, float(box_height))) / 4.0
    gray = _to_grayscale(frame[y1:y2, x1:x2])
    edge = _sobel_edges(gray) if gray.size else np.zeros((1, 1), dtype=np.float32)

    signature = np.array(
        [
            float(gray.mean()) if gray.size else 0.0,
            float(gray.std()) if gray.size else 0.0,
            float(edge.mean()),
            float(edge.std()),
            float(area_ratio),
            float(aspect_ratio),
        ],
        dtype=np.float32,
    )
    return _normalize_signature(signature)


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
    if frame.ndim == 2:
        return frame
    return 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]


def _sobel_edges(image: np.ndarray) -> np.ndarray:
    kernel_x = np.array(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    kernel_y = np.array(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    grad_x = _convolve2d(image, kernel_x)
    grad_y = _convolve2d(image, kernel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    max_value = float(magnitude.max())
    return magnitude / max(max_value, 1e-6)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)
    for iy in range(kernel.shape[0]):
        for ix in range(kernel.shape[1]):
            output += kernel[iy, ix] * padded[iy : iy + image.shape[0], ix : ix + image.shape[1]]
    return output


def _box_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = np.full((kernel_size, kernel_size), 1.0 / (kernel_size * kernel_size), dtype=np.float32)
    pad = kernel_size // 2
    padded = np.pad(image.astype(np.float32), ((pad, pad), (pad, pad)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)
    for iy in range(kernel_size):
        for ix in range(kernel_size):
            output += kernel[iy, ix] * padded[iy : iy + image.shape[0], ix : ix + image.shape[1]]
    return output


def _normalize_signature(signature: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(signature))
    if norm < 1e-6:
        return np.zeros_like(signature, dtype=np.float32)
    return signature.astype(np.float32) / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(_normalize_signature(a), _normalize_signature(b)), -1.0, 1.0))
