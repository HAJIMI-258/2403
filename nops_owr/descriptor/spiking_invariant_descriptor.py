"""Sparse spike-like descriptors for bounded object memory.

This module is SNN-inspired rather than a trainable SNN. It converts an
ObjectFile plus the current pseudo-spike encoding into fixed-size sparse
signatures suitable for event-driven long-term memory lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nops_owr.cognition.object_file import ObjectFile

Box = tuple[int, int, int, int]


@dataclass(slots=True)
class SpikingInvariantDescriptor:
    spike_signature: np.ndarray
    shape_signature: np.ndarray
    appearance_signature: np.ndarray
    topology_signature: np.ndarray
    deformation_signature: np.ndarray
    binary_hash: np.ndarray
    spike_density: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SpikingInvariantDescriptorBuilder:
    def __init__(
        self,
        spike_dim: int = 128,
        hash_bits: int = 128,
        on_threshold: float = 0.55,
        off_threshold: float = 0.35,
        seed: int = 1337,
    ) -> None:
        self.spike_dim = int(spike_dim)
        self.hash_bits = int(hash_bits)
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.seed = int(seed)
        self._projection_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def build(self, object_file: ObjectFile, encoding: Any) -> SpikingInvariantDescriptor:
        box = _clip_box(object_file.box, encoding.current_gray.shape)
        support_box = _clip_box(object_file.support_box, encoding.current_gray.shape)
        gray = _crop(encoding.current_gray, box)
        edge = _crop(encoding.edge_map, box)
        spikes = _crop(encoding.spike_response, box)
        support_spikes = _crop(encoding.spike_response, support_box)

        appearance = np.asarray(
            _patch_stats(gray) + _patch_stats(edge) + _patch_stats(spikes),
            dtype=np.float32,
        )
        shape = _shape_signature(object_file, encoding.current_gray.shape)
        topology = _topology_signature(gray, edge, spikes, support_spikes)
        deformation = _deformation_signature(object_file, encoding.current_gray.shape)
        continuous = _normalize_vector(np.concatenate([appearance, shape, topology, deformation]).astype(np.float32))

        spike_projection = self._projection(input_dim=continuous.size, output_dim=self.spike_dim, salt=11)
        spike_drive = _normalize_vector(spike_projection @ continuous)
        spike_signature = (spike_drive > self.on_threshold).astype(np.float32)
        if float(spike_signature.mean()) < 0.02:
            spike_signature = (spike_drive >= max(self.off_threshold, float(np.quantile(spike_drive, 0.90)))).astype(np.float32)
        spike_signature = _cap_density(spike_signature, spike_drive, max_density=0.20)

        hash_projection = self._projection(input_dim=continuous.size, output_dim=self.hash_bits, salt=29)
        binary_hash = (hash_projection @ continuous >= 0.0).astype(np.uint8)
        spike_density = float(np.mean(spike_signature > 0.0)) if spike_signature.size else 0.0
        return SpikingInvariantDescriptor(
            spike_signature=spike_signature.astype(np.float32),
            shape_signature=shape.astype(np.float32),
            appearance_signature=appearance.astype(np.float32),
            topology_signature=topology.astype(np.float32),
            deformation_signature=deformation.astype(np.float32),
            binary_hash=binary_hash,
            spike_density=spike_density,
            metadata={
                "object_file_id": object_file.object_file_id,
                "frame_index": int(object_file.frame_index),
                "box": tuple(int(v) for v in box),
                "support_box": tuple(int(v) for v in support_box),
                "proposal_source": object_file.proposal_source,
            },
        )

    def _projection(self, *, input_dim: int, output_dim: int, salt: int) -> np.ndarray:
        key = (int(input_dim), int(output_dim), int(salt))
        if key not in self._projection_cache:
            rng = np.random.default_rng(self.seed + 1009 * salt + 17 * input_dim + output_dim)
            matrix = rng.normal(0.0, 1.0 / np.sqrt(max(1, input_dim)), size=(output_dim, input_dim)).astype(np.float32)
            self._projection_cache[key] = matrix
        return self._projection_cache[key]


def _shape_signature(object_file: ObjectFile, frame_shape: tuple[int, int]) -> np.ndarray:
    height, width = frame_shape
    x1, y1, x2, y2 = object_file.box
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))
    area_ratio = float(object_file.area) / max(1.0, float(height * width))
    support = object_file.support_mask_summary
    signature = np.asarray(
        [
            box_w / max(1.0, float(width)),
            box_h / max(1.0, float(height)),
            box_w / box_h,
            area_ratio,
            float(support.fill_ratio),
            float(support.compactness),
            float(support.boundary_smoothness),
            float(object_file.quality_score),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(signature, nan=0.0, posinf=1.0, neginf=0.0)


def _deformation_signature(object_file: ObjectFile, frame_shape: tuple[int, int]) -> np.ndarray:
    height, width = frame_shape
    x1, y1, x2, y2 = object_file.box
    raw_w = max(1.0, float(object_file.raw_box[2] - object_file.raw_box[0]))
    raw_h = max(1.0, float(object_file.raw_box[3] - object_file.raw_box[1]))
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))
    frame_area = max(1.0, float(height * width))
    return np.asarray(
        [
            box_w / max(1.0, raw_w),
            box_h / max(1.0, raw_h),
            (box_w * box_h) / frame_area,
            float(object_file.area) / frame_area,
            float(object_file.support_mask_summary.fill_ratio),
            float(object_file.support_mask_summary.compactness),
        ],
        dtype=np.float32,
    )


def _topology_signature(gray: np.ndarray, edge: np.ndarray, spikes: np.ndarray, support_spikes: np.ndarray) -> np.ndarray:
    pooled = []
    for patch in (gray, edge, spikes):
        pooled.extend(_pool_grid(patch, grid=4).tolist())
    pooled.extend(_edge_histogram(edge, bins=8).tolist())
    pooled.extend(_patch_stats(support_spikes))
    return np.asarray(pooled, dtype=np.float32)


def _pool_grid(patch: np.ndarray, grid: int) -> np.ndarray:
    if patch.size == 0:
        return np.zeros(grid * grid, dtype=np.float32)
    h, w = patch.shape[:2]
    output = np.zeros((grid, grid), dtype=np.float32)
    for gy in range(grid):
        y1 = int(round(gy * h / grid))
        y2 = int(round((gy + 1) * h / grid))
        for gx in range(grid):
            x1 = int(round(gx * w / grid))
            x2 = int(round((gx + 1) * w / grid))
            cell = patch[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
            output[gy, gx] = float(np.mean(cell)) if cell.size else 0.0
    return output.reshape(-1)


def _edge_histogram(edge: np.ndarray, bins: int) -> np.ndarray:
    if edge.size == 0:
        return np.zeros(bins, dtype=np.float32)
    values = np.clip(edge.reshape(-1).astype(np.float32), 0.0, 1.0)
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    return hist / max(1.0, float(hist.sum()))


def _patch_stats(patch: np.ndarray) -> list[float]:
    if patch.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    values = patch.reshape(-1).astype(np.float32)
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.quantile(values, 0.25)),
        float(np.quantile(values, 0.50)),
        float(np.quantile(values, 0.75)),
    ]


def _crop(array: np.ndarray, box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return array[y1:y2, x1:x2].astype(np.float32, copy=False)


def _clip_box(box: Box, frame_shape: tuple[int, int]) -> Box:
    height, width = frame_shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, max(0, width - 1)))
    y1 = max(0, min(y1, max(0, height - 1)))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return (x1, y1, x2, y2)


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return values
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-6:
        return values - mean
    return (values - mean) / std


def _cap_density(spikes: np.ndarray, drive: np.ndarray, max_density: float) -> np.ndarray:
    max_active = max(1, int(np.floor(float(max_density) * spikes.size)))
    active = int(spikes.sum())
    if active <= max_active:
        return spikes
    indices = np.argsort(drive)[-max_active:]
    capped = np.zeros_like(spikes, dtype=np.float32)
    capped[indices] = 1.0
    return capped
