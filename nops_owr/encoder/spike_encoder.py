"""Minimal spike encoder for Phase 1 experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SpikeEncoding:
    prev_gray: np.ndarray
    current_gray: np.ndarray
    frame_diff: np.ndarray
    edge_map: np.ndarray
    on_spikes: np.ndarray
    off_spikes: np.ndarray
    spike_response: np.ndarray


class MinimalSpikeEncoder:
    """Frame difference + Sobel edge + ON/OFF threshold encoder."""

    def __init__(self, edge_weight: float = 0.5, on_off_threshold: float = 0.1) -> None:
        self.edge_weight = float(edge_weight)
        self.on_off_threshold = float(on_off_threshold)

    def encode(self, prev_frame: np.ndarray, current_frame: np.ndarray) -> SpikeEncoding:
        prev_gray = _to_grayscale(prev_frame)
        current_gray = _to_grayscale(current_frame)
        frame_diff = current_gray - prev_gray
        edge_map = _sobel_edges(current_gray)

        on_drive = frame_diff + self.edge_weight * edge_map
        off_drive = -frame_diff + self.edge_weight * edge_map
        on_spikes = (on_drive > self.on_off_threshold).astype(np.float32)
        off_spikes = (off_drive > self.on_off_threshold).astype(np.float32)

        diff_energy = _normalize(np.abs(frame_diff))
        spike_response = _normalize(0.65 * (on_spikes + off_spikes) + 0.35 * diff_energy)

        return SpikeEncoding(
            prev_gray=prev_gray,
            current_gray=current_gray,
            frame_diff=frame_diff,
            edge_map=edge_map,
            on_spikes=on_spikes,
            off_spikes=off_spikes,
            spike_response=spike_response,
        )


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
    return _normalize(magnitude)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)

    for iy in range(kernel.shape[0]):
        for ix in range(kernel.shape[1]):
            output += kernel[iy, ix] * padded[iy : iy + image.shape[0], ix : ix + image.shape[1]]

    return output


def _normalize(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value - min_value < 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)

