"""Object-file abstraction for the visual cognitive loop.

An ObjectFile is the short-lived object-centric unit that sits between
objectness proposals and longer-term memory. It summarizes what the system
currently sees without forcing full masks or raw frames into long-term memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nops_owr.encoder.spike_encoder import SpikeEncoding
from nops_owr.objectness.field import ObjectnessOutput, Proposal

Box = tuple[int, int, int, int]


@dataclass(slots=True)
class SupportMaskSummary:
    area: float
    bbox: Box
    fill_ratio: float
    compactness: float
    boundary_smoothness: float


@dataclass(slots=True)
class ObjectFile:
    object_file_id: str
    frame_index: int
    proposal_index: int
    box: Box
    raw_box: Box
    support_box: Box
    centroid: tuple[float, float]
    area: float
    score: float
    quality_score: float
    support_mask_summary: SupportMaskSummary
    appearance_signature: np.ndarray
    shape_signature: np.ndarray
    context_signature: np.ndarray
    motion_signature: np.ndarray
    novelty_score: float = 1.0
    familiarity_score: float = 0.0
    prediction_error: float = 0.0
    linked_track_id: int | None = None
    linked_prototype_id: int | None = None
    linked_concept_id: int | None = None
    linked_episode_ids: list[int] = field(default_factory=list)
    state: str = "unknown"
    confidence: float = 0.0
    proposal_source: str = "unknown"
    proposal_source_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ObjectFileBuilder:
    """Build object files from objectness proposals and pseudo-spike encoding."""

    def build(
        self,
        objectness_output: ObjectnessOutput,
        encoding: SpikeEncoding,
        frame_index: int,
        current_frame: np.ndarray | None = None,
    ) -> list[ObjectFile]:
        del current_frame  # Signatures intentionally use current minimal pipeline outputs.
        object_files: list[ObjectFile] = []
        frame_shape = encoding.current_gray.shape
        proposal_count = len(objectness_output.proposals)
        for proposal_index, proposal in enumerate(objectness_output.proposals):
            object_files.append(
                ObjectFile(
                    object_file_id=f"of:{frame_index}:{proposal_index}",
                    frame_index=int(frame_index),
                    proposal_index=int(proposal_index),
                    box=proposal.box,
                    raw_box=proposal.raw_box,
                    support_box=proposal.support_box,
                    centroid=(float(proposal.centroid[0]), float(proposal.centroid[1])),
                    area=float(proposal.area),
                    score=float(proposal.score),
                    quality_score=float(proposal.quality_score),
                    support_mask_summary=_summarize_support_mask(proposal),
                    appearance_signature=_appearance_signature(proposal.box, encoding),
                    shape_signature=_shape_signature(proposal, frame_shape),
                    context_signature=_context_signature(proposal, frame_shape, proposal_count),
                    motion_signature=np.zeros(0, dtype=np.float32),
                    novelty_score=float(np.clip(1.0 - proposal.quality_score, 0.0, 1.0)),
                    familiarity_score=0.0,
                    prediction_error=float(np.clip(abs(proposal.score - proposal.quality_score), 0.0, 1.0)),
                    confidence=float(np.clip(0.5 * proposal.score + 0.5 * proposal.quality_score, 0.0, 1.0)),
                    proposal_source=str(getattr(proposal, "source", "component")),
                    proposal_source_score=float(getattr(proposal, "source_score", proposal.score)),
                    metadata={
                        **dict(getattr(proposal, "metadata", {}) or {}),
                        "proposal_source": str(getattr(proposal, "source", "component")),
                        "proposal_source_score": float(getattr(proposal, "source_score", proposal.score)),
                        "template_gray_16": _template_patch(encoding.current_gray, proposal.box, size=16).tolist(),
                        "template_edge_16": _template_patch(encoding.edge_map, proposal.box, size=16).tolist(),
                    },
                )
            )
        return object_files


def _summarize_support_mask(proposal: Proposal) -> SupportMaskSummary:
    support_mask = proposal.support_mask
    mask_area = float(np.count_nonzero(support_mask)) if support_mask.size else 0.0
    return SupportMaskSummary(
        area=mask_area,
        bbox=proposal.support_box,
        fill_ratio=float(proposal.fill_ratio),
        compactness=float(proposal.compactness),
        boundary_smoothness=float(proposal.boundary_smoothness),
    )


def _shape_signature(proposal: Proposal, frame_shape: tuple[int, int]) -> np.ndarray:
    height, width = frame_shape
    x1, y1, x2, y2 = proposal.box
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))
    frame_area = max(1.0, float(height * width))
    signature = np.array(
        [
            box_w / max(1.0, float(width)),
            box_h / max(1.0, float(height)),
            box_w / box_h,
            float(proposal.area) / frame_area,
            float(proposal.fill_ratio),
            float(proposal.compactness),
            float(proposal.boundary_smoothness),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(signature, nan=0.0, posinf=1.0, neginf=0.0)


def _appearance_signature(box: Box, encoding: SpikeEncoding) -> np.ndarray:
    patches = [
        _crop_2d(encoding.current_gray, box),
        _crop_2d(encoding.edge_map, box),
        _crop_2d(encoding.spike_response, box),
    ]
    stats: list[float] = []
    for patch in patches:
        stats.extend(_patch_stats(patch))
    return np.asarray(stats, dtype=np.float32)


def _context_signature(proposal: Proposal, frame_shape: tuple[int, int], proposal_count: int) -> np.ndarray:
    height, width = frame_shape
    cx, cy = proposal.centroid
    x1, y1, x2, y2 = proposal.box
    near_horizontal = min(x1, max(0, width - x2)) / max(1.0, float(width))
    near_vertical = min(y1, max(0, height - y2)) / max(1.0, float(height))
    signature = np.array(
        [
            float(cx) / max(1.0, float(width)),
            float(cy) / max(1.0, float(height)),
            float(proposal_count),
            float(proposal.near_boundary),
            float(near_horizontal),
            float(near_vertical),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(signature, nan=0.0, posinf=1.0, neginf=0.0)


def _crop_2d(array: np.ndarray, box: Box) -> np.ndarray:
    height, width = array.shape[:2]
    x1, y1, x2, y2 = box
    x1 = int(np.clip(x1, 0, width))
    x2 = int(np.clip(x2, 0, width))
    y1 = int(np.clip(y1, 0, height))
    y2 = int(np.clip(y2, 0, height))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0), dtype=np.float32)
    return array[y1:y2, x1:x2].astype(np.float32, copy=False)


def _template_patch(array: np.ndarray, box: Box, size: int = 16) -> np.ndarray:
    patch = _crop_2d(array, box)
    if patch.size == 0:
        return np.zeros((size, size), dtype=np.float32)
    return _resize_nearest(patch, size, size)


def _resize_nearest(patch: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = patch.shape[:2]
    if in_h <= 0 or in_w <= 0:
        return np.zeros((out_h, out_w), dtype=np.float32)
    y_idx = np.clip(np.round(np.linspace(0, in_h - 1, out_h)).astype(np.int32), 0, in_h - 1)
    x_idx = np.clip(np.round(np.linspace(0, in_w - 1, out_w)).astype(np.int32), 0, in_w - 1)
    return patch[np.ix_(y_idx, x_idx)].astype(np.float32, copy=False)


def _patch_stats(patch: np.ndarray) -> list[float]:
    if patch.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    values = patch.astype(np.float32, copy=False).reshape(-1)
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.quantile(values, 0.25)),
        float(np.quantile(values, 0.50)),
        float(np.quantile(values, 0.75)),
    ]
