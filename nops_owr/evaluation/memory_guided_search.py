"""Memory-guided search proposals for external re-entry diagnostics.

This module does not alter the default objectness/tracking pipeline. It builds
eval-time ObjectFile candidates from old episodic memory and the current
objectness heatmap, so we can test whether re-entry failures are caused by
global proposal recall or by later retrieval/decision layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nops_owr.cognition.object_file import (
    ObjectFile,
    SupportMaskSummary,
    _appearance_signature,
    _context_signature,
    _shape_signature,
)
from nops_owr.encoder.spike_encoder import SpikeEncoding
from nops_owr.evaluation.reentry_audit import bbox_iou
from nops_owr.memory.episodic_memory import EpisodicBundle
from nops_owr.objectness.field import Proposal

Box = tuple[int, int, int, int]


@dataclass(slots=True)
class MemoryGuidedSearchConfig:
    top_episode_count: int = 8
    heatmap_peak_count: int = 12
    max_candidates: int = 32
    nms_iou: float = 0.55
    min_box_size: int = 8
    fallback_box_frac: float = 0.22
    include_context_center: bool = True


def build_memory_guided_object_files(
    *,
    encoding: SpikeEncoding,
    heatmap: np.ndarray,
    bundles: list[EpisodicBundle],
    frame_index: int,
    config: MemoryGuidedSearchConfig | None = None,
) -> list[ObjectFile]:
    """Build candidate ObjectFiles from old memory and current heatmap peaks.

    Candidate generation is GT-free. It uses old episode size/context summaries
    and current heatmap peaks. Candidate scoring is also GT-free; downstream
    eval scripts may compare boxes with GT only for audit.
    """

    config = config or MemoryGuidedSearchConfig()
    height, width = heatmap.shape
    eligible = [
        bundle
        for bundle in bundles
        if int(getattr(bundle, "frame_start", 0)) < int(frame_index)
        and int(
            getattr(bundle, "last_observed_frame", None)
            if getattr(bundle, "last_observed_frame", None) is not None
            else getattr(bundle, "frame_end", 0)
        )
        < int(frame_index)
    ]
    eligible.sort(key=_episode_search_priority, reverse=True)
    peaks = _heatmap_peaks(heatmap, count=config.heatmap_peak_count, radius=max(4, min(height, width) // 12))
    object_files: list[ObjectFile] = []
    for bundle in eligible[: config.top_episode_count]:
        box_w, box_h = _bundle_box_size(bundle, frame_shape=(height, width), config=config)
        centers = list(peaks)
        if config.include_context_center:
            centers.insert(0, _bundle_context_center(bundle, frame_shape=(height, width)))
        for center_rank, center in enumerate(centers):
            box = _box_from_center(center, box_w, box_h, frame_shape=(height, width))
            if any(bbox_iou(existing.box, box) >= config.nms_iou for existing in object_files):
                continue
            object_files.append(_object_file_from_box(box, encoding, heatmap, bundle, frame_index, center_rank))
            if len(object_files) >= config.max_candidates:
                return _sort_candidates(object_files)
    return _sort_candidates(object_files)


def _object_file_from_box(
    box: Box,
    encoding: SpikeEncoding,
    heatmap: np.ndarray,
    bundle: EpisodicBundle,
    frame_index: int,
    center_rank: int,
) -> ObjectFile:
    patch = heatmap[box[1] : box[3], box[0] : box[2]].astype(np.float32)
    if patch.size == 0:
        patch = np.zeros((1, 1), dtype=np.float32)
    threshold = float(np.quantile(patch, 0.65)) if patch.size > 1 else float(patch.mean())
    support_mask = patch >= threshold
    if int(support_mask.sum()) == 0:
        support_mask = np.ones_like(patch, dtype=bool)
    score = float(0.70 * patch.mean() + 0.30 * patch.max())
    area = int(max(1, support_mask.sum()))
    fill_ratio = float(area / max(1, patch.shape[0] * patch.shape[1]))
    proposal = Proposal(
        box=box,
        raw_box=box,
        support_box=box,
        area=area,
        raw_area=area,
        score=score,
        quality_score=float(score + 0.05 * fill_ratio),
        centroid=((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5),
        support_mask=support_mask.astype(bool),
        fill_ratio=fill_ratio,
        compactness=0.0,
        boundary_smoothness=0.0,
        near_boundary=int(box[0] <= 4 or box[1] <= 4 or box[2] >= heatmap.shape[1] - 4 or box[3] >= heatmap.shape[0] - 4),
    )
    search_priority = float(_episode_search_priority(bundle) + score - 0.01 * center_rank)
    object_file = ObjectFile(
        object_file_id=f"memsearch:{frame_index}:{bundle.episode_id}:{center_rank}",
        frame_index=int(frame_index),
        proposal_index=-1000 - int(bundle.episode_id),
        box=box,
        raw_box=box,
        support_box=box,
        centroid=proposal.centroid,
        area=float(area),
        score=score,
        quality_score=float(proposal.quality_score),
        support_mask_summary=SupportMaskSummary(
            area=float(area),
            bbox=box,
            fill_ratio=fill_ratio,
            compactness=0.0,
            boundary_smoothness=0.0,
        ),
        appearance_signature=_appearance_signature(box, encoding),
        shape_signature=_shape_signature(proposal, heatmap.shape),
        context_signature=_context_signature(proposal, heatmap.shape, proposal_count=1),
        motion_signature=np.zeros(0, dtype=np.float32),
        novelty_score=0.0,
        familiarity_score=float(bundle.accessibility_score),
        prediction_error=0.0,
        linked_episode_ids=[int(bundle.episode_id)],
        state="familiar_unresolved",
        confidence=float(np.clip(search_priority, 0.0, 1.0)),
        metadata={
            "memory_search": 1,
            "source_episode_id": int(bundle.episode_id),
            "source_episode_closed": int(bool(bundle.closed)),
            "source_episode_gt_instance_id_eval_only": bundle.metadata.get("gt_instance_id"),
            "memory_search_score": search_priority,
            "memory_search_heatmap_score": score,
            "memory_search_center_rank": int(center_rank),
        },
    )
    return object_file


def _sort_candidates(object_files: list[ObjectFile]) -> list[ObjectFile]:
    return sorted(
        object_files,
        key=lambda item: (
            float(item.metadata.get("memory_search_score", 0.0)),
            item.quality_score,
            item.area,
        ),
        reverse=True,
    )


def _episode_search_priority(bundle: EpisodicBundle) -> float:
    return float(
        0.45 * float(bundle.accessibility_score)
        + 0.20 * float(bundle.replay_priority)
        + 0.15 * min(float(bundle.observation_count) / 12.0, 1.0)
        + 0.10 * float(bundle.reactivation_count > 0)
        + 0.10 * float(bool(bundle.closed))
    )


def _bundle_box_size(
    bundle: EpisodicBundle,
    *,
    frame_shape: tuple[int, int],
    config: MemoryGuidedSearchConfig,
) -> tuple[int, int]:
    height, width = frame_shape
    support = np.asarray(bundle.support_signature, dtype=np.float32).reshape(-1)
    if support.size >= 2 and support[0] > 0.0 and support[1] > 0.0:
        box_w = int(round(float(support[0]) * width))
        box_h = int(round(float(support[1]) * height))
    else:
        fallback = int(round(config.fallback_box_frac * min(height, width)))
        box_w = fallback
        box_h = fallback
    box_w = max(config.min_box_size, min(box_w, width))
    box_h = max(config.min_box_size, min(box_h, height))
    # Slightly expand old boxes because re-entry pose/scale is uncertain.
    box_w = max(config.min_box_size, min(int(round(box_w * 1.25)), width))
    box_h = max(config.min_box_size, min(int(round(box_h * 1.25)), height))
    return box_w, box_h


def _bundle_context_center(bundle: EpisodicBundle, *, frame_shape: tuple[int, int]) -> tuple[float, float]:
    height, width = frame_shape
    context = np.asarray(bundle.context_signature, dtype=np.float32).reshape(-1)
    if context.size >= 2:
        return (float(context[0]) * width, float(context[1]) * height)
    return (width * 0.5, height * 0.5)


def _heatmap_peaks(heatmap: np.ndarray, *, count: int, radius: int) -> list[tuple[float, float]]:
    if heatmap.size == 0 or count <= 0:
        return []
    flat_indices = np.argsort(heatmap.reshape(-1))[::-1]
    height, width = heatmap.shape
    peaks: list[tuple[float, float]] = []
    for flat in flat_indices:
        y = int(flat // width)
        x = int(flat % width)
        if any((x - px) ** 2 + (y - py) ** 2 <= radius * radius for px, py in peaks):
            continue
        peaks.append((float(x), float(y)))
        if len(peaks) >= count:
            break
    if not peaks:
        peaks.append((width * 0.5, height * 0.5))
    return peaks


def _box_from_center(center: tuple[float, float], box_w: int, box_h: int, *, frame_shape: tuple[int, int]) -> Box:
    height, width = frame_shape
    cx, cy = center
    x1 = int(round(cx - box_w * 0.5))
    y1 = int(round(cy - box_h * 0.5))
    x1 = max(0, min(x1, width - box_w))
    y1 = max(0, min(y1, height - box_h))
    x2 = max(x1 + 1, min(width, x1 + box_w))
    y2 = max(y1 + 1, min(height, y1 + box_h))
    return (int(x1), int(y1), int(x2), int(y2))

