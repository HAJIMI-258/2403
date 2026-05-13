"""Top-down proposal augmentation from closed episodic memory.

The augmenter is optional and GT-free. It uses closed episodic bundles as
search cues for real re-entry diagnostics: old size/content/context summaries
generate candidate windows, then the current heatmap and cue compatibility rank
the windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nops_owr.encoder.spike_encoder import SpikeEncoding
from nops_owr.memory.episodic_memory import EpisodicMemory, EpisodicBundle
from nops_owr.objectness.field import ObjectnessOutput, Proposal

Box = tuple[int, int, int, int]


@dataclass(slots=True)
class MemoryGuidedProposalConfig:
    max_memory_episodes: int = 8
    windows_per_episode: int = 8
    max_added_proposals: int = 16
    max_total_proposals: int = 48
    nms_iou: float = 0.55
    min_episode_gap: int = 8
    default_window_fracs: tuple[float, ...] = (0.14, 0.22, 0.32)


class MemoryGuidedProposalAugmenter:
    """Append memory-guided proposal windows to bottom-up proposals."""

    def __init__(self, config: MemoryGuidedProposalConfig | None = None) -> None:
        self.config = config or MemoryGuidedProposalConfig()

    def augment(
        self,
        *,
        objectness_output: ObjectnessOutput,
        encoding: SpikeEncoding,
        current_frame: np.ndarray | None,
        episodic_memory: EpisodicMemory,
        frame_index: int,
    ) -> list[Proposal]:
        del current_frame
        base = list(objectness_output.proposals)
        candidates: list[Proposal] = []
        for bundle in self._candidate_episodes(episodic_memory, frame_index):
            candidates.extend(self._episode_windows(bundle, objectness_output.heatmap, encoding, frame_index))
        candidates.sort(key=lambda proposal: (proposal.source_score, proposal.quality_score, proposal.score), reverse=True)
        added: list[Proposal] = []
        for proposal in candidates:
            if len(added) >= self.config.max_added_proposals:
                break
            if any(_box_iou(proposal.box, existing.box) >= self.config.nms_iou for existing in base + added):
                continue
            added.append(proposal)
        merged = base + added
        merged.sort(
            key=lambda proposal: (
                proposal.source_score if proposal.source == "memory_guided_window" else proposal.quality_score,
                proposal.quality_score,
                proposal.score,
            ),
            reverse=True,
        )
        return merged[: self.config.max_total_proposals]

    def _candidate_episodes(self, episodic_memory: EpisodicMemory, frame_index: int) -> list[EpisodicBundle]:
        bundles = []
        for bundle in episodic_memory.bundles:
            last_observed = bundle.last_observed_frame if bundle.last_observed_frame is not None else bundle.frame_end
            gap = int(frame_index) - int(last_observed)
            if not bundle.closed:
                continue
            if gap < self.config.min_episode_gap:
                continue
            bundles.append(bundle)
        bundles.sort(key=_episode_priority, reverse=True)
        return bundles[: self.config.max_memory_episodes]

    def _episode_windows(
        self,
        bundle: EpisodicBundle,
        heatmap: np.ndarray,
        encoding: SpikeEncoding,
        frame_index: int,
    ) -> list[Proposal]:
        height, width = heatmap.shape
        sizes = _episode_window_sizes(bundle, (height, width), self.config.default_window_fracs)
        peaks = _heatmap_peaks(heatmap, count=max(4, self.config.windows_per_episode), radius=max(4, min(height, width) // 10))
        centers: list[tuple[float, float]] = []
        centers.append(_context_center(bundle, (height, width)))
        centers.extend(peaks)
        proposals: list[Proposal] = []
        for box_w, box_h in sizes:
            for center in centers:
                if len(proposals) >= self.config.windows_per_episode:
                    return proposals
                box = _box_from_center(center, box_w, box_h, (height, width))
                proposal = _proposal_from_window(box, bundle, heatmap, encoding, frame_index)
                proposals.append(proposal)
        return proposals


def _proposal_from_window(
    box: Box,
    bundle: EpisodicBundle,
    heatmap: np.ndarray,
    encoding: SpikeEncoding,
    frame_index: int,
) -> Proposal:
    x1, y1, x2, y2 = box
    patch = heatmap[y1:y2, x1:x2].astype(np.float32)
    if patch.size == 0:
        patch = np.zeros((1, 1), dtype=np.float32)
    mask = patch >= (float(np.quantile(patch, 0.65)) if patch.size > 1 else float(patch.mean()))
    if int(mask.sum()) == 0:
        mask = np.ones_like(patch, dtype=bool)
    area = int(max(1, mask.sum()))
    heatmap_score = float(0.70 * patch.mean() + 0.30 * patch.max())
    appearance_score = _cosine(_appearance_signature(box, encoding), bundle.content_signature)
    shape_score = _shape_score(box, heatmap.shape, bundle.support_signature)
    closed_prior = 1.0 if bundle.closed else 0.0
    final_score = float(
        0.30 * appearance_score
        + 0.25 * shape_score
        + 0.25 * heatmap_score
        + 0.10 * float(bundle.accessibility_score)
        + 0.10 * closed_prior
    )
    fill_ratio = float(area / max(1, patch.shape[0] * patch.shape[1]))
    return Proposal(
        box=box,
        raw_box=box,
        support_box=box,
        area=area,
        raw_area=area,
        score=heatmap_score,
        quality_score=final_score,
        centroid=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
        support_mask=mask.astype(bool),
        fill_ratio=fill_ratio,
        compactness=0.0,
        boundary_smoothness=0.0,
        near_boundary=int(x1 <= 4 or y1 <= 4 or x2 >= heatmap.shape[1] - 4 or y2 >= heatmap.shape[0] - 4),
        source="memory_guided_window",
        source_score=final_score,
        metadata={
            "memory_guided_window": 1,
            "source_episode_id": int(bundle.episode_id),
            "source_track_id": "" if bundle.track_id is None else int(bundle.track_id),
            "source_prototype_id": "" if bundle.prototype_id is None else int(bundle.prototype_id),
            "source_concept_id": "" if bundle.concept_id is None else int(bundle.concept_id),
            "source_last_observed_frame": "" if bundle.last_observed_frame is None else int(bundle.last_observed_frame),
            "memory_guided_score": final_score,
            "memory_appearance_score": appearance_score,
            "memory_shape_score": shape_score,
            "memory_heatmap_score": heatmap_score,
            "memory_frame_index": int(frame_index),
        },
    )


def _episode_priority(bundle: EpisodicBundle) -> float:
    return float(
        0.42 * float(bundle.accessibility_score)
        + 0.22 * min(float(bundle.observation_count) / 12.0, 1.0)
        + 0.18 * float(bundle.replay_priority)
        + 0.10 * float(bundle.reactivation_count > 0)
        + 0.08 * float(bundle.closed)
    )


def _episode_window_sizes(
    bundle: EpisodicBundle,
    frame_shape: tuple[int, int],
    defaults: tuple[float, ...],
) -> list[tuple[int, int]]:
    height, width = frame_shape
    support = np.asarray(bundle.support_signature, dtype=np.float32).reshape(-1)
    sizes: list[tuple[int, int]] = []
    if support.size >= 2 and support[0] > 0.0 and support[1] > 0.0:
        base_w = int(round(float(support[0]) * width))
        base_h = int(round(float(support[1]) * height))
        for scale in (0.85, 1.15, 1.50):
            sizes.append((_clip_size(base_w * scale, width), _clip_size(base_h * scale, height)))
    for frac in defaults:
        size = int(round(float(frac) * min(height, width)))
        sizes.append((_clip_size(size, width), _clip_size(size, height)))
    deduped: list[tuple[int, int]] = []
    for size in sizes:
        if size not in deduped:
            deduped.append(size)
    return deduped


def _clip_size(value: float, limit: int) -> int:
    return int(max(8, min(round(value), int(limit))))


def _context_center(bundle: EpisodicBundle, frame_shape: tuple[int, int]) -> tuple[float, float]:
    height, width = frame_shape
    context = np.asarray(bundle.context_signature, dtype=np.float32).reshape(-1)
    if context.size >= 2:
        return (float(context[0]) * width, float(context[1]) * height)
    return (width * 0.5, height * 0.5)


def _heatmap_peaks(heatmap: np.ndarray, *, count: int, radius: int) -> list[tuple[float, float]]:
    if heatmap.size == 0:
        return []
    height, width = heatmap.shape
    indices = np.argsort(heatmap.reshape(-1))[::-1]
    peaks: list[tuple[float, float]] = []
    for flat in indices:
        y = int(flat // width)
        x = int(flat % width)
        if any((x - px) ** 2 + (y - py) ** 2 <= radius * radius for px, py in peaks):
            continue
        peaks.append((float(x), float(y)))
        if len(peaks) >= count:
            break
    return peaks or [(width * 0.5, height * 0.5)]


def _box_from_center(center: tuple[float, float], box_w: int, box_h: int, frame_shape: tuple[int, int]) -> Box:
    height, width = frame_shape
    cx, cy = center
    x1 = int(round(cx - box_w * 0.5))
    y1 = int(round(cy - box_h * 0.5))
    x1 = max(0, min(x1, width - box_w))
    y1 = max(0, min(y1, height - box_h))
    return (x1, y1, min(width, x1 + box_w), min(height, y1 + box_h))


def _appearance_signature(box: Box, encoding: SpikeEncoding) -> np.ndarray:
    patches = [
        _crop_2d(encoding.current_gray, box),
        _crop_2d(encoding.edge_map, box),
        _crop_2d(encoding.spike_response, box),
    ]
    stats: list[float] = []
    for patch in patches:
        if patch.size == 0:
            stats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            values = patch.reshape(-1).astype(np.float32, copy=False)
            stats.extend(
                [
                    float(np.mean(values)),
                    float(np.std(values)),
                    float(np.quantile(values, 0.25)),
                    float(np.quantile(values, 0.50)),
                    float(np.quantile(values, 0.75)),
                ]
            )
    return np.asarray(stats, dtype=np.float32)


def _shape_score(box: Box, frame_shape: tuple[int, int], support_signature: np.ndarray) -> float:
    height, width = frame_shape
    support = np.asarray(support_signature, dtype=np.float32).reshape(-1)
    if support.size < 2:
        return 0.5
    bw = max(1.0, float(box[2] - box[0])) / max(1.0, float(width))
    bh = max(1.0, float(box[3] - box[1])) / max(1.0, float(height))
    distance = abs(bw - float(support[0])) + abs(bh - float(support[1]))
    return float(np.clip(1.0 - 2.5 * distance, 0.0, 1.0))


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


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    dim = min(left.size, right.size)
    a = left.reshape(-1)[:dim].astype(np.float32, copy=False)
    b = right.reshape(-1)[:dim].astype(np.float32, copy=False)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip((float(np.dot(a, b)) / denom + 1.0) * 0.5, 0.0, 1.0))


def _box_iou(left: Box, right: Box) -> float:
    lx1, ly1, lx2, ly2 = [float(v) for v in left]
    rx1, ry1, rx2, ry2 = [float(v) for v in right]
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - inter
    return 0.0 if union <= 0.0 else float(inter / union)

