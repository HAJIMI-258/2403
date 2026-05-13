"""Continuous objectness field and proposal extraction for Phase 2A."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nops_owr.encoder.spike_encoder import SpikeEncoding

Box = tuple[int, int, int, int]


@dataclass(slots=True)
class Proposal:
    box: Box
    raw_box: Box
    support_box: Box
    area: int
    raw_area: int
    score: float
    quality_score: float
    centroid: tuple[float, float]
    support_mask: np.ndarray
    fill_ratio: float
    compactness: float
    boundary_smoothness: float
    near_boundary: int
    source: str = "component"
    source_score: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ObjectnessOutput:
    activation_map: np.ndarray
    boundary_term: np.ndarray
    persistence_term: np.ndarray
    temporal_term: np.ndarray
    surprise_term: np.ndarray
    habituation_map: np.ndarray
    habituation_term: np.ndarray
    habituation_response: np.ndarray
    residual_term: np.ndarray
    raw_objectness: np.ndarray
    normalized_objectness: np.ndarray
    heatmap: np.ndarray
    binary_mask: np.ndarray
    threshold_map: np.ndarray
    threshold: float
    proposals: list[Proposal]


class MinimalObjectnessField:
    """Continuous boundary/persistence/surprise/habituation objectness field."""

    def __init__(
        self,
        wb: float = 0.45,
        wt: float = 0.20,
        wq: float = 0.15,
        wr: float = 0.20,
        tau_obj: float = 0.58,
        ema_eta: float = 0.82,
        hab_rho: float = 0.94,
        hab_lambda: float = 0.85,
        threshold_mode: str = "fixed",
        q_obj: float = 0.82,
        local_k: float = 0.80,
        smoothing_kernel_size: int = 5,
        min_area: int = 96,
        max_proposals: int = 8,
        saliency_window_count: int = 0,
        saliency_window_fracs: tuple[float, ...] | None = None,
        saliency_nms_iou: float = 0.65,
    ) -> None:
        self.wb = float(wb)
        self.wt = float(wt)
        self.wq = float(wq)
        self.wr = float(wr)
        self.tau_obj = float(tau_obj)
        self.ema_eta = float(ema_eta)
        self.hab_rho = float(hab_rho)
        self.hab_lambda = float(hab_lambda)
        self.threshold_mode = str(threshold_mode)
        self.q_obj = float(q_obj)
        self.local_k = float(local_k)
        self.smoothing_kernel_size = int(max(1, smoothing_kernel_size))
        self.min_area = int(min_area)
        self.max_proposals = int(max_proposals)
        self.saliency_window_count = int(max(0, saliency_window_count))
        self.saliency_window_fracs = tuple(float(v) for v in (saliency_window_fracs or (0.18, 0.28)))
        self.saliency_nms_iou = float(np.clip(saliency_nms_iou, 0.0, 1.0))
        self._activation_memory: np.ndarray | None = None
        self._surprise_reference: np.ndarray | None = None
        self._habituation_memory: np.ndarray | None = None

    def reset(self) -> None:
        self._activation_memory = None
        self._surprise_reference = None
        self._habituation_memory = None

    def compute(self, encoding: SpikeEncoding) -> ObjectnessOutput:
        spike_sum = np.clip(0.5 * (encoding.on_spikes + encoding.off_spikes), 0.0, 1.0)
        diff_energy = _normalize(np.abs(encoding.frame_diff))
        spike_drive = _normalize(
            0.45 * encoding.spike_response + 0.25 * diff_energy + 0.20 * spike_sum + 0.10 * encoding.edge_map
        )

        if self._activation_memory is None:
            activation_map = spike_drive.copy()
        else:
            activation_map = (
                self.ema_eta * self._activation_memory + (1.0 - self.ema_eta) * spike_drive
            ).astype(np.float32)
        self._activation_memory = activation_map

        boundary_seed = (
            0.65 * _normalize(encoding.edge_map * (0.25 + 0.75 * activation_map))
            + 0.35 * _normalize(_gradient_magnitude(activation_map))
        )
        boundary_term = _normalize(_box_blur(boundary_seed, self.smoothing_kernel_size))

        persistence_term = _normalize(activation_map)

        if self._surprise_reference is None:
            self._surprise_reference = spike_drive.copy()
        surprise_term = _normalize(np.abs(spike_drive - self._surprise_reference))
        self._surprise_reference = (
            self.ema_eta * self._surprise_reference + (1.0 - self.ema_eta) * spike_drive
        ).astype(np.float32)

        if self._habituation_memory is None:
            self._habituation_memory = np.zeros_like(activation_map, dtype=np.float32)
        habituation_map = self._habituation_memory.copy()
        habituation_response = _normalize(
            np.clip(activation_map - self.hab_lambda * habituation_map, 0.0, None)
        )
        self._habituation_memory = (
            self.hab_rho * self._habituation_memory + (1.0 - self.hab_rho) * activation_map
        ).astype(np.float32)

        raw_objectness = (
            self.wb * boundary_term
            + self.wt * persistence_term
            + self.wq * surprise_term
            + self.wr * habituation_response
        ).astype(np.float32)
        normalized_objectness = _normalize(_box_blur(raw_objectness, self.smoothing_kernel_size))

        threshold_map, threshold_value = self._build_threshold(normalized_objectness)
        binary_mask = normalized_objectness >= threshold_map
        proposals = _extract_proposals(
            binary_mask=binary_mask,
            score_map=normalized_objectness,
            min_area=self.min_area,
            max_proposals=self.max_proposals,
        )
        if self.saliency_window_count > 0:
            proposals = _append_saliency_window_proposals(
                proposals=proposals,
                score_map=normalized_objectness,
                window_fracs=self.saliency_window_fracs,
                window_count=self.saliency_window_count,
                max_proposals=self.max_proposals,
                nms_iou=self.saliency_nms_iou,
            )

        return ObjectnessOutput(
            activation_map=activation_map,
            boundary_term=boundary_term,
            persistence_term=persistence_term,
            temporal_term=persistence_term,
            surprise_term=surprise_term,
            habituation_map=habituation_map,
            habituation_term=habituation_map,
            habituation_response=habituation_response,
            residual_term=habituation_response,
            raw_objectness=raw_objectness,
            normalized_objectness=normalized_objectness,
            heatmap=normalized_objectness,
            binary_mask=binary_mask,
            threshold_map=threshold_map,
            threshold=threshold_value,
            proposals=proposals,
        )

    def _build_threshold(self, normalized_objectness: np.ndarray) -> tuple[np.ndarray, float]:
        if self.threshold_mode == "fixed":
            threshold_map = np.full_like(normalized_objectness, self.tau_obj, dtype=np.float32)
            return threshold_map, self.tau_obj

        if self.threshold_mode == "quantile_local":
            window_size = max(7, self.smoothing_kernel_size * 3)
            local_mean = _box_blur(normalized_objectness, window_size)
            local_sq_mean = _box_blur(normalized_objectness**2, window_size)
            local_var = np.maximum(local_sq_mean - local_mean**2, 0.0)
            local_std = np.sqrt(local_var, dtype=np.float32)
            quantile_threshold = float(np.quantile(normalized_objectness, self.q_obj))
            threshold_map = np.maximum(quantile_threshold, local_mean + self.local_k * local_std).astype(np.float32)
            return threshold_map, float(threshold_map.mean())

        raise ValueError(f"Unsupported threshold_mode: {self.threshold_mode}")


def _extract_proposals(
    binary_mask: np.ndarray,
    score_map: np.ndarray,
    min_area: int,
    max_proposals: int,
) -> list[Proposal]:
    visited = np.zeros_like(binary_mask, dtype=bool)
    height, width = binary_mask.shape
    proposals: list[Proposal] = []

    for y in range(height):
        for x in range(width):
            if not binary_mask[y, x] or visited[y, x]:
                continue

            queue = [(y, x)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []

            while queue:
                cy, cx = queue.pop()
                pixels.append((cy, cx))

                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if visited[ny, nx] or not binary_mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if len(pixels) < min_area:
                continue

            ys = np.array([py for py, _ in pixels], dtype=np.int32)
            xs = np.array([px for _, px in pixels], dtype=np.int32)
            raw_box = (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            )
            refined = _build_refined_proposal(
                raw_box=raw_box,
                ys=ys,
                xs=xs,
                score_map=score_map,
                frame_shape=binary_mask.shape,
            )
            if refined is None:
                continue

            proposals.append(
                Proposal(
                    box=refined["box"],
                    raw_box=raw_box,
                    support_box=refined["support_box"],
                    area=int(refined["area"]),
                    raw_area=len(pixels),
                    score=float(refined["score"]),
                    quality_score=float(refined["quality_score"]),
                    centroid=refined["centroid"],
                    support_mask=refined["support_mask"],
                    fill_ratio=float(refined["fill_ratio"]),
                    compactness=float(refined["compactness"]),
                    boundary_smoothness=float(refined["boundary_smoothness"]),
                    near_boundary=int(refined["near_boundary"]),
                    source="component",
                    source_score=float(refined["score"]),
                    metadata={"component_proposal": 1},
                )
            )

    proposals.sort(
        key=lambda proposal: (
            proposal.score + 0.35 * (proposal.quality_score - proposal.score),
            proposal.quality_score,
            proposal.area,
        ),
        reverse=True,
    )
    return proposals[:max_proposals]


def _append_saliency_window_proposals(
    *,
    proposals: list[Proposal],
    score_map: np.ndarray,
    window_fracs: tuple[float, ...],
    window_count: int,
    max_proposals: int,
    nms_iou: float,
) -> list[Proposal]:
    """Add GT-free heatmap window proposals for fragmented real-video targets.

    Connected components are still the primary proposal source. These windows
    are an optional external-eval recall profile for cases where the target
    only leaves fragmented edge/surprise evidence and no component covers the
    object well.
    """

    if window_count <= 0 or max_proposals <= 0:
        return proposals[:max_proposals]
    height, width = score_map.shape
    candidates: list[Proposal] = []
    for frac in window_fracs:
        window = int(round(float(frac) * min(height, width)))
        window = max(8, min(window, height, width))
        stride = max(4, window // 2)
        for y1 in range(0, max(1, height - window + 1), stride):
            for x1 in range(0, max(1, width - window + 1), stride):
                x2 = min(width, x1 + window)
                y2 = min(height, y1 + window)
                if x2 <= x1 or y2 <= y1:
                    continue
                candidates.append(_saliency_window_proposal((x1, y1, x2, y2), score_map))
        # Make sure the right/bottom border can still generate a candidate.
        if height > window or width > window:
            for y1 in {max(0, height - window)}:
                for x1 in {max(0, width - window)}:
                    candidates.append(_saliency_window_proposal((x1, y1, min(width, x1 + window), min(height, y1 + window)), score_map))

    candidates.sort(key=lambda proposal: (proposal.quality_score, proposal.score, proposal.area), reverse=True)
    merged = list(proposals)
    added = 0
    for candidate in candidates:
        if added >= int(window_count):
            break
        if any(_box_iou(candidate.box, existing.box) >= nms_iou for existing in merged):
            continue
        merged.append(candidate)
        added += 1
    merged.sort(
        key=lambda proposal: (
            proposal.score + 0.35 * (proposal.quality_score - proposal.score),
            proposal.quality_score,
            proposal.area,
        ),
        reverse=True,
    )
    return merged[:max_proposals]


def _saliency_window_proposal(box: Box, score_map: np.ndarray) -> Proposal:
    x1, y1, x2, y2 = box
    patch = score_map[y1:y2, x1:x2].astype(np.float32)
    if patch.size == 0:
        patch = np.zeros((1, 1), dtype=np.float32)
    threshold = float(np.quantile(patch, 0.65)) if patch.size > 1 else float(patch.mean())
    support_mask = patch >= threshold
    if int(support_mask.sum()) == 0:
        support_mask = np.ones_like(patch, dtype=bool)
    area = int(max(1, support_mask.sum()))
    score = float(0.70 * patch.mean() + 0.30 * patch.max())
    fill_ratio = float(area / max(1, patch.shape[0] * patch.shape[1]))
    compactness, boundary_smoothness = _region_shape_metrics(support_mask)
    near_boundary = int(x1 <= 4 or y1 <= 4 or x2 >= score_map.shape[1] - 4 or y2 >= score_map.shape[0] - 4)
    quality_score = _proposal_quality_score(
        score=score,
        fill_ratio=fill_ratio,
        compactness=compactness,
        aspect_ratio=_box_aspect_ratio(box),
        near_boundary=near_boundary,
    )
    return Proposal(
        box=box,
        raw_box=box,
        support_box=box,
        area=area,
        raw_area=area,
        score=score,
        quality_score=quality_score,
        centroid=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
        support_mask=support_mask.astype(bool),
        fill_ratio=fill_ratio,
        compactness=compactness,
        boundary_smoothness=boundary_smoothness,
        near_boundary=near_boundary,
        source="saliency_window",
        source_score=score,
        metadata={"window_proposal": 1},
    )


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


def _build_refined_proposal(
    *,
    raw_box: Box,
    ys: np.ndarray,
    xs: np.ndarray,
    score_map: np.ndarray,
    frame_shape: tuple[int, int],
) -> dict[str, object] | None:
    x1, y1, x2, y2 = raw_box
    height, width = frame_shape
    px1, py1, px2, py2 = x1, y1, x2, y2
    score_patch = score_map[py1:py2, px1:px2].astype(np.float32)
    seed_mask = np.zeros_like(score_patch, dtype=bool)
    seed_mask[ys - py1, xs - px1] = True
    if int(seed_mask.sum()) == 0:
        return None

    refined_mask = _refine_support_mask(seed_mask, score_patch)
    if int(refined_mask.sum()) == 0:
        refined_mask = seed_mask.copy()

    support_box = (px1, py1, px2, py2)
    refined_box = _mask_to_refined_box(refined_mask, score_patch, support_box, raw_box, frame_shape)
    local_ys, local_xs = np.nonzero(refined_mask)
    global_xs = local_xs + px1
    global_ys = local_ys + py1
    component_scores = score_patch[refined_mask]
    centroid = (float(global_xs.mean()), float(global_ys.mean()))
    bbox_area = max(1, (refined_box[2] - refined_box[0]) * (refined_box[3] - refined_box[1]))
    fill_ratio = float(refined_mask.sum() / bbox_area)
    compactness, boundary_smoothness = _region_shape_metrics(refined_mask)
    near_boundary = int(
        refined_box[0] <= 4
        or refined_box[1] <= 4
        or refined_box[2] >= width - 4
        or refined_box[3] >= height - 4
    )
    aspect_ratio = _box_aspect_ratio(refined_box)
    quality_score = _proposal_quality_score(
        score=float(component_scores.mean()),
        fill_ratio=fill_ratio,
        compactness=compactness,
        aspect_ratio=aspect_ratio,
        near_boundary=near_boundary,
    )
    return {
        "box": refined_box,
        "support_box": support_box,
        "support_mask": refined_mask,
        "area": int(refined_mask.sum()),
        "score": float(component_scores.mean()),
        "quality_score": quality_score,
        "centroid": centroid,
        "fill_ratio": fill_ratio,
        "compactness": compactness,
        "boundary_smoothness": boundary_smoothness,
        "near_boundary": near_boundary,
    }


def _refine_support_mask(mask: np.ndarray, score_patch: np.ndarray) -> np.ndarray:
    area = int(mask.sum())
    if area <= 0:
        return mask.astype(bool)

    bbox_area = max(1, mask.shape[0] * mask.shape[1])
    refined = mask.astype(bool).copy()
    if area >= 8:
        refined = _binary_closing(refined, radius=1)
    if area >= 24 and max(mask.shape[0], mask.shape[1]) >= 36:
        refined = _binary_closing(refined, radius=2)

    # Preserve the original support while repairing gaps and filling object interior.
    refined |= mask
    refined = _fill_small_holes(refined, max_hole_area=max(8, int(0.60 * bbox_area)))
    refined = _largest_connected_component(refined, reference_mask=mask)
    refined |= mask

    # When the component is still edge-like, thicken it inside the existing support box
    # instead of shrinking the final box around a thin boundary trace.
    if float(refined.sum()) / bbox_area < 0.18:
        refined = _binary_dilate(refined, radius=1)
        refined = _largest_connected_component(refined | mask, reference_mask=mask)
        refined |= mask

    if int(refined.sum()) == 0:
        return mask.astype(bool)
    return refined.astype(bool)


def _masked_seed_fill(candidate_mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    if not candidate_mask.any() or not seed_mask.any():
        return np.zeros_like(candidate_mask, dtype=bool)

    height, width = candidate_mask.shape
    result = np.zeros_like(candidate_mask, dtype=bool)
    seed_points = list(zip(*np.nonzero(seed_mask & candidate_mask)))
    if not seed_points:
        seed_points = list(zip(*np.nonzero(seed_mask)))
    queue = [(int(y), int(x)) for y, x in seed_points]
    for y, x in queue:
        if 0 <= y < height and 0 <= x < width and candidate_mask[y, x]:
            result[y, x] = True

    while queue:
        cy, cx = queue.pop()
        for ny in range(max(0, cy - 1), min(height, cy + 2)):
            for nx in range(max(0, cx - 1), min(width, cx + 2)):
                if result[ny, nx] or not candidate_mask[ny, nx]:
                    continue
                result[ny, nx] = True
                queue.append((ny, nx))
    return result


def _binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0:
        return mask.astype(bool)
    padded = np.pad(mask.astype(bool), ((radius, radius), (radius, radius)), mode="constant", constant_values=False)
    output = np.zeros_like(mask, dtype=bool)
    kernel_size = 2 * radius + 1
    for iy in range(kernel_size):
        for ix in range(kernel_size):
            output |= padded[iy : iy + mask.shape[0], ix : ix + mask.shape[1]]
    return output


def _binary_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0:
        return mask.astype(bool)
    padded = np.pad(mask.astype(bool), ((radius, radius), (radius, radius)), mode="constant", constant_values=True)
    output = np.ones_like(mask, dtype=bool)
    kernel_size = 2 * radius + 1
    for iy in range(kernel_size):
        for ix in range(kernel_size):
            output &= padded[iy : iy + mask.shape[0], ix : ix + mask.shape[1]]
    return output


def _binary_opening(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return _binary_dilate(_binary_erode(mask, radius=radius), radius=radius)


def _binary_closing(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return _binary_erode(_binary_dilate(mask, radius=radius), radius=radius)


def _fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    if max_hole_area <= 0:
        return mask.astype(bool)
    inverse = ~mask.astype(bool)
    height, width = inverse.shape
    visited = np.zeros_like(inverse, dtype=bool)
    queue: list[tuple[int, int]] = []

    def _push(y: int, x: int) -> None:
        if 0 <= y < height and 0 <= x < width and inverse[y, x] and not visited[y, x]:
            visited[y, x] = True
            queue.append((y, x))

    for x in range(width):
        _push(0, x)
        _push(height - 1, x)
    for y in range(height):
        _push(y, 0)
        _push(y, width - 1)

    while queue:
        cy, cx = queue.pop()
        for ny in range(max(0, cy - 1), min(height, cy + 2)):
            for nx in range(max(0, cx - 1), min(width, cx + 2)):
                if inverse[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    holes = inverse & ~visited
    if not holes.any():
        return mask.astype(bool)

    hole_visited = np.zeros_like(holes, dtype=bool)
    filled = mask.astype(bool).copy()
    for y in range(height):
        for x in range(width):
            if not holes[y, x] or hole_visited[y, x]:
                continue
            component = _flood_component(holes, hole_visited, y, x)
            if len(component) <= max_hole_area:
                ys = np.array([py for py, _ in component], dtype=np.int32)
                xs = np.array([px for _, px in component], dtype=np.int32)
                filled[ys, xs] = True
    return filled


def _mask_to_refined_box(
    mask: np.ndarray,
    score_patch: np.ndarray,
    support_box: Box,
    raw_box: Box,
    frame_shape: tuple[int, int],
) -> Box:
    px1, py1, _, _ = support_box
    local_ys, local_xs = np.nonzero(mask)
    if local_xs.size == 0:
        return raw_box

    tight_box = (
        int(px1 + local_xs.min()),
        int(py1 + local_ys.min()),
        int(px1 + local_xs.max()) + 1,
        int(py1 + local_ys.max()) + 1,
    )

    raw_width = max(1, raw_box[2] - raw_box[0])
    raw_height = max(1, raw_box[3] - raw_box[1])
    max_trim_x = max(1, int(round(0.08 * raw_width)))
    max_trim_y = max(1, int(round(0.08 * raw_height)))
    refined_box = (
        max(raw_box[0], min(raw_box[0] + max_trim_x, tight_box[0])),
        max(raw_box[1], min(raw_box[1] + max_trim_y, tight_box[1])),
        min(raw_box[2], max(raw_box[2] - max_trim_x, tight_box[2])),
        min(raw_box[3], max(raw_box[3] - max_trim_y, tight_box[3])),
    )
    x1, y1, x2, y2 = refined_box
    x1 = max(0, min(x1, frame_shape[1] - 1))
    y1 = max(0, min(y1, frame_shape[0] - 1))
    x2 = max(x1 + 1, min(x2, frame_shape[1]))
    y2 = max(y1 + 1, min(y2, frame_shape[0]))
    return (x1, y1, x2, y2)


def _blend_boxes(refined_box: Box, raw_box: Box, alpha: float, frame_shape: tuple[int, int]) -> Box:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    x1 = int(round(alpha * refined_box[0] + (1.0 - alpha) * raw_box[0]))
    y1 = int(round(alpha * refined_box[1] + (1.0 - alpha) * raw_box[1]))
    x2 = int(round(alpha * refined_box[2] + (1.0 - alpha) * raw_box[2]))
    y2 = int(round(alpha * refined_box[3] + (1.0 - alpha) * raw_box[3]))
    x1 = max(0, min(x1, frame_shape[1] - 1))
    y1 = max(0, min(y1, frame_shape[0] - 1))
    x2 = max(x1 + 1, min(x2, frame_shape[1]))
    y2 = max(y1 + 1, min(y2, frame_shape[0]))
    return (x1, y1, x2, y2)


def _region_shape_metrics(mask: np.ndarray) -> tuple[float, float]:
    area = int(mask.sum())
    if area <= 0:
        return 0.0, 0.0
    perimeter = component_perimeter(mask)
    compactness = float((4.0 * np.pi * area) / max(perimeter * perimeter, 1.0))
    boundary_smoothness = float(perimeter / max(np.sqrt(area), 1.0))
    return compactness, boundary_smoothness


def _box_aspect_ratio(box: Box) -> float:
    width = max(1, box[2] - box[0])
    height = max(1, box[3] - box[1])
    return float(max(width / height, height / width))


def _proposal_quality_score(
    *,
    score: float,
    fill_ratio: float,
    compactness: float,
    aspect_ratio: float,
    near_boundary: int,
) -> float:
    aspect_penalty = max(0.0, min(1.0, (aspect_ratio - 1.8) / 3.0))
    boundary_penalty = 1.0 if near_boundary else 0.0
    return float(
        score
        + 0.12 * np.clip(compactness, 0.0, 1.0)
        + 0.10 * np.clip(fill_ratio, 0.0, 1.0)
        - 0.08 * boundary_penalty
        - 0.06 * aspect_penalty
    )


def _flood_component(
    binary_mask: np.ndarray,
    visited: np.ndarray,
    start_y: int,
    start_x: int,
) -> list[tuple[int, int]]:
    height, width = binary_mask.shape
    queue = [(start_y, start_x)]
    visited[start_y, start_x] = True
    pixels: list[tuple[int, int]] = []
    while queue:
        cy, cx = queue.pop()
        pixels.append((cy, cx))
        for ny in range(max(0, cy - 1), min(height, cy + 2)):
            for nx in range(max(0, cx - 1), min(width, cx + 2)):
                if visited[ny, nx] or not binary_mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))
    return pixels


def _largest_connected_component(mask: np.ndarray, reference_mask: np.ndarray | None = None) -> np.ndarray:
    binary_mask = mask.astype(bool)
    if not binary_mask.any():
        return binary_mask

    visited = np.zeros_like(binary_mask, dtype=bool)
    best_component: list[tuple[int, int]] | None = None
    best_score = -1.0
    reference_mask = None if reference_mask is None else reference_mask.astype(bool)

    for y in range(binary_mask.shape[0]):
        for x in range(binary_mask.shape[1]):
            if visited[y, x] or not binary_mask[y, x]:
                continue
            component = _flood_component(binary_mask, visited, y, x)
            overlap = 0
            if reference_mask is not None:
                overlap = sum(1 for py, px in component if reference_mask[py, px])
            score = float(overlap * 10 + len(component))
            if score > best_score:
                best_score = score
                best_component = component

    if not best_component:
        return binary_mask

    output = np.zeros_like(binary_mask, dtype=bool)
    ys = np.array([py for py, _ in best_component], dtype=np.int32)
    xs = np.array([px for _, px in best_component], dtype=np.int32)
    output[ys, xs] = True
    return output


def component_perimeter(mask: np.ndarray) -> float:
    padded = np.pad(mask.astype(bool), ((1, 1), (1, 1)), mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    perimeter = 0
    perimeter += np.logical_and(center, ~padded[:-2, 1:-1]).sum()
    perimeter += np.logical_and(center, ~padded[2:, 1:-1]).sum()
    perimeter += np.logical_and(center, ~padded[1:-1, :-2]).sum()
    perimeter += np.logical_and(center, ~padded[1:-1, 2:]).sum()
    return float(perimeter)


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
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
    return np.sqrt(grad_x**2 + grad_y**2, dtype=np.float32)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(image.astype(np.float32), ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)

    for iy in range(kernel.shape[0]):
        for ix in range(kernel.shape[1]):
            output += kernel[iy, ix] * padded[iy : iy + image.shape[0], ix : ix + image.shape[1]]

    return output


def _box_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    if kernel_size == 1:
        return image.astype(np.float32)

    kernel = np.full((kernel_size, kernel_size), 1.0 / (kernel_size * kernel_size), dtype=np.float32)
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(image.astype(np.float32), ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
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
