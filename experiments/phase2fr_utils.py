"""Utilities for Phase 2F-R proposal attribution and localization audit."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from datasets import SyntheticStreamGenerator, load_synth_dataset_config
from experiments.scenario_presets import build_phase3_track_scenarios
from nops_owr.encoder import MinimalSpikeEncoder
from nops_owr.objectness import MinimalObjectnessField


@dataclass(slots=True)
class RegionDescriptor:
    component_id: int
    frame_id: int
    scenario_name: str
    sequence_id: int
    box: tuple[int, int, int, int]
    region_area: int
    bbox_area: int
    bbox_aspect_ratio: float
    centroid_x: float
    centroid_y: float
    proposal_score: float
    raw_objectness_score: float
    boundary_mean: float
    persistence_mean: float
    surprise_mean: float
    habituation_mean: float
    gradient_mean: float
    region_fill_ratio: float
    region_compactness: float
    boundary_smoothness: float
    near_boundary: int
    dominant_term: str
    local_mask: np.ndarray


def load_config_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_field_config(config_payload: dict[str, Any]) -> dict[str, Any]:
    if "field" in config_payload:
        return dict(config_payload["field"])
    return dict(config_payload["model"]["objectness"])


def build_track_scenarios(config_path: str | Path) -> list[dict[str, Any]]:
    base_config = load_synth_dataset_config(config_path)
    return build_phase3_track_scenarios(base_config)


def encode_objectness_sequence(
    *,
    config_payload: dict[str, Any],
    scenario_name: str,
    scenario_config: Any,
    seed: int,
    sequence_id: int = 0,
) -> dict[str, Any]:
    sequence = SyntheticStreamGenerator(scenario_config, seed=seed).generate_sequence(sequence_id)
    encoder = MinimalSpikeEncoder(**config_payload["model"]["spike_encoder"])
    objectness = MinimalObjectnessField(**get_field_config(config_payload))
    field_config = get_field_config(config_payload)

    frame_records: list[dict[str, Any]] = []
    for frame_offset in range(1, len(sequence.frames)):
        prev_frame = sequence.frames[frame_offset - 1]
        current_frame = sequence.frames[frame_offset]
        encoding = encoder.encode(prev_frame.frame, current_frame.frame)
        objectness_output = objectness.compute(encoding)
        components = extract_region_descriptors(
            objectness_output=objectness_output,
            frame=current_frame.frame,
            frame_id=int(current_frame.frame_index),
            scenario_name=scenario_name,
            sequence_id=sequence_id,
            min_area=int(field_config.get("min_area", 0)),
            max_proposals=int(field_config.get("max_proposals", 8)),
        )
        frame_records.append(
            {
                "frame_index": int(current_frame.frame_index),
                "frame": current_frame.frame,
                "gt_boxes": list(current_frame.boxes),
                "gt_masks": list(current_frame.masks),
                "instance_ids": list(current_frame.instance_ids),
                "concept_ids": list(current_frame.concept_ids),
                "drift_strength": float(current_frame.drift_strength),
                "blur_level": float(current_frame.blur_level),
                "noise_level": float(current_frame.noise_level),
                "reentry_event": int(bool(current_frame.reentry_event)),
                "objectness_output": objectness_output,
                "components": components,
            }
        )

    return {
        "scenario_name": scenario_name,
        "sequence_id": sequence_id,
        "seed": seed,
        "sequence": sequence,
        "frame_records": frame_records,
        "field_config": field_config,
    }


def extract_region_descriptors(
    *,
    objectness_output: Any,
    frame: np.ndarray,
    frame_id: int,
    scenario_name: str,
    sequence_id: int,
    min_area: int,
    max_proposals: int,
) -> list[RegionDescriptor]:
    binary_mask = np.asarray(objectness_output.binary_mask, dtype=bool)
    visited = np.zeros_like(binary_mask, dtype=bool)
    height, width = binary_mask.shape
    gradient = frame_gradient(frame)
    descriptors: list[RegionDescriptor] = []

    for y in range(height):
        for x in range(width):
            if not binary_mask[y, x] or visited[y, x]:
                continue
            pixels = _flood_component(binary_mask, visited, y, x)
            if len(pixels) < min_area:
                continue

            ys = np.array([py for py, _ in pixels], dtype=np.int32)
            xs = np.array([px for _, px in pixels], dtype=np.int32)
            x1 = int(xs.min())
            x2 = int(xs.max()) + 1
            y1 = int(ys.min())
            y2 = int(ys.max()) + 1
            bbox_area = int(max(1, (x2 - x1) * (y2 - y1)))
            local_mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
            local_mask[ys - y1, xs - x1] = True
            perimeter = component_perimeter(local_mask)
            fill_ratio = float(len(pixels) / max(bbox_area, 1))
            compactness = float((4.0 * math.pi * len(pixels)) / max(perimeter * perimeter, 1.0))
            smoothness = float(perimeter / max(math.sqrt(len(pixels)), 1.0))
            width_extent = float(max(1, x2 - x1))
            height_extent = float(max(1, y2 - y1))
            aspect_ratio = float(max(width_extent / height_extent, height_extent / width_extent))
            near_boundary = int(x1 <= 4 or y1 <= 4 or x2 >= width - 4 or y2 >= height - 4)

            boundary_mean = float(objectness_output.boundary_term[ys, xs].mean())
            persistence_mean = float(objectness_output.persistence_term[ys, xs].mean())
            surprise_mean = float(objectness_output.surprise_term[ys, xs].mean())
            habituation_mean = float(objectness_output.habituation_response[ys, xs].mean())
            dominant_term = max(
                (
                    ("boundary", boundary_mean),
                    ("persistence", persistence_mean),
                    ("surprise", surprise_mean),
                    ("habituation", habituation_mean),
                ),
                key=lambda item: item[1],
            )[0]

            descriptors.append(
                RegionDescriptor(
                    component_id=-1,
                    frame_id=frame_id,
                    scenario_name=scenario_name,
                    sequence_id=sequence_id,
                    box=(x1, y1, x2, y2),
                    region_area=int(len(pixels)),
                    bbox_area=bbox_area,
                    bbox_aspect_ratio=aspect_ratio,
                    centroid_x=float(xs.mean()),
                    centroid_y=float(ys.mean()),
                    proposal_score=float(objectness_output.normalized_objectness[ys, xs].mean()),
                    raw_objectness_score=float(objectness_output.raw_objectness[ys, xs].mean()),
                    boundary_mean=boundary_mean,
                    persistence_mean=persistence_mean,
                    surprise_mean=surprise_mean,
                    habituation_mean=habituation_mean,
                    gradient_mean=float(gradient[ys, xs].mean()),
                    region_fill_ratio=fill_ratio,
                    region_compactness=compactness,
                    boundary_smoothness=smoothness,
                    near_boundary=near_boundary,
                    dominant_term=dominant_term,
                    local_mask=local_mask,
                )
            )

    descriptors.sort(key=lambda item: (item.proposal_score, item.region_area), reverse=True)
    top_descriptors = descriptors[:max_proposals]
    for component_id, descriptor in enumerate(top_descriptors):
        descriptor.component_id = component_id
    return top_descriptors


def frame_gradient(frame: np.ndarray) -> np.ndarray:
    gray = frame.astype(np.float32).mean(axis=2) / 255.0
    grad_y, grad_x = np.gradient(gray)
    return np.sqrt(grad_x**2 + grad_y**2, dtype=np.float32)


def component_perimeter(mask: np.ndarray) -> float:
    padded = np.pad(mask.astype(bool), ((1, 1), (1, 1)), mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    perimeter = 0
    perimeter += np.logical_and(center, ~padded[:-2, 1:-1]).sum()
    perimeter += np.logical_and(center, ~padded[2:, 1:-1]).sum()
    perimeter += np.logical_and(center, ~padded[1:-1, :-2]).sum()
    perimeter += np.logical_and(center, ~padded[1:-1, 2:]).sum()
    return float(perimeter)


def _flood_component(binary_mask: np.ndarray, visited: np.ndarray, start_y: int, start_x: int) -> list[tuple[int, int]]:
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


def box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = max(1, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection)
    return float(intersection / union)


def mask_iou(local_mask_a: np.ndarray, box_a: tuple[int, int, int, int], local_mask_b: np.ndarray, box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    crop_a = local_mask_a[inter_y1 - ay1 : inter_y2 - ay1, inter_x1 - ax1 : inter_x2 - ax1]
    crop_b = local_mask_b[inter_y1 - by1 : inter_y2 - by1, inter_x1 - bx1 : inter_x2 - bx1]
    intersection = np.logical_and(crop_a, crop_b).sum()
    union = np.logical_or(crop_a, crop_b).sum()
    return 0.0 if union <= 0 else float(intersection / union)


def serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            serialized[key] = value.item()
        elif isinstance(value, (tuple, list)):
            serialized[key] = json.dumps(value)
        else:
            serialized[key] = value
    return serialized


def copy_config(config: Any) -> Any:
    return copy.deepcopy(config)
