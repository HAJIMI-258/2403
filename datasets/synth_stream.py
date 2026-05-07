"""Synthetic streaming dataset for Phase 1 of NOPS-OWR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

Box = tuple[int, int, int, int]


@dataclass(slots=True)
class BackgroundDriftConfig:
    enabled: bool
    brightness_amplitude: float
    texture_noise_std: float


@dataclass(slots=True)
class AppearancePerturbationConfig:
    enabled: bool
    scale_jitter: float
    intensity_jitter: float
    edge_blur_probability: float


@dataclass(slots=True)
class BridgeSyntheticConfig:
    enabled: bool = False
    difficulty_preset: str = "simple"
    background_repeat_density: float = 0.0
    background_texture_strength: float = 0.0
    illumination_drift_strength: float = 0.0
    local_noise_std: float = 0.0
    local_blur_probability: float = 0.0
    camera_jitter_std: float = 0.0
    occlusion_duration_range: tuple[int, int] = (16, 32)
    reentry_gap_range: tuple[int, int] = (12, 24)
    crossing_probability: float = 0.0
    target_deformation_strength: float = 0.0
    low_contrast_probability: float = 0.0


@dataclass(slots=True)
class SynthDatasetConfig:
    name: str
    resolution: tuple[int, int]
    sequence_length: int
    num_sequences: int
    num_objects_range: tuple[int, int]
    shapes: tuple[str, ...]
    object_scale_range: tuple[int, int]
    velocity_range: tuple[float, float]
    spawn_margin: int
    occlusion_probability: float
    reentry_probability: float
    background_drift: BackgroundDriftConfig
    appearance_perturbation: AppearancePerturbationConfig
    bridge_synthetic: BridgeSyntheticConfig
    outputs: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict) -> "SynthDatasetConfig":
        dataset = payload["dataset"]
        bridge_payload = dataset.get("bridge_synthetic", {})
        return cls(
            name=dataset["name"],
            resolution=tuple(dataset["resolution"]),
            sequence_length=dataset["sequence_length"],
            num_sequences=dataset["num_sequences"],
            num_objects_range=tuple(dataset["num_objects_range"]),
            shapes=tuple(dataset["shapes"]),
            object_scale_range=tuple(dataset["object_scale_range"]),
            velocity_range=tuple(dataset["velocity_range"]),
            spawn_margin=dataset["spawn_margin"],
            occlusion_probability=dataset["occlusion_probability"],
            reentry_probability=dataset["reentry_probability"],
            background_drift=BackgroundDriftConfig(**dataset["background_drift"]),
            appearance_perturbation=AppearancePerturbationConfig(**dataset["appearance_perturbation"]),
            bridge_synthetic=BridgeSyntheticConfig(
                occlusion_duration_range=tuple(bridge_payload.get("occlusion_duration_range", (16, 32))),
                reentry_gap_range=tuple(bridge_payload.get("reentry_gap_range", (12, 24))),
                **{key: value for key, value in bridge_payload.items() if key not in {"occlusion_duration_range", "reentry_gap_range"}},
            ),
            outputs=tuple(dataset.get("outputs", ("frame", "boxes", "masks", "instance_id", "concept_id"))),
        )


@dataclass(slots=True)
class FrameSample:
    frame_index: int
    frame: np.ndarray
    boxes: list[Box]
    masks: list[np.ndarray]
    instance_ids: list[int]
    concept_ids: list[int]
    visibility_flags: list[bool] = field(default_factory=list)
    occlusion_ratio: float = 0.0
    drift_strength: float = 0.0
    blur_level: float = 0.0
    noise_level: float = 0.0
    reentry_event: bool = False


@dataclass(slots=True)
class SequenceSample:
    sequence_id: int
    frames: list[FrameSample]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class _ObjectState:
    instance_id: int
    concept_id: int
    shape: str
    center: np.ndarray
    velocity: np.ndarray
    base_scale: float
    base_intensity: float
    active: bool
    exit_frame: int | None
    return_frame: int | None
    exit_edge: str | None
    return_edge: str | None
    contrast_gain: float
    deformation_phase: float


@dataclass(slots=True)
class _OcclusionEvent:
    pair: tuple[int, int]
    start: int
    end: int
    target: np.ndarray


class SyntheticStreamGenerator:
    """Generate Phase 1 synthetic streaming sequences."""

    def __init__(self, config: SynthDatasetConfig, seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        self._concept_lookup = {shape: idx for idx, shape in enumerate(config.shapes)}
        self._color_lookup = {
            "circle": np.array([240, 110, 100], dtype=np.float32),
            "square": np.array([100, 190, 245], dtype=np.float32),
            "triangle": np.array([245, 210, 95], dtype=np.float32),
        }

    def generate_sequence(self, sequence_id: int = 0) -> SequenceSample:
        rng = np.random.default_rng(self.seed + sequence_id)
        objects = self._initialize_objects(rng)
        occlusion_event = self._sample_occlusion_event(rng, len(objects))

        frames: list[FrameSample] = []
        for frame_idx in range(self.config.sequence_length):
            returned_ids = self._update_objects(objects, rng, frame_idx, occlusion_event)
            camera_jitter = self._sample_camera_jitter(rng)
            frame, drift_strength = self._build_background(rng, frame_idx, camera_jitter)
            frame_sample = self._render_frame(
                frame,
                objects,
                rng,
                frame_idx,
                camera_jitter,
                drift_strength,
                reentry_event=bool(returned_ids),
            )
            frames.append(frame_sample)

        metadata = {
            "difficulty_preset": self.config.bridge_synthetic.difficulty_preset,
            "active_concept_count": len({obj.concept_id for obj in objects}),
            "planned_reentry_events": sum(int(obj.return_frame is not None) for obj in objects),
            "planned_long_occlusion_events": int(
                occlusion_event is not None
                and (occlusion_event.end - occlusion_event.start) >= max(24, self.config.sequence_length // 18)
            ),
        }
        return SequenceSample(sequence_id=sequence_id, frames=frames, metadata=metadata)

    def iter_sequences(self) -> Iterable[SequenceSample]:
        for sequence_id in range(self.config.num_sequences):
            yield self.generate_sequence(sequence_id)

    def _initialize_objects(self, rng: np.random.Generator) -> list[_ObjectState]:
        height, width = self.config.resolution
        num_objects = int(rng.integers(self.config.num_objects_range[0], self.config.num_objects_range[1] + 1))
        objects: list[_ObjectState] = []

        for instance_id in range(num_objects):
            shape = self.config.shapes[instance_id % len(self.config.shapes)]
            concept_id = self._concept_lookup[shape]
            center = np.array(
                [
                    rng.uniform(self.config.spawn_margin, width - self.config.spawn_margin),
                    rng.uniform(self.config.spawn_margin, height - self.config.spawn_margin),
                ],
                dtype=np.float32,
            )
            velocity = self._sample_velocity(rng)
            base_scale = float(rng.uniform(*self.config.object_scale_range))
            base_intensity = float(rng.uniform(0.85, 1.05))
            exit_frame, return_frame, exit_edge, return_edge = self._sample_reentry_plan(rng)
            contrast_gain = (
                float(rng.uniform(0.25, 0.55))
                if self.config.bridge_synthetic.enabled and rng.random() < self.config.bridge_synthetic.low_contrast_probability
                else 1.0
            )
            deformation_phase = float(rng.uniform(0.0, 2.0 * np.pi))

            objects.append(
                _ObjectState(
                    instance_id=instance_id,
                    concept_id=concept_id,
                    shape=shape,
                    center=center,
                    velocity=velocity,
                    base_scale=base_scale,
                    base_intensity=base_intensity,
                    active=True,
                    exit_frame=exit_frame,
                    return_frame=return_frame,
                    exit_edge=exit_edge,
                    return_edge=return_edge,
                    contrast_gain=contrast_gain,
                    deformation_phase=deformation_phase,
                )
            )

        return objects

    def _sample_reentry_plan(
        self, rng: np.random.Generator
    ) -> tuple[int | None, int | None, str | None, str | None]:
        if rng.random() > self.config.reentry_probability:
            return None, None, None, None

        leave_frame = int(rng.integers(max(12, self.config.sequence_length // 5), max(18, self.config.sequence_length // 2)))
        if self.config.bridge_synthetic.enabled:
            gap_low, gap_high = self.config.bridge_synthetic.reentry_gap_range
            hidden_duration = int(rng.integers(gap_low, max(gap_low + 1, gap_high + 1)))
        else:
            hidden_duration = int(rng.integers(12, max(18, self.config.sequence_length // 6)))
        return_frame = min(self.config.sequence_length - 1, leave_frame + hidden_duration)
        edges = ("left", "right", "top", "bottom")
        exit_edge = str(rng.choice(edges))
        return_edge = str(rng.choice(edges))
        return leave_frame, return_frame, exit_edge, return_edge

    def _sample_occlusion_event(
        self, rng: np.random.Generator, num_objects: int
    ) -> _OcclusionEvent | None:
        occlusion_gate = max(self.config.occlusion_probability, self.config.bridge_synthetic.crossing_probability)
        if num_objects < 2 or rng.random() > occlusion_gate:
            return None

        pair = tuple(int(idx) for idx in rng.choice(num_objects, size=2, replace=False))
        start = int(rng.integers(self.config.sequence_length // 4, max(self.config.sequence_length // 3 + 1, self.config.sequence_length // 2)))
        if self.config.bridge_synthetic.enabled:
            duration_low, duration_high = self.config.bridge_synthetic.occlusion_duration_range
            duration = int(rng.integers(duration_low, max(duration_low + 1, duration_high + 1)))
        else:
            duration = int(rng.integers(16, 32))
        end = min(self.config.sequence_length - 1, start + duration)
        height, width = self.config.resolution
        target = np.array(
            [rng.uniform(0.35 * width, 0.65 * width), rng.uniform(0.35 * height, 0.65 * height)],
            dtype=np.float32,
        )
        return _OcclusionEvent(pair=pair, start=start, end=end, target=target)

    def _update_objects(
        self,
        objects: list[_ObjectState],
        rng: np.random.Generator,
        frame_idx: int,
        occlusion_event: _OcclusionEvent | None,
    ) -> list[int]:
        height, width = self.config.resolution
        margin = self.config.spawn_margin
        returned_ids: list[int] = []

        for obj in objects:
            if obj.return_frame is not None and frame_idx == obj.return_frame:
                obj.center, obj.velocity = self._spawn_from_edge(rng, obj.return_edge or "left")
                obj.active = True
                returned_ids.append(obj.instance_id)

            if not obj.active:
                continue

            velocity = obj.velocity.copy()
            velocity += rng.normal(0.0, 0.08, size=2).astype(np.float32)
            exiting = False

            if obj.exit_frame is not None and obj.return_frame is not None and obj.exit_frame <= frame_idx < obj.return_frame:
                exiting = True
                remaining_frames = max(1, obj.return_frame - frame_idx)
                velocity = self._steer_to_edge(
                    obj.center,
                    obj.exit_edge or "left",
                    remaining_frames=remaining_frames,
                )

            if occlusion_event and occlusion_event.start <= frame_idx <= occlusion_event.end:
                if obj.instance_id in occlusion_event.pair:
                    direction = occlusion_event.target - obj.center
                    norm = float(np.linalg.norm(direction)) or 1.0
                    speed = float(np.linalg.norm(velocity)) or rng.uniform(*self.config.velocity_range)
                    velocity = 0.55 * velocity + 0.45 * direction / norm * speed

            obj.center += velocity
            obj.velocity = velocity

            if exiting:
                if self._is_outside(obj.center, width, height, margin):
                    obj.active = False
                continue

            if obj.center[0] < margin or obj.center[0] > width - margin:
                obj.velocity[0] *= -1.0
                obj.center[0] = np.clip(obj.center[0], margin, width - margin)
            if obj.center[1] < margin or obj.center[1] > height - margin:
                obj.velocity[1] *= -1.0
                obj.center[1] = np.clip(obj.center[1], margin, height - margin)
        return returned_ids

    def _build_background(
        self,
        rng: np.random.Generator,
        frame_idx: int,
        camera_jitter: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        height, width = self.config.resolution
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, height, dtype=np.float32),
            np.linspace(0.0, 1.0, width, dtype=np.float32),
            indexing="ij",
        )
        background = np.full((height, width, 3), 28.0, dtype=np.float32)

        if self.config.background_drift.enabled:
            phase = 2.0 * np.pi * frame_idx / max(self.config.sequence_length, 1)
            drift = self.config.background_drift.brightness_amplitude * 255.0 * np.sin(phase)
            gradient = (xx * 14.0) + (yy * 10.0)
            texture = rng.normal(
                0.0,
                self.config.background_drift.texture_noise_std * 255.0,
                size=(height, width, 3),
            ).astype(np.float32)
            background += drift + gradient[..., None] + texture
            drift_strength = abs(float(drift)) / 255.0
        else:
            background += rng.normal(0.0, 2.0, size=(height, width, 3)).astype(np.float32)
            drift_strength = 0.0

        if self.config.bridge_synthetic.enabled:
            density = 2.0 + 10.0 * self.config.bridge_synthetic.background_repeat_density
            jitter_x = float(camera_jitter[0]) / max(width, 1)
            jitter_y = float(camera_jitter[1]) / max(height, 1)
            repeated = (
                np.sin((xx + jitter_x) * np.pi * density)
                + np.cos((yy + jitter_y) * np.pi * density * 1.27)
            )
            checker = np.sign(
                np.sin((xx + jitter_x) * np.pi * density * 0.75)
                * np.sin((yy + jitter_y) * np.pi * density * 0.85)
            )
            structured = 0.65 * repeated + 0.35 * checker
            structured = structured[..., None] * (95.0 * self.config.bridge_synthetic.background_texture_strength)
            background += structured

            illumination_phase = 2.0 * np.pi * frame_idx / max(48, self.config.sequence_length // 4)
            spotlight_x = 0.5 + 0.22 * np.sin(illumination_phase * 0.73)
            spotlight_y = 0.5 + 0.18 * np.cos(illumination_phase * 0.91)
            gaussian = np.exp(
                -(
                    ((xx - spotlight_x) ** 2) / 0.045
                    + ((yy - spotlight_y) ** 2) / 0.038
                )
            )
            illumination = (
                2.0 * np.sin(illumination_phase) - 0.8 * np.cos(illumination_phase * 0.5)
            ) * 70.0 * self.config.bridge_synthetic.illumination_drift_strength
            background += gaussian[..., None] * illumination
            drift_strength += abs(float(illumination)) / 255.0

            if self.config.bridge_synthetic.local_noise_std > 0.0:
                patch_h = int(rng.integers(max(24, height // 8), max(25, height // 3)))
                patch_w = int(rng.integers(max(24, width // 8), max(25, width // 3)))
                y0 = int(rng.integers(0, max(1, height - patch_h)))
                x0 = int(rng.integers(0, max(1, width - patch_w)))
                noise_patch = rng.normal(
                    0.0,
                    self.config.bridge_synthetic.local_noise_std * 255.0,
                    size=(patch_h, patch_w, 3),
                ).astype(np.float32)
                background[y0 : y0 + patch_h, x0 : x0 + patch_w] += noise_patch

        return np.clip(background, 0.0, 255.0), float(drift_strength)

    def _render_frame(
        self,
        background: np.ndarray,
        objects: list[_ObjectState],
        rng: np.random.Generator,
        frame_idx: int,
        camera_jitter: np.ndarray,
        drift_strength: float,
        reentry_event: bool,
    ) -> FrameSample:
        height, width = self.config.resolution
        owner_map = np.full((height, width), -1, dtype=np.int32)
        shape_masks: dict[int, np.ndarray] = {}
        colors: dict[int, np.ndarray] = {}
        frame = background.copy()

        active_objects = [obj for obj in objects if obj.active]
        draw_order = sorted(active_objects, key=lambda obj: (obj.center[1], obj.instance_id))

        for obj in draw_order:
            mask, color = self._shape_mask_and_color(obj, rng, frame_idx, camera_jitter)
            shape_masks[obj.instance_id] = mask
            colors[obj.instance_id] = color
            owner_map[mask] = obj.instance_id

        boxes: list[Box] = []
        masks: list[np.ndarray] = []
        instance_ids: list[int] = []
        concept_ids: list[int] = []

        for obj in draw_order:
            visible_mask = owner_map == obj.instance_id
            if not np.any(visible_mask):
                continue

            frame[visible_mask] = colors[obj.instance_id]
            box = self._mask_to_box(visible_mask)
            boxes.append(box)
            masks.append(visible_mask)
            instance_ids.append(obj.instance_id)
            concept_ids.append(obj.concept_id)

        blur_level = 0.0
        if self.config.bridge_synthetic.enabled and rng.random() < self.config.bridge_synthetic.local_blur_probability:
            frame = self._apply_local_blur(frame, rng)
            blur_level = 1.0

        noise_level = 0.0
        if self.config.bridge_synthetic.enabled and self.config.bridge_synthetic.local_noise_std > 0.0:
            noise_std = self.config.bridge_synthetic.local_noise_std * 255.0
            frame += rng.normal(0.0, noise_std, size=frame.shape).astype(np.float32)
            noise_level = float(self.config.bridge_synthetic.local_noise_std)

        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
        return FrameSample(
            frame_index=frame_idx,
            frame=frame,
            boxes=boxes,
            masks=masks,
            instance_ids=instance_ids,
            concept_ids=concept_ids,
            visibility_flags=[True] * len(boxes),
            occlusion_ratio=self._frame_occlusion_ratio(boxes),
            drift_strength=float(drift_strength),
            blur_level=float(blur_level),
            noise_level=float(noise_level),
            reentry_event=bool(reentry_event),
        )

    def _shape_mask_and_color(
        self,
        obj: _ObjectState,
        rng: np.random.Generator,
        frame_idx: int,
        camera_jitter: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = self.config.resolution
        scale = obj.base_scale
        intensity = obj.base_intensity
        deformation = self.config.bridge_synthetic.target_deformation_strength if self.config.bridge_synthetic.enabled else 0.0

        if self.config.appearance_perturbation.enabled:
            scale *= 1.0 + rng.normal(0.0, self.config.appearance_perturbation.scale_jitter * 0.35)
            intensity *= 1.0 + rng.normal(0.0, self.config.appearance_perturbation.intensity_jitter * 0.35)
            if rng.random() < self.config.appearance_perturbation.edge_blur_probability:
                scale *= 0.96

        scale = float(np.clip(scale, self.config.object_scale_range[0] * 0.7, self.config.object_scale_range[1] * 1.3))
        color = self._color_lookup.get(obj.shape, np.array([220, 220, 220], dtype=np.float32)).copy()
        color *= intensity
        color += np.array([0.0, np.sin(frame_idx * 0.07) * 6.0, np.cos(frame_idx * 0.05) * 6.0], dtype=np.float32)
        if obj.contrast_gain < 1.0:
            color = 58.0 + obj.contrast_gain * (color - 58.0)

        yy, xx = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        cx, cy = obj.center + camera_jitter
        stretch_x = 1.0 + deformation * 0.6 * np.sin(frame_idx * 0.05 + obj.deformation_phase)
        stretch_y = 1.0 + deformation * 0.6 * np.cos(frame_idx * 0.04 + obj.deformation_phase)
        scale_x = max(4.0, scale * stretch_x)
        scale_y = max(4.0, scale * stretch_y)

        if obj.shape == "circle":
            mask = ((xx - cx) / scale_x) ** 2 + ((yy - cy) / scale_y) ** 2 <= 1.0
        elif obj.shape == "square":
            mask = (np.abs(xx - cx) <= scale_x) & (np.abs(yy - cy) <= scale_y)
        else:
            top = np.array([cx, cy - scale_y], dtype=np.float32)
            left = np.array([cx - 0.92 * scale_x, cy + 0.85 * scale_y], dtype=np.float32)
            right = np.array([cx + 0.92 * scale_x, cy + 0.85 * scale_y], dtype=np.float32)
            mask = _triangle_mask(xx, yy, top, left, right)

        return mask, np.clip(color, 0.0, 255.0)

    def _sample_velocity(self, rng: np.random.Generator) -> np.ndarray:
        speed = float(rng.uniform(*self.config.velocity_range))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        return np.array([np.cos(angle) * speed, np.sin(angle) * speed], dtype=np.float32)

    def _spawn_from_edge(self, rng: np.random.Generator, edge: str) -> tuple[np.ndarray, np.ndarray]:
        height, width = self.config.resolution
        margin = self.config.spawn_margin

        if edge == "left":
            center = np.array([margin, rng.uniform(margin, height - margin)], dtype=np.float32)
            direction = np.array([1.0, rng.uniform(-0.5, 0.5)], dtype=np.float32)
        elif edge == "right":
            center = np.array([width - margin, rng.uniform(margin, height - margin)], dtype=np.float32)
            direction = np.array([-1.0, rng.uniform(-0.5, 0.5)], dtype=np.float32)
        elif edge == "top":
            center = np.array([rng.uniform(margin, width - margin), margin], dtype=np.float32)
            direction = np.array([rng.uniform(-0.5, 0.5), 1.0], dtype=np.float32)
        else:
            center = np.array([rng.uniform(margin, width - margin), height - margin], dtype=np.float32)
            direction = np.array([rng.uniform(-0.5, 0.5), -1.0], dtype=np.float32)

        speed = float(rng.uniform(*self.config.velocity_range))
        direction /= float(np.linalg.norm(direction)) or 1.0
        return center, direction * speed

    def _steer_to_edge(self, center: np.ndarray, edge: str, remaining_frames: int) -> np.ndarray:
        height, width = self.config.resolution
        margin = self.config.spawn_margin
        if edge == "left":
            direction = np.array([-1.0, 0.0], dtype=np.float32)
            distance = float(center[0] + margin)
        elif edge == "right":
            direction = np.array([1.0, 0.0], dtype=np.float32)
            distance = float(width + margin - center[0])
        elif edge == "top":
            direction = np.array([0.0, -1.0], dtype=np.float32)
            distance = float(center[1] + margin)
        else:
            direction = np.array([0.0, 1.0], dtype=np.float32)
            distance = float(height + margin - center[1])
        speed = max(np.mean(self.config.velocity_range) * 1.5, distance / max(1, remaining_frames))
        return direction * speed

    @staticmethod
    def _is_outside(center: np.ndarray, width: int, height: int, margin: int) -> bool:
        return bool(
            center[0] < -margin
            or center[0] > width + margin
            or center[1] < -margin
            or center[1] > height + margin
        )

    @staticmethod
    def _mask_to_box(mask: np.ndarray) -> Box:
        ys, xs = np.where(mask)
        x1 = int(xs.min())
        x2 = int(xs.max()) + 1
        y1 = int(ys.min())
        y2 = int(ys.max()) + 1
        return (x1, y1, x2, y2)

    def _sample_camera_jitter(self, rng: np.random.Generator) -> np.ndarray:
        if not self.config.bridge_synthetic.enabled or self.config.bridge_synthetic.camera_jitter_std <= 0.0:
            return np.zeros(2, dtype=np.float32)
        return rng.normal(0.0, self.config.bridge_synthetic.camera_jitter_std, size=2).astype(np.float32)

    @staticmethod
    def _apply_local_blur(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        height, width = frame.shape[:2]
        patch_h = int(rng.integers(max(24, height // 8), max(25, height // 3)))
        patch_w = int(rng.integers(max(24, width // 8), max(25, width // 3)))
        y0 = int(rng.integers(0, max(1, height - patch_h)))
        x0 = int(rng.integers(0, max(1, width - patch_w)))
        patch = frame[y0 : y0 + patch_h, x0 : x0 + patch_w].astype(np.float32)
        blurred = (
            patch
            + np.roll(patch, 1, axis=0)
            + np.roll(patch, -1, axis=0)
            + np.roll(patch, 1, axis=1)
            + np.roll(patch, -1, axis=1)
        ) / 5.0
        result = frame.copy()
        result[y0 : y0 + patch_h, x0 : x0 + patch_w] = blurred
        return result

    @staticmethod
    def _frame_occlusion_ratio(boxes: list[Box]) -> float:
        if len(boxes) < 2:
            return 0.0
        overlap = 0.0
        area_total = 0.0
        for index, (ax1, ay1, ax2, ay2) in enumerate(boxes):
            area_total += max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            for bx1, by1, bx2, by2 in boxes[index + 1 :]:
                inter_x1 = max(ax1, bx1)
                inter_y1 = max(ay1, by1)
                inter_x2 = min(ax2, bx2)
                inter_y2 = min(ay2, by2)
                overlap += max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        return float(overlap / max(area_total, 1.0))


def load_synth_dataset_config(path: str | Path) -> SynthDatasetConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return SynthDatasetConfig.from_dict(payload)


def _triangle_mask(
    xx: np.ndarray,
    yy: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    px = xx
    py = yy

    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    denominator = denominator if abs(float(denominator)) > 1e-6 else 1.0

    w1 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / denominator
    w2 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / denominator
    w3 = 1.0 - w1 - w2
    return (w1 >= 0.0) & (w2 >= 0.0) & (w3 >= 0.0)
