"""Controlled morphology re-entry eval for bounded spiking object memory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.permanence_metrics import (  # noqa: E402
    bytes_per_capsule,
    deformation_tolerance_curve,
    false_resurrection_rate,
    mean_spike_density,
    memory_growth_rate,
    same_instance_reentry_recall,
)
from nops_owr.cognition.object_file import ObjectFile, SupportMaskSummary  # noqa: E402
from nops_owr.cognition.permanence_recognizer import PermanenceRecognizer  # noqa: E402
from nops_owr.descriptor.spiking_invariant_descriptor import SpikingInvariantDescriptorBuilder  # noqa: E402
from nops_owr.encoder.spike_encoder import MinimalSpikeEncoder, SpikeEncoding  # noqa: E402
from nops_owr.memory.spiking_object_memory import SpikingObjectMemoryBank  # noqa: E402


EVENT_FIELDS = [
    "event_id",
    "object_id",
    "frame_index",
    "phase",
    "scale_change",
    "aspect_change",
    "brightness_drift",
    "occlusion",
    "distractor_level",
    "decision_type",
    "matched_capsule_id",
    "true_capsule_id",
    "same_instance_success",
    "false_resurrection",
    "score",
    "spike_score",
    "deformation_score",
    "identity_score",
    "gray_appearance_score",
    "chromatic_score",
    "hash_score",
    "conflict_score",
    "top1_margin",
    "false_resurrection_risk",
    "top1_is_true_capsule",
    "top2_capsule_id",
    "top2_score",
    "true_capsule_rank",
    "true_capsule_score",
    "score_gap_top1_minus_true",
    "true_spike_score",
    "true_deformation_score",
    "true_identity_score",
    "true_chromatic_score",
    "true_hash_score",
    "delta_top1_minus_true_spike",
    "delta_top1_minus_true_deformation",
    "delta_top1_minus_true_identity",
    "delta_top1_minus_true_chromatic",
    "delta_top1_minus_true_hash",
    "memory_bytes",
    "capsule_count",
    "spike_density",
]

MATCH_FIELDS = [
    "event_id",
    "object_id",
    "frame_index",
    "rank",
    "capsule_id",
    "true_capsule_id",
    "is_true_capsule",
    "score",
    "base_score",
    "identity_score",
    "spike_score",
    "deformation_score",
    "gray_appearance_score",
    "chromatic_score",
    "hash_score",
    "shape_score",
    "topology_score",
    "stability_bonus",
    "conflict_score",
    "top1_margin",
    "scale_change",
    "aspect_change",
    "brightness_drift",
    "occlusion",
    "distractor_level",
]


def run_eval(
    output_dir: str | Path = "results/spiking_morph_permanence_eval",
    seed: int = 7,
    object_count: int = 16,
    events_per_object: int = 4,
    max_capsules: int = 32,
    spike_dim: int = 128,
    max_frames: int = 800,
    same_object_threshold: float = 0.90,
    same_object_margin_threshold: float = 0.04,
    false_resurrection_risk_threshold: float = 0.30,
    match_profile: str = "current",
) -> dict[str, Any]:
    del max_frames
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    encoder = MinimalSpikeEncoder()
    builder = SpikingInvariantDescriptorBuilder(spike_dim=spike_dim, hash_bits=spike_dim, seed=seed)
    bank = SpikingObjectMemoryBank(max_capsules=max_capsules, spike_dim=spike_dim, match_profile=match_profile)
    recognizer = PermanenceRecognizer(
        same_object_threshold=same_object_threshold,
        same_object_margin_threshold=same_object_margin_threshold,
        false_resurrection_risk_threshold=false_resurrection_risk_threshold,
    )
    rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    true_capsules: dict[int, int] = {}
    object_specs = [_object_spec(object_id, rng) for object_id in range(int(object_count))]
    frame_index = 1

    for object_id in range(int(object_count)):
        spec = object_specs[object_id]
        prev, current, box = _render_observation(spec, scale=1.0, aspect=1.0, brightness=1.0, occlusion=0.0)
        encoding = encoder.encode(prev, current)
        object_file = _object_file(object_id, frame_index, box, encoding, source="context", current_frame=current)
        descriptor = builder.build(object_file, encoding)
        capsule_id = bank.create_capsule(descriptor, frame_index=frame_index, metadata={"object_id_eval_only": object_id})
        true_capsules[object_id] = capsule_id
        rows.append(_row("context", object_id, frame_index, "context", 1.0, 1.0, 1.0, 0.0, "none", "same_object", capsule_id, capsule_id, True, False, 1.0, 1.0, 1.0, bank))
        frame_index += 1

    scale_values = [1.0, 1.2, 1.5, 2.0]
    aspect_values = [1.0, 1.25, 1.5]
    distractors = ["none", "low", "high"]
    for object_id in range(int(object_count)):
        spec = object_specs[object_id]
        for event_idx in range(int(events_per_object)):
            scale = scale_values[event_idx % len(scale_values)]
            aspect = aspect_values[(object_id + event_idx) % len(aspect_values)]
            brightness = 1.0 + (0.10 if event_idx % 2 else -0.08)
            occlusion = 0.0 if event_idx % 3 else 0.25
            distractor_level = distractors[(object_id + event_idx) % len(distractors)]
            prev, current, box = _render_observation(spec, scale=scale, aspect=aspect, brightness=brightness, occlusion=occlusion)
            encoding = encoder.encode(prev, current)
            object_file = _object_file(object_id, frame_index, box, encoding, source="reentry", current_frame=current)
            descriptor = builder.build(object_file, encoding)
            matches = bank.match(descriptor, frame_index=frame_index, top_k=5)
            decision = recognizer.decide(object_file, matches)
            matched_capsule_id = decision.capsule_id
            true_capsule_id = true_capsules[object_id]
            match_diagnostics = _match_diagnostics(matches, true_capsule_id)
            event_id = f"obj{object_id}_event{event_idx}"
            match_rows.extend(
                _match_rows(
                    event_id,
                    object_id,
                    frame_index,
                    matches,
                    true_capsule_id,
                    scale,
                    aspect,
                    brightness,
                    occlusion,
                    distractor_level,
                )
            )
            success = decision.decision_type in {"same_object", "familiar_but_deformed"} and matched_capsule_id == true_capsule_id
            false_res = decision.decision_type in {"same_object", "familiar_but_deformed"} and matched_capsule_id not in {None, true_capsule_id}
            if success:
                bank.update_capsule(true_capsule_id, descriptor, frame_index=frame_index, confidence=decision.confidence)
            elif decision.decision_type == "new_object" and distractor_level == "high":
                bank.write_or_update(descriptor, frame_index=frame_index, confidence=0.5, metadata={"object_id_eval_only": object_id})
            rows.append(
                _row(
                    event_id,
                    object_id,
                    frame_index,
                    "reentry",
                    scale,
                    aspect,
                    brightness,
                    occlusion,
                    distractor_level,
                    decision.decision_type,
                    matched_capsule_id,
                    true_capsule_id,
                    success,
                    false_res,
                    decision.score,
                    decision.spike_score,
                    decision.deformation_score,
                    bank,
                    {**decision.metadata, **match_diagnostics},
                    decision.false_resurrection_risk,
                )
            )
            frame_index += 1

    summary = _summary(
        rows,
        bank,
        {
            "same_object_threshold": float(same_object_threshold),
            "same_object_margin_threshold": float(same_object_margin_threshold),
            "false_resurrection_risk_threshold": float(false_resurrection_risk_threshold),
            "match_profile": str(match_profile),
        },
    )
    _write_csv(out / "events.csv", rows, EVENT_FIELDS)
    _write_csv(out / "matches.csv", match_rows, MATCH_FIELDS)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _object_spec(object_id: int, rng: np.random.Generator) -> dict[str, Any]:
    base = rng.uniform(0.35, 0.90, size=3)
    return {
        "color": np.roll(base, object_id % 3),
        "width": 12 + (object_id % 5) * 3,
        "height": 12 + (object_id % 4) * 4,
        "x": 20 + (object_id % 4) * 18,
        "y": 18 + (object_id % 3) * 18,
        "texture_period": 3 + (object_id % 4),
        "texture_axis": object_id % 3,
        "texture_phase": (object_id * 2) % 5,
        "texture_strength": 18 + (object_id % 4) * 8,
    }


def _render_observation(
    spec: dict[str, Any],
    *,
    scale: float,
    aspect: float,
    brightness: float,
    occlusion: float,
    size: int = 96,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    prev = np.zeros((size, size, 3), dtype=np.uint8)
    current = np.zeros_like(prev)
    width = max(4, int(round(float(spec["width"]) * float(scale) * float(aspect))))
    height = max(4, int(round(float(spec["height"]) * float(scale) / max(0.5, float(aspect)))))
    x1 = int(np.clip(int(spec["x"]) - width // 2, 0, size - 2))
    y1 = int(np.clip(int(spec["y"]) - height // 2, 0, size - 2))
    x2 = int(np.clip(x1 + width, x1 + 1, size))
    y2 = int(np.clip(y1 + height, y1 + 1, size))
    color = np.clip(np.asarray(spec["color"]) * 255.0 * float(brightness), 0, 255).astype(np.uint8)
    current[y1:y2, x1:x2, :] = color
    _apply_object_texture(current, (x1, y1, x2, y2), spec, color)
    if occlusion > 0.0:
        occ_w = max(1, int(round((x2 - x1) * float(occlusion))))
        current[y1:y2, x2 - occ_w : x2, :] = 0
    return prev, current, (x1, y1, x2, y2)


def _apply_object_texture(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    spec: dict[str, Any],
    color: np.ndarray,
) -> None:
    x1, y1, x2, y2 = box
    period = max(2, int(spec.get("texture_period", 4)))
    axis = int(spec.get("texture_axis", 0))
    phase = int(spec.get("texture_phase", 0))
    strength = float(spec.get("texture_strength", 24.0))
    bright = np.clip(color.astype(np.float32) + strength, 0, 255).astype(np.uint8)
    dark = np.clip(color.astype(np.float32) - 0.6 * strength, 0, 255).astype(np.uint8)
    if axis == 0:
        frame[y1 + phase % period : y2 : period, x1:x2, :] = bright
        frame[y1 + (phase + period // 2) % period : y2 : period, x1:x2, :] = dark
    elif axis == 1:
        frame[y1:y2, x1 + phase % period : x2 : period, :] = bright
        frame[y1:y2, x1 + (phase + period // 2) % period : x2 : period, :] = dark
    else:
        yy, xx = np.indices((max(1, y2 - y1), max(1, x2 - x1)))
        mask = ((xx + yy + phase) % period) == 0
        patch = frame[y1:y2, x1:x2, :]
        patch[mask, :] = bright
        frame[y1:y2, x1:x2, :] = patch


def _object_file(
    object_id: int,
    frame_index: int,
    box: tuple[int, int, int, int],
    encoding: SpikeEncoding,
    *,
    source: str,
    current_frame: np.ndarray | None = None,
) -> ObjectFile:
    x1, y1, x2, y2 = box
    area = float((x2 - x1) * (y2 - y1))
    frame_area = float(encoding.current_gray.shape[0] * encoding.current_gray.shape[1])
    shape = np.asarray(
        [(x2 - x1) / 96.0, (y2 - y1) / 96.0, (x2 - x1) / max(1.0, y2 - y1), area / max(1.0, frame_area), 1.0, 0.6, 0.6],
        dtype=np.float32,
    )
    appearance = _appearance(box, encoding)
    context = np.asarray([(x1 + x2) / 192.0, (y1 + y2) / 192.0, 1.0, 0.0, x1 / 96.0, y1 / 96.0], dtype=np.float32)
    return ObjectFile(
        object_file_id=f"morph:{object_id}:{frame_index}:{source}",
        frame_index=int(frame_index),
        proposal_index=0,
        box=box,
        raw_box=box,
        support_box=box,
        centroid=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
        area=area,
        score=1.0,
        quality_score=1.0,
        support_mask_summary=SupportMaskSummary(area=area, bbox=box, fill_ratio=1.0, compactness=0.6, boundary_smoothness=0.6),
        appearance_signature=appearance,
        shape_signature=shape,
        context_signature=context,
        motion_signature=np.zeros(0, dtype=np.float32),
        confidence=1.0,
        metadata={
            "object_id_eval_only": object_id,
            "chromatic_signature": _chromatic_signature(box, current_frame).tolist(),
            "chromatic_grid_signature": _chromatic_grid_signature(box, current_frame, grid=4).tolist(),
        },
    )


def _appearance(box: tuple[int, int, int, int], encoding: SpikeEncoding) -> np.ndarray:
    x1, y1, x2, y2 = box
    stats = []
    for array in (encoding.current_gray, encoding.edge_map, encoding.spike_response):
        patch = array[y1:y2, x1:x2]
        values = patch.reshape(-1).astype(np.float32)
        stats.extend([float(np.mean(values)), float(np.std(values)), float(np.quantile(values, 0.25)), float(np.quantile(values, 0.50)), float(np.quantile(values, 0.75))])
    return np.asarray(stats, dtype=np.float32)


def _chromatic_signature(box: tuple[int, int, int, int], current_frame: np.ndarray | None) -> np.ndarray:
    if current_frame is None or current_frame.ndim < 3 or current_frame.shape[2] < 3:
        return np.zeros(12, dtype=np.float32)
    x1, y1, x2, y2 = box
    patch = current_frame[y1:y2, x1:x2, :3]
    if patch.size == 0:
        return np.zeros(12, dtype=np.float32)
    values = patch.reshape(-1, 3).astype(np.float32) / 255.0
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0)
    intensity = np.sum(values, axis=1, keepdims=True) + 1e-6
    chroma = values / intensity
    return np.nan_to_num(
        np.concatenate([means, stds, np.mean(chroma, axis=0), np.std(chroma, axis=0)]).astype(np.float32),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )


def _chromatic_grid_signature(box: tuple[int, int, int, int], current_frame: np.ndarray | None, grid: int = 4) -> np.ndarray:
    if current_frame is None or current_frame.ndim < 3 or current_frame.shape[2] < 3:
        return np.zeros(grid * grid * 3, dtype=np.float32)
    x1, y1, x2, y2 = box
    patch = current_frame[y1:y2, x1:x2, :3]
    if patch.size == 0:
        return np.zeros(grid * grid * 3, dtype=np.float32)
    patch = patch.astype(np.float32, copy=False) / 255.0
    h, w = patch.shape[:2]
    output = np.zeros((grid, grid, 3), dtype=np.float32)
    for gy in range(grid):
        cy1 = int(round(gy * h / grid))
        cy2 = int(round((gy + 1) * h / grid))
        for gx in range(grid):
            cx1 = int(round(gx * w / grid))
            cx2 = int(round((gx + 1) * w / grid))
            cell = patch[cy1:max(cy1 + 1, cy2), cx1:max(cx1 + 1, cx2), :]
            if cell.size:
                output[gy, gx, :] = np.mean(cell.reshape(-1, 3), axis=0)
    return np.nan_to_num(output.reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)


def _row(
    event_id: str,
    object_id: int,
    frame_index: int,
    phase: str,
    scale: float,
    aspect: float,
    brightness: float,
    occlusion: float,
    distractor: str,
    decision_type: str,
    matched_capsule_id: int | None,
    true_capsule_id: int,
    success: bool,
    false_resurrection: bool,
    score: float,
    spike_score: float,
    deformation_score: float,
    bank: SpikingObjectMemoryBank,
    decision_metadata: dict[str, Any] | None = None,
    false_resurrection_risk: float = 0.0,
) -> dict[str, Any]:
    metadata = dict(decision_metadata or {})
    top1_is_true = int(matched_capsule_id == true_capsule_id) if matched_capsule_id not in {None, ""} else 0
    return {
        "event_id": event_id,
        "object_id": int(object_id),
        "frame_index": int(frame_index),
        "phase": phase,
        "scale_change": float(scale),
        "aspect_change": float(aspect),
        "brightness_drift": float(brightness),
        "occlusion": float(occlusion),
        "distractor_level": distractor,
        "decision_type": decision_type,
        "matched_capsule_id": "" if matched_capsule_id is None else int(matched_capsule_id),
        "true_capsule_id": int(true_capsule_id),
        "same_instance_success": int(success),
        "false_resurrection": int(false_resurrection),
        "score": float(score),
        "spike_score": float(spike_score),
        "deformation_score": float(deformation_score),
        "identity_score": float(metadata.get("identity_score", 0.0)),
        "gray_appearance_score": float(metadata.get("gray_appearance_score", 0.0)),
        "chromatic_score": float(metadata.get("chromatic_score", 0.0)),
        "hash_score": float(metadata.get("hash_score", 0.0)),
        "conflict_score": float(metadata.get("conflict_score", 0.0)),
        "top1_margin": float(metadata.get("top1_margin", 0.0)),
        "false_resurrection_risk": float(false_resurrection_risk),
        "top1_is_true_capsule": top1_is_true,
        "top2_capsule_id": metadata.get("top2_capsule_id", ""),
        "top2_score": float(metadata.get("top2_score", 0.0)),
        "true_capsule_rank": int(metadata.get("true_capsule_rank", 1 if top1_is_true else 0)),
        "true_capsule_score": float(metadata.get("true_capsule_score", score if top1_is_true else 0.0)),
        "score_gap_top1_minus_true": float(metadata.get("score_gap_top1_minus_true", 0.0 if top1_is_true else score)),
        "true_spike_score": float(metadata.get("true_spike_score", spike_score if top1_is_true else 0.0)),
        "true_deformation_score": float(metadata.get("true_deformation_score", deformation_score if top1_is_true else 0.0)),
        "true_identity_score": float(metadata.get("true_identity_score", metadata.get("identity_score", 0.0) if top1_is_true else 0.0)),
        "true_chromatic_score": float(metadata.get("true_chromatic_score", metadata.get("chromatic_score", 0.0) if top1_is_true else 0.0)),
        "true_hash_score": float(metadata.get("true_hash_score", metadata.get("hash_score", 0.0) if top1_is_true else 0.0)),
        "delta_top1_minus_true_spike": float(metadata.get("delta_top1_minus_true_spike", 0.0)),
        "delta_top1_minus_true_deformation": float(metadata.get("delta_top1_minus_true_deformation", 0.0)),
        "delta_top1_minus_true_identity": float(metadata.get("delta_top1_minus_true_identity", 0.0)),
        "delta_top1_minus_true_chromatic": float(metadata.get("delta_top1_minus_true_chromatic", 0.0)),
        "delta_top1_minus_true_hash": float(metadata.get("delta_top1_minus_true_hash", 0.0)),
        "memory_bytes": int(bank.memory_bytes()),
        "capsule_count": int(len(bank)),
        "spike_density": float(bank.mean_spike_density()),
        "is_reentry": int(phase == "reentry"),
    }


def _match_diagnostics(matches: list[Any], true_capsule_id: int) -> dict[str, Any]:
    top1 = matches[0] if matches else None
    top1_score = float(top1.score) if top1 is not None else 0.0
    top2 = matches[1] if len(matches) > 1 else None
    true_match = None
    true_rank = 0
    true_score = 0.0
    for match in matches:
        if int(match.capsule_id) == int(true_capsule_id):
            true_match = match
            true_rank = int(match.rank)
            true_score = float(match.score)
            break
    true_spike = float(true_match.spike_score) if true_match is not None else 0.0
    true_deformation = float(true_match.deformation_score) if true_match is not None else 0.0
    true_identity = float(true_match.identity_score) if true_match is not None else 0.0
    true_chromatic = float(true_match.metadata.get("chromatic_score", 0.0)) if true_match is not None else 0.0
    true_hash = float(true_match.hash_score) if true_match is not None else 0.0
    top1_spike = float(top1.spike_score) if top1 is not None else 0.0
    top1_deformation = float(top1.deformation_score) if top1 is not None else 0.0
    top1_identity = float(top1.identity_score) if top1 is not None else 0.0
    top1_chromatic = float(top1.metadata.get("chromatic_score", 0.0)) if top1 is not None else 0.0
    top1_hash = float(top1.hash_score) if top1 is not None else 0.0
    return {
        "top2_capsule_id": "" if top2 is None else int(top2.capsule_id),
        "top2_score": 0.0 if top2 is None else float(top2.score),
        "true_capsule_rank": true_rank,
        "true_capsule_score": true_score,
        "score_gap_top1_minus_true": float(top1_score - true_score) if true_rank > 0 else top1_score,
        "true_spike_score": true_spike,
        "true_deformation_score": true_deformation,
        "true_identity_score": true_identity,
        "true_chromatic_score": true_chromatic,
        "true_hash_score": true_hash,
        "delta_top1_minus_true_spike": top1_spike - true_spike if true_rank > 0 else top1_spike,
        "delta_top1_minus_true_deformation": top1_deformation - true_deformation if true_rank > 0 else top1_deformation,
        "delta_top1_minus_true_identity": top1_identity - true_identity if true_rank > 0 else top1_identity,
        "delta_top1_minus_true_chromatic": top1_chromatic - true_chromatic if true_rank > 0 else top1_chromatic,
        "delta_top1_minus_true_hash": top1_hash - true_hash if true_rank > 0 else top1_hash,
    }


def _match_rows(
    event_id: str,
    object_id: int,
    frame_index: int,
    matches: list[Any],
    true_capsule_id: int,
    scale: float,
    aspect: float,
    brightness: float,
    occlusion: float,
    distractor: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in matches:
        rows.append(
            {
                "event_id": event_id,
                "object_id": int(object_id),
                "frame_index": int(frame_index),
                "rank": int(match.rank),
                "capsule_id": int(match.capsule_id),
                "true_capsule_id": int(true_capsule_id),
                "is_true_capsule": int(int(match.capsule_id) == int(true_capsule_id)),
                "score": float(match.score),
                "base_score": float(match.metadata.get("base_score", match.score)),
                "identity_score": float(match.identity_score),
                "spike_score": float(match.spike_score),
                "deformation_score": float(match.deformation_score),
                "gray_appearance_score": float(match.metadata.get("gray_appearance_score", 0.0)),
                "chromatic_score": float(match.metadata.get("chromatic_score", 0.0)),
                "hash_score": float(match.hash_score),
                "shape_score": float(match.metadata.get("shape_score", 0.0)),
                "topology_score": float(match.metadata.get("topology_score", 0.0)),
                "stability_bonus": float(match.metadata.get("stability_bonus", 0.0)),
                "conflict_score": float(match.conflict_score),
                "top1_margin": float(match.metadata.get("top1_margin", 0.0)),
                "scale_change": float(scale),
                "aspect_change": float(aspect),
                "brightness_drift": float(brightness),
                "occlusion": float(occlusion),
                "distractor_level": distractor,
            }
        )
    return rows


def _summary(rows: list[dict[str, Any]], bank: SpikingObjectMemoryBank, config: dict[str, float] | None = None) -> dict[str, Any]:
    reentry = [row for row in rows if row["phase"] == "reentry"]
    memory_sizes = [row["capsule_count"] for row in rows]
    accepted = [row for row in reentry if row["decision_type"] in {"same_object", "familiar_but_deformed"}]
    top1_true = [row for row in reentry if int(row.get("top1_is_true_capsule", 0)) == 1]
    true_in_top3 = [row for row in reentry if 1 <= int(row.get("true_capsule_rank", 0)) <= 3]
    true_in_top5 = [row for row in reentry if 1 <= int(row.get("true_capsule_rank", 0)) <= 5]
    top1_true_rejected = [
        row
        for row in top1_true
        if row["decision_type"] not in {"same_object", "familiar_but_deformed"}
    ]
    return {
        **dict(config or {}),
        "same_instance_reentry_recall": same_instance_reentry_recall(reentry),
        "false_resurrection_rate": false_resurrection_rate(reentry),
        "accepted_reentry_decision_count": int(len(accepted)),
        "false_resurrection_count": int(sum(int(row.get("false_resurrection", 0)) for row in reentry)),
        "top1_true_capsule_rate": _safe_rate(len(top1_true), len(reentry)),
        "true_capsule_top3_rate": _safe_rate(len(true_in_top3), len(reentry)),
        "true_capsule_top5_rate": _safe_rate(len(true_in_top5), len(reentry)),
        "top1_true_but_not_accepted_rate": _safe_rate(len(top1_true_rejected), len(reentry)),
        "uncertain_hold_rate": _safe_rate(sum(1 for row in reentry if row["decision_type"] == "uncertain_hold"), len(reentry)),
        "false_resurrection_risk_decision_rate": _safe_rate(sum(1 for row in reentry if row["decision_type"] == "false_resurrection_risk"), len(reentry)),
        "mean_top1_margin": float(np.mean([row["top1_margin"] for row in reentry])) if reentry else 0.0,
        "mean_score_gap_top1_minus_true": float(np.mean([row["score_gap_top1_minus_true"] for row in true_in_top5])) if true_in_top5 else 0.0,
        "mean_false_resurrection_risk": float(np.mean([row["false_resurrection_risk"] for row in reentry])) if reentry else 0.0,
        "mean_memory_bytes": float(np.mean([row["memory_bytes"] for row in rows])) if rows else 0.0,
        "bytes_per_capsule": bytes_per_capsule(bank.memory_bytes(), len(bank)),
        "mean_spike_density": mean_spike_density([row["spike_density"] for row in rows]),
        "memory_growth_rate": memory_growth_rate(memory_sizes),
        "final_capsule_count": int(len(bank)),
        "final_memory_bytes": int(bank.memory_bytes()),
        "recall_by_scale_change": deformation_tolerance_curve(reentry, field="scale_change"),
        "recall_by_aspect_change": deformation_tolerance_curve(reentry, field="aspect_change"),
        "false_resurrection_by_distractor_level": _false_by_group(reentry, "distractor_level"),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def _false_by_group(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[field]), []).append(row)
    return {key: false_resurrection_rate(group) for key, group in sorted(groups.items())}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _report(summary: dict[str, Any]) -> str:
    return (
        "# Spiking Morph Permanence Eval\n\n"
        f"- same_instance_reentry_recall: {summary['same_instance_reentry_recall']:.4f}\n"
        f"- false_resurrection_rate: {summary['false_resurrection_rate']:.4f}\n"
        f"- bytes_per_capsule: {summary['bytes_per_capsule']:.2f}\n"
        f"- mean_spike_density: {summary['mean_spike_density']:.4f}\n"
        f"- final_capsule_count: {summary['final_capsule_count']}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/spiking_morph_permanence_eval")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--object-count", type=int, default=16)
    parser.add_argument("--events-per-object", type=int, default=4)
    parser.add_argument("--max-capsules", type=int, default=32)
    parser.add_argument("--spike-dim", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=800)
    parser.add_argument("--same-object-threshold", type=float, default=0.90)
    parser.add_argument("--same-object-margin-threshold", type=float, default=0.04)
    parser.add_argument("--false-resurrection-risk-threshold", type=float, default=0.30)
    parser.add_argument("--match-profile", default="current")
    summary = run_eval(**vars(parser.parse_args()))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
