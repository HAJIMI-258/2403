"""Bounded sparse long-term object memory.

The bank stores one fixed-size capsule per remembered object. It is
SNN-inspired: signatures are sparse, matching is event-like, and online
plasticity updates capsule statistics without storing raw frames or per-frame
descriptor history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nops_owr.descriptor.spiking_invariant_descriptor import SpikingInvariantDescriptor


@dataclass(slots=True)
class SpikingObjectMemoryCapsule:
    capsule_id: int
    spike_mu: np.ndarray
    spike_var: np.ndarray
    shape_mu: np.ndarray
    shape_var: np.ndarray
    appearance_mu: np.ndarray
    appearance_var: np.ndarray
    topology_mu: np.ndarray
    topology_var: np.ndarray
    deformation_mu: np.ndarray
    deformation_var: np.ndarray
    binary_hash: np.ndarray
    observation_count: int
    reactivation_count: int
    created_frame: int
    last_seen_frame: int
    stability: float
    plasticity: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpikingMemoryMatch:
    capsule_id: int
    score: float
    rank: int
    identity_score: float
    deformation_score: float
    spike_score: float
    hash_score: float
    novelty_score: float
    conflict_score: float
    decision_hint: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SpikingObjectMemoryBank:
    def __init__(
        self,
        max_capsules: int = 128,
        spike_dim: int = 128,
        plasticity: float = 0.08,
        min_match_score: float = 0.68,
        min_same_object_margin: float = 0.06,
        max_spike_density: float = 0.20,
        match_profile: str = "current",
    ) -> None:
        self.max_capsules = int(max_capsules)
        self.spike_dim = int(spike_dim)
        self.plasticity = float(plasticity)
        self.min_match_score = float(min_match_score)
        self.min_same_object_margin = float(min_same_object_margin)
        self.max_spike_density = float(max_spike_density)
        self.match_profile = str(match_profile)
        self.capsules: dict[int, SpikingObjectMemoryCapsule] = {}
        self._next_capsule_id = 1

    def __len__(self) -> int:
        return len(self.capsules)

    def match(
        self,
        descriptor: SpikingInvariantDescriptor,
        frame_index: int,
        top_k: int = 5,
    ) -> list[SpikingMemoryMatch]:
        del frame_index
        scored = []
        for capsule in self.capsules.values():
            spike_score = _spike_similarity(descriptor.spike_signature, capsule.spike_mu)
            shape_score = _gaussian_similarity(descriptor.shape_signature, capsule.shape_mu, capsule.shape_var)
            gray_appearance_score, chromatic_score = _appearance_component_scores(
                descriptor.appearance_signature,
                capsule.appearance_mu,
            )
            topology_score = _cosine_similarity(descriptor.topology_signature, capsule.topology_mu)
            identity_score = float(
                0.20 * shape_score
                + 0.125 * gray_appearance_score
                + 0.375 * chromatic_score
                + 0.30 * topology_score
            )
            deformation_score = _gaussian_similarity(
                descriptor.deformation_signature,
                capsule.deformation_mu,
                capsule.deformation_var,
            )
            hash_score = _hash_similarity(descriptor.binary_hash, capsule.binary_hash)
            stability_bonus = float(np.clip(capsule.stability, 0.0, 1.0))
            base_score = _profile_base_score(
                self.match_profile,
                spike_score=spike_score,
                shape_score=shape_score,
                gray_appearance_score=gray_appearance_score,
                chromatic_score=chromatic_score,
                topology_score=topology_score,
                deformation_score=deformation_score,
                hash_score=hash_score,
                identity_score=identity_score,
                stability_bonus=stability_bonus,
            )
            scored.append(
                {
                    "capsule": capsule,
                    "base_score": float(base_score),
                    "identity_score": identity_score,
                    "gray_appearance_score": gray_appearance_score,
                    "chromatic_score": chromatic_score,
                    "deformation_score": deformation_score,
                    "spike_score": spike_score,
                    "hash_score": hash_score,
                    "shape_score": shape_score,
                    "topology_score": topology_score,
                    "stability_bonus": stability_bonus,
                }
            )
        scored.sort(key=lambda row: row["base_score"], reverse=True)
        adjusted = []
        for index, row in enumerate(scored):
            rival_score = 0.0
            if len(scored) > 1:
                rival_score = float(scored[1]["base_score"] if index == 0 else scored[0]["base_score"])
            margin_to_rival = float(row["base_score"]) - rival_score
            raw_conflict = max(0.0, 1.0 - (margin_to_rival / max(self.min_same_object_margin, 1e-6))) if len(scored) > 1 else 0.0
            conflict_score = float(np.clip(raw_conflict, 0.0, 1.0))
            row["conflict_score"] = conflict_score
            row["score"] = float(row["base_score"] - 0.15 * conflict_score)
            adjusted.append(row)
        adjusted.sort(key=lambda row: row["score"], reverse=True)

        output: list[SpikingMemoryMatch] = []
        for rank, row in enumerate(adjusted[: max(0, int(top_k))], start=1):
            next_score = float(adjusted[rank]["score"]) if rank < len(adjusted) else 0.0
            margin = float(row["score"]) - next_score
            hint = "same_object_candidate" if row["score"] >= self.min_match_score and margin >= self.min_same_object_margin else "ambiguous_or_novel"
            output.append(
                SpikingMemoryMatch(
                    capsule_id=int(row["capsule"].capsule_id),
                    score=float(row["score"]),
                    rank=rank,
                    identity_score=float(row["identity_score"]),
                    deformation_score=float(row["deformation_score"]),
                    spike_score=float(row["spike_score"]),
                    hash_score=float(row["hash_score"]),
                    novelty_score=float(np.clip(1.0 - row["identity_score"], 0.0, 1.0)),
                    conflict_score=float(row["conflict_score"]),
                    decision_hint=hint,
                    metadata={
                        "base_score": float(row["base_score"]),
                        "top1_margin": margin if rank == 1 else 0.0,
                        "gray_appearance_score": float(row["gray_appearance_score"]),
                        "chromatic_score": float(row["chromatic_score"]),
                        "shape_score": float(row["shape_score"]),
                        "topology_score": float(row["topology_score"]),
                        "stability_bonus": float(row["stability_bonus"]),
                        "match_profile": self.match_profile,
                    },
                )
            )
        return output

    def write_or_update(
        self,
        descriptor: SpikingInvariantDescriptor,
        frame_index: int,
        confirmed_capsule_id: int | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        if confirmed_capsule_id is not None and int(confirmed_capsule_id) in self.capsules:
            self.update_capsule(int(confirmed_capsule_id), descriptor, frame_index, confidence=confidence)
            return int(confirmed_capsule_id)
        matches = self.match(descriptor, frame_index=frame_index, top_k=2)
        if matches:
            top = matches[0]
            margin = float(top.metadata.get("top1_margin", 0.0))
            if top.score >= self.min_match_score and margin >= self.min_same_object_margin:
                self.update_capsule(top.capsule_id, descriptor, frame_index, confidence=confidence)
                return top.capsule_id
        return self.create_capsule(descriptor, frame_index, metadata=metadata)

    def update_capsule(
        self,
        capsule_id: int,
        descriptor: SpikingInvariantDescriptor,
        frame_index: int,
        confidence: float = 1.0,
    ) -> None:
        capsule = self.capsules[int(capsule_id)]
        eta = float(np.clip(capsule.plasticity * confidence, 0.005, 0.25))
        _ema_update(capsule.spike_mu, capsule.spike_var, descriptor.spike_signature, eta)
        _ema_update(capsule.shape_mu, capsule.shape_var, descriptor.shape_signature, eta)
        _ema_update(capsule.appearance_mu, capsule.appearance_var, descriptor.appearance_signature, eta)
        _ema_update(capsule.topology_mu, capsule.topology_var, descriptor.topology_signature, eta)
        deform_eta = eta * (0.5 if _gaussian_similarity(descriptor.deformation_signature, capsule.deformation_mu, capsule.deformation_var) < 0.60 else 1.0)
        _ema_update(capsule.deformation_mu, capsule.deformation_var, descriptor.deformation_signature, deform_eta)
        capsule.spike_mu[:] = _sparsify_mu(capsule.spike_mu, self.max_spike_density)
        capsule.binary_hash = _majority_hash(capsule.binary_hash, descriptor.binary_hash, eta)
        capsule.observation_count += 1
        capsule.reactivation_count += 1
        capsule.last_seen_frame = int(frame_index)
        capsule.confidence = float(np.clip(0.92 * capsule.confidence + 0.08 * confidence, 0.0, 1.0))
        capsule.stability = float(np.clip(capsule.stability + 0.02 * confidence, 0.0, 1.0))

    def create_capsule(
        self,
        descriptor: SpikingInvariantDescriptor,
        frame_index: int,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        capsule_id = self._next_capsule_id
        self._next_capsule_id += 1
        capsule = SpikingObjectMemoryCapsule(
            capsule_id=capsule_id,
            spike_mu=descriptor.spike_signature.astype(np.float32).copy(),
            spike_var=np.full_like(descriptor.spike_signature, 0.05, dtype=np.float32),
            shape_mu=descriptor.shape_signature.astype(np.float32).copy(),
            shape_var=np.full_like(descriptor.shape_signature, 0.05, dtype=np.float32),
            appearance_mu=descriptor.appearance_signature.astype(np.float32).copy(),
            appearance_var=np.full_like(descriptor.appearance_signature, 0.05, dtype=np.float32),
            topology_mu=descriptor.topology_signature.astype(np.float32).copy(),
            topology_var=np.full_like(descriptor.topology_signature, 0.05, dtype=np.float32),
            deformation_mu=descriptor.deformation_signature.astype(np.float32).copy(),
            deformation_var=np.full_like(descriptor.deformation_signature, 0.05, dtype=np.float32),
            binary_hash=descriptor.binary_hash.astype(np.uint8).copy(),
            observation_count=1,
            reactivation_count=0,
            created_frame=int(frame_index),
            last_seen_frame=int(frame_index),
            stability=0.10,
            plasticity=self.plasticity,
            confidence=1.0,
            metadata={**dict(descriptor.metadata), **dict(metadata or {})},
        )
        capsule.spike_mu[:] = _sparsify_mu(capsule.spike_mu, self.max_spike_density)
        self.capsules[capsule_id] = capsule
        self.enforce_budget()
        return capsule_id

    def memory_bytes(self) -> int:
        total = 0
        for capsule in self.capsules.values():
            for array in (
                capsule.spike_mu,
                capsule.spike_var,
                capsule.shape_mu,
                capsule.shape_var,
                capsule.appearance_mu,
                capsule.appearance_var,
                capsule.topology_mu,
                capsule.topology_var,
                capsule.deformation_mu,
                capsule.deformation_var,
                capsule.binary_hash,
            ):
                total += int(array.nbytes)
        return total

    def mean_spike_density(self) -> float:
        if not self.capsules:
            return 0.0
        return float(np.mean([np.mean(capsule.spike_mu > 0.0) for capsule in self.capsules.values()]))

    def enforce_budget(self) -> None:
        while len(self.capsules) > self.max_capsules:
            victim = min(self.capsules.values(), key=self._eviction_utility)
            del self.capsules[int(victim.capsule_id)]

    def _eviction_utility(self, capsule: SpikingObjectMemoryCapsule) -> float:
        age_penalty = max(0.0, (self._next_capsule_id - capsule.capsule_id) * 0.001)
        return (
            float(capsule.confidence)
            + float(capsule.stability)
            + 0.05 * float(capsule.reactivation_count)
            + 0.01 * float(capsule.observation_count)
            - age_penalty
        )


def _ema_update(mu: np.ndarray, var: np.ndarray, sample: np.ndarray, eta: float) -> None:
    sample = sample.astype(np.float32, copy=False)
    delta = sample - mu
    mu += eta * delta
    var[:] = np.clip((1.0 - eta) * (var + eta * delta * delta), 1e-4, 10.0)


def _spike_similarity(left: np.ndarray, right_mu: np.ndarray) -> float:
    left = left.astype(np.float32, copy=False)
    right = (right_mu > 0.05).astype(np.float32)
    overlap = float(np.dot(left, right))
    denom = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return 0.0 if denom <= 1e-6 else float(np.clip(overlap / denom, 0.0, 1.0))


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float32, copy=False)
    right = right.astype(np.float32, copy=False)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-6:
        return 0.0
    return float(np.clip(0.5 + 0.5 * float(np.dot(left, right) / denom), 0.0, 1.0))


def _appearance_component_scores(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left = left.astype(np.float32, copy=False)
    right = right.astype(np.float32, copy=False)
    if left.size >= 27 and right.size >= 27:
        gray_score = _cosine_similarity(left[:15], right[:15])
        chromatic_score = _cosine_similarity(left[15:], right[15:])
        return gray_score, chromatic_score
    score = _cosine_similarity(left, right)
    return score, 0.5


def _profile_base_score(
    match_profile: str,
    *,
    spike_score: float,
    shape_score: float,
    gray_appearance_score: float,
    chromatic_score: float,
    topology_score: float,
    deformation_score: float,
    hash_score: float,
    identity_score: float,
    stability_bonus: float,
) -> float:
    if match_profile == "hash_chroma_deform":
        return float(
            0.30 * hash_score
            + 0.25 * chromatic_score
            + 0.20 * deformation_score
            + 0.15 * spike_score
            + 0.10 * identity_score
        )
    if match_profile == "identity_hash_chroma":
        return float(
            0.30 * identity_score
            + 0.25 * hash_score
            + 0.25 * chromatic_score
            + 0.10 * deformation_score
            + 0.10 * spike_score
        )
    return float(
        0.20 * spike_score
        + 0.08 * shape_score
        + 0.05 * gray_appearance_score
        + 0.15 * chromatic_score
        + 0.12 * topology_score
        + 0.15 * deformation_score
        + 0.20 * hash_score
        + 0.05 * stability_bonus
    )


def _gaussian_similarity(sample: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    if sample.size == 0:
        return 0.0
    diff = sample.astype(np.float32, copy=False) - mu.astype(np.float32, copy=False)
    scaled = (diff * diff) / np.maximum(var.astype(np.float32, copy=False), 1e-4)
    return float(np.exp(-0.5 * float(np.mean(np.clip(scaled, 0.0, 25.0)))))


def _hash_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    length = min(left.size, right.size)
    return float(np.mean(left[:length].astype(np.uint8) == right[:length].astype(np.uint8)))


def _majority_hash(old_hash: np.ndarray, new_hash: np.ndarray, eta: float) -> np.ndarray:
    # Keep this deterministic and conservative; hash bits update only under
    # moderately strong plasticity.
    if eta < 0.10:
        return old_hash.astype(np.uint8, copy=True)
    return np.where(new_hash.astype(np.uint8) > 0, old_hash | new_hash, old_hash).astype(np.uint8)


def _sparsify_mu(values: np.ndarray, max_density: float) -> np.ndarray:
    max_active = max(1, int(np.floor(float(max_density) * values.size)))
    if int(np.count_nonzero(values > 0.05)) <= max_active:
        return np.clip(values, 0.0, 1.0)
    threshold = float(np.partition(values, -max_active)[-max_active])
    return np.where(values >= threshold, np.clip(values, 0.0, 1.0), 0.0).astype(np.float32)
