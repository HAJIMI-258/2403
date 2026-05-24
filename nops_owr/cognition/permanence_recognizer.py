"""Decision layer for spiking long-term object permanence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nops_owr.cognition.object_file import ObjectFile
from nops_owr.memory.spiking_object_memory import SpikingMemoryMatch


@dataclass(slots=True)
class PermanenceDecision:
    object_file_id: str
    decision_type: str
    capsule_id: int | None
    score: float
    confidence: float
    novelty_score: float
    deformation_score: float
    spike_score: float
    false_resurrection_risk: float
    metadata: dict[str, Any] = field(default_factory=dict)


class PermanenceRecognizer:
    def __init__(
        self,
        same_object_threshold: float = 0.86,
        uncertain_threshold: float = 0.55,
        same_object_margin_threshold: float = 0.04,
        false_resurrection_risk_threshold: float = 0.30,
    ) -> None:
        self.same_object_threshold = float(same_object_threshold)
        self.uncertain_threshold = float(uncertain_threshold)
        self.same_object_margin_threshold = float(same_object_margin_threshold)
        self.false_resurrection_risk_threshold = float(false_resurrection_risk_threshold)

    def decide(
        self,
        object_file: ObjectFile,
        matches: list[SpikingMemoryMatch],
    ) -> PermanenceDecision:
        if not matches:
            return PermanenceDecision(
                object_file_id=object_file.object_file_id,
                decision_type="new_object",
                capsule_id=None,
                score=0.0,
                confidence=1.0,
                novelty_score=1.0,
                deformation_score=0.0,
                spike_score=0.0,
                false_resurrection_risk=0.0,
                metadata={"reason": "no_matches"},
            )
        top = matches[0]
        second = matches[1] if len(matches) > 1 else None
        margin = float(top.score - second.score) if second is not None else float(top.score)
        risk = _false_resurrection_risk(top, margin)
        decision_type = "uncertain_hold"
        if risk >= self.false_resurrection_risk_threshold:
            decision_type = "false_resurrection_risk"
        elif top.score >= self.same_object_threshold and margin >= self.same_object_margin_threshold and top.deformation_score >= 0.45:
            decision_type = "same_object"
        elif top.score >= self.uncertain_threshold and top.spike_score >= 0.55 and top.deformation_score < 0.45:
            decision_type = "familiar_but_deformed"
        elif top.score < self.uncertain_threshold and top.novelty_score >= 0.45:
            decision_type = "new_object"
        confidence = _confidence(decision_type, top, margin, risk)
        return PermanenceDecision(
            object_file_id=object_file.object_file_id,
            decision_type=decision_type,
            capsule_id=top.capsule_id,
            score=float(top.score),
            confidence=confidence,
            novelty_score=float(top.novelty_score),
            deformation_score=float(top.deformation_score),
            spike_score=float(top.spike_score),
            false_resurrection_risk=risk,
            metadata={
                "top1_margin": margin,
                "identity_score": float(top.identity_score),
                "hash_score": float(top.hash_score),
                "gray_appearance_score": float(top.metadata.get("gray_appearance_score", 0.0)),
                "chromatic_score": float(top.metadata.get("chromatic_score", 0.0)),
                "conflict_score": float(top.conflict_score),
                "decision_hint": top.decision_hint,
                "same_object_threshold": self.same_object_threshold,
                "same_object_margin_threshold": self.same_object_margin_threshold,
                "false_resurrection_risk_threshold": self.false_resurrection_risk_threshold,
            },
        )


def _false_resurrection_risk(match: SpikingMemoryMatch, margin: float) -> float:
    low_margin = max(0.0, 1.0 - margin / 0.08)
    evidence_conflict = max(0.0, match.spike_score - match.deformation_score)
    hash_conflict = max(0.0, match.spike_score - match.hash_score)
    return float(min(1.0, 0.50 * match.conflict_score + 0.25 * low_margin + 0.15 * evidence_conflict + 0.10 * hash_conflict))


def _confidence(decision_type: str, match: SpikingMemoryMatch, margin: float, risk: float) -> float:
    if decision_type == "new_object":
        return float(max(0.2, match.novelty_score))
    if decision_type == "false_resurrection_risk":
        return float(risk)
    return float(max(0.0, min(1.0, 0.50 * match.score + 0.30 * margin + 0.20 * (1.0 - risk))))
