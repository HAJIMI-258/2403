from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RetrievalState(str, Enum):
    OLD_RECALL_CANDIDATE = "old_recall_candidate"
    UNCERTAIN_NEED_MORE_EVIDENCE = "uncertain_need_more_evidence"


@dataclass(frozen=True)
class MemoryDecisionConfig:
    uncertainty_margin_threshold: float = 0.0194
    bounded_wait_horizon_frames: int = 10


@dataclass(frozen=True)
class MemoryDecision:
    retrieval_state: RetrievalState
    top1_margin: float
    threshold: float
    memory_update_allowed: bool
    evidence_queue_enqueued: bool
    attach_allowed: bool = False
    promotion_allowed: bool = False
    head_update_allowed: bool = False


def decide_memory_retrieval(top1_margin: float, config: MemoryDecisionConfig | None = None) -> MemoryDecision:
    """Map online-visible retrieval confidence to a safe memory action.

    The policy intentionally does not decide identity attachment or promotion.
    Low-margin recalls are routed to evidence acquisition / delayed resolution.
    """
    cfg = config or MemoryDecisionConfig()
    if top1_margin < cfg.uncertainty_margin_threshold:
        return MemoryDecision(
            retrieval_state=RetrievalState.UNCERTAIN_NEED_MORE_EVIDENCE,
            top1_margin=float(top1_margin),
            threshold=cfg.uncertainty_margin_threshold,
            memory_update_allowed=False,
            evidence_queue_enqueued=True,
        )
    return MemoryDecision(
        retrieval_state=RetrievalState.OLD_RECALL_CANDIDATE,
        top1_margin=float(top1_margin),
        threshold=cfg.uncertainty_margin_threshold,
        memory_update_allowed=True,
        evidence_queue_enqueued=False,
    )


def can_release_after_wait(
    *,
    wait_frames: int,
    release_margin: float,
    config: MemoryDecisionConfig | None = None,
) -> bool:
    """Return whether an uncertain recall can be released after bounded wait.

    This uses only online-visible wait length and release margin. Correctness is
    evaluated elsewhere and must not be used by the policy.
    """
    cfg = config or MemoryDecisionConfig()
    return 0 < int(wait_frames) <= cfg.bounded_wait_horizon_frames and float(release_margin) >= cfg.uncertainty_margin_threshold


def assert_safe_side_effects(decision: MemoryDecision) -> None:
    """Raise if a decision violates the CORE-1AQ safety contract."""
    if decision.retrieval_state == RetrievalState.UNCERTAIN_NEED_MORE_EVIDENCE:
        if decision.memory_update_allowed or decision.attach_allowed or decision.promotion_allowed or decision.head_update_allowed:
            raise ValueError("uncertain retrieval cannot update memory, attach, promote, or update head")
        if not decision.evidence_queue_enqueued:
            raise ValueError("uncertain retrieval must enqueue evidence acquisition or delayed resolution")
    if decision.attach_allowed or decision.promotion_allowed or decision.head_update_allowed:
        raise ValueError("CORE-1 policy must not trigger attach, promotion, or head update")
