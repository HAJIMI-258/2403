"""Event ledger rows emitted by the visual cognitive loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CognitiveEvent:
    event_id: str
    frame_index: int
    event_type: str
    object_file_id: str | None = None
    track_id: int | None = None
    prototype_id: int | None = None
    concept_id: int | None = None
    episode_id: int | None = None
    decision_type: str | None = None
    confidence: float = 0.0
    prediction_error: float = 0.0
    familiarity_score: float = 0.0
    novelty_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
