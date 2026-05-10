"""Runtime retrieval context for episodic memory queries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievalContext:
    frame_index: int
    query_track_id: int | None = None
    query_prototype_id: int | None = None
    query_concept_id: int | None = None
    active_track_ids: set[int] = field(default_factory=set)
    mode: str = "general"
    min_reentry_gap: int = 8
    prefer_closed_episodes: bool = False
    suppress_active_conflicts: bool = True
