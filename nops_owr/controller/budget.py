"""Streaming budget checks for evaluation and protocol enforcement."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BudgetFrameRecord:
    frame_index: int
    proposals: int
    active_tracks: int
    dormant_tracks: int
    memory_size: int
    proposal_violation: bool
    track_violation: bool
    memory_violation: bool


@dataclass(slots=True)
class BudgetSourceSuppressionRecord:
    frame_index: int
    source_object_id: str
    source_kind: str
    source_runtime_owner_id: int | None
    source_continuity_owner_id: int | None
    reason: str


@dataclass(slots=True)
class BudgetReport:
    memory_budget: int
    proposal_budget: int | None
    track_budget: int | None
    total_frames: int
    violation_frames: int
    proposal_violation_frames: int
    track_violation_frames: int
    memory_violation_frames: int
    peak_proposals: int
    peak_active_tracks: int
    peak_dormant_tracks: int
    peak_memory_size: int
    frame_records: list[BudgetFrameRecord]
    source_suppression_records: list[BudgetSourceSuppressionRecord]


class BudgetEnforcer:
    """Record frame-level budget pressure and protocol violations."""

    def __init__(
        self,
        *,
        memory_budget: int,
        proposal_budget: int | None = None,
        track_budget: int | None = None,
    ) -> None:
        self.memory_budget = int(memory_budget)
        self.proposal_budget = None if proposal_budget is None else int(proposal_budget)
        self.track_budget = None if track_budget is None else int(track_budget)
        self._records: list[BudgetFrameRecord] = []
        self._source_suppressions: list[BudgetSourceSuppressionRecord] = []

    def observe(
        self,
        *,
        frame_index: int,
        proposals: int,
        active_tracks: int,
        dormant_tracks: int,
        memory_size: int,
    ) -> BudgetFrameRecord:
        record = BudgetFrameRecord(
            frame_index=int(frame_index),
            proposals=int(proposals),
            active_tracks=int(active_tracks),
            dormant_tracks=int(dormant_tracks),
            memory_size=int(memory_size),
            proposal_violation=(self.proposal_budget is not None and proposals > self.proposal_budget),
            track_violation=(self.track_budget is not None and active_tracks > self.track_budget),
            memory_violation=memory_size > self.memory_budget,
        )
        self._records.append(record)
        return record

    def finalize(self) -> BudgetReport:
        return BudgetReport(
            memory_budget=self.memory_budget,
            proposal_budget=self.proposal_budget,
            track_budget=self.track_budget,
            total_frames=len(self._records),
            violation_frames=sum(
                int(record.proposal_violation or record.track_violation or record.memory_violation)
                for record in self._records
            ),
            proposal_violation_frames=sum(int(record.proposal_violation) for record in self._records),
            track_violation_frames=sum(int(record.track_violation) for record in self._records),
            memory_violation_frames=sum(int(record.memory_violation) for record in self._records),
            peak_proposals=max((record.proposals for record in self._records), default=0),
            peak_active_tracks=max((record.active_tracks for record in self._records), default=0),
            peak_dormant_tracks=max((record.dormant_tracks for record in self._records), default=0),
            peak_memory_size=max((record.memory_size for record in self._records), default=0),
            frame_records=list(self._records),
            source_suppression_records=list(self._source_suppressions),
        )

    def log_source_suppression(
        self,
        *,
        frame_index: int,
        source_object_id: str,
        source_kind: str,
        source_runtime_owner_id: int | None,
        source_continuity_owner_id: int | None,
        reason: str,
    ) -> BudgetSourceSuppressionRecord:
        record = BudgetSourceSuppressionRecord(
            frame_index=int(frame_index),
            source_object_id=str(source_object_id),
            source_kind=str(source_kind),
            source_runtime_owner_id=(
                None if source_runtime_owner_id is None else int(source_runtime_owner_id)
            ),
            source_continuity_owner_id=(
                None if source_continuity_owner_id is None else int(source_continuity_owner_id)
            ),
            reason=str(reason),
        )
        self._source_suppressions.append(record)
        return record
