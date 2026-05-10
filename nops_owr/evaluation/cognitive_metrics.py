"""Cognitive-loop metrics for object memory evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np


def same_instance_recovery_rate(decisions: Iterable[Any]) -> float:
    rows = list(decisions)
    if not rows:
        return 0.0
    return _count_decisions(rows, "same_instance") / len(rows)


def false_resurrection_rate(records: Iterable[Any]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    false_count = sum(int(bool(_get(row, "false_resurrection", False))) for row in rows)
    return false_count / len(rows)


def uncertain_hold_rate(decisions: Iterable[Any]) -> float:
    rows = list(decisions)
    if not rows:
        return 0.0
    return _count_decisions(rows, "uncertain_hold") / len(rows)


def memory_write_rate(written_count: int, object_count: int) -> float:
    if object_count <= 0:
        return 0.0
    return float(written_count) / float(object_count)


def episodic_reactivation_rate(reactivated_count: int, candidate_count: int) -> float:
    if candidate_count <= 0:
        return 0.0
    return float(reactivated_count) / float(candidate_count)


def prediction_error_mean(decisions: Iterable[Any]) -> float:
    errors = [float(_get(row, "prediction_error", 0.0)) for row in decisions]
    return float(np.mean(errors)) if errors else 0.0


def attended_object_ratio(attended_count: int, object_count: int) -> float:
    if object_count <= 0:
        return 0.0
    return float(attended_count) / float(object_count)


def memory_compression_ratio(raw_observation_count: int, memory_item_count: int) -> float:
    if raw_observation_count <= 0:
        return 0.0
    return float(memory_item_count) / float(raw_observation_count)


def long_gap_reentry_success_rate(events: Iterable[Any], min_gap: int = 10) -> float:
    long_gap = [event for event in events if int(_get(event, "gap_length", 0)) >= min_gap]
    if not long_gap:
        return 0.0
    successes = sum(int(bool(_get(event, "success", _get(event, "recovered", False)))) for event in long_gap)
    return successes / len(long_gap)


def concept_fragmentation_rate(records: Iterable[Any]) -> float:
    concept_to_instances: dict[Any, set[Any]] = defaultdict(set)
    for record in records:
        concept_id = _get(record, "concept_id", None)
        prototype_id = _get(record, "prototype_id", None)
        if concept_id is None or prototype_id is None:
            continue
        concept_to_instances[concept_id].add(prototype_id)
    if not concept_to_instances:
        return 0.0
    fragmented = sum(int(len(prototypes) > 1) for prototypes in concept_to_instances.values())
    return fragmented / len(concept_to_instances)


def decision_histogram(decisions: Iterable[Any]) -> dict[str, int]:
    return dict(Counter(str(_get(decision, "decision_type", "unknown")) for decision in decisions))


def _count_decisions(rows: list[Any], decision_type: str) -> int:
    return sum(int(_get(row, "decision_type", None) == decision_type) for row in rows)


def _get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)
