"""Metrics for bounded object permanence memory."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def same_instance_reentry_recall(rows: Iterable[dict[str, Any]]) -> float:
    rows = [row for row in rows if _is_reentry(row)]
    return _mean([bool(row.get("same_instance_success", False)) for row in rows])


def false_resurrection_rate(rows: Iterable[dict[str, Any]]) -> float:
    rows = [row for row in rows if _is_reentry(row)]
    return _mean([bool(row.get("false_resurrection", False)) for row in rows])


def memory_growth_rate(memory_sizes: Iterable[int | float]) -> float:
    values = [float(v) for v in memory_sizes]
    if len(values) <= 1:
        return 0.0
    return float((values[-1] - values[0]) / (len(values) - 1))


def bytes_per_capsule(memory_bytes: int | float, capsule_count: int | float) -> float:
    return 0.0 if float(capsule_count) <= 0.0 else float(memory_bytes) / float(capsule_count)


def mean_spike_density(spike_densities: Iterable[int | float]) -> float:
    return _mean([float(value) for value in spike_densities])


def deformation_tolerance_curve(rows: Iterable[dict[str, Any]], field: str = "scale_change") -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, ""))].append(row)
    return {key: same_instance_reentry_recall(group) for key, group in sorted(grouped.items())}


def stability_plasticity_score(rows: Iterable[dict[str, Any]]) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    recall = same_instance_reentry_recall(rows)
    false_rate = false_resurrection_rate(rows)
    growth = memory_growth_rate([row.get("capsule_count", 0) for row in rows])
    return float(max(0.0, recall - false_rate - 0.01 * max(0.0, growth)))


def _is_reentry(row: dict[str, Any]) -> bool:
    phase = str(row.get("phase", ""))
    return phase in {"reentry", "reappear", "query"} or bool(row.get("is_reentry", False))


def _mean(values: Iterable[Any]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))
