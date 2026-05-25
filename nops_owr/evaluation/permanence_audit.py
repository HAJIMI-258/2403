"""Shared audit utilities for spiking object permanence evaluation.

The helpers in this file may use GT ids for reporting, but online spiking
memory matching and permanence decisions must not consume those ids.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def find_target_capsule_from_bank(bank: Any, instance_id: int, reappear_frame: int) -> Any | None:
    """Find an old eval-tagged capsule for an instance before reappearance."""

    capsules = getattr(bank, "capsules", {}) or {}
    candidates = []
    for capsule in capsules.values():
        metadata = getattr(capsule, "metadata", {}) or {}
        object_id = metadata.get("object_id_eval_only", metadata.get("gt_instance_id"))
        if object_id != int(instance_id):
            continue
        if int(getattr(capsule, "created_frame", 10**9)) >= int(reappear_frame):
            continue
        candidates.append(capsule)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda capsule: (
            int(getattr(capsule, "last_seen_frame", -1)),
            int(getattr(capsule, "observation_count", 0)),
            float(getattr(capsule, "confidence", 0.0)),
        ),
    )


def permanence_failure_bucket(
    *,
    gt_box_present: bool,
    matched_object: bool,
    object_attended: bool,
    target_capsule: bool,
    target_rank: int,
    decision_type: str,
    success: bool,
    false_resurrection: bool,
) -> str:
    if success:
        return "success"
    if not gt_box_present:
        return "no_gt_box_at_reentry"
    if not matched_object:
        return "no_object_file_matched"
    if not object_attended:
        return "attention_missed_object"
    if not target_capsule:
        return "target_capsule_missing"
    if int(target_rank) == 0:
        return "target_capsule_not_in_top5"
    if int(target_rank) > 1:
        return "target_capsule_low_rank"
    if false_resurrection:
        return "false_resurrection"
    if decision_type == "false_resurrection_risk":
        return "false_resurrection_risk_blocked"
    if decision_type not in {"same_object", "familiar_but_deformed"}:
        return "decision_underconfident"
    return "target_capsule_low_rank"


def summarize_permanence_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    sequence_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    count = len(rows)
    matched_count = sum(int(str(row.get("matched_object_iou", 0.0)) not in {"", "0", "0.0"}) for row in rows)
    attended_count = sum(int(row.get("object_attended", 0)) for row in rows)
    target_present = sum(int(row.get("target_capsule_exists", 0)) for row in rows)
    target_top1 = sum(int(int(row.get("target_capsule_rank", 0)) == 1) for row in rows)
    target_top3 = sum(int(1 <= int(row.get("target_capsule_rank", 0)) <= 3) for row in rows)
    target_top5 = sum(int(1 <= int(row.get("target_capsule_rank", 0)) <= 5) for row in rows)
    success = sum(int(row.get("same_instance_success", 0)) for row in rows)
    false_resurrection = sum(int(row.get("false_resurrection", 0)) for row in rows)
    risk_block = sum(int(row.get("permanence_decision_type", "") == "false_resurrection_risk") for row in rows)
    buckets = Counter(str(row.get("failure_bucket", "unknown")) for row in rows)
    summary = {
        "dataset_name": dataset_name,
        "sequence_count": int(sequence_count),
        "evaluated_event_count": int(count),
        "reentry_event_count": int(count),
        "proposal_recall_at_reentry": _safe_div(matched_count, count),
        "attention_recall_at_reentry": _safe_div(attended_count, count),
        "target_capsule_presence_rate": _safe_div(target_present, count),
        "target_capsule_top1_rate": _safe_div(target_top1, count),
        "target_capsule_top3_rate": _safe_div(target_top3, count),
        "target_capsule_top5_rate": _safe_div(target_top5, count),
        "same_instance_recall_at_reentry": _safe_div(success, count),
        "false_resurrection_rate_at_reentry": _safe_div(false_resurrection, count),
        "false_resurrection_risk_block_rate": _safe_div(risk_block, count),
        "mean_capsule_count": _mean(row.get("capsule_count_after_reentry", 0) for row in rows),
        "mean_spiking_memory_bytes": _mean(row.get("spiking_memory_bytes", 0) for row in rows),
        "mean_spike_density": _mean(row.get("mean_spike_density", 0.0) for row in rows),
        "failure_buckets": dict(buckets),
        "category_metrics": _group_metrics(rows, "category"),
        "gap_bucket_metrics": _group_metrics(rows, "gap_bucket"),
        "benchmark_status": "valid" if count > 0 else "invalid_no_reentry_events",
    }
    if extra:
        summary.update(extra)
    return summary


def write_permanence_report(
    path: str | Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    title: str,
) -> None:
    buckets = Counter(str(row.get("failure_bucket", "unknown")) for row in rows)
    total = max(1, len(rows))
    table = ["| bucket | count | rate |", "|---|---:|---:|"]
    for bucket, count in buckets.most_common():
        table.append(f"| {bucket} | {count} | {count / total:.4f} |")
    text = (
        f"# {title}\n\n"
        "This is an eval-only permanence audit. GT boxes/ids are used for ledger "
        "and failure buckets, not for normal online spiking memory decisions.\n\n"
        f"- mode: {summary.get('mode', '')}\n"
        f"- evaluated_event_count: {summary.get('evaluated_event_count', 0)}\n"
        f"- target_capsule_top5_rate: {summary.get('target_capsule_top5_rate', 0.0):.4f}\n"
        f"- same_instance_recall_at_reentry: {summary.get('same_instance_recall_at_reentry', 0.0):.4f}\n"
        f"- false_resurrection_rate_at_reentry: {summary.get('false_resurrection_rate_at_reentry', 0.0):.4f}\n"
        f"- mean_spiking_memory_bytes: {summary.get('mean_spiking_memory_bytes', 0.0):.2f}\n\n"
        "## Failure Buckets\n\n"
        + "\n".join(table)
        + "\n\n## Summary JSON\n\n```json\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n```\n"
    )
    Path(path).write_text(text, encoding="utf-8")


def _safe_div(num: int | float, den: int | float) -> float:
    return 0.0 if float(den) == 0.0 else float(num) / float(den)


def _mean(values: Any) -> float:
    values = [float(value) for value in values]
    return 0.0 if not values else float(sum(values) / len(values))


def _group_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "unknown"))].append(row)
    output: dict[str, dict[str, float]] = {}
    for key, group in sorted(grouped.items()):
        count = len(group)
        output[key] = {
            "count": float(count),
            "target_capsule_top5_rate": _safe_div(
                sum(int(1 <= int(row.get("target_capsule_rank", 0)) <= 5) for row in group),
                count,
            ),
            "same_instance_recall": _safe_div(
                sum(int(row.get("same_instance_success", 0)) for row in group),
                count,
            ),
            "false_resurrection_rate": _safe_div(
                sum(int(row.get("false_resurrection", 0)) for row in group),
                count,
            ),
        }
    return output
