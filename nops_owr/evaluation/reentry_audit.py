"""Shared utilities for re-entry memory evaluation.

These helpers keep synthetic and external-video re-entry reports on the same
metric vocabulary. Ground-truth identifiers may be passed into these utilities
for audit/reporting, but the online cognitive loop must not consume them for
retrieval or recognition decisions.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


Box = tuple[float, float, float, float]


def bbox_iou(left: Box | None, right: Box | None) -> float:
    if left is None or right is None:
        return 0.0
    lx1, ly1, lx2, ly2 = [float(v) for v in left]
    rx1, ry1, rx2, ry2 = [float(v) for v in right]
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def gap_bucket(gap_length: int) -> str:
    gap_length = int(gap_length)
    if gap_length <= 3:
        return "gap_1_3"
    if gap_length <= 7:
        return "gap_4_7"
    if gap_length <= 15:
        return "gap_8_15"
    return "gap_16_plus"


def failure_bucket(
    *,
    gt_box_present: bool,
    matched_object: bool,
    object_attended: bool,
    target_episode: bool,
    target_rank: int,
    decision_type: str,
    rejection_reason: str,
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
    if not target_episode:
        return "target_episode_missing"
    if int(target_rank) == 0:
        return "target_not_in_topk"
    if int(target_rank) > 1:
        return "target_in_topk_but_low_rank"
    if rejection_reason == "low_retrieval_margin":
        return "low_margin_uncertain"
    if rejection_reason == "active_conflict":
        return "active_conflict_blocked"
    if false_resurrection:
        return "false_resurrection"
    if decision_type != "same_instance":
        return "decision_underconfident"
    return "target_in_topk_but_low_rank"


def find_target_episode_from_bundles(
    bundles: Iterable[Any],
    instance_id: int,
    reappear_frame: int,
) -> Any | None:
    """Find the old episode for an instance at the reappearance frame.

    The selected episode must pre-date the reappearance frame. This prevents an
    eval-only audit from counting a new post-reentry write as successful memory.
    """

    candidates = [
        bundle
        for bundle in bundles
        if getattr(bundle, "metadata", {}).get("gt_instance_id") == int(instance_id)
        and int(getattr(bundle, "frame_start", 0)) < int(reappear_frame)
        and int(
            getattr(bundle, "last_observed_frame", None)
            if getattr(bundle, "last_observed_frame", None) is not None
            else getattr(bundle, "frame_end", 0)
        )
        < int(reappear_frame)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda bundle: (
            int(bool(getattr(bundle, "closed", False))),
            int(
                getattr(bundle, "last_observed_frame", None)
                if getattr(bundle, "last_observed_frame", None) is not None
                else getattr(bundle, "frame_end", 0)
            ),
            int(getattr(bundle, "observation_count", 0)),
        ),
    )


def summarize_reentry_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    sequence_count: int,
    evaluated_sequence_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    count = len(rows)
    matched_count = sum(int(str(row.get("matched_object_file_id", "")) != "") for row in rows)
    attended_count = sum(int(row.get("object_attended", 0)) for row in rows)
    target_present = sum(int(row.get("target_episode_exists", 0)) for row in rows)
    target_top1 = sum(int(row.get("target_episode_rank", 0) == 1) for row in rows)
    target_top3 = sum(int(1 <= int(row.get("target_episode_rank", 0)) <= 3) for row in rows)
    target_top5 = sum(int(1 <= int(row.get("target_episode_rank", 0)) <= 5) for row in rows)
    successes = sum(int(row.get("success_same_instance", 0)) for row in rows)
    false_resurrections = sum(int(row.get("false_resurrection", 0)) for row in rows)
    same_instance = sum(int(row.get("decision_type", "") == "same_instance") for row in rows)
    unresolved_target = sum(int(row.get("unresolved_but_target_in_topk", 0)) for row in rows)
    buckets = Counter(str(row.get("failure_bucket", "unknown")) for row in rows)
    category_metrics = _group_metrics(rows, "category")
    gap_metrics = _group_metrics(rows, "gap_bucket")
    summary = {
        "dataset_name": dataset_name,
        "sequence_count": int(sequence_count),
        "evaluated_sequence_count": int(evaluated_sequence_count),
        "reentry_event_count": int(count),
        "proposal_recall_at_reentry": _safe_div(matched_count, count),
        "attention_recall_at_reentry": _safe_div(attended_count, count),
        "target_episode_presence_rate": _safe_div(target_present, count),
        "target_episode_top1_rate": _safe_div(target_top1, count),
        "target_episode_top3_rate": _safe_div(target_top3, count),
        "target_episode_top5_rate": _safe_div(target_top5, count),
        "same_instance_precision_at_reentry": _safe_div(successes, same_instance),
        "same_instance_recall_at_reentry": _safe_div(successes, count),
        "false_resurrection_rate_at_reentry": _safe_div(false_resurrections, count),
        "unresolved_but_target_in_topk_rate": _safe_div(unresolved_target, count),
        "failure_buckets": dict(buckets),
        "category_metrics": category_metrics,
        "gap_bucket_metrics": gap_metrics,
        "benchmark_status": "valid" if count > 0 else "invalid_no_reentry_events",
    }
    if extra:
        summary.update(extra)
    return summary


def write_reentry_report(
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
    false_examples = [row for row in rows if int(row.get("false_resurrection", 0))][:5]
    examples = "\n".join(
        f"- sequence={row.get('sequence_id')} category={row.get('category', '')} "
        f"instance={row.get('instance_id')} top1_episode={row.get('top1_episode_id')} "
        f"top1_gt={row.get('top1_gt_instance_id')} margin={float(row.get('top1_margin', 0.0)):.4f}"
        for row in false_examples
    ) or "- none"
    text = (
        f"# {title}\n\n"
        "This is an eval-only re-entry audit. GT boxes/ids are used for ledger and "
        "failure buckets, not for online retrieval or recognition decisions.\n\n"
        f"- benchmark_status: {summary.get('benchmark_status')}\n"
        f"- reentry_event_count: {summary.get('reentry_event_count', 0)}\n"
        f"- proposal_recall_at_reentry: {summary.get('proposal_recall_at_reentry', 0.0):.4f}\n"
        f"- attention_recall_at_reentry: {summary.get('attention_recall_at_reentry', 0.0):.4f}\n"
        f"- target_episode_presence_rate: {summary.get('target_episode_presence_rate', 0.0):.4f}\n"
        f"- target_episode_top5_rate: {summary.get('target_episode_top5_rate', 0.0):.4f}\n"
        f"- same_instance_recall_at_reentry: {summary.get('same_instance_recall_at_reentry', 0.0):.4f}\n"
        f"- false_resurrection_rate_at_reentry: {summary.get('false_resurrection_rate_at_reentry', 0.0):.4f}\n\n"
        "## Failure Buckets\n\n"
        + "\n".join(table)
        + "\n\n"
        "## False Resurrection Examples\n\n"
        + examples
        + "\n\n"
        "## Summary JSON\n\n```json\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n```\n"
    )
    Path(path).write_text(text, encoding="utf-8")


def _safe_div(num: int | float, den: int | float) -> float:
    return 0.0 if float(den) == 0.0 else float(num) / float(den)


def _group_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "unknown"))].append(row)
    output: dict[str, dict[str, float]] = {}
    for key, group in sorted(grouped.items()):
        count = len(group)
        output[key] = {
            "count": float(count),
            "success_rate": _safe_div(sum(int(row.get("success_same_instance", 0)) for row in group), count),
            "false_resurrection_rate": _safe_div(
                sum(int(row.get("false_resurrection", 0)) for row in group),
                count,
            ),
            "target_top5_rate": _safe_div(sum(int(row.get("topk_contains_target", 0)) for row in group), count),
        }
    return output
