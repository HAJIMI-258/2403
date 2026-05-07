from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from experiments.phase3r_utils import write_csv

TRACK_A_NAME = "track_a_bridge"
TRACK_C_NAME = "track_c_long_horizon"


def write_yaml(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def scenario_short_name(scenario_name: str) -> str:
    lookup = {
        TRACK_A_NAME: "TA",
        TRACK_C_NAME: "TC",
    }
    return lookup.get(str(scenario_name), str(scenario_name)[:2].upper())


def classify_event_type(row: dict[str, Any]) -> str:
    old_lineage = row.get("old_lineage_id")
    old_continuity = row.get("old_continuity_lineage_id")
    gap_length = int(row.get("gap_length", 0) or 0)
    continuation_related = any(
        int(row.get(key, 0) or 0) > 0
        for key in (
            "continuation_bank_exists",
            "continuation_attempted",
            "continuation_success",
            "resurrected_from_continuation",
        )
    )
    if old_lineage is not None and old_continuity is not None and int(old_lineage) != int(old_continuity):
        return "continuity_owner_conflict"
    if continuation_related:
        return "reentry_after_archive"
    if gap_length >= 48:
        return "long_gap_reentry"
    return "reentry"


def build_event_ledger_entries(
    scenario_name: str,
    event_rows: list[dict[str, Any]],
    *,
    recovery_window: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    short = scenario_short_name(scenario_name)
    for index, row in enumerate(event_rows, start=1):
        event_type = classify_event_type(row)
        family_prefix = {
            "continuity_owner_conflict": "M-OWN",
            "reentry_after_archive": "M-ARC",
            "long_gap_reentry": "M-LG",
            "reentry": "M-RE",
        }.get(event_type, "M-RE")
        gap_length = int(row.get("gap_length", 0) or 0)
        ambiguity_level = "high" if short == "TC" and gap_length >= 48 else "medium"
        entries.append(
            {
                "event_id": f"{family_prefix}-{short}-{index:03d}",
                "runtime_event_index": int(row.get("event_id", index - 1)),
                "sequence_id": int(row.get("sequence_id", 0) or 0),
                "scenario_name": str(scenario_name),
                "event_family": "M",
                "event_type": str(event_type),
                "frame_start": int(row.get("disappear_frame", 0) or 0),
                "frame_end": int(row.get("reappear_frame", 0) or 0) + int(recovery_window),
                "target_instance_id": int(row.get("instance_id", -1)),
                "target_lineage_id": row.get("old_continuity_lineage_id")
                if row.get("old_continuity_lineage_id") is not None
                else row.get("old_lineage_id"),
                "target_track_id": row.get("old_track_id"),
                "target_prototype_id": row.get("old_prototype_id"),
                "gap_length": gap_length,
                "gap_bucket": str(row.get("gap_bucket", "")),
                "expected_outcome": "same-lineage_then_same-track",
                "ambiguity_level": ambiguity_level,
                "required_traces": [
                    "source_pool",
                    "claim_ranking",
                    "attach_state",
                    "promotion_state",
                    "owner_transition",
                ],
                "notes": "",
            }
        )
    return entries


def classify_failure_bucket(row: dict[str, Any]) -> tuple[str, str, str]:
    proposal_detected = int(row.get("proposal_detected", 0) or 0)
    candidate_pool_nonempty = int(row.get("candidate_pool_nonempty", 0) or 0)
    continuation_bank_exists = int(row.get("continuation_bank_exists", 0) or 0)
    same_lineage = int(row.get("same_lineage_id", 0) or 0)
    same_track_after_concept = int(row.get("same_track_after_concept_recovery", 0) or 0)
    same_prototype = int(row.get("same_prototype_id", 0) or 0)
    gap_length = int(row.get("gap_length", 0) or 0)
    continuity_exists = bool(row.get("old_continuity_lineage_id") is not None)

    if proposal_detected == 0:
        return ("F0", "perception", "proposal_missing")
    if candidate_pool_nonempty == 0 and continuation_bank_exists == 0:
        if continuity_exists and gap_length >= 48:
            return ("F9", "governance", "continuity_source_missing_under_long_gap")
        return ("F4", "source_visibility", "target_continuity_source_not_visible")
    if same_lineage == 0:
        return ("F3", "lineage_routing", "wrong_lineage_after_reentry")
    if same_track_after_concept == 0:
        return ("F7", "identity_attach", "lineage_recovered_but_old_track_not_restored")
    if same_prototype == 0:
        return ("F8", "prototype_head", "old_track_restored_but_strict_prototype_continuity_lost")
    return ("PASS", "success", "expected_recovery_path")


def build_stage_e1_event_rows(
    scenario_name: str,
    event_rows: list[dict[str, Any]],
    *,
    recovery_window: int,
) -> list[dict[str, Any]]:
    ledger_entries = build_event_ledger_entries(
        scenario_name,
        event_rows,
        recovery_window=recovery_window,
    )
    runtime_to_ledger = {
        int(entry["runtime_event_index"]): str(entry["event_id"]) for entry in ledger_entries
    }
    rows: list[dict[str, Any]] = []
    for row in event_rows:
        failure_code, failure_layer, failure_reason = classify_failure_bucket(row)
        rows.append(
            {
                "ledger_event_id": runtime_to_ledger.get(int(row.get("event_id", -1)), ""),
                "runtime_event_index": int(row.get("event_id", -1)),
                "scenario_name": str(scenario_name),
                "sequence_id": int(row.get("sequence_id", 0) or 0),
                "instance_id": int(row.get("instance_id", -1)),
                "frame_id": int(row.get("frame_id", 0) or 0),
                "disappear_frame": int(row.get("disappear_frame", 0) or 0),
                "reappear_frame": int(row.get("reappear_frame", 0) or 0),
                "gap_length": int(row.get("gap_length", 0) or 0),
                "gap_bucket": str(row.get("gap_bucket", "")),
                "event_type": str(classify_event_type(row)),
                "old_track_id": row.get("old_track_id"),
                "old_prototype_id": row.get("old_prototype_id"),
                "old_lineage_id": row.get("old_lineage_id"),
                "old_continuity_lineage_id": row.get("old_continuity_lineage_id"),
                "matched_prototype_id": row.get("matched_prototype_id"),
                "matched_lineage_id": row.get("matched_lineage_id"),
                "matched_continuity_lineage_id": row.get("matched_continuity_lineage_id"),
                "proposal_detected": int(row.get("proposal_detected", 0) or 0),
                "proposal_detect_rate_proxy": float(int(row.get("proposal_detected", 0) or 0)),
                "objectness_at_reentry": float(row.get("objectness_at_reentry", 0.0) or 0.0),
                "candidate_pool_size": int(row.get("candidate_pool_size", 0) or 0),
                "live_candidate_pool_size": int(row.get("live_candidate_pool_size", 0) or 0),
                "continuation_bank_size": int(row.get("continuation_bank_size", 0) or 0),
                "candidate_pool_nonempty": int(row.get("candidate_pool_nonempty", 0) or 0),
                "continuation_bank_exists": int(row.get("continuation_bank_exists", 0) or 0),
                "continuation_attempted": int(row.get("continuation_attempted", 0) or 0),
                "continuation_success": int(row.get("continuation_success", 0) or 0),
                "concept_recovered": int(row.get("concept_recovered", 0) or 0),
                "same_prototype_id": int(row.get("same_prototype_id", 0) or 0),
                "same_lineage_id": int(row.get("same_lineage_id", 0) or 0),
                "same_continuity_lineage_id": int(row.get("same_continuity_lineage_id", 0) or 0),
                "same_track": int(row.get("matched_same_track", 0) or 0),
                "same_track_after_concept": int(row.get("same_track_after_concept_recovery", 0) or 0),
                "same_track_after_lineage": int(row.get("same_track_after_lineage_recovery", 0) or 0),
                "concept_only_recovery": int(row.get("concept_only_recovery", 0) or 0),
                "new_track_created": int(row.get("new_track_created", 0) or 0),
                "new_prototype_created": int(row.get("new_prototype_created", 0) or 0),
                "pfr_delta_if_any": int(row.get("pfr_delta_if_any", 0) or 0),
                "idsw_after_reentry_window": int(row.get("idsw_after_reentry_window", 0) or 0),
                "failure_code": failure_code,
                "failure_layer": failure_layer,
                "failure_reason": failure_reason,
            }
        )
    return rows


def build_stage_e1_failure_rows(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scenario_layer: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in event_rows:
        key = (str(row["scenario_name"]), str(row["failure_layer"]))
        by_scenario_layer[key][str(row["failure_code"])] += 1
    rows: list[dict[str, Any]] = []
    for (scenario_name, failure_layer), counter in sorted(by_scenario_layer.items()):
        total = sum(counter.values())
        for failure_code, count in sorted(counter.items()):
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "failure_layer": failure_layer,
                    "failure_code": failure_code,
                    "count": int(count),
                    "ratio_within_layer": float(count / total) if total else 0.0,
                }
            )
    return rows


def summarize_stage_e1(
    scenario_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "scenarios": {},
        "overall": {},
    }
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        by_scenario[str(row["scenario_name"])].append(row)

    for row in scenario_rows:
        scenario_name = str(row["scenario_name"])
        events = by_scenario.get(scenario_name, [])
        failure_counter = Counter(str(item["failure_layer"]) for item in events)
        summary["scenarios"][scenario_name] = {
            "u_recall": float(row["u_recall"]),
            "pfr": float(row["pfr"]),
            "track_idsw": int(row["track_idsw"]),
            "memory_growth": float(row["memory_growth"]),
            "reentry_events": len(events),
            "svr_proxy": _mean_int(events, "candidate_pool_nonempty"),
            "same_lineage": _mean_int(events, "same_lineage_id"),
            "same_prototype": _mean_int(events, "same_prototype_id"),
            "same_track": _mean_int(events, "same_track"),
            "same_track_after_concept": _mean_int(events, "same_track_after_concept"),
            "failure_layers": dict(failure_counter),
        }

    summary["overall"] = {
        "num_events": len(event_rows),
        "svr_proxy": _mean_int(event_rows, "candidate_pool_nonempty"),
        "same_lineage": _mean_int(event_rows, "same_lineage_id"),
        "same_prototype": _mean_int(event_rows, "same_prototype_id"),
        "same_track": _mean_int(event_rows, "same_track"),
        "same_track_after_concept": _mean_int(event_rows, "same_track_after_concept"),
    }
    return summary


def render_stage_e1_report(summary: dict[str, Any], event_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage E1 Report",
        "",
        "## 目标",
        "",
        "对当前系统做 baseline forensic audit。先看失败主层分布，不做机制修补。",
        "",
        "## 总览",
        "",
        f"- `num_events = {int(summary['overall']['num_events'])}`",
        f"- `SVR proxy = {float(summary['overall']['svr_proxy']):.4f}`",
        f"- `same_lineage = {float(summary['overall']['same_lineage']):.4f}`",
        f"- `same_prototype = {float(summary['overall']['same_prototype']):.4f}`",
        f"- `same_track = {float(summary['overall']['same_track']):.4f}`",
        f"- `STAC = {float(summary['overall']['same_track_after_concept']):.4f}`",
        "",
        "## Track A / Track C",
        "",
    ]
    for scenario_name, row in summary["scenarios"].items():
        lines.extend(
            [
                f"### {scenario_name}",
                "",
                f"- `U-Recall = {float(row['u_recall']):.4f}`",
                f"- `PFR = {float(row['pfr']):.4f}`",
                f"- `IDSW = {int(row['track_idsw'])}`",
                f"- `reentry_events = {int(row['reentry_events'])}`",
                f"- `SVR proxy = {float(row['svr_proxy']):.4f}`",
                f"- `same_lineage = {float(row['same_lineage']):.4f}`",
                f"- `same_prototype = {float(row['same_prototype']):.4f}`",
                f"- `same_track = {float(row['same_track']):.4f}`",
                f"- `STAC = {float(row['same_track_after_concept']):.4f}`",
                f"- `failure_layers = {row['failure_layers']}`",
                "",
            ]
        )

    top_failures = sorted(
        [row for row in event_rows if str(row["failure_code"]) != "PASS"],
        key=lambda item: (
            int(item.get("candidate_pool_nonempty", 0)),
            int(item.get("same_track_after_concept", 0)),
            -int(item.get("gap_length", 0)),
            -int(item.get("pfr_delta_if_any", 0)),
        ),
    )[:10]
    lines.extend(
        [
            "## Top Failures",
            "",
            "| ledger_event_id | scenario | gap | failure_layer | failure_reason | same_lineage | same_proto | same_track |",
            "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in top_failures:
        lines.append(
            f"| {row['ledger_event_id']} | {row['scenario_name']} | {int(row['gap_length'])} | "
            f"{row['failure_layer']} | {row['failure_reason']} | {int(row['same_lineage_id'])} | "
            f"{int(row['same_prototype_id'])} | {int(row['same_track'])} |"
        )
    if not top_failures:
        lines.append("| none | none | 0 | none | none | 0 | 0 | 0 |")
    return "\n".join(lines) + "\n"


def _mean_int(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(int(row.get(key, 0) or 0) for row in rows) / len(rows))


__all__ = [
    "TRACK_A_NAME",
    "TRACK_C_NAME",
    "build_event_ledger_entries",
    "build_stage_e1_event_rows",
    "build_stage_e1_failure_rows",
    "classify_event_type",
    "render_stage_e1_report",
    "scenario_short_name",
    "summarize_stage_e1",
    "write_csv",
    "write_yaml",
]
