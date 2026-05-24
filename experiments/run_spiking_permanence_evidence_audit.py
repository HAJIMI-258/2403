"""Evidence audit for spiking object permanence decisions.

This script is intentionally diagnostic-only. It does not change the capsule
memory, recognizer thresholds, or matching scores. It separates re-entry rows
by eval-only correctness labels and measures whether online evidence fields can
distinguish true top-1 matches from false top-1 matches.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_spiking_morph_permanence_eval import run_eval  # noqa: E402


EVIDENCE_FIELDS = [
    "score",
    "top1_margin",
    "false_resurrection_risk",
    "spike_score",
    "deformation_score",
    "identity_score",
    "gray_appearance_score",
    "chromatic_score",
    "hash_score",
    "conflict_score",
    "true_capsule_score",
    "score_gap_top1_minus_true",
    "true_spike_score",
    "true_deformation_score",
    "true_identity_score",
    "true_chromatic_score",
    "true_hash_score",
    "delta_top1_minus_true_spike",
    "delta_top1_minus_true_deformation",
    "delta_top1_minus_true_identity",
    "delta_top1_minus_true_chromatic",
    "delta_top1_minus_true_hash",
]

AUDIT_FIELDS = [
    "event_id",
    "object_id",
    "scale_change",
    "aspect_change",
    "brightness_drift",
    "occlusion",
    "distractor_level",
    "decision_type",
    "matched_capsule_id",
    "true_capsule_id",
    "top1_is_true_capsule",
    "accepted_as_same",
    "same_instance_success",
    "false_resurrection",
    "top1_true_but_rejected",
    "top1_false_but_accepted",
    "true_capsule_rank",
    *EVIDENCE_FIELDS,
]

DISTRIBUTION_FIELDS = [
    "group",
    "field",
    "count",
    "mean",
    "median",
    "min",
    "p10",
    "p25",
    "p75",
    "p90",
    "max",
]

PREDICATE_FIELDS = [
    "predicate_id",
    "score_threshold",
    "margin_threshold",
    "risk_threshold",
    "chromatic_threshold",
    "hash_threshold",
    "deformation_threshold",
    "accepted_count",
    "true_accept_count",
    "false_accept_count",
    "precision",
    "recall",
    "false_resurrection_rate",
    "selected_best_zero_false",
    "selected_best_safe",
]


def run_audit(
    events_csv: str | Path = "results/spiking_morph_permanence_eval/events.csv",
    output_dir: str | Path = "results/spiking_permanence_evidence_audit",
    rerun_eval: bool = False,
    seed: int = 7,
    object_count: int = 16,
    events_per_object: int = 4,
    max_capsules: int = 32,
    spike_dim: int = 128,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_csv = Path(events_csv)
    if rerun_eval or not source_csv.exists():
        eval_dir = out / "source_eval"
        run_eval(
            output_dir=eval_dir,
            seed=seed,
            object_count=object_count,
            events_per_object=events_per_object,
            max_capsules=max_capsules,
            spike_dim=spike_dim,
        )
        source_csv = eval_dir / "events.csv"

    rows = _read_events(source_csv)
    reentry = [row for row in rows if row.get("phase") == "reentry"]
    audit_rows = [_audit_row(row) for row in reentry]
    distribution_rows = _distribution_rows(audit_rows)
    predicate_rows = _predicate_rows(audit_rows)
    best_zero_false = _select_best_zero_false(predicate_rows)
    best_safe = _select_best_safe(predicate_rows)
    if best_zero_false is not None:
        best_zero_false["selected_best_zero_false"] = 1
    if best_safe is not None:
        best_safe["selected_best_safe"] = 1

    summary = _summary(audit_rows, distribution_rows, predicate_rows, best_zero_false, best_safe, source_csv)
    _write_csv(out / "evidence_audit.csv", audit_rows, AUDIT_FIELDS)
    _write_csv(out / "evidence_distribution_summary.csv", distribution_rows, DISTRIBUTION_FIELDS)
    _write_csv(out / "predicate_candidate_summary.csv", predicate_rows, PREDICATE_FIELDS)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _read_events(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _audit_row(row: dict[str, Any]) -> dict[str, Any]:
    accepted = row.get("decision_type") in {"same_object", "familiar_but_deformed"}
    top1_true = _to_int(row.get("top1_is_true_capsule", 0)) == 1
    false_resurrection = _to_int(row.get("false_resurrection", 0)) == 1
    output = {
        "event_id": row.get("event_id", ""),
        "object_id": row.get("object_id", ""),
        "scale_change": _to_float(row.get("scale_change", 0.0)),
        "aspect_change": _to_float(row.get("aspect_change", 0.0)),
        "brightness_drift": _to_float(row.get("brightness_drift", 0.0)),
        "occlusion": _to_float(row.get("occlusion", 0.0)),
        "distractor_level": row.get("distractor_level", ""),
        "decision_type": row.get("decision_type", ""),
        "matched_capsule_id": row.get("matched_capsule_id", ""),
        "true_capsule_id": row.get("true_capsule_id", ""),
        "top1_is_true_capsule": int(top1_true),
        "accepted_as_same": int(accepted),
        "same_instance_success": _to_int(row.get("same_instance_success", 0)),
        "false_resurrection": int(false_resurrection),
        "top1_true_but_rejected": int(top1_true and not accepted),
        "top1_false_but_accepted": int((not top1_true) and accepted),
        "true_capsule_rank": _to_int(row.get("true_capsule_rank", 1 if top1_true else 0)),
    }
    for field in EVIDENCE_FIELDS:
        output[field] = _to_float(row.get(field, 0.0))
    return output


def _distribution_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        "all_reentry": rows,
        "top1_true": [row for row in rows if int(row["top1_is_true_capsule"]) == 1],
        "top1_false": [row for row in rows if int(row["top1_is_true_capsule"]) == 0],
        "accepted_true": [row for row in rows if int(row["same_instance_success"]) == 1],
        "accepted_false": [row for row in rows if int(row["false_resurrection"]) == 1],
        "rejected_true": [row for row in rows if int(row["top1_true_but_rejected"]) == 1],
    }
    output: list[dict[str, Any]] = []
    for group, group_rows in groups.items():
        for field in EVIDENCE_FIELDS:
            output.append({"group": group, "field": field, **_stats([_to_float(row[field]) for row in group_rows])})
    return output


def _predicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    score_thresholds = [0.72, 0.80, 0.86, 0.90, 0.92]
    margin_thresholds = [0.04, 0.06, 0.10, 0.14]
    risk_thresholds = [0.25, 0.30, 0.35, 0.45, 0.60]
    chromatic_thresholds = [0.00, 0.50, 0.70]
    hash_thresholds = [0.00, 0.55, 0.65]
    deformation_thresholds = [0.00, 0.45, 0.60]
    output: list[dict[str, Any]] = []
    total_true = sum(1 for row in rows if int(row["top1_is_true_capsule"]) == 1)
    total_events = max(1, len(rows))
    predicate_index = 0
    for score_t in score_thresholds:
        for margin_t in margin_thresholds:
            for risk_t in risk_thresholds:
                for chromatic_t in chromatic_thresholds:
                    for hash_t in hash_thresholds:
                        for deformation_t in deformation_thresholds:
                            accepted = [
                                row
                                for row in rows
                                if float(row["score"]) >= score_t
                                and float(row["top1_margin"]) >= margin_t
                                and float(row["false_resurrection_risk"]) < risk_t
                                and float(row["chromatic_score"]) >= chromatic_t
                                and float(row["hash_score"]) >= hash_t
                                and float(row["deformation_score"]) >= deformation_t
                            ]
                            true_accept = sum(1 for row in accepted if int(row["top1_is_true_capsule"]) == 1)
                            false_accept = len(accepted) - true_accept
                            output.append(
                                {
                                    "predicate_id": f"P{predicate_index:04d}",
                                    "score_threshold": score_t,
                                    "margin_threshold": margin_t,
                                    "risk_threshold": risk_t,
                                    "chromatic_threshold": chromatic_t,
                                    "hash_threshold": hash_t,
                                    "deformation_threshold": deformation_t,
                                    "accepted_count": len(accepted),
                                    "true_accept_count": true_accept,
                                    "false_accept_count": false_accept,
                                    "precision": _safe_rate(true_accept, len(accepted)),
                                    "recall": _safe_rate(true_accept, total_true),
                                    "false_resurrection_rate": float(false_accept) / float(total_events),
                                    "selected_best_zero_false": 0,
                                    "selected_best_safe": 0,
                                }
                            )
                            predicate_index += 1
    return output


def _select_best_zero_false(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if int(row["false_accept_count"]) == 0 and int(row["true_accept_count"]) > 0
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row["recall"]),
            int(row["accepted_count"]),
            -float(row["score_threshold"]),
            -float(row["margin_threshold"]),
        ),
    )


def _select_best_safe(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    nonzero = [row for row in rows if int(row["true_accept_count"]) > 0]
    candidates = nonzero if nonzero else rows
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            int(row["false_accept_count"]),
            -float(row["recall"]),
            -float(row["precision"]),
            -int(row["accepted_count"]),
        ),
    )


def _summary(
    rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    predicate_rows: list[dict[str, Any]],
    best_zero_false: dict[str, Any] | None,
    best_safe: dict[str, Any] | None,
    source_csv: Path,
) -> dict[str, Any]:
    reentry_count = len(rows)
    top1_true_count = sum(1 for row in rows if int(row["top1_is_true_capsule"]) == 1)
    true_top3_count = sum(1 for row in rows if 1 <= int(row.get("true_capsule_rank", 0)) <= 3)
    true_top5_count = sum(1 for row in rows if 1 <= int(row.get("true_capsule_rank", 0)) <= 5)
    accepted_true_count = sum(1 for row in rows if int(row["same_instance_success"]) == 1)
    accepted_false_count = sum(1 for row in rows if int(row["false_resurrection"]) == 1)
    rejected_true_count = sum(1 for row in rows if int(row["top1_true_but_rejected"]) == 1)
    top1_false_count = reentry_count - top1_true_count
    separability = _separability(distribution_rows)
    return {
        "source_events_csv": str(source_csv),
        "reentry_event_count": reentry_count,
        "top1_true_count": top1_true_count,
        "top1_true_rate": _safe_rate(top1_true_count, reentry_count),
        "true_capsule_top3_count": true_top3_count,
        "true_capsule_top3_rate": _safe_rate(true_top3_count, reentry_count),
        "true_capsule_top5_count": true_top5_count,
        "true_capsule_top5_rate": _safe_rate(true_top5_count, reentry_count),
        "top1_false_count": top1_false_count,
        "accepted_true_count": accepted_true_count,
        "accepted_false_count": accepted_false_count,
        "accepted_precision": _safe_rate(accepted_true_count, accepted_true_count + accepted_false_count),
        "accepted_recall_vs_top1_true": _safe_rate(accepted_true_count, top1_true_count),
        "top1_true_but_rejected_count": rejected_true_count,
        "top1_true_but_rejected_rate": _safe_rate(rejected_true_count, reentry_count),
        "predicate_candidate_count": len(predicate_rows),
        "best_zero_false_predicate": dict(best_zero_false or {}),
        "best_safe_predicate": dict(best_safe or {}),
        "evidence_separability": separability,
        "main_diagnosis": _diagnosis(top1_true_count, rejected_true_count, accepted_false_count, separability),
    }


def _separability(distribution_rows: list[dict[str, Any]]) -> dict[str, float]:
    by_key = {(row["group"], row["field"]): row for row in distribution_rows}
    output: dict[str, float] = {}
    for field in EVIDENCE_FIELDS:
        true_mean = _to_float(by_key.get(("top1_true", field), {}).get("mean", 0.0))
        false_mean = _to_float(by_key.get(("top1_false", field), {}).get("mean", 0.0))
        output[f"{field}_true_minus_false_mean"] = true_mean - false_mean
    return output


def _diagnosis(
    top1_true_count: int,
    rejected_true_count: int,
    accepted_false_count: int,
    separability: dict[str, float],
) -> str:
    if top1_true_count <= 0:
        return "retrieval_top1_alignment_failed"
    if accepted_false_count > 0 and separability.get("score_true_minus_false_mean", 0.0) < 0.05:
        return "true_false_evidence_overlap_blocks_safe_release"
    if rejected_true_count > 0:
        return "decision_gate_overblocks_true_top1"
    return "predicate_boundary_appears_clean"


def _stats(values: list[float]) -> dict[str, float | int]:
    clean = [float(value) for value in values if value == value]
    if not clean:
        return {key: 0.0 for key in ("mean", "median", "min", "p10", "p25", "p75", "p90", "max")} | {"count": 0}
    sorted_values = sorted(clean)
    return {
        "count": len(clean),
        "mean": float(sum(clean) / len(clean)),
        "median": float(median(clean)),
        "min": float(sorted_values[0]),
        "p10": _percentile(sorted_values, 0.10),
        "p25": _percentile(sorted_values, 0.25),
        "p75": _percentile(sorted_values, 0.75),
        "p90": _percentile(sorted_values, 0.90),
        "max": float(sorted_values[-1]),
    }


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    pos = float(q) * float(len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return float(sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac)


def _to_float(value: Any) -> float:
    try:
        if value in {None, ""}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    try:
        if value in {None, ""}:
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _report(summary: dict[str, Any]) -> str:
    best_zero = summary.get("best_zero_false_predicate", {})
    best_safe = summary.get("best_safe_predicate", {})
    return (
        "# Spiking Permanence Evidence Audit\n\n"
        f"- reentry_event_count: {summary.get('reentry_event_count', 0)}\n"
        f"- top1_true_rate: {float(summary.get('top1_true_rate', 0.0)):.4f}\n"
        f"- accepted_precision: {float(summary.get('accepted_precision', 0.0)):.4f}\n"
        f"- accepted_recall_vs_top1_true: {float(summary.get('accepted_recall_vs_top1_true', 0.0)):.4f}\n"
        f"- top1_true_but_rejected_count: {summary.get('top1_true_but_rejected_count', 0)}\n"
        f"- main_diagnosis: {summary.get('main_diagnosis', '')}\n\n"
        "## Best Zero-False Predicate\n\n"
        f"- predicate_id: {best_zero.get('predicate_id', '')}\n"
        f"- recall: {float(best_zero.get('recall', 0.0)):.4f}\n"
        f"- precision: {float(best_zero.get('precision', 0.0)):.4f}\n\n"
        "## Best Safe Predicate\n\n"
        f"- predicate_id: {best_safe.get('predicate_id', '')}\n"
        f"- recall: {float(best_safe.get('recall', 0.0)):.4f}\n"
        f"- false_resurrection_rate: {float(best_safe.get('false_resurrection_rate', 0.0)):.4f}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-csv", default="results/spiking_morph_permanence_eval/events.csv")
    parser.add_argument("--output-dir", default="results/spiking_permanence_evidence_audit")
    parser.add_argument("--rerun-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--object-count", type=int, default=16)
    parser.add_argument("--events-per-object", type=int, default=4)
    parser.add_argument("--max-capsules", type=int, default=32)
    parser.add_argument("--spike-dim", type=int, default=128)
    summary = run_audit(**vars(parser.parse_args()))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
