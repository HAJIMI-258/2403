"""Eval-only rerank sweep for spiking permanence top-k matches.

The sweep uses only online-available match evidence for scoring. Eval-only
capsule labels are used only after reranking to measure whether the true capsule
would move from top-k to top-1.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_spiking_morph_permanence_eval import run_eval  # noqa: E402


SUMMARY_FIELDS = [
    "profile_name",
    "event_count",
    "top1_true_rate",
    "true_top3_rate",
    "true_top5_rate",
    "mean_selected_score",
    "mean_selected_margin",
    "mean_true_rank",
    "high_conf_accept_rate",
    "high_conf_true_accept_rate",
    "high_conf_false_accept_rate",
    "selected_as_best",
]

RERANK_PROFILES: dict[str, dict[str, float]] = {
    "R0_current_adjusted": {"score": 1.0},
    "R1_base_score": {"base_score": 1.0},
    "R2_hash_chroma_deform": {"hash_score": 0.30, "chromatic_score": 0.25, "deformation_score": 0.20, "spike_score": 0.15, "identity_score": 0.10},
    "R3_identity_hash_chroma": {"identity_score": 0.30, "hash_score": 0.25, "chromatic_score": 0.25, "deformation_score": 0.10, "spike_score": 0.10},
    "R4_spike_hash_low_conflict": {"spike_score": 0.30, "hash_score": 0.25, "chromatic_score": 0.20, "deformation_score": 0.15, "identity_score": 0.10, "conflict_score": -0.10},
    "R5_deformation_identity": {"deformation_score": 0.30, "identity_score": 0.25, "hash_score": 0.20, "chromatic_score": 0.15, "score": 0.10},
    "R6_topology_shape_hash": {"topology_score": 0.25, "shape_score": 0.20, "hash_score": 0.25, "chromatic_score": 0.20, "deformation_score": 0.10},
}


def run_sweep(
    matches_csv: str | Path = "results/spiking_morph_permanence_eval/matches.csv",
    output_dir: str | Path = "results/spiking_permanence_rerank_sweep",
    rerun_eval: bool = False,
    seed: int = 7,
    object_count: int = 16,
    events_per_object: int = 4,
    max_capsules: int = 32,
    spike_dim: int = 128,
    high_conf_score_threshold: float = 0.90,
    high_conf_margin_threshold: float = 0.04,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_matches = Path(matches_csv)
    if rerun_eval or not source_matches.exists():
        eval_dir = out / "source_eval"
        run_eval(
            output_dir=eval_dir,
            seed=seed,
            object_count=object_count,
            events_per_object=events_per_object,
            max_capsules=max_capsules,
            spike_dim=spike_dim,
        )
        source_matches = eval_dir / "matches.csv"

    match_rows = _read_matches(source_matches)
    grouped = _group_by_event(match_rows)
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for profile_name, weights in RERANK_PROFILES.items():
        profile_details = [_rerank_event(profile_name, weights, event_id, rows, high_conf_score_threshold, high_conf_margin_threshold) for event_id, rows in grouped.items()]
        detail_rows.extend(profile_details)
        summary_rows.append(_profile_summary(profile_name, profile_details))

    best = _select_best(summary_rows)
    if best is not None:
        best["selected_as_best"] = 1
    summary = {
        "source_matches_csv": str(source_matches),
        "profile_count": len(summary_rows),
        "event_count": len(grouped),
        "best_profile": dict(best or {}),
        "current_profile": next((dict(row) for row in summary_rows if row["profile_name"] == "R0_current_adjusted"), {}),
        "main_diagnosis": _diagnosis(summary_rows),
    }
    _write_csv(out / "rerank_profile_summary.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(out / "rerank_event_details.csv", detail_rows, list(detail_rows[0].keys()) if detail_rows else [])
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _read_matches(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _group_by_event(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("event_id", "")), []).append(row)
    return grouped


def _rerank_event(
    profile_name: str,
    weights: dict[str, float],
    event_id: str,
    rows: list[dict[str, Any]],
    high_conf_score_threshold: float,
    high_conf_margin_threshold: float,
) -> dict[str, Any]:
    scored = []
    for row in rows:
        score = sum(float(weight) * _to_float(row.get(field, 0.0)) for field, weight in weights.items())
        scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected_score, selected = scored[0] if scored else (0.0, {})
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    selected_margin = float(selected_score - second_score)
    true_rank = 0
    for index, (_, row) in enumerate(scored, start=1):
        if _to_int(row.get("is_true_capsule", 0)) == 1:
            true_rank = index
            break
    high_conf_accept = selected_score >= high_conf_score_threshold and selected_margin >= high_conf_margin_threshold
    selected_true = _to_int(selected.get("is_true_capsule", 0)) == 1
    return {
        "profile_name": profile_name,
        "event_id": event_id,
        "selected_capsule_id": selected.get("capsule_id", ""),
        "true_capsule_id": selected.get("true_capsule_id", ""),
        "selected_is_true": int(selected_true),
        "true_rank": true_rank,
        "true_in_top3": int(1 <= true_rank <= 3),
        "true_in_top5": int(1 <= true_rank <= 5),
        "selected_score": float(selected_score),
        "selected_margin": selected_margin,
        "high_conf_accept": int(high_conf_accept),
        "high_conf_true_accept": int(high_conf_accept and selected_true),
        "high_conf_false_accept": int(high_conf_accept and not selected_true),
        "scale_change": _to_float(selected.get("scale_change", 0.0)),
        "aspect_change": _to_float(selected.get("aspect_change", 0.0)),
        "distractor_level": selected.get("distractor_level", ""),
    }


def _profile_summary(profile_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    true_ranks = [int(row["true_rank"]) for row in rows if int(row["true_rank"]) > 0]
    true_accept = sum(int(row["high_conf_true_accept"]) for row in rows)
    false_accept = sum(int(row["high_conf_false_accept"]) for row in rows)
    return {
        "profile_name": profile_name,
        "event_count": count,
        "top1_true_rate": _safe_rate(sum(int(row["selected_is_true"]) for row in rows), count),
        "true_top3_rate": _safe_rate(sum(int(row["true_in_top3"]) for row in rows), count),
        "true_top5_rate": _safe_rate(sum(int(row["true_in_top5"]) for row in rows), count),
        "mean_selected_score": _mean([float(row["selected_score"]) for row in rows]),
        "mean_selected_margin": _mean([float(row["selected_margin"]) for row in rows]),
        "mean_true_rank": _mean(true_ranks),
        "high_conf_accept_rate": _safe_rate(true_accept + false_accept, count),
        "high_conf_true_accept_rate": _safe_rate(true_accept, count),
        "high_conf_false_accept_rate": _safe_rate(false_accept, count),
        "selected_as_best": 0,
    }


def _select_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row["top1_true_rate"]),
            -float(row["high_conf_false_accept_rate"]),
            float(row["true_top3_rate"]),
            float(row["mean_selected_margin"]),
        ),
    )


def _diagnosis(rows: list[dict[str, Any]]) -> str:
    current = next((row for row in rows if row["profile_name"] == "R0_current_adjusted"), None)
    best = _select_best(rows)
    if current is None or best is None:
        return "missing_rerank_rows"
    gain = float(best["top1_true_rate"]) - float(current["top1_true_rate"])
    if gain <= 0.0:
        return "existing_evidence_weights_already_best_or_tied"
    if float(best["high_conf_false_accept_rate"]) > float(current["high_conf_false_accept_rate"]):
        return "rerank_improves_top1_but_increases_high_conf_false"
    return "rerank_profile_improves_top1_without_extra_high_conf_false"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _report(summary: dict[str, Any]) -> str:
    best = summary.get("best_profile", {})
    current = summary.get("current_profile", {})
    return (
        "# Spiking Permanence Rerank Sweep\n\n"
        f"- event_count: {summary.get('event_count', 0)}\n"
        f"- current_profile_top1_true_rate: {float(current.get('top1_true_rate', 0.0)):.4f}\n"
        f"- best_profile: {best.get('profile_name', '')}\n"
        f"- best_top1_true_rate: {float(best.get('top1_true_rate', 0.0)):.4f}\n"
        f"- best_high_conf_false_accept_rate: {float(best.get('high_conf_false_accept_rate', 0.0)):.4f}\n"
        f"- main_diagnosis: {summary.get('main_diagnosis', '')}\n"
    )


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


def _mean(values: list[float | int]) -> float:
    return 0.0 if not values else float(sum(float(value) for value in values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches-csv", default="results/spiking_morph_permanence_eval/matches.csv")
    parser.add_argument("--output-dir", default="results/spiking_permanence_rerank_sweep")
    parser.add_argument("--rerun-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--object-count", type=int, default=16)
    parser.add_argument("--events-per-object", type=int, default=4)
    parser.add_argument("--max-capsules", type=int, default=32)
    parser.add_argument("--spike-dim", type=int, default=128)
    parser.add_argument("--high-conf-score-threshold", type=float, default=0.90)
    parser.add_argument("--high-conf-margin-threshold", type=float, default=0.04)
    summary = run_sweep(**vars(parser.parse_args()))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
