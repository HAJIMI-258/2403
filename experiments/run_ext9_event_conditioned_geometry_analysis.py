from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ext1_utils import read_csv, write_csv


def as_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def gap_bin(gap: int) -> str:
    if gap <= 3:
        return "gap_<=3"
    if gap <= 5:
        return "gap_4_5"
    if gap <= 10:
        return "gap_6_10"
    if gap <= 20:
        return "gap_11_20"
    return "gap_>20"


def candidate_bin(n: int) -> str:
    if n <= 2:
        return "cand_<=2"
    if n <= 4:
        return "cand_3_4"
    if n <= 6:
        return "cand_5_6"
    return "cand_>6"


def load_event_pairs() -> list[dict[str, Any]]:
    rows = read_csv(ROOT / "results" / "ext5" / "stage_EXT5_event_results_v1.csv")
    by_event: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("variant") in {"A0_nops_geometry_passive", "A5_external_trajectory_heavy"}:
            by_event[row["event_id"]][row["variant"]] = row
    out: list[dict[str, Any]] = []
    for event_id, variants in sorted(by_event.items()):
        a0 = variants.get("A0_nops_geometry_passive")
        a5 = variants.get("A5_external_trajectory_heavy")
        if not a0 or not a5:
            continue
        gap = as_int(a0.get("gap_length"))
        cand = as_int(a0.get("candidate_count"))
        a0_success = as_int(a0.get("top1"))
        a5_success = as_int(a5.get("top1"))
        if a0_success and a5_success:
            delta_class = "both_success"
        elif (not a0_success) and a5_success:
            delta_class = "external_rescued"
        elif a0_success and (not a5_success):
            delta_class = "external_regressed"
        else:
            delta_class = "both_failure"
        out.append({
            "event_id": event_id,
            "category": a0.get("category", ""),
            "sequence_id": a0.get("sequence_id", ""),
            "gap_length": gap,
            "gap_bin": gap_bin(gap),
            "candidate_count": cand,
            "candidate_bin": candidate_bin(cand),
            "target_instance_id_eval_only": a0.get("target_instance_id_eval_only", ""),
            "geometry_top1_id": a0.get("predicted_memory_id", ""),
            "external_top1_id": a5.get("predicted_memory_id", ""),
            "geometry_success": a0_success,
            "external_success": a5_success,
            "delta_class": delta_class,
            "observable_gate_features": "gap_length,candidate_count,category",
        })
    return out


def summarize_group(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    out = []
    for name, rs in sorted(groups.items()):
        n = len(rs)
        geom = sum(as_int(r["geometry_success"]) for r in rs) / max(n, 1)
        ext = sum(as_int(r["external_success"]) for r in rs) / max(n, 1)
        out.append({
            "group_by": key,
            "group_value": name,
            "num_events": n,
            "geometry_top1": geom,
            "external_branch_top1": ext,
            "external_minus_geometry": ext - geom,
            "external_rescued_count": sum(1 for r in rs if r["delta_class"] == "external_rescued"),
            "external_regressed_count": sum(1 for r in rs if r["delta_class"] == "external_regressed"),
            "both_failure_count": sum(1 for r in rs if r["delta_class"] == "both_failure"),
            "both_success_count": sum(1 for r in rs if r["delta_class"] == "both_success"),
        })
    return out


def gate_defs() -> list[tuple[str, Callable[[dict[str, Any]], bool], str]]:
    return [
        ("A0_all_geometry", lambda r: False, "never use external branch"),
        ("A1_all_external", lambda r: True, "always use external branch"),
        ("G_gap_ge_6", lambda r: as_int(r["gap_length"]) >= 6, "use external branch if gap >= 6"),
        ("G_gap_ge_10", lambda r: as_int(r["gap_length"]) >= 10, "use external branch if gap >= 10"),
        ("G_candidate_ge_4", lambda r: as_int(r["candidate_count"]) >= 4, "use external branch if candidate_count >= 4"),
        ("G_candidate_ge_6", lambda r: as_int(r["candidate_count"]) >= 6, "use external branch if candidate_count >= 6"),
        ("G_gap_ge_6_or_candidate_ge_4", lambda r: as_int(r["gap_length"]) >= 6 or as_int(r["candidate_count"]) >= 4, "use external branch if gap >= 6 or candidates >= 4"),
        ("G_gap_ge_10_or_candidate_ge_6", lambda r: as_int(r["gap_length"]) >= 10 or as_int(r["candidate_count"]) >= 6, "use external branch if gap >= 10 or candidates >= 6"),
        ("G_category_coin_kite_motorcycle", lambda r: r["category"] in {"coin", "kite", "motorcycle"}, "diagnostic category gate; not integration-ready"),
    ]


def gate_analysis(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    base_success = {r["event_id"]: as_int(r["geometry_success"]) for r in rows}
    for gate_name, pred, desc in gate_defs():
        successes = []
        applied = 0
        improved = 0
        regressed = 0
        for row in rows:
            use_external = bool(pred(row))
            applied += int(use_external)
            chosen_success = as_int(row["external_success"] if use_external else row["geometry_success"])
            base = base_success[row["event_id"]]
            successes.append(chosen_success)
            if chosen_success and not base:
                delta = "improved"
                improved += 1
            elif base and not chosen_success:
                delta = "regressed"
                regressed += 1
            elif chosen_success and base:
                delta = "unchanged_success"
            else:
                delta = "unchanged_failure"
            event_rows.append({
                "gate_name": gate_name,
                "event_id": row["event_id"],
                "category": row["category"],
                "gap_length": row["gap_length"],
                "candidate_count": row["candidate_count"],
                "use_external_branch": int(use_external),
                "geometry_success": row["geometry_success"],
                "external_success": row["external_success"],
                "chosen_success": chosen_success,
                "delta_vs_geometry": delta,
            })
        n = len(rows)
        top1 = sum(successes) / max(n, 1)
        summary.append({
            "gate_name": gate_name,
            "description": desc,
            "num_events": n,
            "external_branch_applied_count": applied,
            "top1": top1,
            "delta_vs_geometry": top1 - (sum(base_success.values()) / max(n, 1)),
            "improved_count": improved,
            "regressed_count": regressed,
            "eligible_for_integration": 0,
            "reason_not_eligible": "same-subset diagnostic only; no train/dev/test split and synthetic regression gate not checked",
        })
    return summary, event_rows


def main() -> None:
    out_dir = ROOT / "results" / "ext9"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_event_pairs()
    group_rows = []
    for key in ("category", "gap_bin", "candidate_bin"):
        group_rows.extend(summarize_group(rows, key))
    gates, gate_events = gate_analysis(rows)
    best_gate = max(gates, key=lambda r: (as_float(r["top1"]), -as_int(r["regressed_count"])), default={})
    n = len(rows)
    geometry_top1 = sum(as_int(r["geometry_success"]) for r in rows) / max(n, 1)
    external_top1 = sum(as_int(r["external_success"]) for r in rows) / max(n, 1)
    delta_counts = Counter(r["delta_class"] for r in rows)
    compact = {
        "stage": "EXT-9",
        "num_events": n,
        "geometry_passive_top1": geometry_top1,
        "external_branch_top1": external_top1,
        "external_minus_geometry": external_top1 - geometry_top1,
        "delta_counts": dict(delta_counts),
        "best_diagnostic_gate": best_gate.get("gate_name", ""),
        "best_diagnostic_gate_top1": best_gate.get("top1", 0),
        "best_diagnostic_gate_delta_vs_geometry": best_gate.get("delta_vs_geometry", 0),
        "best_diagnostic_gate_regressed_count": best_gate.get("regressed_count", 0),
        "gate_integration_allowed": 0,
        "next_recommendation": "use this as failure analysis only; next either expand full-pixel events or create train/dev split before event-conditioned geometry routing",
    }
    write_csv(out_dir / "stage_EXT9_event_delta_v1.csv", rows)
    write_csv(out_dir / "stage_EXT9_stratified_summary_v1.csv", group_rows)
    write_csv(out_dir / "stage_EXT9_gate_summary_v1.csv", gates)
    write_csv(out_dir / "stage_EXT9_gate_event_trace_v1.csv", gate_events)
    write_json(out_dir / "stage_EXT9_compact_for_gpt_v1.json", compact)
    report = [
        "# EXT-9 Event-Conditioned Geometry Failure Analysis",
        "",
        "## Result",
        "",
        f"- Events: `{n}`",
        f"- Geometry passive top1: `{geometry_top1:.4f}`",
        f"- External branch top1: `{external_top1:.4f}`",
        f"- Delta counts: `{dict(delta_counts)}`",
        f"- Best diagnostic gate: `{compact['best_diagnostic_gate']}`",
        f"- Best diagnostic gate top1: `{as_float(compact['best_diagnostic_gate_top1']):.4f}`",
        "",
        "## Decision",
        "",
        "No gate is integration-ready. These are same-subset diagnostics only.",
        compact["next_recommendation"],
    ]
    (out_dir / "stage_EXT9_report_v1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
