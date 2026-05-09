from __future__ import annotations

import hashlib
import itertools
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


def split_for_sequence(seq: str) -> str:
    # Sequence-level split prevents event leakage across frames from the same video.
    h = int(hashlib.sha1(seq.encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < 60:
        return "train"
    if h < 80:
        return "dev"
    return "test"


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


def load_rows() -> list[dict[str, Any]]:
    rows = read_csv(ROOT / "results" / "ext9" / "stage_EXT9_event_delta_v1.csv")
    out = []
    for r in rows:
        gap = as_int(r.get("gap_length"))
        cand = as_int(r.get("candidate_count"))
        seq = r.get("sequence_id", "")
        row = dict(r)
        row.update({
            "gap_length": gap,
            "candidate_count": cand,
            "gap_bin": r.get("gap_bin") or gap_bin(gap),
            "candidate_bin": r.get("candidate_bin") or candidate_bin(cand),
            "split": split_for_sequence(seq),
            "geometry_success": as_int(r.get("geometry_success")),
            "external_success": as_int(r.get("external_success")),
        })
        out.append(row)
    return out


class Gate:
    def __init__(self, name: str, description: str, fn: Callable[[dict[str, Any]], bool], diagnostic_only: int = 0) -> None:
        self.name = name
        self.description = description
        self.fn = fn
        self.diagnostic_only = diagnostic_only


def build_gates(rows: list[dict[str, Any]]) -> list[Gate]:
    gates: list[Gate] = [
        Gate("A0_all_geometry", "never use external branch", lambda r: False),
        Gate("A1_all_external", "always use external branch", lambda r: True),
    ]
    for t in [4, 5, 6, 8, 10, 12, 15, 20, 30]:
        gates.append(Gate(f"G_gap_ge_{t}", f"use external if gap >= {t}", lambda r, t=t: as_int(r["gap_length"]) >= t))
    for t in [2, 3, 4, 5, 6, 7, 8]:
        gates.append(Gate(f"G_candidate_ge_{t}", f"use external if candidate_count >= {t}", lambda r, t=t: as_int(r["candidate_count"]) >= t))
    for g in [6, 10, 20]:
        for c in [4, 6, 8]:
            gates.append(Gate(
                f"G_gap_ge_{g}_or_candidate_ge_{c}",
                f"use external if gap >= {g} or candidate_count >= {c}",
                lambda r, g=g, c=c: as_int(r["gap_length"]) >= g or as_int(r["candidate_count"]) >= c,
            ))
            gates.append(Gate(
                f"G_gap_ge_{g}_and_candidate_ge_{c}",
                f"use external if gap >= {g} and candidate_count >= {c}",
                lambda r, g=g, c=c: as_int(r["gap_length"]) >= g and as_int(r["candidate_count"]) >= c,
            ))
    cats = sorted({r["category"] for r in rows})
    for k in range(1, min(3, len(cats)) + 1):
        for subset in itertools.combinations(cats, k):
            name = "G_category_" + "_".join(subset)
            gates.append(Gate(
                name,
                "diagnostic category gate; category may not be reliably online in full perception",
                lambda r, subset=set(subset): r["category"] in subset,
                diagnostic_only=1,
            ))
    return gates


def eval_gate(rows: list[dict[str, Any]], gate: Gate, split: str) -> dict[str, Any]:
    rs = [r for r in rows if r["split"] == split]
    n = len(rs)
    top1 = 0
    applied = 0
    improved = 0
    regressed = 0
    unchanged_success = 0
    unchanged_failure = 0
    for r in rs:
        use_ext = bool(gate.fn(r))
        applied += int(use_ext)
        chosen = as_int(r["external_success"] if use_ext else r["geometry_success"])
        base = as_int(r["geometry_success"])
        top1 += chosen
        if chosen and not base:
            improved += 1
        elif base and not chosen:
            regressed += 1
        elif chosen and base:
            unchanged_success += 1
        else:
            unchanged_failure += 1
    geom = sum(as_int(r["geometry_success"]) for r in rs) / max(n, 1)
    ext = sum(as_int(r["external_success"]) for r in rs) / max(n, 1)
    return {
        "gate_name": gate.name,
        "split": split,
        "description": gate.description,
        "diagnostic_only": gate.diagnostic_only,
        "num_events": n,
        "external_branch_applied_count": applied,
        "top1": top1 / max(n, 1),
        "geometry_top1": geom,
        "all_external_top1": ext,
        "delta_vs_geometry": top1 / max(n, 1) - geom,
        "delta_vs_all_external": top1 / max(n, 1) - ext,
        "improved_count": improved,
        "regressed_count": regressed,
        "unchanged_success_count": unchanged_success,
        "unchanged_failure_count": unchanged_failure,
    }


def event_predictions(rows: list[dict[str, Any]], gate: Gate) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        use_ext = bool(gate.fn(r))
        chosen = as_int(r["external_success"] if use_ext else r["geometry_success"])
        base = as_int(r["geometry_success"])
        if chosen and not base:
            delta = "improved"
        elif base and not chosen:
            delta = "regressed"
        elif chosen and base:
            delta = "unchanged_success"
        else:
            delta = "unchanged_failure"
        out.append({
            "event_id": r["event_id"],
            "sequence_id": r["sequence_id"],
            "category": r["category"],
            "split": r["split"],
            "gap_length": r["gap_length"],
            "candidate_count": r["candidate_count"],
            "selected_gate": gate.name,
            "use_external_branch": int(use_ext),
            "geometry_success": r["geometry_success"],
            "external_success": r["external_success"],
            "chosen_success": chosen,
            "delta_vs_geometry": delta,
        })
    return out


def choose_gate(summary: list[dict[str, Any]]) -> tuple[str, str]:
    # Select on dev, but require train not to be worse than geometry.
    by_gate: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary:
        by_gate[row["gate_name"]][row["split"]] = row
    eligible = []
    for name, splits in by_gate.items():
        train = splits.get("train")
        dev = splits.get("dev")
        if not train or not dev:
            continue
        if as_float(train["delta_vs_geometry"]) < 0:
            continue
        eligible.append((name, dev, train))
    if not eligible:
        return "A0_all_geometry", "no gate improved train without regression"
    eligible.sort(key=lambda x: (as_float(x[1]["top1"]), as_float(x[1]["delta_vs_all_external"]), -as_int(x[1]["regressed_count"])), reverse=True)
    return eligible[0][0], "selected by best dev top1 among train-safe gates"


def split_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_seq[r["sequence_id"]].append(r)
    out = []
    for seq, rs in sorted(by_seq.items()):
        out.append({
            "sequence_id": seq,
            "split": rs[0]["split"],
            "category": rs[0]["category"],
            "num_events": len(rs),
            "geometry_top1": sum(as_int(r["geometry_success"]) for r in rs) / max(len(rs), 1),
            "external_top1": sum(as_int(r["external_success"]) for r in rs) / max(len(rs), 1),
        })
    return out


def main() -> None:
    out_dir = ROOT / "results" / "ext10"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    gates = build_gates(rows)
    summary = []
    for gate in gates:
        for split in ("train", "dev", "test"):
            summary.append(eval_gate(rows, gate, split))
    selected_name, selection_reason = choose_gate(summary)
    selected_gate = next(g for g in gates if g.name == selected_name)
    preds = event_predictions(rows, selected_gate)
    test_row = next(r for r in summary if r["gate_name"] == selected_name and r["split"] == "test")
    test_all_external = next(r for r in summary if r["gate_name"] == "A1_all_external" and r["split"] == "test")
    test_geometry = next(r for r in summary if r["gate_name"] == "A0_all_geometry" and r["split"] == "test")
    integration_ready = int(
        as_float(test_row["top1"]) > max(as_float(test_all_external["top1"]), as_float(test_geometry["top1"]))
        and as_int(test_row["regressed_count"]) <= as_int(test_all_external["regressed_count"])
        and as_int(test_row["num_events"]) >= 30
        and not selected_gate.diagnostic_only
    )
    compact = {
        "stage": "EXT-10",
        "num_events": len(rows),
        "selected_gate": selected_name,
        "selection_reason": selection_reason,
        "selected_gate_diagnostic_only": selected_gate.diagnostic_only,
        "test_num_events": test_row["num_events"],
        "test_geometry_top1": test_geometry["top1"],
        "test_all_external_top1": test_all_external["top1"],
        "test_selected_gate_top1": test_row["top1"],
        "test_selected_delta_vs_geometry": test_row["delta_vs_geometry"],
        "test_selected_delta_vs_all_external": test_row["delta_vs_all_external"],
        "test_selected_regressed_count": test_row["regressed_count"],
        "routing_integration_ready": integration_ready,
        "next_recommendation": (
            "routing gate passed independent test; still requires synthetic regression before integration"
            if integration_ready
            else "routing gate not integration-ready; expand full-pixel data or keep all-external isolated branch"
        ),
    }
    write_csv(out_dir / "stage_EXT10_split_manifest_v1.csv", split_manifest(rows))
    write_csv(out_dir / "stage_EXT10_gate_candidate_summary_v1.csv", summary)
    write_csv(out_dir / "stage_EXT10_selected_gate_event_predictions_v1.csv", preds)
    write_json(out_dir / "stage_EXT10_compact_for_gpt_v1.json", compact)
    report = [
        "# EXT-10 Geometry Routing Split Gate",
        "",
        "## Result",
        "",
        f"- Selected gate: `{selected_name}`",
        f"- Test events: `{test_row['num_events']}`",
        f"- Test geometry top1: `{as_float(test_geometry['top1']):.4f}`",
        f"- Test all-external top1: `{as_float(test_all_external['top1']):.4f}`",
        f"- Test selected gate top1: `{as_float(test_row['top1']):.4f}`",
        f"- Integration ready: `{integration_ready}`",
        "",
        "## Decision",
        "",
        compact["next_recommendation"],
    ]
    (out_dir / "stage_EXT10_report_v1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
