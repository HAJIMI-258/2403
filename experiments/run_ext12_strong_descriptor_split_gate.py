from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ext1_utils import read_csv, write_csv


VARIANTS = [
    "A6_external_trajectory_heavy",
    "A7_external_trajectory_plus_strong_w005",
    "A8_external_trajectory_plus_strong_w010",
    "A9_external_trajectory_plus_strong_w020",
]
CONTROL_MODES = ["real", "within_event_shuffled", "category_shuffled"]


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
    h = int(hashlib.sha1(seq.encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < 60:
        return "train"
    if h < 80:
        return "dev"
    return "test"


def load_rows() -> list[dict[str, Any]]:
    rows = read_csv(ROOT / "results" / "ext6" / "stage_EXT6_event_results_v1.csv")
    out = []
    for r in rows:
        if r.get("variant") not in VARIANTS or r.get("control_mode") not in CONTROL_MODES:
            continue
        rr = dict(r)
        rr["split"] = split_for_sequence(r.get("sequence_id", ""))
        rr["top1"] = as_int(r.get("top1"))
        rr["top3"] = as_int(r.get("top3"))
        rr["top5"] = as_int(r.get("top5"))
        rr["candidate_count"] = as_int(r.get("candidate_count"))
        rr["gap_length"] = as_int(r.get("gap_length"))
        out.append(rr)
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_key[(r["control_mode"], r["variant"], r["split"])].append(r)
    out = []
    for mode in CONTROL_MODES:
        for variant in VARIANTS:
            for split in ("train", "dev", "test"):
                rs = by_key.get((mode, variant, split), [])
                n = len(rs)
                out.append({
                    "control_mode": mode,
                    "variant": variant,
                    "split": split,
                    "num_events": n,
                    "top1": sum(as_int(r["top1"]) for r in rs) / max(n, 1),
                    "top3": sum(as_int(r["top3"]) for r in rs) / max(n, 1),
                    "top5": sum(as_int(r["top5"]) for r in rs) / max(n, 1),
                })
    base_by_split_mode = {
        (r["control_mode"], r["split"]): r
        for r in out
        if r["variant"] == "A6_external_trajectory_heavy"
    }
    for r in out:
        base = base_by_split_mode.get((r["control_mode"], r["split"]), {})
        r["delta_vs_external_branch"] = as_float(r["top1"]) - as_float(base.get("top1"))
    return out


def choose_variant(summary: list[dict[str, Any]]) -> tuple[str, str]:
    train = {r["variant"]: r for r in summary if r["control_mode"] == "real" and r["split"] == "train"}
    dev = [r for r in summary if r["control_mode"] == "real" and r["split"] == "dev"]
    eligible = []
    for r in dev:
        tr = train.get(r["variant"])
        if not tr:
            continue
        if as_float(tr["delta_vs_external_branch"]) < 0:
            continue
        eligible.append(r)
    if not eligible:
        return "A6_external_trajectory_heavy", "no train-safe strong descriptor variant"
    eligible.sort(key=lambda r: (as_float(r["top1"]), as_float(r["delta_vs_external_branch"]), as_float(r["top3"])), reverse=True)
    return eligible[0]["variant"], "selected by dev top1 among train-safe variants"


def lookup(summary: list[dict[str, Any]], mode: str, variant: str, split: str) -> dict[str, Any]:
    return next((r for r in summary if r["control_mode"] == mode and r["variant"] == variant and r["split"] == split), {})


def event_delta(rows: list[dict[str, Any]], selected: str) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in rows:
        if r["control_mode"] == "real" and r["variant"] in {"A6_external_trajectory_heavy", selected}:
            by_key[(r["event_id"], r["variant"], r["split"])] = r
    out = []
    for (event_id, variant, split), r in sorted(by_key.items()):
        if variant != selected:
            continue
        base = by_key.get((event_id, "A6_external_trajectory_heavy", split))
        if not base:
            continue
        b = as_int(base["top1"])
        s = as_int(r["top1"])
        if s and not b:
            delta = "improved"
        elif b and not s:
            delta = "regressed"
        elif s and b:
            delta = "unchanged_success"
        else:
            delta = "unchanged_failure"
        out.append({
            "event_id": event_id,
            "sequence_id": r.get("sequence_id", ""),
            "category": r.get("category", ""),
            "split": split,
            "selected_variant": selected,
            "external_branch_top1": b,
            "selected_top1": s,
            "delta_vs_external_branch": delta,
            "gap_length": r.get("gap_length", ""),
            "candidate_count": r.get("candidate_count", ""),
        })
    return out


def main() -> None:
    out_dir = ROOT / "results" / "ext12"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    summary = summarize(rows)
    selected, reason = choose_variant(summary)
    test_sel = lookup(summary, "real", selected, "test")
    test_base = lookup(summary, "real", "A6_external_trajectory_heavy", "test")
    test_within = lookup(summary, "within_event_shuffled", selected, "test")
    test_cat = lookup(summary, "category_shuffled", selected, "test")
    delta_real = as_float(test_sel.get("delta_vs_external_branch"))
    delta_within = as_float(test_within.get("delta_vs_external_branch"))
    delta_cat = as_float(test_cat.get("delta_vs_external_branch"))
    ready = int(
        selected != "A6_external_trajectory_heavy"
        and delta_real > 0
        and delta_real > max(delta_within, delta_cat)
        and as_int(test_sel.get("num_events")) >= 30
    )
    deltas = event_delta(rows, selected)
    compact = {
        "stage": "EXT-12",
        "selected_variant": selected,
        "selection_reason": reason,
        "test_num_events": test_sel.get("num_events", 0),
        "test_external_branch_top1": test_base.get("top1", 0),
        "test_selected_top1": test_sel.get("top1", 0),
        "test_selected_delta_vs_external_branch": delta_real,
        "test_within_event_shuffled_delta": delta_within,
        "test_category_shuffled_delta": delta_cat,
        "strong_auxiliary_split_gate_passed": ready,
        "safe_for_external_auxiliary": ready,
        "safe_for_main_merge": 0,
        "next_recommendation": (
            "define isolated strong-descriptor external auxiliary candidate; still needs synthetic guard"
            if ready else
            "do not integrate strong descriptor auxiliary; split/control gate failed"
        ),
    }
    write_csv(out_dir / "stage_EXT12_variant_split_summary_v1.csv", summary)
    write_csv(out_dir / "stage_EXT12_selected_event_delta_v1.csv", deltas)
    write_json(out_dir / "stage_EXT12_compact_for_gpt_v1.json", compact)
    report = [
        "# EXT-12 Strong Descriptor External Auxiliary Split Gate",
        "",
        "## Result",
        "",
        f"- Selected variant: `{selected}`",
        f"- Test external branch top1: `{as_float(test_base.get('top1')):.4f}`",
        f"- Test selected top1: `{as_float(test_sel.get('top1')):.4f}`",
        f"- Test real delta: `{delta_real:.4f}`",
        f"- Test within-event shuffled delta: `{delta_within:.4f}`",
        f"- Test category-shuffled delta: `{delta_cat:.4f}`",
        f"- Split gate passed: `{ready}`",
        "",
        "## Decision",
        "",
        compact["next_recommendation"],
    ]
    (out_dir / "stage_EXT12_report_v1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
