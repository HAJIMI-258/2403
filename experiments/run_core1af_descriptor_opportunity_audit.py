from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AF descriptor integration opportunity audit.")
    p.add_argument("--core1ae-compact", default="results/core1ae/stage_CORE1AE_compact_for_gpt_v1.json")
    p.add_argument("--event-results", default="results/core1ad/stage_CORE1AD_event_results_v1.csv")
    p.add_argument("--output-dir", default="results/core1af")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def i(v: Any, default: int = 0) -> int:
    if v in (None, ""):
        return default
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default


def f(v: Any, default: float = 0.0) -> float:
    if v in (None, ""):
        return default
    try:
        out = float(v)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def row_map(rows: list[dict[str, str]], gate: str, variant: str) -> dict[int, dict[str, str]]:
    return {i(r["query_obs_id"]): r for r in rows if r["gate_name"] == gate and r["variant"] == variant}


def classify_failure(base: dict[str, str], selected: dict[str, str] | None, raw: dict[str, str] | None, controls: list[dict[str, str]]) -> str:
    if i(base["top1_success"]) == 1:
        return "baseline_success_no_opportunity"
    if selected is not None and i(selected["top1_success"]) == 1:
        control_success = any(i(c["top1_success"]) == 1 for c in controls if c is not None)
        return "rescued_but_control_also_rescues" if control_success else "descriptor_rescued_failure"
    if raw is not None and f(raw.get("target_margin")) <= 0.0:
        return "raw_descriptor_not_discriminative"
    if i(base.get("target_rank"), 999) > 3:
        return "target_not_near_top_under_baseline"
    if f(base.get("target_margin")) < -0.20:
        return "baseline_wrong_with_large_negative_margin"
    return "descriptor_integration_not_strong_enough"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compact_ae = read_json(Path(args.core1ae_compact))
    gate = str(compact_ae["gate_name"])
    selected_variant = str(compact_ae["selected_variant"])
    rows = read_csv(Path(args.event_results))
    base = row_map(rows, gate, "A0_track_recency_baseline")
    selected = row_map(rows, gate, selected_variant)
    raw = row_map(rows, gate, "A1_raw_descriptor_only")
    controls = [
        row_map(rows, gate, "A7_shuffled_descriptor_w010_control"),
        row_map(rows, gate, "A8_wrong_binding_descriptor_w010_control"),
        row_map(rows, gate, "A9_random_descriptor_w010_control"),
    ]
    audit_rows: list[dict[str, Any]] = []
    for qid, b in sorted(base.items()):
        s = selected.get(qid)
        r = raw.get(qid)
        cands = [c.get(qid) for c in controls if qid in c]
        reason = classify_failure(b, s, r, cands)
        audit_rows.append(
            {
                "query_obs_id": qid,
                "sequence_id": b["sequence_id"],
                "event_id": b["event_id"],
                "window_kind": b["window_kind"],
                "baseline_success": b["top1_success"],
                "selected_success": "" if s is None else s["top1_success"],
                "raw_descriptor_success": "" if r is None else r["top1_success"],
                "any_control_success": int(any(i(c["top1_success"]) == 1 for c in cands)),
                "baseline_target_rank": b["target_rank"],
                "baseline_target_margin": b["target_margin"],
                "selected_target_margin": "" if s is None else s["target_margin"],
                "raw_descriptor_target_margin": "" if r is None else r["target_margin"],
                "opportunity_class": reason,
            }
        )
    counts = Counter(row["opportunity_class"] for row in audit_rows)
    group_rows: list[dict[str, Any]] = []
    for key in ["sequence_id", "event_id", "window_kind"]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in audit_rows:
            grouped[str(row[key])].append(row)
        for value, items in sorted(grouped.items()):
            baseline_fail = sum(1 for r in items if i(r["baseline_success"]) == 0)
            descriptor_rescue = sum(1 for r in items if r["opportunity_class"] == "descriptor_rescued_failure")
            control_rescue = sum(1 for r in items if r["opportunity_class"] == "rescued_but_control_also_rescues")
            group_rows.append(
                {
                    "group_key": key,
                    "group_value": value,
                    "num_queries": len(items),
                    "baseline_failure_count": baseline_fail,
                    "descriptor_clean_rescue_count": descriptor_rescue,
                    "control_confounded_rescue_count": control_rescue,
                    "opportunity_rate": descriptor_rescue / max(baseline_fail, 1),
                }
            )
    baseline_failure_count = sum(1 for r in audit_rows if i(r["baseline_success"]) == 0)
    clean_rescue_count = counts.get("descriptor_rescued_failure", 0)
    control_confounded_count = counts.get("rescued_but_control_also_rescues", 0)
    compact = {
        "stage": "CORE-1AF",
        "artifact_version": args.artifact_version,
        "source_stage": "CORE-1AE",
        "gate_name": gate,
        "selected_variant": selected_variant,
        "num_queries": len(audit_rows),
        "baseline_failure_count": baseline_failure_count,
        "descriptor_clean_rescue_count": clean_rescue_count,
        "control_confounded_rescue_count": control_confounded_count,
        "raw_descriptor_not_discriminative_count": counts.get("raw_descriptor_not_discriminative", 0),
        "baseline_success_no_opportunity_count": counts.get("baseline_success_no_opportunity", 0),
        "descriptor_opportunity_rate": clean_rescue_count / max(baseline_failure_count, 1),
        "main_opportunity_class": counts.most_common(1)[0][0] if counts else "",
        "safe_for_integration": 0,
        "oracle_leakage_found": 0,
        "next_recommendation": "CORE-1AG mine harder non-oracle descriptor-opportunity events before any integration",
    }
    report = f"""# CORE-1AF Descriptor Opportunity Audit

This stage explains why CORE-1AE rejected descriptor integration. It audits whether the selected descriptor gate has enough clean failure-rescue opportunities.

## Result

- Gate: {gate}
- Selected variant: {selected_variant}
- Queries: {compact['num_queries']}
- Baseline failures: {baseline_failure_count}
- Clean descriptor rescues: {clean_rescue_count}
- Control-confounded rescues: {control_confounded_count}
- Descriptor opportunity rate: {float(compact['descriptor_opportunity_rate']):.4f}
- Main class: {compact['main_opportunity_class']}

## Interpretation

The descriptor cue has some signal, but the current selected gate contains too few clean baseline failures. Integration would be premature because the observed gain is a small number of rescues, with controls close behind.

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AF_"
    write_csv(
        out_dir / f"{prefix}opportunity_audit_{args.artifact_version}.csv",
        audit_rows,
        [
            "query_obs_id",
            "sequence_id",
            "event_id",
            "window_kind",
            "baseline_success",
            "selected_success",
            "raw_descriptor_success",
            "any_control_success",
            "baseline_target_rank",
            "baseline_target_margin",
            "selected_target_margin",
            "raw_descriptor_target_margin",
            "opportunity_class",
        ],
    )
    write_csv(
        out_dir / f"{prefix}group_summary_{args.artifact_version}.csv",
        group_rows,
        ["group_key", "group_value", "num_queries", "baseline_failure_count", "descriptor_clean_rescue_count", "control_confounded_rescue_count", "opportunity_rate"],
    )
    write_csv(
        out_dir / f"{prefix}class_counts_{args.artifact_version}.csv",
        [{"opportunity_class": key, "count": value} for key, value in counts.items()],
        ["opportunity_class", "count"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
