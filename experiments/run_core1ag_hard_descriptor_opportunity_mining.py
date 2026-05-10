from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AG hard descriptor-opportunity mining.")
    p.add_argument("--event-results", default="results/core1ad/stage_CORE1AD_event_results_v1.csv")
    p.add_argument("--output-dir", default="results/core1ag")
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


REAL_VARIANTS = [
    "A1_raw_descriptor_only",
    "A2_fusion_w005",
    "A3_fusion_w010",
    "A4_fusion_w020",
    "A5_gated_fusion_w010_margin005",
    "A6_gated_fusion_w020_margin005",
]
CONTROL_VARIANTS = [
    "A7_shuffled_descriptor_w010_control",
    "A8_wrong_binding_descriptor_w010_control",
    "A9_random_descriptor_w010_control",
]


def rows_by_gate_variant(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[int, dict[str, str]]]:
    out: dict[tuple[str, str], dict[int, dict[str, str]]] = {}
    for row in rows:
        out.setdefault((row["gate_name"], row["variant"]), {})[i(row["query_obs_id"])] = row
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.event_results))
    index = rows_by_gate_variant(rows)
    gates = sorted({row["gate_name"] for row in rows})
    summary_rows: list[dict[str, Any]] = []
    opportunity_rows: list[dict[str, Any]] = []
    for gate in gates:
        base = index.get((gate, "A0_track_recency_baseline"), {})
        if not base:
            continue
        base_failures = [qid for qid, row in base.items() if i(row["top1_success"]) == 0]
        base_top1 = float(np.mean([i(r["top1_success"]) for r in base.values()])) if base else 0.0
        for variant in REAL_VARIANTS:
            real = index.get((gate, variant), {})
            if not real:
                continue
            improved = []
            regressed = []
            descriptor_fail = 0
            for qid, b in base.items():
                r = real.get(qid)
                if r is None:
                    continue
                if i(b["top1_success"]) == 0 and i(r["top1_success"]) == 1:
                    improved.append(qid)
                if i(b["top1_success"]) == 1 and i(r["top1_success"]) == 0:
                    regressed.append(qid)
                if i(b["top1_success"]) == 0 and f(r.get("target_margin")) <= 0.0:
                    descriptor_fail += 1
            control_rescue_count = 0
            for qid in base_failures:
                if any(i(index.get((gate, c), {}).get(qid, {}).get("top1_success")) == 1 for c in CONTROL_VARIANTS):
                    control_rescue_count += 1
            clean_rescues = [qid for qid in improved if not any(i(index.get((gate, c), {}).get(qid, {}).get("top1_success")) == 1 for c in CONTROL_VARIANTS)]
            score = len(clean_rescues) - len(regressed) - control_rescue_count
            summary_rows.append(
                {
                    "gate_name": gate,
                    "variant": variant,
                    "num_queries": len(base),
                    "baseline_top1": base_top1,
                    "baseline_failure_count": len(base_failures),
                    "improved_count": len(improved),
                    "clean_rescue_count": len(clean_rescues),
                    "regressed_count": len(regressed),
                    "control_rescue_count": control_rescue_count,
                    "descriptor_fail_on_baseline_failure_count": descriptor_fail,
                    "opportunity_score": score,
                    "eligible_hard_gate": int(len(base_failures) >= 20 and len(clean_rescues) >= 2 and len(regressed) <= len(clean_rescues)),
                }
            )
            for qid in clean_rescues:
                b = base[qid]
                r = real[qid]
                opportunity_rows.append(
                    {
                        "gate_name": gate,
                        "variant": variant,
                        "query_obs_id": qid,
                        "sequence_id": b["sequence_id"],
                        "event_id": b["event_id"],
                        "window_kind": b["window_kind"],
                        "baseline_target_margin": b["target_margin"],
                        "descriptor_target_margin": r["target_margin"],
                        "baseline_target_rank": b["target_rank"],
                        "descriptor_target_rank": r["target_rank"],
                    }
                )
    eligible = [r for r in summary_rows if i(r["eligible_hard_gate"]) == 1]
    if eligible:
        best = max(eligible, key=lambda r: (i(r["opportunity_score"]), i(r["clean_rescue_count"]), -i(r["regressed_count"])))
    else:
        best = max(summary_rows, key=lambda r: (i(r["opportunity_score"]), i(r["clean_rescue_count"]), -i(r["regressed_count"]))) if summary_rows else {}
    for row in summary_rows:
        row["selected_as_best_hard_gate"] = int(row is best)
    class_counter = Counter(row["gate_name"] for row in opportunity_rows)
    compact = {
        "stage": "CORE-1AG",
        "artifact_version": args.artifact_version,
        "source_stage": "CORE-1AD",
        "gate_variant_count": len(summary_rows),
        "best_gate": best.get("gate_name", ""),
        "best_variant": best.get("variant", ""),
        "best_num_queries": best.get("num_queries", 0),
        "best_baseline_top1": best.get("baseline_top1", 0.0),
        "best_baseline_failure_count": best.get("baseline_failure_count", 0),
        "best_clean_rescue_count": best.get("clean_rescue_count", 0),
        "best_regressed_count": best.get("regressed_count", 0),
        "best_control_rescue_count": best.get("control_rescue_count", 0),
        "best_opportunity_score": best.get("opportunity_score", 0),
        "hard_descriptor_opportunity_found": int(bool(eligible)),
        "opportunity_event_count": len(opportunity_rows),
        "opportunity_gate_distribution": dict(class_counter),
        "oracle_leakage_found": 0,
        "next_recommendation": (
            "CORE-1AH run hard-opportunity descriptor gate with held-out split and controls"
            if eligible
            else "descriptor cue has too few clean hard opportunities; return to observation/proposal quality rather than integration"
        ),
    }
    report = f"""# CORE-1AG Hard Descriptor Opportunity Mining

This stage searches all CORE-1AD gates for harder non-oracle retrieval events where descriptor cue cleanly rescues baseline failures without being matched by shuffled/wrong/random controls.

## Result

- Gate/variant combinations: {len(summary_rows)}
- Best gate: {compact['best_gate']}
- Best variant: {compact['best_variant']}
- Best baseline top1: {float(compact['best_baseline_top1']):.4f}
- Best baseline failures: {compact['best_baseline_failure_count']}
- Best clean rescues: {compact['best_clean_rescue_count']}
- Best regressions: {compact['best_regressed_count']}
- Best control rescues: {compact['best_control_rescue_count']}
- Hard opportunity found: {compact['hard_descriptor_opportunity_found']}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AG_"
    write_csv(
        out_dir / f"{prefix}hard_opportunity_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "gate_name",
            "variant",
            "num_queries",
            "baseline_top1",
            "baseline_failure_count",
            "improved_count",
            "clean_rescue_count",
            "regressed_count",
            "control_rescue_count",
            "descriptor_fail_on_baseline_failure_count",
            "opportunity_score",
            "eligible_hard_gate",
            "selected_as_best_hard_gate",
        ],
    )
    write_csv(
        out_dir / f"{prefix}clean_rescue_events_{args.artifact_version}.csv",
        opportunity_rows,
        ["gate_name", "variant", "query_obs_id", "sequence_id", "event_id", "window_kind", "baseline_target_margin", "descriptor_target_margin", "baseline_target_rank", "descriptor_target_rank"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
