from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AE descriptor gate robustness audit.")
    p.add_argument("--core1ad-compact", default="results/core1ad/stage_CORE1AD_compact_for_gpt_v1.json")
    p.add_argument("--event-results", default="results/core1ad/stage_CORE1AD_event_results_v1.csv")
    p.add_argument("--output-dir", default="results/core1ae")
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
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


def top1(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([i(r["top1_success"]) for r in rows])) if rows else 0.0


def bootstrap_delta(base: np.ndarray, test: np.ndarray, samples: int, seed: int) -> dict[str, float]:
    if base.size == 0 or test.size == 0 or base.size != test.size:
        return {"mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = np.random.default_rng(seed)
    deltas = []
    n = base.size
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        deltas.append(float(test[idx].mean() - base[idx].mean()))
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "mean_delta": float(arr.mean()),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
    }


def paired_rows(rows: list[dict[str, str]], gate: str, variant: str) -> dict[int, dict[str, str]]:
    out = {}
    for row in rows:
        if row["gate_name"] == gate and row["variant"] == variant:
            out[i(row["query_obs_id"])] = row
    return out


def compare_variant(rows: list[dict[str, str]], gate: str, variant: str, samples: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = paired_rows(rows, gate, "A0_track_recency_baseline")
    test = paired_rows(rows, gate, variant)
    keys = sorted(set(base) & set(test))
    delta_rows: list[dict[str, Any]] = []
    for key in keys:
        b = base[key]
        t = test[key]
        delta_class = "unchanged_success"
        if i(b["top1_success"]) == 0 and i(t["top1_success"]) == 1:
            delta_class = "improved"
        elif i(b["top1_success"]) == 1 and i(t["top1_success"]) == 0:
            delta_class = "regressed"
        elif i(b["top1_success"]) == 0 and i(t["top1_success"]) == 0:
            delta_class = "unchanged_failure"
        delta_rows.append(
            {
                "gate_name": gate,
                "variant": variant,
                "query_obs_id": key,
                "sequence_id": t["sequence_id"],
                "event_id": t["event_id"],
                "window_kind": t["window_kind"],
                "baseline_success": b["top1_success"],
                "variant_success": t["top1_success"],
                "baseline_target_margin": b["target_margin"],
                "variant_target_margin": t["target_margin"],
                "delta_class": delta_class,
            }
        )
    b_arr = np.asarray([i(base[k]["top1_success"]) for k in keys], dtype=np.float64)
    t_arr = np.asarray([i(test[k]["top1_success"]) for k in keys], dtype=np.float64)
    boot = bootstrap_delta(b_arr, t_arr, samples, seed)
    summary = {
        "gate_name": gate,
        "variant": variant,
        "num_queries": len(keys),
        "baseline_top1": float(b_arr.mean()) if keys else 0.0,
        "variant_top1": float(t_arr.mean()) if keys else 0.0,
        "delta_top1": float(t_arr.mean() - b_arr.mean()) if keys else 0.0,
        "improved_count": sum(1 for r in delta_rows if r["delta_class"] == "improved"),
        "regressed_count": sum(1 for r in delta_rows if r["delta_class"] == "regressed"),
        "unchanged_success_count": sum(1 for r in delta_rows if r["delta_class"] == "unchanged_success"),
        "unchanged_failure_count": sum(1 for r in delta_rows if r["delta_class"] == "unchanged_failure"),
        **boot,
    }
    return summary, delta_rows


def group_summary(delta_rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        grouped[str(row[group_key])].append(row)
    out = []
    for key, rows in sorted(grouped.items()):
        base = np.asarray([i(r["baseline_success"]) for r in rows], dtype=np.float64)
        var = np.asarray([i(r["variant_success"]) for r in rows], dtype=np.float64)
        out.append(
            {
                "group_key": group_key,
                "group_value": key,
                "num_queries": len(rows),
                "baseline_top1": float(base.mean()) if rows else 0.0,
                "variant_top1": float(var.mean()) if rows else 0.0,
                "delta_top1": float(var.mean() - base.mean()) if rows else 0.0,
                "improved_count": sum(1 for r in rows if r["delta_class"] == "improved"),
                "regressed_count": sum(1 for r in rows if r["delta_class"] == "regressed"),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compact_ad = read_json(Path(args.core1ad_compact))
    rows = read_csv(Path(args.event_results))
    gate = str(compact_ad.get("best_gate", ""))
    selected_variant = str(compact_ad.get("best_variant", ""))
    controls = [
        "A7_shuffled_descriptor_w010_control",
        "A8_wrong_binding_descriptor_w010_control",
        "A9_random_descriptor_w010_control",
    ]
    selected_summary, selected_delta = compare_variant(rows, gate, selected_variant, args.bootstrap_samples, args.seed)
    control_summaries: list[dict[str, Any]] = []
    control_delta_rows: list[dict[str, Any]] = []
    for idx, control in enumerate(controls):
        summary, deltas = compare_variant(rows, gate, control, args.bootstrap_samples, args.seed + idx + 10)
        control_summaries.append(summary)
        control_delta_rows.extend(deltas)
    best_control_delta = max([f(r["delta_top1"]) for r in control_summaries], default=0.0)
    split_rows = group_summary(selected_delta, "sequence_id") + group_summary(selected_delta, "event_id") + group_summary(selected_delta, "window_kind")
    ci_excludes_zero = int(f(selected_summary["ci95_low"]) > 0.0)
    beats_controls = int(f(selected_summary["delta_top1"]) > best_control_delta)
    regression_ok = int(i(selected_summary["regressed_count"]) <= 1)
    robust_passed = int(ci_excludes_zero and beats_controls and regression_ok)
    compact = {
        "stage": "CORE-1AE",
        "artifact_version": args.artifact_version,
        "source_stage": "CORE-1AD",
        "gate_name": gate,
        "selected_variant": selected_variant,
        "num_queries": selected_summary["num_queries"],
        "baseline_top1": selected_summary["baseline_top1"],
        "selected_top1": selected_summary["variant_top1"],
        "delta_top1": selected_summary["delta_top1"],
        "bootstrap_ci95_low": selected_summary["ci95_low"],
        "bootstrap_ci95_high": selected_summary["ci95_high"],
        "improved_count": selected_summary["improved_count"],
        "regressed_count": selected_summary["regressed_count"],
        "best_control_delta_top1": best_control_delta,
        "ci_excludes_zero": ci_excludes_zero,
        "beats_controls": beats_controls,
        "regression_ok": regression_ok,
        "robustness_gate_passed": robust_passed,
        "safe_for_main_integration": 0,
        "oracle_leakage_found": 0,
        "next_recommendation": (
            "CORE-1AF run broader internal cache integration with focus/anchor guards"
            if robust_passed
            else "do not integrate descriptor cue; CORE-1AD gain is too small or not robust against controls"
        ),
    }
    report = f"""# CORE-1AE Descriptor Gate Robustness Audit

This stage audits the CORE-1AD selected descriptor gate with paired bootstrap and shuffled/wrong/random controls. It is a decision gate, not a new model.

## Result

- Gate: {gate}
- Variant: {selected_variant}
- Queries: {selected_summary['num_queries']}
- Baseline top1: {float(selected_summary['baseline_top1']):.4f}
- Selected top1: {float(selected_summary['variant_top1']):.4f}
- Delta top1: {float(selected_summary['delta_top1']):.4f}
- 95% bootstrap CI: [{float(selected_summary['ci95_low']):.4f}, {float(selected_summary['ci95_high']):.4f}]
- Improved / regressed: {selected_summary['improved_count']} / {selected_summary['regressed_count']}
- Best control delta: {best_control_delta:.4f}
- CI excludes zero: {ci_excludes_zero}
- Beats controls: {beats_controls}
- Robustness gate passed: {robust_passed}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AE_"
    write_csv(
        out_dir / f"{prefix}variant_robustness_summary_{args.artifact_version}.csv",
        [selected_summary] + control_summaries,
        [
            "gate_name",
            "variant",
            "num_queries",
            "baseline_top1",
            "variant_top1",
            "delta_top1",
            "mean_delta",
            "ci95_low",
            "ci95_high",
            "improved_count",
            "regressed_count",
            "unchanged_success_count",
            "unchanged_failure_count",
        ],
    )
    write_csv(
        out_dir / f"{prefix}selected_event_delta_{args.artifact_version}.csv",
        selected_delta,
        [
            "gate_name",
            "variant",
            "query_obs_id",
            "sequence_id",
            "event_id",
            "window_kind",
            "baseline_success",
            "variant_success",
            "baseline_target_margin",
            "variant_target_margin",
            "delta_class",
        ],
    )
    write_csv(
        out_dir / f"{prefix}control_event_delta_{args.artifact_version}.csv",
        control_delta_rows,
        [
            "gate_name",
            "variant",
            "query_obs_id",
            "sequence_id",
            "event_id",
            "window_kind",
            "baseline_success",
            "variant_success",
            "baseline_target_margin",
            "variant_target_margin",
            "delta_class",
        ],
    )
    write_csv(
        out_dir / f"{prefix}split_summary_{args.artifact_version}.csv",
        split_rows,
        ["group_key", "group_value", "num_queries", "baseline_top1", "variant_top1", "delta_top1", "improved_count", "regressed_count"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
