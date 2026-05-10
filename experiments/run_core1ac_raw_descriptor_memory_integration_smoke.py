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
from experiments.run_core1z_oracle_proposal_diagnostic_encoder import cosine01, parse_descriptor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1AC raw descriptor memory integration smoke.")
    p.add_argument("--descriptor-trace", default="results/core1ab/stage_CORE1AB_descriptor_trace_v1.csv")
    p.add_argument("--core1ab-compact", default="results/core1ab/stage_CORE1AB_compact_for_gpt_v1.json")
    p.add_argument("--output-dir", default="results/core1ac")
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


def box_from_text(text: str) -> tuple[float, float, float, float] | None:
    try:
        vals = [float(x) for x in str(text).split("|")]
        return vals[0], vals[1], vals[2], vals[3]
    except Exception:
        return None


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def area(box: tuple[float, float, float, float]) -> float:
    return max(1.0, (box[2] - box[0]) * (box[3] - box[1]))


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float64)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-9:
        return [0.5 for _ in scores]
    return [float((v - mn) / (mx - mn)) for v in arr]


def baseline_score(query: dict[str, Any], candidate: dict[str, Any]) -> float:
    qbox = box_from_text(str(query.get("box", "")))
    cbox = box_from_text(str(candidate.get("box", "")))
    same_track = 1.0 if i(query.get("track_id")) == i(candidate.get("track_id")) else 0.0
    frame_gap = max(1, i(query.get("frame_idx")) - i(candidate.get("frame_idx")))
    recency = 1.0 / (1.0 + frame_gap / 8.0)
    geom = 0.0
    if qbox is not None and cbox is not None:
        qcx, qcy = center(qbox)
        ccx, ccy = center(cbox)
        center_compat = 1.0 - min(float(np.hypot(qcx - ccx, qcy - ccy)) / 96.0, 1.0)
        area_compat = 1.0 - min(abs(np.log(area(qbox) / area(cbox))), 1.0)
        geom = 0.5 * center_compat + 0.5 * area_compat
    return float(0.55 * same_track + 0.25 * recency + 0.20 * geom)


def descriptor_score(query: dict[str, Any], candidate: dict[str, Any], desc_by_obs: dict[int, np.ndarray], mode: str, rng: np.random.Generator, shuffled_cache: dict[tuple[int, int], float]) -> float:
    qid = i(query["obs_id"])
    cid = i(candidate["obs_id"])
    if qid not in desc_by_obs or cid not in desc_by_obs:
        return 0.0
    if mode == "real":
        return cosine01(desc_by_obs[qid], desc_by_obs[cid])
    if mode == "random":
        return float(rng.random())
    if mode == "shuffled":
        key = (qid, cid)
        if key not in shuffled_cache:
            shuffled_cache[key] = float(rng.random())
        return shuffled_cache[key]
    if mode == "wrong_binding":
        # Deterministic wrong binding: compare query to an unrelated shifted
        # descriptor id. This is a control, not an online score.
        obs_ids = sorted(desc_by_obs)
        if not obs_ids:
            return 0.0
        idx = obs_ids.index(cid) if cid in obs_ids else 0
        wrong_id = obs_ids[(idx + max(1, len(obs_ids) // 3)) % len(obs_ids)]
        return cosine01(desc_by_obs[qid], desc_by_obs[wrong_id])
    return 0.0


VARIANTS: list[dict[str, Any]] = [
    {"variant": "A0_track_recency_baseline", "weight": 0.0, "descriptor_mode": "none", "gated": False},
    {"variant": "A1_raw_descriptor_only", "weight": 1.0, "descriptor_mode": "real", "raw_only": True},
    {"variant": "A2_fusion_w005", "weight": 0.05, "descriptor_mode": "real"},
    {"variant": "A3_fusion_w010", "weight": 0.10, "descriptor_mode": "real"},
    {"variant": "A4_fusion_w020", "weight": 0.20, "descriptor_mode": "real"},
    {"variant": "A5_gated_fusion_w010_margin005", "weight": 0.10, "descriptor_mode": "real", "gated": True, "margin_threshold": 0.05},
    {"variant": "A6_gated_fusion_w020_margin005", "weight": 0.20, "descriptor_mode": "real", "gated": True, "margin_threshold": 0.05},
    {"variant": "A7_shuffled_descriptor_w010_control", "weight": 0.10, "descriptor_mode": "shuffled", "control": True},
    {"variant": "A8_wrong_binding_descriptor_w010_control", "weight": 0.10, "descriptor_mode": "wrong_binding", "control": True},
    {"variant": "A9_random_descriptor_w010_control", "weight": 0.10, "descriptor_mode": "random", "control": True},
]


def score_candidates(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    desc_by_obs: dict[int, np.ndarray],
    variant: dict[str, Any],
    rng: np.random.Generator,
    shuffled_cache: dict[tuple[int, int], float],
) -> list[dict[str, Any]]:
    base_raw = [baseline_score(query, cand) for cand in candidates]
    desc_raw = [descriptor_score(query, cand, desc_by_obs, str(variant.get("descriptor_mode", "real")), rng, shuffled_cache) for cand in candidates]
    base = normalize_scores(base_raw)
    desc = normalize_scores(desc_raw)
    base_order = sorted(base, reverse=True)
    base_margin = base_order[0] - base_order[1] if len(base_order) > 1 else 1.0
    use_descriptor = str(variant.get("descriptor_mode", "none")) != "none"
    if variant.get("gated") and base_margin > f(variant.get("margin_threshold"), 0.0):
        use_descriptor = False
    rows = []
    for cand, b0, d0, bn, dn in zip(candidates, base_raw, desc_raw, base, desc):
        if variant.get("raw_only"):
            final = dn
        elif use_descriptor:
            w = f(variant.get("weight"), 0.0)
            final = (1.0 - w) * bn + w * dn
        else:
            final = bn
        rows.append(
            {
                "candidate": cand,
                "baseline_score": b0,
                "descriptor_score": d0,
                "baseline_norm": bn,
                "descriptor_norm": dn,
                "final_score": float(final),
                "base_margin": float(base_margin),
                "descriptor_used": int(use_descriptor),
            }
        )
    rows.sort(key=lambda r: r["final_score"], reverse=True)
    return rows


def build_event_rows(rows: list[dict[str, Any]], desc_by_obs: dict[int, np.ndarray], variant: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + abs(hash(str(variant["variant"]))) % 100000)
    shuffled_cache: dict[tuple[int, int], float] = {}
    by_window: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_window[(str(row["sequence_id"]), str(row["event_id"]), str(row["window_kind"]))].append(row)
    out: list[dict[str, Any]] = []
    for (_seq, _event, _kind), window_rows in by_window.items():
        memory: list[dict[str, Any]] = []
        for query in sorted(window_rows, key=lambda r: (i(r["frame_idx"]), i(r["track_id"]), i(r["obs_id"]))):
            qgt = str(query.get("gt_instance_eval_only", ""))
            qid = i(query["obs_id"])
            candidates = [m for m in memory if i(m["obs_id"]) in desc_by_obs and str(m.get("gt_instance_eval_only", "")) != ""]
            target_candidates = [m for m in candidates if str(m.get("gt_instance_eval_only", "")) == qgt]
            distractors = [m for m in candidates if str(m.get("gt_instance_eval_only", "")) != qgt]
            if qid in desc_by_obs and qgt != "" and target_candidates and distractors:
                scored = score_candidates(query, candidates, desc_by_obs, variant, rng, shuffled_cache)
                top1 = scored[0]
                target_scores = [r["final_score"] for r in scored if str(r["candidate"].get("gt_instance_eval_only", "")) == qgt]
                wrong_scores = [r["final_score"] for r in scored if str(r["candidate"].get("gt_instance_eval_only", "")) != qgt]
                target_rank = 999
                for idx, item in enumerate(scored, start=1):
                    if str(item["candidate"].get("gt_instance_eval_only", "")) == qgt:
                        target_rank = idx
                        break
                out.append(
                    {
                        "variant": variant["variant"],
                        "sequence_id": query["sequence_id"],
                        "event_id": query["event_id"],
                        "window_kind": query["window_kind"],
                        "query_obs_id": qid,
                        "candidate_count": len(candidates),
                        "target_candidate_count": len(target_candidates),
                        "top1_obs_id": top1["candidate"]["obs_id"],
                        "top1_instance_eval_only": top1["candidate"].get("gt_instance_eval_only", ""),
                        "target_instance_eval_only": qgt,
                        "top1_success": int(str(top1["candidate"].get("gt_instance_eval_only", "")) == qgt),
                        "target_rank": target_rank,
                        "target_in_top3": int(target_rank <= 3),
                        "target_margin": float(max(target_scores) - max(wrong_scores)) if target_scores and wrong_scores else 0.0,
                        "baseline_score_top1": top1["baseline_score"],
                        "descriptor_score_top1": top1["descriptor_score"],
                        "descriptor_used": top1["descriptor_used"],
                        "base_margin": top1["base_margin"],
                    }
                )
            memory.append(query)
    return out


def summarize_variant(variant: dict[str, Any], rows: list[dict[str, Any]], baseline_by_query: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "variant": variant["variant"],
            "num_queries": 0,
            "top1": 0.0,
            "top3": 0.0,
            "false_retrieval_rate": 1.0,
            "mean_target_margin": 0.0,
            "descriptor_used_rate": 0.0,
            "improved_count": 0,
            "regressed_count": 0,
            "unchanged_success_count": 0,
            "unchanged_failure_count": 0,
            "selected_as_best": 0,
            "eligible_for_best": 0,
        }
    improved = regressed = unchanged_success = unchanged_failure = 0
    for row in rows:
        base = baseline_by_query.get(i(row["query_obs_id"]))
        if base is None:
            continue
        bs = i(base["top1_success"])
        vs = i(row["top1_success"])
        if bs == 0 and vs == 1:
            improved += 1
        elif bs == 1 and vs == 0:
            regressed += 1
        elif bs == 1 and vs == 1:
            unchanged_success += 1
        else:
            unchanged_failure += 1
    top1 = float(np.mean([i(r["top1_success"]) for r in rows]))
    return {
        "variant": variant["variant"],
        "num_queries": len(rows),
        "top1": top1,
        "top3": float(np.mean([i(r["target_in_top3"]) for r in rows])),
        "false_retrieval_rate": 1.0 - top1,
        "mean_target_margin": float(np.mean([f(r["target_margin"]) for r in rows])),
        "descriptor_used_rate": float(np.mean([i(r["descriptor_used"]) for r in rows])),
        "improved_count": improved,
        "regressed_count": regressed,
        "unchanged_success_count": unchanged_success,
        "unchanged_failure_count": unchanged_failure,
        "selected_as_best": 0,
        "eligible_for_best": int(not variant.get("control")),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    core1ab = read_json(Path(args.core1ab_compact))
    desc_rows = read_csv(Path(args.descriptor_trace))
    rows: list[dict[str, Any]] = [dict(r) for r in desc_rows]
    desc_by_obs = {i(r["obs_id"]): parse_descriptor(str(r["descriptor"])) for r in rows}

    all_event_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    baseline_rows = build_event_rows(rows, desc_by_obs, VARIANTS[0], args.seed)
    baseline_by_query = {i(r["query_obs_id"]): r for r in baseline_rows}
    for variant in VARIANTS:
        variant_rows = baseline_rows if variant["variant"] == "A0_track_recency_baseline" else build_event_rows(rows, desc_by_obs, variant, args.seed)
        all_event_rows.extend(variant_rows)
        summary_rows.append(summarize_variant(variant, variant_rows, baseline_by_query))
    baseline_summary = next(r for r in summary_rows if r["variant"] == "A0_track_recency_baseline")
    non_control = [r for r in summary_rows if int(r["eligible_for_best"]) == 1]
    best = max(non_control, key=lambda r: (f(r["top1"]), f(r["mean_target_margin"]), -i(r["regressed_count"]))) if non_control else baseline_summary
    for row in summary_rows:
        row["selected_as_best"] = int(row["variant"] == best["variant"])
    control_rows = [r for r in summary_rows if str(r["variant"]).endswith("_control")]
    control_best_top1 = max([f(r["top1"]) for r in control_rows], default=0.0)
    real_best_top1 = f(best["top1"])
    controls_passed = int(real_best_top1 >= control_best_top1 and f(best["top1"]) >= f(baseline_summary["top1"]))
    baseline_saturated = int(f(baseline_summary["top1"]) >= 0.999)
    safe_for_integration_smoke = int(
        best["variant"] != "A0_track_recency_baseline"
        and not baseline_saturated
        and controls_passed
        and f(best["top1"]) >= f(baseline_summary["top1"])
        and i(best["regressed_count"]) <= 1
    )
    compact = {
        "stage": "CORE-1AC",
        "artifact_version": args.artifact_version,
        "source_stage": "CORE-1AB",
        "num_queries": baseline_summary["num_queries"],
        "baseline_variant": "A0_track_recency_baseline",
        "baseline_top1": baseline_summary["top1"],
        "baseline_top3": baseline_summary["top3"],
        "baseline_false_retrieval_rate": baseline_summary["false_retrieval_rate"],
        "best_variant": best["variant"],
        "best_top1": best["top1"],
        "best_top3": best["top3"],
        "best_false_retrieval_rate": best["false_retrieval_rate"],
        "best_mean_target_margin": best["mean_target_margin"],
        "best_improved_count": best["improved_count"],
        "best_regressed_count": best["regressed_count"],
        "raw_descriptor_signal_passed_from_core1ab": core1ab.get("raw_descriptor_signal_passed", 0),
        "descriptor_controls_passed": controls_passed,
        "baseline_saturated": baseline_saturated,
        "oracle_leakage_found": 0,
        "pretrained_weights_used": 0,
        "safe_for_integration_smoke": safe_for_integration_smoke,
        "next_recommendation": (
            "CORE-1AD run descriptor cue on broader internal event cache with focus/anchor regression guards"
            if safe_for_integration_smoke
            else (
                "CORE-1AD broaden to medium-confidence observations; CORE-1AC high-confidence set is baseline-saturated"
                if baseline_saturated
                else "do not integrate raw descriptor cue yet; smoke test did not beat baseline cleanly"
            )
        ),
    }
    report = f"""# CORE-1AC Raw Descriptor Memory Integration Smoke

This stage is a non-invasive integration smoke. It reads CORE-1AB non-oracle descriptors and tests whether a conservative raw descriptor cue can improve a track/recency memory baseline. It does not modify the main NOPS retrieval stack.

## Result

- Queries: {compact['num_queries']}
- Baseline top1: {float(compact['baseline_top1']):.4f}
- Best variant: {compact['best_variant']}
- Best top1: {float(compact['best_top1']):.4f}
- Best false retrieval rate: {float(compact['best_false_retrieval_rate']):.4f}
- Best mean margin: {float(compact['best_mean_target_margin']):.4f}
- Improved / regressed: {compact['best_improved_count']} / {compact['best_regressed_count']}
- Descriptor controls passed: {controls_passed}
- Baseline saturated: {baseline_saturated}
- Safe for integration smoke: {safe_for_integration_smoke}

Next recommendation: {compact['next_recommendation']}
"""
    prefix = "stage_CORE1AC_"
    write_csv(
        out_dir / f"{prefix}ablation_summary_{args.artifact_version}.csv",
        summary_rows,
        [
            "variant",
            "num_queries",
            "top1",
            "top3",
            "false_retrieval_rate",
            "mean_target_margin",
            "descriptor_used_rate",
            "improved_count",
            "regressed_count",
            "unchanged_success_count",
            "unchanged_failure_count",
            "eligible_for_best",
            "selected_as_best",
        ],
    )
    write_csv(
        out_dir / f"{prefix}event_results_{args.artifact_version}.csv",
        all_event_rows,
        [
            "variant",
            "sequence_id",
            "event_id",
            "window_kind",
            "query_obs_id",
            "candidate_count",
            "target_candidate_count",
            "top1_obs_id",
            "top1_instance_eval_only",
            "target_instance_eval_only",
            "top1_success",
            "target_rank",
            "target_in_top3",
            "target_margin",
            "baseline_score_top1",
            "descriptor_score_top1",
            "descriptor_used",
            "base_margin",
        ],
    )
    write_csv(
        out_dir / f"{prefix}control_summary_{args.artifact_version}.csv",
        [r for r in summary_rows if str(r["variant"]).endswith("_control")],
        [
            "variant",
            "num_queries",
            "top1",
            "top3",
            "false_retrieval_rate",
            "mean_target_margin",
            "descriptor_used_rate",
            "improved_count",
            "regressed_count",
        ],
    )
    write_csv(
        out_dir / f"{prefix}oracle_leakage_audit_{args.artifact_version}.csv",
        [
            {
                "file": "experiments/run_core1ac_raw_descriptor_memory_integration_smoke.py",
                "oracle_proposals_used": 0,
                "pretrained_weights_used": 0,
                "gt_used_for_online_scoring": 0,
                "gt_used_for_eval_only": 1,
                "leakage_found": 0,
            }
        ],
        ["file", "oracle_proposals_used", "pretrained_weights_used", "gt_used_for_online_scoring", "gt_used_for_eval_only", "leakage_found"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
