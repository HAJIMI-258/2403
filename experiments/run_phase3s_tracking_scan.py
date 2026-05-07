"""Run the Phase 3S prototype-continuation scan."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3r_utils import write_csv
from experiments.phase3s_utils import (
    TRACK_A_NAME,
    TRACK_C_NAME,
    evaluate_phase3s_bundle,
    load_phase3r3_before_lookup,
    pick_best_scan_row,
)

TRACK_A_BEFORE = load_phase3r3_before_lookup().get(TRACK_A_NAME, {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 3S tracking scan.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml", help="Path to config.")
    parser.add_argument("--output-dir", default="results/phase3s", help="Directory for artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--max-workers", type=int, default=3, help="Parallel workers for candidate evaluation.")
    parser.add_argument("--stage", default="all", choices=["all", "stage1", "stage2"], help="Scan stage to run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_csv = output_dir / "tracking_scan_stage1_v4.csv"
    stage1_best_json = output_dir / "tracking_scan_stage1_best_v1.json"
    final_csv = output_dir / "tracking_scan_v4.csv"

    before_lookup = load_phase3r3_before_lookup()
    rows: list[dict[str, object]] = []

    stage1_candidates = []
    for continuation_max_gap in [64, 96, 128]:
        for tau_continuation in [0.58, 0.62, 0.66]:
            for continuation_margin in [0.05, 0.08, 0.12]:
                stage1_candidates.append(
                    {
                        "stage": "continuation_core",
                        "continuation_topk_per_proto": 4,
                        "continuation_max_gap": continuation_max_gap,
                        "tau_continuation": tau_continuation,
                        "continuation_margin": continuation_margin,
                        "min_track_age_for_continuation": 4,
                    }
                )

    stage1_rows: list[dict[str, object]] = []
    if args.stage in {"all", "stage1"}:
        stage1_rows = _evaluate_candidates(args.config, stage1_candidates, args.seed, args.max_workers)
        stage1_best = pick_best_scan_row(stage1_rows, before_lookup)
        write_csv(stage1_csv, stage1_rows)
        stage1_best_json.write_text(json.dumps(stage1_best, indent=2), encoding="utf-8")
        print(f"saved_stage1={stage1_csv}")
        print(f"saved_stage1_best={stage1_best_json}")
        if args.stage == "stage1":
            return
    else:
        stage1_best = json.loads(stage1_best_json.read_text(encoding="utf-8"))

    stage2_candidates = []
    for continuation_topk_per_proto in [2, 4, 6]:
        for min_track_age_for_continuation in [3, 4, 5]:
            stage2_candidates.append(
                {
                    "stage": "continuation_archive_policy",
                    "continuation_topk_per_proto": continuation_topk_per_proto,
                    "continuation_max_gap": int(stage1_best["continuation_max_gap"]),
                    "tau_continuation": float(stage1_best["tau_continuation"]),
                    "continuation_margin": float(stage1_best["continuation_margin"]),
                    "min_track_age_for_continuation": min_track_age_for_continuation,
                }
            )

    stage2_rows = _evaluate_candidates(args.config, stage2_candidates, args.seed, args.max_workers)
    rows.extend(stage1_rows)
    rows.extend(stage2_rows)
    final_best = pick_best_scan_row(stage2_rows, before_lookup)

    for row in rows:
        row["is_best"] = int(
            row["stage"] == final_best["stage"]
            and int(row["continuation_topk_per_proto"]) == int(final_best["continuation_topk_per_proto"])
            and int(row["continuation_max_gap"]) == int(final_best["continuation_max_gap"])
            and abs(float(row["tau_continuation"]) - float(final_best["tau_continuation"])) < 1e-9
            and abs(float(row["continuation_margin"]) - float(final_best["continuation_margin"])) < 1e-9
            and int(row["min_track_age_for_continuation"]) == int(final_best["min_track_age_for_continuation"])
        )

    write_csv(final_csv, rows)
    print(f"saved_scan={final_csv}")
    print(
        "best="
        f"stage={final_best['stage']}, "
        f"topk={int(final_best['continuation_topk_per_proto'])}, "
        f"max_gap={int(final_best['continuation_max_gap'])}, "
        f"tau_continuation={float(final_best['tau_continuation']):.2f}, "
        f"continuation_margin={float(final_best['continuation_margin']):.2f}, "
        f"min_track_age={int(final_best['min_track_age_for_continuation'])}, "
        f"track_c_cont_bank={float(final_best['track_c_continuation_bank_nonempty_rate']):.4f}, "
        f"track_c_pool={float(final_best['track_c_candidate_pool_nonempty_rate']):.4f}, "
        f"track_c_track_after={float(final_best['track_c_same_track_after_concept_recovery']):.4f}, "
        f"track_c_same_track={float(final_best['track_c_same_track_reentry_recovery']):.4f}, "
        f"track_c_same_proto={float(final_best['track_c_same_prototype_reentry_recovery']):.4f}, "
        f"track_c_pfr={float(final_best['track_c_pfr']):.4f}, "
        f"track_c_idsw={int(final_best['track_c_track_idsw'])}"
    )


def _evaluate_candidate(config_path: str, candidate: dict[str, object], seed: int) -> dict[str, object]:
    bundle = evaluate_phase3s_bundle(
        config_path,
        tracking_override={
            "tau_continuation": float(candidate["tau_continuation"]),
            "continuation_margin": float(candidate["continuation_margin"]),
            "enable_identity_slots": False,
        },
        memory_override={
            "continuation_topk_per_proto": int(candidate["continuation_topk_per_proto"]),
            "continuation_max_gap": int(candidate["continuation_max_gap"]),
            "min_track_age_for_continuation": int(candidate["min_track_age_for_continuation"]),
            "enable_continuation_bank": True,
        },
        seed=seed,
        scenario_names=[TRACK_C_NAME],
        frame_record_mode="lite",
    )
    row_lookup = {row["scenario_name"]: row for row in bundle["rows"]}
    track_c = row_lookup[TRACK_C_NAME]
    return {
        "stage": candidate["stage"],
        "continuation_topk_per_proto": int(candidate["continuation_topk_per_proto"]),
        "continuation_max_gap": int(candidate["continuation_max_gap"]),
        "tau_continuation": float(candidate["tau_continuation"]),
        "continuation_margin": float(candidate["continuation_margin"]),
        "min_track_age_for_continuation": int(candidate["min_track_age_for_continuation"]),
        "track_c_continuation_bank_nonempty_rate": float(track_c["continuation_bank_nonempty_rate"]),
        "track_c_candidate_pool_nonempty_rate": float(track_c["candidate_pool_nonempty_rate"]),
        "track_c_continuation_attempt_rate": float(track_c["continuation_attempt_rate"]),
        "track_c_continuation_success_rate": float(track_c["continuation_success_rate"]),
        "track_c_same_track_after_concept_recovery": float(track_c["same_track_after_concept_recovery"]),
        "track_c_same_track_reentry_recovery": float(track_c["same_track_reentry_recovery"]),
        "track_c_same_prototype_reentry_recovery": float(track_c["same_prototype_reentry_recovery"]),
        "track_c_pfr": float(track_c["pfr"]),
        "track_c_track_idsw": int(track_c["track_idsw"]),
        "track_c_memory_growth": float(track_c["memory_growth"]),
        "track_a_u_recall": float(TRACK_A_BEFORE.get("u_recall", 0.0)),
        "track_a_same_prototype_reentry_recovery": float(
            TRACK_A_BEFORE.get("same_prototype_reentry_recovery", 0.0)
        ),
        "track_a_memory_growth": float(TRACK_A_BEFORE.get("memory_growth", 0.0)),
    }


def _evaluate_candidates(
    config_path: str,
    candidates: list[dict[str, object]],
    seed: int,
    max_workers: int,
) -> list[dict[str, object]]:
    if max_workers <= 1:
        return [_evaluate_candidate(config_path, candidate, seed) for candidate in candidates]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_evaluate_candidate, config_path, candidate, seed) for candidate in candidates]
        return [future.result() for future in futures]


if __name__ == "__main__":
    main()
