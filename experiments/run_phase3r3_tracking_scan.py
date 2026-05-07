"""Run the Phase 3R.3 local identity-slot scan."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3r3_utils import (
    TRACK_A_NAME,
    TRACK_C_NAME,
    evaluate_phase3r3_bundle,
    load_phase3r2_before_lookup,
    pick_best_scan_row,
)
from experiments.phase3r_utils import write_csv

TRACK_A_BEFORE = load_phase3r2_before_lookup().get(TRACK_A_NAME, {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 3R.3 tracking scan.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml", help="Path to config.")
    parser.add_argument("--output-dir", default="results/phase3r3", help="Directory for artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--max-workers", type=int, default=3, help="Parallel workers for candidate evaluation.")
    parser.add_argument("--stage", default="all", choices=["all", "stage1", "stage2"], help="Scan stage to run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_csv = output_dir / "tracking_scan_stage1_v3.csv"
    stage1_best_json = output_dir / "tracking_scan_stage1_best_v1.json"
    final_csv = output_dir / "tracking_scan_v3.csv"

    before_lookup = load_phase3r2_before_lookup()
    rows: list[dict[str, object]] = []

    stage1_candidates = []
    for slot_max_gap in [64, 96, 128]:
        for slot_tau in [0.58, 0.62, 0.66]:
            for slot_margin in [0.05, 0.08, 0.12]:
                stage1_candidates.append(
                    {
                        "stage": "slot_resurrection_core",
                        "dormant_frames": 16,
                        "ghost_frames": 80,
                        "tau_g": 12,
                        "tau_res_short": 0.56,
                        "tau_res_long": 0.68,
                        "slot_topk_per_proto": 4,
                        "slot_max_gap": slot_max_gap,
                        "slot_tau": slot_tau,
                        "slot_margin": slot_margin,
                        "min_track_age_for_slot": 4,
                    }
                )

    stage1_rows = []
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
    for slot_topk_per_proto in [2, 4, 6]:
        for min_track_age_for_slot in [3, 4, 5]:
            stage2_candidates.append(
                {
                    "stage": "slot_archive_policy",
                    "dormant_frames": 16,
                    "ghost_frames": 80,
                    "tau_g": 12,
                    "tau_res_short": 0.56,
                    "tau_res_long": 0.68,
                    "slot_topk_per_proto": slot_topk_per_proto,
                    "slot_max_gap": int(stage1_best["slot_max_gap"]),
                    "slot_tau": float(stage1_best["slot_tau"]),
                    "slot_margin": float(stage1_best["slot_margin"]),
                    "min_track_age_for_slot": min_track_age_for_slot,
                }
            )

    stage2_rows = _evaluate_candidates(args.config, stage2_candidates, args.seed, args.max_workers)
    rows.extend(stage1_rows)
    rows.extend(stage2_rows)
    final_best = pick_best_scan_row(stage2_rows, before_lookup)

    for row in rows:
        row["is_best"] = int(
            row["stage"] == final_best["stage"]
            and int(row["slot_topk_per_proto"]) == int(final_best["slot_topk_per_proto"])
            and int(row["slot_max_gap"]) == int(final_best["slot_max_gap"])
            and abs(float(row["slot_tau"]) - float(final_best["slot_tau"])) < 1e-9
            and abs(float(row["slot_margin"]) - float(final_best["slot_margin"])) < 1e-9
            and int(row["min_track_age_for_slot"]) == int(final_best["min_track_age_for_slot"])
        )

    write_csv(final_csv, rows)

    print(f"saved_scan={final_csv}")
    print(
        "best="
        f"stage={final_best['stage']}, "
        f"slot_topk={int(final_best['slot_topk_per_proto'])}, "
        f"slot_max_gap={int(final_best['slot_max_gap'])}, "
        f"slot_tau={float(final_best['slot_tau']):.2f}, "
        f"slot_margin={float(final_best['slot_margin']):.2f}, "
        f"min_track_age={int(final_best['min_track_age_for_slot'])}, "
        f"track_c_pool={float(final_best['track_c_candidate_pool_nonempty_rate']):.4f}, "
        f"track_c_track_after={float(final_best['track_c_same_track_after_concept_recovery']):.4f}, "
        f"track_c_same_track={float(final_best['track_c_same_track_reentry_recovery']):.4f}, "
        f"track_c_same_proto={float(final_best['track_c_same_prototype_reentry_recovery']):.4f}, "
        f"track_c_pfr={float(final_best['track_c_pfr']):.4f}, "
        f"track_c_idsw={int(final_best['track_c_track_idsw'])}"
    )


def _evaluate_candidate(config_path: str, candidate: dict[str, object], seed: int) -> dict[str, object]:
    bundle = evaluate_phase3r3_bundle(
        config_path,
        tracking_override={
            "dormant_frames": int(candidate["dormant_frames"]),
            "ghost_frames": int(candidate["ghost_frames"]),
            "tau_g": float(candidate["tau_g"]),
            "tau_res_short": float(candidate["tau_res_short"]),
            "tau_res_long": float(candidate["tau_res_long"]),
            "slot_topk_per_proto": int(candidate["slot_topk_per_proto"]),
            "slot_max_gap": int(candidate["slot_max_gap"]),
            "slot_tau": float(candidate["slot_tau"]),
            "slot_margin": float(candidate["slot_margin"]),
            "min_track_age_for_slot": int(candidate["min_track_age_for_slot"]),
        },
        seed=seed,
        scenario_names=[TRACK_C_NAME],
        frame_record_mode="lite",
    )
    row_lookup = {row["scenario_name"]: row for row in bundle["rows"]}
    track_c = row_lookup[TRACK_C_NAME]
    return {
        "stage": candidate["stage"],
        "dormant_frames": int(candidate["dormant_frames"]),
        "ghost_frames": int(candidate["ghost_frames"]),
        "tau_g": int(candidate["tau_g"]),
        "tau_res_short": float(candidate["tau_res_short"]),
        "tau_res_long": float(candidate["tau_res_long"]),
        "slot_topk_per_proto": int(candidate["slot_topk_per_proto"]),
        "slot_max_gap": int(candidate["slot_max_gap"]),
        "slot_tau": float(candidate["slot_tau"]),
        "slot_margin": float(candidate["slot_margin"]),
        "min_track_age_for_slot": int(candidate["min_track_age_for_slot"]),
        "track_c_candidate_pool_nonempty_rate": float(track_c["candidate_pool_nonempty_rate"]),
        "track_c_slot_pool_nonempty_rate": float(track_c["slot_pool_nonempty_rate"]),
        "track_c_slot_resurrection_attempt_rate": float(track_c["slot_resurrection_attempt_rate"]),
        "track_c_slot_resurrection_success_rate": float(track_c["slot_resurrection_success_rate"]),
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
