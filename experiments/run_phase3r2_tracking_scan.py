"""Run the Phase 3R.2 local resurrection-parameter scan."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3r2_utils import (
    TRACK_A_NAME,
    TRACK_C_NAME,
    evaluate_phase3r2_bundle,
    load_phase3r_before_lookup,
    pick_best_scan_row,
)
from experiments.phase3r_utils import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 3R.2 tracking scan.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml", help="Path to config.")
    parser.add_argument("--output-dir", default="results/phase3r2", help="Directory for artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--max-workers", type=int, default=3, help="Parallel workers for candidate evaluation.")
    parser.add_argument("--stage", default="all", choices=["all", "stage1", "stage2"], help="Scan stage to run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_csv = output_dir / "tracking_scan_stage1_v2.csv"
    stage1_best_json = output_dir / "tracking_scan_stage1_best_v1.json"
    final_csv = output_dir / "tracking_scan_v2.csv"

    before_lookup = load_phase3r_before_lookup()
    rows: list[dict[str, object]] = []

    stage1_candidates = []
    for tau_g in [8, 12, 16]:
        for tau_res_short in [0.56, 0.60, 0.64]:
            for tau_res_long in [0.68, 0.72, 0.76]:
                if tau_res_long < tau_res_short:
                    continue
                stage1_candidates.append(
                    {
                        "stage": "resurrection_core",
                        "dormant_frames": 24,
                        "ghost_frames": 64,
                        "tau_g": tau_g,
                        "tau_res_short": tau_res_short,
                        "tau_res_long": tau_res_long,
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
    for dormant_frames in [16, 24, 32]:
        for ghost_frames in [48, 64, 80]:
            stage2_candidates.append(
                {
                    "stage": "state_lifetime",
                    "dormant_frames": dormant_frames,
                    "ghost_frames": ghost_frames,
                    "tau_g": int(stage1_best["tau_g"]),
                    "tau_res_short": float(stage1_best["tau_res_short"]),
                    "tau_res_long": float(stage1_best["tau_res_long"]),
                }
            )

    stage2_rows = _evaluate_candidates(args.config, stage2_candidates, args.seed, args.max_workers)
    rows.extend(stage1_rows)
    rows.extend(stage2_rows)
    final_best = pick_best_scan_row(stage2_rows, before_lookup)

    for row in rows:
        row["is_best"] = int(
            row["stage"] == final_best["stage"]
            and int(row["dormant_frames"]) == int(final_best["dormant_frames"])
            and int(row["ghost_frames"]) == int(final_best["ghost_frames"])
            and int(row["tau_g"]) == int(final_best["tau_g"])
            and abs(float(row["tau_res_short"]) - float(final_best["tau_res_short"])) < 1e-9
            and abs(float(row["tau_res_long"]) - float(final_best["tau_res_long"])) < 1e-9
        )

    write_csv(final_csv, rows)

    print(f"saved_scan={final_csv}")
    print(
        "best="
        f"stage={final_best['stage']}, dormant={int(final_best['dormant_frames'])}, ghost={int(final_best['ghost_frames'])}, "
        f"tau_g={int(final_best['tau_g'])}, tau_res_short={float(final_best['tau_res_short']):.2f}, "
        f"tau_res_long={float(final_best['tau_res_long']):.2f}, "
        f"track_c_same_track={float(final_best['track_c_same_track_reentry_recovery']):.4f}, "
        f"track_c_same_proto={float(final_best['track_c_same_prototype_reentry_recovery']):.4f}, "
        f"track_c_track_after_concept={float(final_best['track_c_same_track_after_concept_recovery']):.4f}, "
        f"track_c_pfr={float(final_best['track_c_pfr']):.4f}, "
        f"track_c_idsw={int(final_best['track_c_track_idsw'])}"
    )


def _evaluate_candidate(config_path: str, candidate: dict[str, object], seed: int) -> dict[str, object]:
    bundle = evaluate_phase3r2_bundle(
        config_path,
        tracking_override={
            "dormant_frames": int(candidate["dormant_frames"]),
            "ghost_frames": int(candidate["ghost_frames"]),
            "tau_g": float(candidate["tau_g"]),
            "tau_res_short": float(candidate["tau_res_short"]),
            "tau_res_long": float(candidate["tau_res_long"]),
        },
        seed=seed,
        scenario_names=[TRACK_A_NAME, TRACK_C_NAME],
    )
    row_lookup = {row["scenario_name"]: row for row in bundle["rows"]}
    track_a = row_lookup[TRACK_A_NAME]
    track_c = row_lookup[TRACK_C_NAME]
    return {
        "stage": candidate["stage"],
        "dormant_frames": int(candidate["dormant_frames"]),
        "ghost_frames": int(candidate["ghost_frames"]),
        "tau_g": int(candidate["tau_g"]),
        "tau_res_short": float(candidate["tau_res_short"]),
        "tau_res_long": float(candidate["tau_res_long"]),
        "track_c_same_track_reentry_recovery": float(track_c["same_track_reentry_recovery"]),
        "track_c_same_prototype_reentry_recovery": float(track_c["same_prototype_reentry_recovery"]),
        "track_c_same_track_after_concept_recovery": float(track_c["same_track_after_concept_recovery"]),
        "track_c_prototype_gated_resurrection_attempt_rate": float(
            track_c["prototype_gated_resurrection_attempt_rate"]
        ),
        "track_c_resurrection_success_given_candidate_exists": float(
            track_c["resurrection_success_given_candidate_exists"]
        ),
        "track_c_pfr": float(track_c["pfr"]),
        "track_c_track_idsw": int(track_c["track_idsw"]),
        "track_c_memory_growth": float(track_c["memory_growth"]),
        "track_a_u_recall": float(track_a["u_recall"]),
        "track_a_same_prototype_reentry_recovery": float(track_a["same_prototype_reentry_recovery"]),
        "track_a_memory_growth": float(track_a["memory_growth"]),
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
