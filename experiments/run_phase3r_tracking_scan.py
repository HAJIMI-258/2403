"""Run the Phase 3R local re-entry parameter scan with strict event-level metrics."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3r_utils import (
    evaluate_phase3_scenarios,
    extract_reentry_events,
    load_config_payload,
    summarize_reentry_events,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3R local tracking scan.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml", help="Path to config.")
    parser.add_argument("--output-dir", default="results/phase3r", help="Directory for outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tracking_scan_v1.csv"
    best_path = output_dir / "tracking_scan_best_v1.json"

    payload = load_config_payload(args.config)
    current_tracking = dict(payload["tracking"])
    current_memory = dict(payload["memory"])
    rows: list[dict[str, object]] = []

    tracking_grid = {
        "dormant_frames": [16, 24, 32],
        "tau_g": [8, 12, 16],
        "tau_react_short": [0.58, 0.62, 0.66],
        "tau_react_long": [0.70, 0.74, 0.78],
        "tau_proto_attach": [0.30, 0.35, 0.40],
        "tau_obj_attach": [0.40, 0.45, 0.50, 0.55],
    }
    memory_grid = {"decay_patience": [16, 24]}

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for param_name, values in tracking_grid.items():
            candidate_rows = _scan_parameter(
                executor=executor,
                config_path=args.config,
                scenario_name="track_c_long_horizon",
                param_name=param_name,
                values=values,
                current_tracking=current_tracking,
                current_memory=current_memory,
                seed=args.seed,
            )
            rows.extend(candidate_rows)
            best_row = max(candidate_rows, key=_candidate_key)
            current_tracking[param_name] = best_row["param_value"]
            if param_name in {"tau_proto_attach", "tau_obj_attach"}:
                current_memory[param_name] = best_row["param_value"]
            rows.append(
                {
                    "stage": "tracking_commit",
                    "param_name": param_name,
                    "param_value": best_row["param_value"],
                    "selected": 1,
                    **_extract_scan_metrics(best_row),
                }
            )
            write_csv(csv_path, rows)

        current_memory["protect_linked_prototypes"] = True
        memory_rows = _scan_parameter(
            executor=executor,
            config_path=args.config,
            scenario_name="track_c_long_horizon",
            param_name="decay_patience",
            values=memory_grid["decay_patience"],
            current_tracking=current_tracking,
            current_memory=current_memory,
            seed=args.seed,
        )
        for row in memory_rows:
            row["stage"] = "memory_scan"
        rows.extend(memory_rows)
        best_memory = max(memory_rows, key=_candidate_key)
        current_memory["decay_patience"] = best_memory["param_value"]
        rows.append(
            {
                "stage": "memory_commit",
                "param_name": "decay_patience",
                "param_value": best_memory["param_value"],
                "selected": 1,
                **_extract_scan_metrics(best_memory),
            }
        )
        write_csv(csv_path, rows)

    final_pair = _evaluate_pair_strict(
        config_path=args.config,
        tracking_override=current_tracking,
        memory_override=current_memory,
        seed=args.seed,
    )
    rows.append(
        {
            "stage": "final_best",
            "param_name": "final",
            "param_value": "best",
            "selected": 1,
            **final_pair,
        }
    )

    write_csv(csv_path, rows)
    best_path.write_text(
        json.dumps({"tracking": current_tracking, "memory": current_memory, "final_metrics": final_pair}, indent=2),
        encoding="utf-8",
    )

    print(f"saved_csv={csv_path}")
    print(f"saved_best={best_path}")
    print(
        "best_track_c: "
        f"same_track={float(final_pair['track_c_same_track_reentry_recovery']):.4f}, "
        f"same_proto={float(final_pair['track_c_same_prototype_reentry_recovery']):.4f}, "
        f"pfr={float(final_pair['track_c_pfr']):.4f}, "
        f"idsw={int(final_pair['track_c_track_idsw'])}"
    )


def _scan_parameter(
    *,
    executor: ProcessPoolExecutor,
    config_path: str,
    scenario_name: str,
    param_name: str,
    values: list[object],
    current_tracking: dict[str, object],
    current_memory: dict[str, object],
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    future_map = {}
    for value in values:
        tracking_override = dict(current_tracking)
        memory_override = dict(current_memory)
        tracking_override[param_name] = value
        if param_name in {"tau_proto_attach", "tau_obj_attach"}:
            memory_override[param_name] = value
        future = executor.submit(
            _evaluate_strict_scenario,
            config_path,
            scenario_name,
            tracking_override,
            memory_override,
            seed,
        )
        future_map[future] = value
    for future in as_completed(future_map):
        value = future_map[future]
        metrics = future.result()
        rows.append(
            {
                "stage": "tracking_scan",
                "param_name": param_name,
                "param_value": value,
                "selected": 0,
                **metrics,
            }
        )
    return sorted(rows, key=lambda row: str(row["param_value"]))


def _evaluate_strict_scenario(
    config_path: str,
    scenario_name: str,
    tracking_override: dict[str, object],
    memory_override: dict[str, object],
    seed: int,
) -> dict[str, object]:
    run = evaluate_phase3_scenarios(
        config_path,
        tracking_override=tracking_override,
        memory_override=memory_override,
        scenario_names=[scenario_name],
        collect_frames=True,
        frame_record_mode="lite",
        seed=seed,
    )[0]
    result = run["result"]
    event_rows, _ = extract_reentry_events(scenario_name, run["sequence"], result)
    reentry = summarize_reentry_events(event_rows)
    return {
        "scenario_name": scenario_name,
        "u_recall": float(result.summary.u_recall),
        "same_track_reentry_recovery": float(reentry["same_track_reentry_recovery"]),
        "same_prototype_reentry_recovery": float(reentry["same_prototype_reentry_recovery"]),
        "proposal_detect_rate": float(reentry["proposal_detect_rate"]),
        "concept_only_recovery_rate": float(reentry["concept_only_recovery_rate"]),
        "reactivation_attempt_rate": float(reentry["reactivation_attempt_rate"]),
        "reentry_events": int(reentry["num_events"]),
        "pfr": float(result.summary.pfr),
        "track_idsw": int(result.primary_monitoring["track_idsw"]),
        "memory_growth": float(result.summary.memory_growth),
    }


def _evaluate_pair_strict(
    *,
    config_path: str,
    tracking_override: dict[str, object],
    memory_override: dict[str, object],
    seed: int,
) -> dict[str, object]:
    runs = evaluate_phase3_scenarios(
        config_path,
        tracking_override=tracking_override,
        memory_override=memory_override,
        scenario_names=["track_a_bridge", "track_c_long_horizon"],
        collect_frames=True,
        frame_record_mode="lite",
        seed=seed,
    )
    row: dict[str, object] = {}
    for run in runs:
        scenario = run["scenario_name"]
        prefix = "track_a" if scenario == "track_a_bridge" else "track_c"
        result = run["result"]
        event_rows, _ = extract_reentry_events(scenario, run["sequence"], result)
        reentry = summarize_reentry_events(event_rows)
        row[f"{prefix}_u_recall"] = float(result.summary.u_recall)
        row[f"{prefix}_same_track_reentry_recovery"] = float(reentry["same_track_reentry_recovery"])
        row[f"{prefix}_same_prototype_reentry_recovery"] = float(reentry["same_prototype_reentry_recovery"])
        row[f"{prefix}_proposal_detect_rate"] = float(reentry["proposal_detect_rate"])
        row[f"{prefix}_concept_only_recovery_rate"] = float(reentry["concept_only_recovery_rate"])
        row[f"{prefix}_reentry_events"] = int(reentry["num_events"])
        row[f"{prefix}_pfr"] = float(result.summary.pfr)
        row[f"{prefix}_track_idsw"] = int(result.primary_monitoring["track_idsw"])
        row[f"{prefix}_memory_growth"] = float(result.summary.memory_growth)
    return row


def _extract_scan_metrics(row: dict[str, object]) -> dict[str, object]:
    return {
        "scenario_name": row.get("scenario_name", "track_c_long_horizon"),
        "u_recall": row.get("u_recall", 0.0),
        "same_track_reentry_recovery": row.get("same_track_reentry_recovery", 0.0),
        "same_prototype_reentry_recovery": row.get("same_prototype_reentry_recovery", 0.0),
        "proposal_detect_rate": row.get("proposal_detect_rate", 0.0),
        "concept_only_recovery_rate": row.get("concept_only_recovery_rate", 0.0),
        "reactivation_attempt_rate": row.get("reactivation_attempt_rate", 0.0),
        "reentry_events": row.get("reentry_events", 0),
        "pfr": row.get("pfr", 0.0),
        "track_idsw": row.get("track_idsw", 0),
        "memory_growth": row.get("memory_growth", 0.0),
    }


def _candidate_key(row: dict[str, object]):
    return (
        float(row["same_track_reentry_recovery"]),
        float(row["same_prototype_reentry_recovery"]),
        -float(row["pfr"]),
        -int(row["track_idsw"]),
        -float(row["memory_growth"]),
        float(row["u_recall"]),
    )


if __name__ == "__main__":
    main()
