"""Resume the remaining Phase 3R tracking scan stages and append to the main csv."""

from __future__ import annotations

import csv
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
    summarize_reentry_events,
    write_csv,
)

CONFIG = "configs/bridge_synth_generic_v1.yaml"
CSV_PATH = Path("results/phase3r/tracking_scan_v1.csv")
BEST_PATH = Path("results/phase3r/tracking_scan_best_v1.json")
SEED = 42


def _evaluate_strict(
    scenario_name: str,
    tracking_override: dict[str, object],
    memory_override: dict[str, object],
) -> dict[str, object]:
    run = evaluate_phase3_scenarios(
        CONFIG,
        tracking_override=tracking_override,
        memory_override=memory_override,
        scenario_names=[scenario_name],
        collect_frames=True,
        frame_record_mode="lite",
        seed=SEED,
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


def _candidate_key(row: dict[str, object]) -> tuple[float, float, float, int, float, float]:
    return (
        float(row["same_track_reentry_recovery"]),
        float(row["same_prototype_reentry_recovery"]),
        -float(row["pfr"]),
        -int(row["track_idsw"]),
        -float(row["memory_growth"]),
        float(row["u_recall"]),
    )


def _scan_param(
    executor: ProcessPoolExecutor,
    *,
    current_tracking: dict[str, object],
    current_memory: dict[str, object],
    param_name: str,
    values: list[object],
    stage: str,
) -> list[dict[str, object]]:
    future_map = {}
    for value in values:
        tracking_override = dict(current_tracking)
        memory_override = dict(current_memory)
        tracking_override[param_name] = value
        if param_name in {"tau_proto_attach", "tau_obj_attach"}:
            memory_override[param_name] = value
        future = executor.submit(_evaluate_strict, "track_c_long_horizon", tracking_override, memory_override)
        future_map[future] = value

    rows: list[dict[str, object]] = []
    for future in as_completed(future_map):
        value = future_map[future]
        metrics = future.result()
        rows.append(
            {
                "stage": stage,
                "param_name": param_name,
                "param_value": value,
                "selected": 0,
                **metrics,
            }
        )
    rows.sort(key=lambda row: str(row["param_value"]))
    return rows


def _commit_row(stage: str, best_row: dict[str, object]) -> dict[str, object]:
    return {
        "stage": stage,
        "param_name": best_row["param_name"],
        "param_value": best_row["param_value"],
        "selected": 1,
        "scenario_name": best_row["scenario_name"],
        "u_recall": best_row["u_recall"],
        "same_track_reentry_recovery": best_row["same_track_reentry_recovery"],
        "same_prototype_reentry_recovery": best_row["same_prototype_reentry_recovery"],
        "proposal_detect_rate": best_row["proposal_detect_rate"],
        "concept_only_recovery_rate": best_row["concept_only_recovery_rate"],
        "reactivation_attempt_rate": best_row["reactivation_attempt_rate"],
        "reentry_events": best_row["reentry_events"],
        "pfr": best_row["pfr"],
        "track_idsw": best_row["track_idsw"],
        "memory_growth": best_row["memory_growth"],
    }


def main() -> None:
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    current_tracking = {
        "beta_iou": 0.35,
        "beta_center": 0.45,
        "beta_feat": 0.2,
        "max_match_cost": 0.62,
        "min_match_similarity": 0.12,
        "missed_match_similarity_boost": 0.05,
        "use_gap_aware_matching": True,
        "keepalive_frames": 8,
        "use_linear_prediction": True,
        "use_dormant_reactivation": True,
        "dormant_frames": 32,
        "tau_g": 12,
        "tau_react_short": 0.62,
        "tau_react_long": 0.78,
        "tau_proto_attach": 0.40,
        "tau_obj_attach": 0.40,
        "reactivation_cost": 0.84,
        "reactivation_proto_sim": 0.32,
        "signature_momentum": 0.70,
        "velocity_momentum": 0.65,
    }
    current_memory = {
        "tau_birth": 0.35,
        "tau_merge": 0.18,
        "tau_sim": 0.20,
        "lr_proto": 0.35,
        "decay_rate": 0.03,
        "decay_patience": 24,
        "memory_budget": 32,
        "decay_floor": 0.10,
        "tau_proto_attach": 0.40,
        "tau_obj_attach": 0.40,
        "use_concept_only_recovery": True,
        "protect_linked_prototypes": True,
    }

    def append(new_rows: list[dict[str, object]]) -> None:
        rows.extend(new_rows)
        write_csv(CSV_PATH, rows)

    with ProcessPoolExecutor(max_workers=4) as executor:
        for param_name, values in (
            ("tau_proto_attach", [0.30, 0.35, 0.40]),
            ("tau_obj_attach", [0.40, 0.45, 0.50, 0.55]),
        ):
            scanned = _scan_param(
                executor,
                current_tracking=current_tracking,
                current_memory=current_memory,
                param_name=param_name,
                values=values,
                stage="tracking_scan",
            )
            append(scanned)
            best = max(scanned, key=_candidate_key)
            current_tracking[param_name] = best["param_value"]
            current_memory[param_name] = best["param_value"]
            append([_commit_row("tracking_commit", best)])

        scanned = _scan_param(
            executor,
            current_tracking=current_tracking,
            current_memory=current_memory,
            param_name="decay_patience",
            values=[16, 24],
            stage="memory_scan",
        )
        append(scanned)
        best = max(scanned, key=_candidate_key)
        current_memory["decay_patience"] = best["param_value"]
        append([_commit_row("memory_commit", best)])

    final_entry: dict[str, object] = {"stage": "final_best", "param_name": "final", "param_value": "best", "selected": 1}
    for scenario_name in ("track_a_bridge", "track_c_long_horizon"):
        metrics = _evaluate_strict(scenario_name, current_tracking, current_memory)
        prefix = "track_a" if scenario_name == "track_a_bridge" else "track_c"
        for key, value in metrics.items():
            if key == "scenario_name":
                continue
            final_entry[f"{prefix}_{key}"] = value
    append([final_entry])

    BEST_PATH.write_text(
        json.dumps({"tracking": current_tracking, "memory": current_memory, "final_metrics": final_entry}, indent=2),
        encoding="utf-8",
    )
    print("tail scan complete")
    print(json.dumps({"tracking": current_tracking, "memory": current_memory}, indent=2))
    print(json.dumps(final_entry, indent=2))


if __name__ == "__main__":
    main()
