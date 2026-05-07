"""Run Phase 2B local coordinate scans for field, tracking, and memory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator
from experiments.phase2b_utils import build_scenarios, evaluate_main_pipeline, get_field_config


FIELD_GRID = {
    "tau_obj": [0.43, 0.47, 0.51],
    "wr": [0.15, 0.20, 0.25],
    "hab_rho": [0.90, 0.94, 0.97],
    "hab_lambda": [0.75, 0.85, 0.95],
}

TRACKING_GRID = {
    "max_match_cost": [0.62, 0.72, 0.82],
    "keepalive_frames": [8, 12, 16],
    "beta_iou": [0.35, 0.45, 0.55],
    "beta_feat": [0.20, 0.30, 0.40],
}

MEMORY_GRID = {
    "tau_birth": [0.35, 0.40, 0.45],
    "tau_merge": [0.18, 0.22, 0.26],
    "tau_sim": [0.20, 0.25, 0.30],
    "lr_proto": [0.35, 0.45, 0.55],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2B local coordinate scans.")
    parser.add_argument("--config", default="configs/synth.yaml", help="Path to the config file.")
    parser.add_argument("--output-dir", default="results/phase2b_param_scan", help="Directory for artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_payload = _load_config_payload(args.config)
    scenarios = build_scenarios(args.config)
    sequences = {
        scenario["name"]: SyntheticStreamGenerator(scenario["config"], seed=args.seed + index * 11).generate_sequence(0)
        for index, scenario in enumerate(scenarios)
    }

    field_config = get_field_config(config_payload)
    tracking_config = dict(config_payload["tracking"])
    memory_config = dict(config_payload["memory"])

    baseline = _evaluate_candidate_bundle(sequences, config_payload, field_config, tracking_config, memory_config)

    field_rows, field_config, field_metrics = _coordinate_scan(
        block_name="field",
        grid=FIELD_GRID,
        sequences=sequences,
        config_payload=config_payload,
        field_config=field_config,
        tracking_config=tracking_config,
        memory_config=memory_config,
        baseline_metrics=baseline,
    )
    tracking_rows, tracking_config, tracking_metrics = _coordinate_scan(
        block_name="tracking",
        grid=TRACKING_GRID,
        sequences=sequences,
        config_payload=config_payload,
        field_config=field_config,
        tracking_config=tracking_config,
        memory_config=memory_config,
        baseline_metrics=field_metrics,
    )
    memory_rows, memory_config, memory_metrics = _coordinate_scan(
        block_name="memory",
        grid=MEMORY_GRID,
        sequences=sequences,
        config_payload=config_payload,
        field_config=field_config,
        tracking_config=tracking_config,
        memory_config=memory_config,
        baseline_metrics=tracking_metrics,
    )

    best_config_payload = deepcopy(config_payload)
    best_config_payload["field"] = field_config
    best_config_payload["tracking"] = tracking_config
    best_config_payload["memory"] = memory_config

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    field_csv = output_dir / "field_scan.csv"
    tracking_csv = output_dir / "tracking_scan.csv"
    memory_csv = output_dir / "memory_scan.csv"
    best_yaml = output_dir / "best_config_v1.yaml"
    summary_json = output_dir / "best_config_v1.json"

    _write_csv(field_csv, field_rows)
    _write_csv(tracking_csv, tracking_rows)
    _write_csv(memory_csv, memory_rows)
    best_yaml.write_text(_dump_yaml(best_config_payload), encoding="utf-8")
    summary_json.write_text(
        json.dumps(
            {
                "field": field_config,
                "tracking": tracking_config,
                "memory": memory_config,
                "final_metrics": memory_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved_field_scan={field_csv}")
    print(f"saved_tracking_scan={tracking_csv}")
    print(f"saved_memory_scan={memory_csv}")
    print(f"saved_best_config={best_yaml}")
    print(
        "best_hard="
        f"u_recall={memory_metrics['hard_drift_occlusion']['u_recall']:.4f},"
        f"idsw={int(memory_metrics['hard_drift_occlusion']['idsw'])},"
        f"pfr={memory_metrics['hard_drift_occlusion']['pfr']:.4f},"
        f"memory_growth={memory_metrics['hard_drift_occlusion']['memory_growth']:.4f}"
    )


def _coordinate_scan(
    *,
    block_name: str,
    grid: dict[str, list[float | int]],
    sequences,
    config_payload: dict,
    field_config: dict,
    tracking_config: dict,
    memory_config: dict,
    baseline_metrics: dict[str, dict[str, float | int]],
) -> tuple[list[dict[str, object]], dict, dict[str, dict[str, float | int]]]:
    rows: list[dict[str, object]] = []
    current_field = dict(field_config)
    current_tracking = dict(tracking_config)
    current_memory = dict(memory_config)
    current_metrics = deepcopy(baseline_metrics)

    for step_index, (param_name, candidates) in enumerate(grid.items(), start=1):
        best_row: dict[str, object] | None = None
        best_metrics: dict[str, dict[str, float | int]] | None = None
        best_configs: tuple[dict, dict, dict] | None = None

        for candidate in candidates:
            trial_field = dict(current_field)
            trial_tracking = dict(current_tracking)
            trial_memory = dict(current_memory)
            if block_name == "field":
                trial_field[param_name] = candidate
            elif block_name == "tracking":
                trial_tracking[param_name] = candidate
                if param_name in {"beta_iou", "beta_feat"}:
                    trial_tracking["beta_center"] = 1.0 - trial_tracking["beta_iou"] - trial_tracking["beta_feat"]
                if trial_tracking.get("beta_center", 0.0) < 0.0:
                    continue
            else:
                trial_memory[param_name] = candidate

            trial_metrics = _evaluate_candidate_bundle(
                sequences,
                config_payload,
                trial_field,
                trial_tracking,
                trial_memory,
            )
            row = _scan_row(
                block_name=block_name,
                step_index=step_index,
                param_name=param_name,
                candidate=candidate,
                field_config=trial_field,
                tracking_config=trial_tracking,
                memory_config=trial_memory,
                metrics_by_scenario=trial_metrics,
                baseline_metrics=baseline_metrics,
            )
            rows.append(row)

            if best_row is None or _score_tuple(row) > _score_tuple(best_row):
                best_row = row
                best_metrics = trial_metrics
                best_configs = (trial_field, trial_tracking, trial_memory)

        if best_row is None or best_metrics is None or best_configs is None:
            raise RuntimeError(f"No valid candidate found for {block_name}.{param_name}")

        best_row["selected"] = 1
        current_field, current_tracking, current_memory = best_configs
        current_metrics = best_metrics

    return rows, {"field": current_field, "tracking": current_tracking, "memory": current_memory}[block_name], current_metrics


def _evaluate_candidate_bundle(
    sequences,
    config_payload: dict,
    field_config: dict,
    tracking_config: dict,
    memory_config: dict,
) -> dict[str, dict[str, float | int]]:
    metrics_by_scenario: dict[str, dict[str, float | int]] = {}
    for scenario_name, sequence in sequences.items():
        result = evaluate_main_pipeline(
            sequence,
            config_payload,
            field_override=field_config,
            tracking_override=tracking_config,
            memory_override=memory_config,
        )
        summary = result["summary"]
        audit = result["audit"]
        metrics_by_scenario[scenario_name] = {
            "u_recall": float(summary.u_recall),
            "purity": float(summary.purity),
            "pfr": float(summary.pfr),
            "idsw": int(summary.idsw),
            "memory_growth": float(summary.memory_growth),
            "final_prototypes": int(audit.final_proto_count),
        }
    return metrics_by_scenario


def _scan_row(
    *,
    block_name: str,
    step_index: int,
    param_name: str,
    candidate: float | int,
    field_config: dict,
    tracking_config: dict,
    memory_config: dict,
    metrics_by_scenario: dict[str, dict[str, float | int]],
    baseline_metrics: dict[str, dict[str, float | int]],
) -> dict[str, object]:
    easy = metrics_by_scenario["easy_single_object"]
    multi = metrics_by_scenario["multi_object_reentry"]
    hard = metrics_by_scenario["hard_drift_occlusion"]
    baseline_easy = baseline_metrics["easy_single_object"]
    baseline_multi = baseline_metrics["multi_object_reentry"]

    guard_ok = int(
        easy["u_recall"] >= baseline_easy["u_recall"] - 0.02
        and multi["u_recall"] >= baseline_multi["u_recall"] - 0.05
        and multi["idsw"] <= baseline_multi["idsw"] + 2
        and multi["final_prototypes"] <= baseline_multi["final_prototypes"] + 1
        and multi["purity"] >= baseline_multi["purity"] - 0.10
    )

    return {
        "block": block_name,
        "step_index": step_index,
        "param_name": param_name,
        "candidate_value": candidate,
        "selected": 0,
        "guard_ok": guard_ok,
        "easy_u_recall": easy["u_recall"],
        "easy_purity": easy["purity"],
        "easy_idsw": easy["idsw"],
        "easy_pfr": easy["pfr"],
        "easy_memory_growth": easy["memory_growth"],
        "easy_final_prototypes": easy["final_prototypes"],
        "multi_u_recall": multi["u_recall"],
        "multi_purity": multi["purity"],
        "multi_idsw": multi["idsw"],
        "multi_pfr": multi["pfr"],
        "multi_memory_growth": multi["memory_growth"],
        "multi_final_prototypes": multi["final_prototypes"],
        "hard_u_recall": hard["u_recall"],
        "hard_purity": hard["purity"],
        "hard_idsw": hard["idsw"],
        "hard_pfr": hard["pfr"],
        "hard_memory_growth": hard["memory_growth"],
        "hard_final_prototypes": hard["final_prototypes"],
        "field_tau_obj": field_config["tau_obj"],
        "field_wr": field_config["wr"],
        "field_hab_rho": field_config["hab_rho"],
        "field_hab_lambda": field_config["hab_lambda"],
        "tracking_max_match_cost": tracking_config["max_match_cost"],
        "tracking_keepalive_frames": tracking_config["keepalive_frames"],
        "tracking_beta_iou": tracking_config["beta_iou"],
        "tracking_beta_center": tracking_config["beta_center"],
        "tracking_beta_feat": tracking_config["beta_feat"],
        "memory_tau_birth": memory_config["tau_birth"],
        "memory_tau_merge": memory_config["tau_merge"],
        "memory_tau_sim": memory_config["tau_sim"],
        "memory_lr_proto": memory_config["lr_proto"],
    }


def _score_tuple(row: dict[str, object]) -> tuple:
    return (
        int(row["guard_ok"]),
        float(row["hard_u_recall"]),
        -int(row["hard_idsw"]),
        -float(row["hard_pfr"]),
        -float(row["hard_memory_growth"]),
        float(row["multi_u_recall"]),
        float(row["multi_purity"]),
        -int(row["multi_idsw"]),
        -int(row["multi_final_prototypes"]),
    )


def _write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _dump_yaml(payload: dict) -> str:
    import yaml

    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def _load_config_payload(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


if __name__ == "__main__":
    main()
