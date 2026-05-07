"""Export Phase 2B final scenario and baseline summaries on the three main scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator, load_synth_dataset_config
from experiments.run_baseline_comparison import (
    _evaluate_edge_cluster_baseline,
    _evaluate_frame_diff_baseline,
    _evaluate_main_pipeline,
)
from experiments.scenario_presets import build_phase1_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Phase 2B final summaries on the three main scenarios.")
    parser.add_argument("--config", default="configs/synth.yaml", help="Path to the config file.")
    parser.add_argument("--output-dir", default="results/phase2b_final", help="Directory for output artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_config_payload(args.config)
    base_config = load_synth_dataset_config(args.config)
    scenarios = build_phase1_scenarios(base_config)
    sequences = {
        scenario["name"]: SyntheticStreamGenerator(scenario["config"], seed=args.seed + index * 11).generate_sequence(0)
        for index, scenario in enumerate(scenarios)
    }

    method_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    methods = {
        "minimal_nops_owr": lambda sequence, seq_id: _evaluate_main_pipeline(sequence, payload, seq_id),
        "baseline_frame_diff_cc": _evaluate_frame_diff_baseline,
        "baseline_edge_cluster": _evaluate_edge_cluster_baseline,
    }

    for sequence_id, (scenario_name, sequence) in enumerate(sequences.items()):
        for method_name, evaluate_fn in methods.items():
            metrics = evaluate_fn(sequence, sequence_id)
            row = {
                "scenario": scenario_name,
                "method": method_name,
                "u_recall": float(metrics["u_recall"]),
                "purity": float(metrics["purity"]),
                "pfr": float(metrics["pfr"]),
                "idsw": int(metrics["idsw"]),
                "churn": float(metrics["churn"]),
                "memory_growth": float(metrics["memory_growth"]),
                "final_prototype_count": int(metrics["final_memory_size"]),
            }
            method_rows.append(row)
            if method_name == "minimal_nops_owr":
                scenario_rows.append({key: value for key, value in row.items() if key != "method"})

    aggregated_rows = _aggregate_by_method(method_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_csv = output_dir / "scenario_summary_v2.csv"
    scenario_json = output_dir / "scenario_summary_v2.json"
    baseline_csv = output_dir / "baseline_comparison_v2.csv"
    baseline_json = output_dir / "baseline_comparison_v2.json"

    _write_csv(
        scenario_csv,
        scenario_rows,
        [
            "scenario",
            "u_recall",
            "purity",
            "pfr",
            "idsw",
            "churn",
            "memory_growth",
            "final_prototype_count",
        ],
    )
    scenario_json.write_text(json.dumps(scenario_rows, indent=2), encoding="utf-8")

    _write_csv(
        baseline_csv,
        method_rows + aggregated_rows,
        [
            "scenario",
            "method",
            "u_recall",
            "purity",
            "pfr",
            "idsw",
            "churn",
            "memory_growth",
            "final_prototype_count",
        ],
    )
    baseline_json.write_text(
        json.dumps({"per_scenario": method_rows, "aggregated": aggregated_rows}, indent=2),
        encoding="utf-8",
    )

    print(f"saved_scenario_csv={scenario_csv}")
    print(f"saved_scenario_json={scenario_json}")
    print(f"saved_baseline_csv={baseline_csv}")
    print(f"saved_baseline_json={baseline_json}")
    for row in scenario_rows:
        print(
            f"{row['scenario']}: "
            f"u_recall={float(row['u_recall']):.4f}, "
            f"purity={float(row['purity']):.4f}, "
            f"pfr={float(row['pfr']):.4f}, "
            f"idsw={int(row['idsw'])}, "
            f"memory_growth={float(row['memory_growth']):.4f}, "
            f"final_prototypes={int(row['final_prototype_count'])}"
        )
    for row in aggregated_rows:
        print(
            f"{row['method']}__mean: "
            f"u_recall={float(row['u_recall']):.4f}, "
            f"pfr={float(row['pfr']):.4f}, "
            f"idsw={float(row['idsw']):.2f}, "
            f"memory_growth={float(row['memory_growth']):.4f}"
        )


def _aggregate_by_method(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    aggregated_rows: list[dict[str, object]] = []
    for method_name, method_rows in grouped.items():
        aggregated_rows.append(
            {
                "scenario": "mean",
                "method": method_name,
                "u_recall": sum(float(row["u_recall"]) for row in method_rows) / len(method_rows),
                "purity": sum(float(row["purity"]) for row in method_rows) / len(method_rows),
                "pfr": sum(float(row["pfr"]) for row in method_rows) / len(method_rows),
                "idsw": sum(float(row["idsw"]) for row in method_rows) / len(method_rows),
                "churn": sum(float(row["churn"]) for row in method_rows) / len(method_rows),
                "memory_growth": sum(float(row["memory_growth"]) for row in method_rows) / len(method_rows),
                "final_prototype_count": sum(float(row["final_prototype_count"]) for row in method_rows) / len(method_rows),
            }
        )
    return aggregated_rows


def _write_csv(csv_path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_config_payload(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


if __name__ == "__main__":
    main()
