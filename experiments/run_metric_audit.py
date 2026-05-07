"""Audit Phase 2B metrics back to raw counts and denominators."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import SyntheticStreamGenerator
from experiments.phase2b_utils import build_scenarios, evaluate_main_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Phase 2B metrics into raw counts.")
    parser.add_argument("--config", default="configs/synth.yaml", help="Path to the config file.")
    parser.add_argument("--output-dir", default="results/phase2b_metric_audit", help="Directory for artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_payload = _load_config_payload(args.config)
    scenarios = build_scenarios(args.config)

    rows: list[dict[str, object]] = []
    for index, scenario in enumerate(scenarios):
        sequence = SyntheticStreamGenerator(scenario["config"], seed=args.seed + index * 11).generate_sequence(0)
        result = evaluate_main_pipeline(sequence, config_payload)
        audit = result["audit"]
        summary = result["summary"]
        rows.append(
            {
                "sequence_id": 0,
                "scenario_name": scenario["name"],
                **audit.to_row(),
                "purity": float(summary.purity),
                "churn": float(summary.churn),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metric_audit_v1.csv"
    json_path = output_dir / "metric_audit_v1.json"

    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"saved_csv={csv_path}")
    print(f"saved_json={json_path}")
    for row in rows:
        print(
            f"{row['scenario_name']}: "
            f"u_recall={float(row['u_recall']):.4f} "
            f"(frame_sum={float(row['u_recall_frame_sum']):.4f}/{int(row['u_recall_frame_denominator'])}), "
            f"pfr={float(row['pfr']):.4f} "
            f"({int(row['pfr_numerator'])}/{int(row['pfr_denominator'])}), "
            f"memory_growth={float(row['memory_growth']):.4f} "
            f"({int(row['memory_growth_numerator'])}/{int(row['memory_growth_denominator'])})"
        )


def _write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sequence_id",
        "scenario_name",
        "total_frames",
        "total_gt_objects",
        "total_matched_objects",
        "u_recall",
        "u_recall_frame_sum",
        "u_recall_frame_denominator",
        "object_recall",
        "object_recall_numerator",
        "object_recall_denominator",
        "fragmented_concepts",
        "extra_prototypes",
        "total_concepts",
        "pfr",
        "pfr_numerator",
        "pfr_denominator",
        "initial_proto_count",
        "final_proto_count",
        "net_proto_growth",
        "memory_growth",
        "memory_growth_numerator",
        "memory_growth_denominator",
        "net_birth_count",
        "net_merge_count",
        "net_decay_count",
        "total_id_switches",
        "idsw",
        "purity",
        "churn",
    ]
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
