"""Run the Phase 3L lineage / continuation binding scan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3l_utils import (
    TRACK_A_NAME,
    TRACK_C_NAME,
    evaluate_phase3l_bundle,
    load_phase3x_before_lookup,
    pick_best_scan_row,
)
from experiments.phase3r_utils import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 3L lineage / binding scan.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--output-dir", default="results/phase3l")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-version", default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    stage1 = [
        {"bind_continuation_to": "prototype", "allow_alias_lineage": True, "continuation_topk_per_lineage": 4},
        {"bind_continuation_to": "lineage", "allow_alias_lineage": True, "continuation_topk_per_lineage": 4},
    ]
    for variant in stage1:
        rows.append(_evaluate_variant("binding_mode", args.config, args.seed, variant))

    stage1_rows = [row for row in rows if row["stage"] == "binding_mode"]
    stage1_best = pick_best_scan_row(stage1_rows, load_phase3x_before_lookup())
    best_binding = str(stage1_best["bind_continuation_to"])

    for allow_alias in [False, True]:
        for topk in [2, 4, 6]:
            rows.append(
                _evaluate_variant(
                    "lineage_binding",
                    args.config,
                    args.seed,
                    {
                        "bind_continuation_to": best_binding,
                        "allow_alias_lineage": allow_alias,
                        "continuation_topk_per_lineage": topk,
                    },
                )
            )

    best_row = pick_best_scan_row(rows, load_phase3x_before_lookup())
    for row in rows:
        row["is_best"] = int(
            str(row["stage"]) == str(best_row["stage"])
            and str(row["bind_continuation_to"]) == str(best_row["bind_continuation_to"])
            and int(row["continuation_topk_per_lineage"]) == int(best_row["continuation_topk_per_lineage"])
            and str(row["allow_alias_lineage"]).lower() == str(best_row["allow_alias_lineage"]).lower()
        )

    scan_path = output_dir / f"tracking_scan_v5.csv"
    best_path = output_dir / f"tracking_scan_stage1_best_{args.artifact_version}.json"
    write_csv(scan_path, rows)
    best_path.write_text(json.dumps(best_row, indent=2), encoding="utf-8")

    print(f"saved_scan={scan_path}")
    print(f"saved_best={best_path}")
    print(
        "best_track_c="
        f"mismatch={float(best_row['track_c_concept_recovered_but_lineage_mismatch_rate']):.4f}, "
        f"access={float(best_row['track_c_continuation_bank_access_rate_given_concept_recovery']):.4f}, "
        f"same_lineage={float(best_row['track_c_same_lineage_prototype_reentry_recovery']):.4f}, "
        f"track_after={float(best_row['track_c_same_track_after_concept_recovery']):.4f}, "
        f"same_proto={float(best_row['track_c_same_prototype_reentry_recovery']):.4f}, "
        f"pfr={float(best_row['track_c_pfr']):.4f}"
    )


def _evaluate_variant(
    stage: str,
    config_path: str,
    seed: int,
    memory_variant: dict[str, object],
) -> dict[str, object]:
    bundle = evaluate_phase3l_bundle(
        config_path,
        seed=seed,
        memory_override=memory_variant,
        frame_record_mode="lite",
    )
    row_lookup = {str(row["scenario_name"]): row for row in bundle["rows"]}
    track_a = row_lookup[TRACK_A_NAME]
    track_c = row_lookup[TRACK_C_NAME]
    return {
        "stage": stage,
        "bind_continuation_to": str(memory_variant["bind_continuation_to"]),
        "allow_alias_lineage": bool(memory_variant["allow_alias_lineage"]),
        "continuation_topk_per_lineage": int(memory_variant["continuation_topk_per_lineage"]),
        "track_c_concept_recovered_but_lineage_mismatch_rate": float(track_c["concept_recovered_but_lineage_mismatch_rate"]),
        "track_c_continuation_bank_access_rate_given_concept_recovery": float(track_c["continuation_bank_access_rate_given_concept_recovery"]),
        "track_c_same_lineage_prototype_reentry_recovery": float(track_c["same_lineage_prototype_reentry_recovery"]),
        "track_c_same_track_after_concept_recovery": float(track_c["same_track_after_concept_recovery"]),
        "track_c_same_prototype_reentry_recovery": float(track_c["same_prototype_reentry_recovery"]),
        "track_c_pfr": float(track_c["pfr"]),
        "track_c_track_idsw": int(track_c["track_idsw"]),
        "track_c_memory_growth": float(track_c["memory_growth"]),
        "track_a_u_recall": float(track_a["u_recall"]),
        "track_a_same_prototype_reentry_recovery": float(track_a["same_prototype_reentry_recovery"]),
        "track_a_memory_growth": float(track_a["memory_growth"]),
    }


if __name__ == "__main__":
    main()
