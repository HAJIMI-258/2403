"""Run Phase 3P ablations on Track C and final guardrail check on Track A."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3p_utils import (
    TRACK_A_NAME,
    TRACK_C_NAME,
    build_phase3p_action_breakdown,
    build_phase3p_event_audit_rows,
    default_phase3p_memory_override,
    default_phase3p_tracking_override,
    evaluate_phase3p_bundle,
)
from experiments.phase3r_utils import extract_reentry_events, write_csv


ABLATIONS = [
    ("keep_head_only", {"enable_phase3p_keep_head_default": True}),
    ("grouped_gating_only", {"enable_phase3p_grouped_gating": True}),
    ("birth_suppression_only", {"enable_phase3p_birth_suppression": True}),
    (
        "keep_head_plus_grouped",
        {
            "enable_phase3p_keep_head_default": True,
            "enable_phase3p_grouped_gating": True,
        },
    ),
    (
        "keep_head_grouped_birth_suppression",
        {
            "enable_phase3p_keep_head_default": True,
            "enable_phase3p_grouped_gating": True,
            "enable_phase3p_birth_suppression": True,
        },
    ),
    ("full_phase3p", {"enable_phase3p_full_stabilization": True}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3P ablation suite.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--output-dir", default="results/phase3p")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_baseline_row(path: Path, *, scenario_name: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("method") == "phase3l_current" and row.get("scenario_name") == scenario_name:
                return row
    raise FileNotFoundError(f"missing phase3l baseline row for {scenario_name}: {path}")


def _run_single_ablation(
    *,
    config_path: str,
    seed: int,
    scenario_names: list[str],
    memory_patch: dict[str, object],
) -> dict[str, object]:
    tracking_override = default_phase3p_tracking_override()
    memory_override = default_phase3p_memory_override()
    memory_override.update(memory_patch)
    bundle = evaluate_phase3p_bundle(
        config_path,
        tracking_override=tracking_override,
        memory_override=memory_override,
        seed=seed,
        scenario_names=scenario_names,
    )
    event_rows = []
    for run in bundle["runs"]:
        events, _ = extract_reentry_events(run["scenario_name"], run["sequence"], run["result"])
        event_rows.extend(events)
    audit_rows = build_phase3p_event_audit_rows(event_rows)
    breakdown_rows, summary = build_phase3p_action_breakdown(audit_rows)
    row_lookup = {str(row["scenario_name"]): row for row in bundle["rows"]}
    track_c = row_lookup.get(TRACK_C_NAME, {})
    track_a = row_lookup.get(TRACK_A_NAME, {})
    return {
        "rows": bundle["rows"],
        "track_c": track_c,
        "track_a": track_a,
        "audit_summary": summary,
        "breakdown_rows": breakdown_rows,
        "memory_override": memory_override,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase3l_summary_path = Path("results/phase3l/phase3l_final_summary_v1.csv")
    baseline_track_c = _load_baseline_row(phase3l_summary_path, scenario_name=TRACK_C_NAME)
    baseline_track_a = _load_baseline_row(phase3l_summary_path, scenario_name=TRACK_A_NAME)

    ablation_rows: list[dict[str, object]] = []
    best_name = ""
    best_score = None
    best_payload: dict[str, object] | None = None

    for ablation_name, patch in ABLATIONS:
        payload = _run_single_ablation(
            config_path=args.config,
            seed=args.seed,
            scenario_names=[TRACK_C_NAME],
            memory_patch=patch,
        )
        track_c = payload["track_c"]
        summary = payload["audit_summary"]
        row = {
            "ablation": ablation_name,
            "same_prototype": float(track_c["same_prototype_reentry_recovery"]),
            "pfr": float(track_c["pfr"]),
            "idsw": int(track_c["track_idsw"]),
            "same_track_after_concept": float(track_c["same_track_after_concept_recovery"]),
            "same_lineage_prototype": float(track_c["same_lineage_prototype_reentry_recovery"]),
            "concept_recovered_but_lineage_mismatch_rate": float(track_c["concept_recovered_but_lineage_mismatch_rate"]),
            "continuation_bank_access_rate_given_concept_recovery": float(track_c["continuation_bank_access_rate_given_concept_recovery"]),
            "head_keep_rate": float(summary["head_keep_rate_given_matched_lineage"]),
            "head_replacement_rate": float(summary["head_replacement_rate_given_matched_lineage"]),
            "new_sibling_birth_rate": float(summary["new_sibling_birth_rate_given_concept_recovery"]),
            "archived_sibling_reactivation_rate": float(summary["archived_sibling_reactivation_rate"]),
        }
        ablation_rows.append(row)
        sort_key = (
            float(row["same_prototype"]),
            -float(row["pfr"]),
            -float(row["idsw"]),
            float(row["same_track_after_concept"]),
        )
        if best_score is None or sort_key > best_score:
            best_score = sort_key
            best_name = ablation_name
            best_payload = payload

    write_csv(output_dir / "phase3p_ablation_summary.csv", ablation_rows)
    (output_dir / "phase3p_best_ablation_v1.json").write_text(
        json.dumps({"best_ablation": best_name}, indent=2),
        encoding="utf-8",
    )

    if best_payload is None:
        raise RuntimeError("no Phase 3P ablation results")

    best_patch = next(patch for name, patch in ABLATIONS if name == best_name)
    final_payload = _run_single_ablation(
        config_path=args.config,
        seed=args.seed,
        scenario_names=[TRACK_A_NAME, TRACK_C_NAME],
        memory_patch=best_patch,
    )
    final_lookup = {str(row["scenario_name"]): row for row in final_payload["rows"]}
    final_rows = [
        {"method": "phase3l_current", **baseline_track_a},
        {"method": "phase3p_current", **final_lookup[TRACK_A_NAME]},
        {"method": "phase3l_current", **baseline_track_c},
        {"method": "phase3p_current", **final_lookup[TRACK_C_NAME]},
    ]
    write_csv(output_dir / "phase3p_final_summary_v1.csv", final_rows)
    print(f"saved_ablation_summary={output_dir / 'phase3p_ablation_summary.csv'}")
    print(f"saved_best={output_dir / 'phase3p_best_ablation_v1.json'}")
    print(f"saved_final_summary={output_dir / 'phase3p_final_summary_v1.csv'}")
    print(f"best_ablation={best_name}")


if __name__ == "__main__":
    main()
