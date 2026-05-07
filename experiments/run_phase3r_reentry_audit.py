"""Run Phase 3R / Phase 3R.2 / Phase 3R.3 event-level re-entry audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3r2_utils import (
    default_phase3r2_memory_override,
    default_phase3r2_tracking_override,
    load_csv_rows,
    load_phase3r_before_lookup,
    pick_best_scan_row,
)
from experiments.phase3r3_utils import (
    default_phase3r3_memory_override,
    default_phase3r3_tracking_override,
    load_phase3r2_before_lookup,
    pick_best_scan_row as pick_phase3r3_scan_row,
)
from experiments.phase3s_utils import (
    default_phase3s_memory_override,
    default_phase3s_tracking_override,
    load_phase3r3_before_lookup,
    pick_best_scan_row as pick_phase3s_scan_row,
)
from experiments.phase3x_utils import (
    default_phase3x_memory_override,
    default_phase3x_tracking_override,
)
from experiments.phase3l_utils import (
    default_phase3l_memory_override,
    default_phase3l_tracking_override,
    load_phase3x_before_lookup,
    pick_best_scan_row as pick_phase3l_scan_row,
)
from experiments.phase3r_utils import (
    REENTRY_WINDOW,
    evaluate_phase3_scenarios,
    extract_reentry_events,
    summarize_reentry_events,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3R re-entry audit.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml", help="Path to config.")
    parser.add_argument("--output-dir", default="results/phase3r", help="Directory for outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--artifact-version", default="v1", help="Artifact suffix such as v1 or v2.")
    parser.add_argument(
        "--phase-label",
        default="phase3r",
        choices=["phase3r", "phase3r2", "phase3r3", "phase3s", "phase3x", "phase3l"],
        help="Experiment phase.",
    )
    parser.add_argument("--tracking-scan", default="", help="Optional tracking scan CSV for Phase 3R.2 / 3R.3.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_override, memory_override = _resolve_overrides(args.phase_label, args.tracking_scan)
    scenario_runs = evaluate_phase3_scenarios(
        args.config,
        scenario_names=["track_a_bridge", "track_c_long_horizon"],
        tracking_override=tracking_override,
        memory_override=memory_override,
        collect_frames=True,
        frame_record_mode="lite",
        seed=args.seed,
    )

    event_rows = []
    frame_log_rows = []
    summary_rows = []
    for scenario_run in scenario_runs:
        events, frame_logs = extract_reentry_events(
            scenario_run["scenario_name"],
            scenario_run["sequence"],
            scenario_run["result"],
            recovery_window=REENTRY_WINDOW,
        )
        event_rows.extend(events)
        frame_log_rows.extend(frame_logs)
        reentry_summary = summarize_reentry_events(events)
        summary_rows.append(
            {
                "scenario_name": scenario_run["scenario_name"],
                "num_events": int(reentry_summary["num_events"]),
                "same_track_reentry_recovery": float(reentry_summary["same_track_reentry_recovery"]),
                "same_prototype_reentry_recovery": float(reentry_summary["same_prototype_reentry_recovery"]),
                "same_track_after_concept_recovery": float(reentry_summary["same_track_after_concept_recovery"]),
                "concept_recovered_events": int(reentry_summary["concept_recovered_events"]),
                "prototype_gated_resurrection_attempt_rate": float(
                    reentry_summary["prototype_gated_resurrection_attempt_rate"]
                ),
                "resurrection_success_given_candidate_exists": float(
                    reentry_summary["resurrection_success_given_candidate_exists"]
                ),
                "candidate_exists_events": int(reentry_summary["candidate_exists_events"]),
                "mean_candidate_pool_size": float(reentry_summary["mean_candidate_pool_size"]),
                "candidate_pool_nonempty_rate": float(reentry_summary["candidate_pool_nonempty_rate"]),
                "continuation_bank_nonempty_rate": float(reentry_summary["continuation_bank_nonempty_rate"]),
                "continuation_attempt_rate": float(reentry_summary["continuation_attempt_rate"]),
                "continuation_success_rate": float(reentry_summary["continuation_success_rate"]),
                "slot_pool_nonempty_rate": float(reentry_summary["slot_pool_nonempty_rate"]),
                "slot_resurrection_attempt_rate": float(reentry_summary["slot_resurrection_attempt_rate"]),
                "slot_resurrection_success_rate": float(reentry_summary["slot_resurrection_success_rate"]),
                "new_track_with_old_prototype_rate": float(reentry_summary["new_track_with_old_prototype_rate"]),
            }
        )

    events_path = output_dir / f"reentry_events_{args.artifact_version}.csv"
    frame_log_path = output_dir / f"reentry_frame_log_{args.artifact_version}.csv"
    summary_path = output_dir / f"reentry_audit_summary_{args.artifact_version}.json"

    write_csv(events_path, event_rows)
    write_csv(frame_log_path, frame_log_rows)
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print(f"saved_events={events_path}")
    print(f"saved_frame_log={frame_log_path}")
    print(f"saved_summary={summary_path}")
    for row in summary_rows:
        print(
            f"{row['scenario_name']}: events={int(row['num_events'])}, "
            f"same_track={float(row['same_track_reentry_recovery']):.4f}, "
            f"same_proto={float(row['same_prototype_reentry_recovery']):.4f}, "
            f"track_after_concept={float(row['same_track_after_concept_recovery']):.4f}, "
            f"pg_attempt={float(row['prototype_gated_resurrection_attempt_rate']):.4f}, "
            f"res_success|cand={float(row['resurrection_success_given_candidate_exists']):.4f}"
        )


def _resolve_overrides(phase_label: str, tracking_scan_path: str) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if phase_label == "phase3r":
        return None, None

    if phase_label == "phase3r2":
        tracking_override = default_phase3r2_tracking_override()
        memory_override = default_phase3r2_memory_override()
        if tracking_scan_path:
            best_row = _load_best_phase3r2_scan_row(tracking_scan_path)
            tracking_override.update(
                {
                    "dormant_frames": int(best_row["dormant_frames"]),
                    "ghost_frames": int(best_row["ghost_frames"]),
                    "tau_g": float(best_row["tau_g"]),
                    "tau_res_short": float(best_row["tau_res_short"]),
                    "tau_res_long": float(best_row["tau_res_long"]),
                }
            )
        return tracking_override, memory_override

    if phase_label == "phase3r3":
        tracking_override = default_phase3r3_tracking_override()
        memory_override = default_phase3r3_memory_override()
        if tracking_scan_path:
            best_row = _load_best_phase3r3_scan_row(tracking_scan_path)
            tracking_override.update(
                {
                    "dormant_frames": int(best_row["dormant_frames"]),
                    "ghost_frames": int(best_row["ghost_frames"]),
                    "tau_g": float(best_row["tau_g"]),
                    "tau_res_short": float(best_row["tau_res_short"]),
                    "tau_res_long": float(best_row["tau_res_long"]),
                    "slot_topk_per_proto": int(best_row["slot_topk_per_proto"]),
                    "slot_max_gap": int(best_row["slot_max_gap"]),
                    "slot_tau": float(best_row["slot_tau"]),
                    "slot_margin": float(best_row["slot_margin"]),
                    "min_track_age_for_slot": int(best_row["min_track_age_for_slot"]),
                }
            )
        return tracking_override, memory_override

    if phase_label == "phase3s":
        tracking_override = default_phase3s_tracking_override()
        memory_override = default_phase3s_memory_override()
        if tracking_scan_path:
            best_row = _load_best_phase3s_scan_row(tracking_scan_path)
            tracking_override.update(
                {
                    "tau_continuation": float(best_row["tau_continuation"]),
                    "continuation_margin": float(best_row["continuation_margin"]),
                    "enable_identity_slots": False,
                }
            )
            memory_override.update(
                {
                    "continuation_topk_per_proto": int(best_row["continuation_topk_per_proto"]),
                    "continuation_max_gap": int(best_row["continuation_max_gap"]),
                    "min_track_age_for_continuation": int(best_row["min_track_age_for_continuation"]),
                    "enable_continuation_bank": True,
                }
            )
        return tracking_override, memory_override

    if phase_label == "phase3l":
        tracking_override = default_phase3l_tracking_override()
        memory_override = default_phase3l_memory_override()
        if tracking_scan_path:
            best_row = _load_best_phase3l_scan_row(tracking_scan_path)
            memory_override.update(
                {
                    "bind_continuation_to": str(best_row["bind_continuation_to"]),
                    "allow_alias_lineage": str(best_row["allow_alias_lineage"]).lower() in {"1", "true", "yes"},
                    "continuation_topk_per_lineage": int(best_row["continuation_topk_per_lineage"]),
                }
            )
        return tracking_override, memory_override

    return default_phase3x_tracking_override(), default_phase3x_memory_override()


def _load_best_phase3r2_scan_row(path: str) -> dict[str, object]:
    rows = load_csv_rows(path)
    if not rows:
        raise FileNotFoundError(f"empty tracking scan: {path}")
    explicit = [row for row in rows if str(row.get("is_best", "0")).lower() in {"1", "true", "yes"}]
    if explicit:
        return explicit[0]
    return pick_best_scan_row(rows, load_phase3r_before_lookup())


def _load_best_phase3r3_scan_row(path: str) -> dict[str, object]:
    rows = load_csv_rows(path)
    if not rows:
        raise FileNotFoundError(f"empty tracking scan: {path}")
    explicit = [row for row in rows if str(row.get("is_best", "0")).lower() in {"1", "true", "yes"}]
    if explicit:
        return explicit[0]
    return pick_phase3r3_scan_row(rows, load_phase3r2_before_lookup())


def _load_best_phase3s_scan_row(path: str) -> dict[str, object]:
    rows = load_csv_rows(path)
    if not rows:
        raise FileNotFoundError(f"empty tracking scan: {path}")
    explicit = [row for row in rows if str(row.get("is_best", "0")).lower() in {"1", "true", "yes"}]
    if explicit:
        return explicit[0]
    return pick_phase3s_scan_row(rows, load_phase3r3_before_lookup())


def _load_best_phase3l_scan_row(path: str) -> dict[str, object]:
    rows = load_csv_rows(path)
    if not rows:
        raise FileNotFoundError(f"empty tracking scan: {path}")
    explicit = [row for row in rows if str(row.get("is_best", "0")).lower() in {"1", "true", "yes"}]
    if explicit:
        return explicit[0]
    return pick_phase3l_scan_row(rows, load_phase3x_before_lookup())


if __name__ == "__main__":
    main()
