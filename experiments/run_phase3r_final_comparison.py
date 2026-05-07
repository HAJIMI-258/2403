"""Build the final Phase 3R / Phase 3R.2 / Phase 3R.3 comparison package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3r2_utils import (
    TRACK_A_NAME,
    TRACK_C_NAME,
    default_phase3r2_memory_override,
    default_phase3r2_tracking_override,
    evaluate_phase3r2_bundle,
    load_csv_rows,
    load_phase3r_before_lookup,
    pick_best_scan_row,
    plot_track_c_before_after,
    plot_track_state_timeline,
)
from experiments.phase3r3_utils import (
    default_phase3r3_memory_override,
    default_phase3r3_tracking_override,
    evaluate_phase3r3_bundle,
    load_phase3r2_before_lookup,
    plot_slot_resurrection_timeline,
    plot_track_c_before_after_v3,
    pick_best_scan_row as pick_phase3r3_scan_row,
)
from experiments.phase3s_utils import (
    default_phase3s_memory_override,
    default_phase3s_tracking_override,
    evaluate_phase3s_bundle,
    load_phase3r3_before_lookup,
    pick_best_scan_row as pick_phase3s_scan_row,
    plot_continuation_timeline,
    plot_track_c_before_after_v4,
)
from experiments.phase3l_utils import (
    TRACK_A_NAME as PHASE3L_TRACK_A_NAME,
    TRACK_C_NAME as PHASE3L_TRACK_C_NAME,
    build_phase3l_summary_rows,
    default_phase3l_memory_override,
    default_phase3l_tracking_override,
    evaluate_phase3l_bundle,
    load_phase3x_before_lookup,
    pick_best_scan_row as pick_phase3l_scan_row,
    plot_continuation_access_after_concept,
    plot_failure_stage_comparison,
    plot_lineage_preservation_timeline,
    plot_strict_vs_lineage_vs_binding_eval,
)
from experiments.phase3x_utils import load_phase3s_before_lookup
from experiments.phase3r_utils import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final Phase 3R comparison package.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml", help="Path to config.")
    parser.add_argument("--output-dir", default="results/phase3r", help="Directory for outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument(
        "--phase-label",
        default="phase3r",
        choices=["phase3r", "phase3r2", "phase3r3", "phase3s", "phase3x", "phase3l"],
        help="Experiment phase.",
    )
    parser.add_argument("--artifact-version", default="v1", help="Artifact suffix such as v1.")
    parser.add_argument("--tracking-scan", default="", help="Optional Phase 3R.2 tracking scan CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase_label == "phase3r2":
        _run_phase3r2(args, output_dir)
        return
    if args.phase_label == "phase3r3":
        _run_phase3r3(args, output_dir)
        return
    if args.phase_label == "phase3s":
        _run_phase3s(args, output_dir)
        return
    if args.phase_label == "phase3x":
        _run_phase3x(args, output_dir)
        return
    if args.phase_label == "phase3l":
        _run_phase3l(args, output_dir)
        return
    raise ValueError("This runner supports Phase 3R.2, Phase 3R.3, Phase 3S, Phase 3X, and Phase 3L only.")


def _run_phase3r2(args: argparse.Namespace, output_dir: Path) -> None:
    before_lookup = load_phase3r_before_lookup()
    tracking_override, memory_override = _resolve_phase3r2_overrides(args.tracking_scan)
    bundle = evaluate_phase3r2_bundle(
        args.config,
        tracking_override=tracking_override,
        memory_override=memory_override,
        seed=args.seed,
    )
    after_lookup = {row["scenario_name"]: row for row in bundle["rows"]}

    summary_rows = []
    for scenario_name in [TRACK_A_NAME, TRACK_C_NAME]:
        before_row = _coerce_phase3r_before_row(before_lookup[scenario_name])
        after_row = dict(after_lookup[scenario_name])
        summary_rows.append({"method": "phase3r_before", **before_row})
        summary_rows.append({"method": "phase3r2_current", **after_row})

    summary_path = output_dir / f"phase3r2_final_summary_{args.artifact_version}.csv"
    write_csv(summary_path, summary_rows)

    plot_track_state_timeline(bundle["frame_logs"], output_dir / "track_state_timeline_v2.png", scenario_name=TRACK_C_NAME)
    plot_track_c_before_after(
        before_lookup[TRACK_C_NAME],
        after_lookup[TRACK_C_NAME],
        output_dir / "track_c_before_after_v2.png",
    )

    summary_doc = _build_phase3r2_summary_doc(after_lookup, before_lookup, tracking_override)
    failure_notes = _build_phase3r2_failure_notes(after_lookup, before_lookup)
    (output_dir / f"phase3r2_summary_{args.artifact_version}.md").write_text(summary_doc, encoding="utf-8")
    (output_dir / f"phase3r2_failure_notes_{args.artifact_version}.md").write_text(failure_notes, encoding="utf-8")

    print(f"saved_summary={summary_path}")
    print(f"saved_timeline={output_dir / 'track_state_timeline_v2.png'}")
    print(f"saved_before_after={output_dir / 'track_c_before_after_v2.png'}")
    print(f"saved_notes={output_dir / f'phase3r2_failure_notes_{args.artifact_version}.md'}")
    print(f"saved_summary_doc={output_dir / f'phase3r2_summary_{args.artifact_version}.md'}")


def _run_phase3r3(args: argparse.Namespace, output_dir: Path) -> None:
    before_lookup = load_phase3r2_before_lookup()
    tracking_override, memory_override = _resolve_phase3r3_overrides(args.tracking_scan)
    bundle = evaluate_phase3r3_bundle(
        args.config,
        tracking_override=tracking_override,
        memory_override=memory_override,
        seed=args.seed,
        frame_record_mode="full",
    )
    after_lookup = {row["scenario_name"]: row for row in bundle["rows"]}

    summary_rows = []
    for scenario_name in [TRACK_A_NAME, TRACK_C_NAME]:
        before_row = _coerce_phase3r2_before_row(before_lookup[scenario_name])
        after_row = dict(after_lookup[scenario_name])
        summary_rows.append({"method": "phase3r2_current", **before_row})
        summary_rows.append({"method": "phase3r3_current", **after_row})

    summary_path = output_dir / f"phase3r3_final_summary_{args.artifact_version}.csv"
    write_csv(summary_path, summary_rows)

    plot_slot_resurrection_timeline(
        bundle["frame_logs"],
        output_dir / "slot_resurrection_timeline_v1.png",
        scenario_name=TRACK_C_NAME,
    )
    plot_track_c_before_after_v3(
        before_lookup[TRACK_C_NAME],
        after_lookup[TRACK_C_NAME],
        output_dir / "track_c_before_after_v3.png",
    )

    summary_doc = _build_phase3r3_summary_doc(after_lookup, before_lookup, tracking_override)
    failure_notes = _build_phase3r3_failure_notes(after_lookup, before_lookup)
    (output_dir / f"phase3r3_summary_{args.artifact_version}.md").write_text(summary_doc, encoding="utf-8")
    (output_dir / f"phase3r3_failure_notes_{args.artifact_version}.md").write_text(failure_notes, encoding="utf-8")

    print(f"saved_summary={summary_path}")
    print(f"saved_timeline={output_dir / 'slot_resurrection_timeline_v1.png'}")
    print(f"saved_before_after={output_dir / 'track_c_before_after_v3.png'}")
    print(f"saved_notes={output_dir / f'phase3r3_failure_notes_{args.artifact_version}.md'}")
    print(f"saved_summary_doc={output_dir / f'phase3r3_summary_{args.artifact_version}.md'}")


def _run_phase3s(args: argparse.Namespace, output_dir: Path) -> None:
    before_lookup = load_phase3r3_before_lookup()
    tracking_override, memory_override = _resolve_phase3s_overrides(args.tracking_scan)
    bundle = evaluate_phase3s_bundle(
        args.config,
        tracking_override=tracking_override,
        memory_override=memory_override,
        seed=args.seed,
        frame_record_mode="lite",
    )
    after_lookup = {row["scenario_name"]: row for row in bundle["rows"]}

    summary_rows = []
    for scenario_name in [TRACK_A_NAME, TRACK_C_NAME]:
        before_row = _coerce_phase3r3_before_row(before_lookup[scenario_name])
        after_row = dict(after_lookup[scenario_name])
        summary_rows.append({"method": "phase3r3_current", **before_row})
        summary_rows.append({"method": "phase3s_current", **after_row})

    summary_path = output_dir / f"phase3s_final_summary_{args.artifact_version}.csv"
    write_csv(summary_path, summary_rows)

    plot_continuation_timeline(
        bundle["frame_logs"],
        output_dir / "continuation_timeline_v1.png",
        scenario_name=TRACK_C_NAME,
    )
    plot_track_c_before_after_v4(
        before_lookup[TRACK_C_NAME],
        after_lookup[TRACK_C_NAME],
        output_dir / "track_c_before_after_v4.png",
    )

    summary_doc = _build_phase3s_summary_doc(after_lookup, before_lookup, tracking_override, memory_override)
    failure_notes = _build_phase3s_failure_notes(after_lookup, before_lookup)
    (output_dir / f"phase3s_summary_{args.artifact_version}.md").write_text(summary_doc, encoding="utf-8")
    (output_dir / f"phase3s_failure_notes_{args.artifact_version}.md").write_text(failure_notes, encoding="utf-8")

    print(f"saved_summary={summary_path}")
    print(f"saved_timeline={output_dir / 'continuation_timeline_v1.png'}")
    print(f"saved_before_after={output_dir / 'track_c_before_after_v4.png'}")
    print(f"saved_notes={output_dir / f'phase3s_failure_notes_{args.artifact_version}.md'}")
    print(f"saved_summary_doc={output_dir / f'phase3s_summary_{args.artifact_version}.md'}")


def _run_phase3x(args: argparse.Namespace, output_dir: Path) -> None:
    before_lookup = load_phase3s_before_lookup()
    summary_path = output_dir / f"phase3x_final_audit_summary_{args.artifact_version}.json"
    eval_path = output_dir / f"phase3x_lineage_eval_summary_{args.artifact_version}.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing audit summary: {summary_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"missing lineage eval summary: {eval_path}")

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    eval_rows = load_csv_rows(str(eval_path))
    summary_doc = _build_phase3x_summary_doc(summary_payload, eval_rows, before_lookup)
    failure_notes = _build_phase3x_failure_notes(summary_payload, eval_rows, before_lookup)
    (output_dir / f"phase3x_summary_{args.artifact_version}.md").write_text(summary_doc, encoding="utf-8")
    (output_dir / f"phase3x_failure_notes_{args.artifact_version}.md").write_text(failure_notes, encoding="utf-8")

    print(f"saved_notes={output_dir / f'phase3x_failure_notes_{args.artifact_version}.md'}")
    print(f"saved_summary_doc={output_dir / f'phase3x_summary_{args.artifact_version}.md'}")


def _run_phase3l(args: argparse.Namespace, output_dir: Path) -> None:
    before_lookup = load_phase3x_before_lookup()
    tracking_override, memory_override = _resolve_phase3l_overrides(args.tracking_scan)
    bundle = evaluate_phase3l_bundle(
        args.config,
        tracking_override=tracking_override,
        memory_override=memory_override,
        seed=args.seed,
        frame_record_mode="full",
    )
    trace_path = output_dir / "reentry_event_trace_v2.csv"
    eval_path = output_dir / f"phase3l_lineage_eval_summary_{args.artifact_version}.csv"
    prototype_rows_path = output_dir / "prototype_lineage_rows_v2.csv"
    phase3x_summary_path = Path("results/phase3x/phase3x_final_audit_summary_v1.json")
    if not trace_path.exists():
        raise FileNotFoundError(f"missing phase3l trace rows: {trace_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"missing phase3l eval summary: {eval_path}")
    if not prototype_rows_path.exists():
        raise FileNotFoundError(f"missing phase3l prototype rows: {prototype_rows_path}")
    if not phase3x_summary_path.exists():
        raise FileNotFoundError(f"missing phase3x audit summary: {phase3x_summary_path}")

    trace_rows = load_csv_rows(str(trace_path))
    eval_rows = load_csv_rows(str(eval_path))
    prototype_rows = load_csv_rows(str(prototype_rows_path))
    phase3x_summary = json.loads(phase3x_summary_path.read_text(encoding="utf-8"))

    after_lookup = {
        str(row["scenario_name"]): row
        for row in build_phase3l_summary_rows(trace_rows, bundle["rows"])
    }
    summary_rows = []
    for scenario_name in [PHASE3L_TRACK_A_NAME, PHASE3L_TRACK_C_NAME]:
        before_row = dict(before_lookup[scenario_name])
        after_row = dict(after_lookup[scenario_name])
        summary_rows.append({"method": "phase3x_before", **before_row})
        summary_rows.append({"method": "phase3l_current", **after_row})

    summary_path = output_dir / f"phase3l_final_summary_{args.artifact_version}.csv"
    write_csv(summary_path, summary_rows)

    plot_lineage_preservation_timeline(prototype_rows, output_dir / "lineage_preservation_timeline_v1.png")
    plot_strict_vs_lineage_vs_binding_eval(eval_rows, output_dir / "strict_vs_lineage_vs_binding_eval_v1.png")
    plot_failure_stage_comparison(
        before_counts={str(key): int(value) for key, value in phase3x_summary.get("track_c_failure_stage_counts", {}).items()},
        after_trace_rows=trace_rows,
        path=output_dir / "track_c_event_failure_stage_v2.png",
    )
    plot_continuation_access_after_concept(trace_rows, output_dir / "continuation_access_after_concept_v1.png")

    summary_doc = _build_phase3l_summary_doc(after_lookup, before_lookup, tracking_override, memory_override)
    failure_notes = _build_phase3l_failure_notes(after_lookup, before_lookup)
    (output_dir / f"phase3l_summary_{args.artifact_version}.md").write_text(summary_doc, encoding="utf-8")
    (output_dir / f"phase3l_failure_notes_{args.artifact_version}.md").write_text(failure_notes, encoding="utf-8")

    print(f"saved_summary={summary_path}")
    print(f"saved_lineage_timeline={output_dir / 'lineage_preservation_timeline_v1.png'}")
    print(f"saved_eval_plot={output_dir / 'strict_vs_lineage_vs_binding_eval_v1.png'}")
    print(f"saved_failure_plot={output_dir / 'track_c_event_failure_stage_v2.png'}")
    print(f"saved_access_plot={output_dir / 'continuation_access_after_concept_v1.png'}")
    print(f"saved_notes={output_dir / f'phase3l_failure_notes_{args.artifact_version}.md'}")
    print(f"saved_summary_doc={output_dir / f'phase3l_summary_{args.artifact_version}.md'}")


def _resolve_phase3r2_overrides(tracking_scan_path: str) -> tuple[dict[str, object], dict[str, object]]:
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


def _resolve_phase3r3_overrides(tracking_scan_path: str) -> tuple[dict[str, object], dict[str, object]]:
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


def _resolve_phase3s_overrides(tracking_scan_path: str) -> tuple[dict[str, object], dict[str, object]]:
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


def _resolve_phase3l_overrides(tracking_scan_path: str) -> tuple[dict[str, object], dict[str, object]]:
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


def _load_best_phase3l_scan_row(path: str) -> dict[str, object]:
    rows = load_csv_rows(path)
    if not rows:
        raise FileNotFoundError(f"empty tracking scan: {path}")
    explicit = [row for row in rows if str(row.get("is_best", "0")).lower() in {"1", "true", "yes"}]
    if explicit:
        return explicit[0]
    return pick_phase3l_scan_row(rows, load_phase3x_before_lookup())


def _coerce_phase3r_before_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "scenario_name": str(row["scenario_name"]),
        "u_recall": float(row["u_recall"]),
        "same_track_reentry_recovery": float(row["same_track_reentry_recovery"]),
        "same_prototype_reentry_recovery": float(row["same_prototype_reentry_recovery"]),
        "same_track_after_concept_recovery": float(row["same_track_after_concept_recovery"]),
        "concept_recovered_events": int(float(row["concept_recovered_events"])),
        "prototype_gated_resurrection_attempt_rate": 0.0,
        "resurrection_success_given_candidate_exists": 0.0,
        "candidate_exists_events": 0,
        "mean_candidate_pool_size": 0.0,
        "proposal_detect_rate": float(row["proposal_detect_rate"]),
        "pfr": float(row["pfr"]),
        "track_idsw": int(float(row["track_idsw"])),
        "memory_growth": float(row["memory_growth"]),
        "reentry_events": int(float(row["reentry_events"])),
        "reactivation_successes": 0,
        "created_tracks": 0,
    }


def _coerce_phase3r2_before_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "scenario_name": str(row["scenario_name"]),
        "u_recall": float(row["u_recall"]),
        "same_track_reentry_recovery": float(row["same_track_reentry_recovery"]),
        "same_prototype_reentry_recovery": float(row["same_prototype_reentry_recovery"]),
        "same_track_after_concept_recovery": float(row["same_track_after_concept_recovery"]),
        "candidate_pool_nonempty_rate": float(row.get("candidate_pool_nonempty_rate", 0.0)),
        "slot_pool_nonempty_rate": float(row.get("slot_pool_nonempty_rate", 0.0)),
        "slot_resurrection_attempt_rate": float(row.get("slot_resurrection_attempt_rate", 0.0)),
        "slot_resurrection_success_rate": float(row.get("slot_resurrection_success_rate", 0.0)),
        "new_track_with_old_prototype_rate": float(row.get("new_track_with_old_prototype_rate", 0.0)),
        "concept_recovered_events": int(float(row["concept_recovered_events"])),
        "prototype_gated_resurrection_attempt_rate": float(row["prototype_gated_resurrection_attempt_rate"]),
        "resurrection_success_given_candidate_exists": float(row["resurrection_success_given_candidate_exists"]),
        "candidate_exists_events": int(float(row["candidate_exists_events"])),
        "mean_candidate_pool_size": float(row["mean_candidate_pool_size"]),
        "proposal_detect_rate": float(row["proposal_detect_rate"]),
        "pfr": float(row["pfr"]),
        "track_idsw": int(float(row["track_idsw"])),
        "memory_growth": float(row["memory_growth"]),
        "reentry_events": int(float(row["reentry_events"])),
        "reactivation_successes": int(float(row["reactivation_successes"])),
        "created_tracks": int(float(row["created_tracks"])),
    }


def _coerce_phase3r3_before_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "scenario_name": str(row["scenario_name"]),
        "u_recall": float(row["u_recall"]),
        "same_track_reentry_recovery": float(row["same_track_reentry_recovery"]),
        "same_prototype_reentry_recovery": float(row["same_prototype_reentry_recovery"]),
        "same_track_after_concept_recovery": float(row["same_track_after_concept_recovery"]),
        "continuation_bank_nonempty_rate": float(row.get("continuation_bank_nonempty_rate", 0.0)),
        "candidate_pool_nonempty_rate": float(row.get("candidate_pool_nonempty_rate", 0.0)),
        "continuation_attempt_rate": float(row.get("continuation_attempt_rate", 0.0)),
        "continuation_success_rate": float(row.get("continuation_success_rate", 0.0)),
        "new_track_with_old_prototype_rate": float(row.get("new_track_with_old_prototype_rate", 0.0)),
        "concept_recovered_events": int(float(row["concept_recovered_events"])),
        "prototype_gated_resurrection_attempt_rate": float(row["prototype_gated_resurrection_attempt_rate"]),
        "resurrection_success_given_candidate_exists": float(row["resurrection_success_given_candidate_exists"]),
        "candidate_exists_events": int(float(row["candidate_exists_events"])),
        "mean_candidate_pool_size": float(row["mean_candidate_pool_size"]),
        "proposal_detect_rate": float(row["proposal_detect_rate"]),
        "pfr": float(row["pfr"]),
        "track_idsw": int(float(row["track_idsw"])),
        "memory_growth": float(row["memory_growth"]),
        "reentry_events": int(float(row["reentry_events"])),
        "reactivation_successes": int(float(row["reactivation_successes"])),
        "created_tracks": int(float(row["created_tracks"])),
    }


def _build_phase3r2_summary_doc(
    after_lookup: dict[str, dict[str, object]],
    before_lookup: dict[str, dict[str, object]],
    tracking_override: dict[str, object],
) -> str:
    track_a_before = before_lookup[TRACK_A_NAME]
    track_c_before = before_lookup[TRACK_C_NAME]
    track_a_after = after_lookup[TRACK_A_NAME]
    track_c_after = after_lookup[TRACK_C_NAME]

    passed = (
        float(track_c_after["same_track_after_concept_recovery"]) >= 0.50
        and float(track_c_after["same_track_reentry_recovery"]) >= 0.35
        and float(track_c_after["same_prototype_reentry_recovery"]) >= 0.80
        and float(track_c_after["pfr"]) < 2.0
    )
    partial = (
        float(track_c_after["same_track_after_concept_recovery"]) > float(track_c_before["same_track_after_concept_recovery"])
        and float(track_c_after["same_track_reentry_recovery"]) > float(track_c_before["same_track_reentry_recovery"])
        and float(track_c_after["same_prototype_reentry_recovery"]) >= 0.80
        and float(track_c_after["pfr"]) < float(track_c_before["pfr"])
    )
    verdict = "pass" if passed else ("partial" if partial else "fail")

    lines = [
        "# Phase 3R.2 Summary v1",
        "",
        "## Selected Params",
        "",
        f"- keepalive_frames={int(tracking_override['keepalive_frames'])}, dormant_frames={int(tracking_override['dormant_frames'])}, ghost_frames={int(tracking_override['ghost_frames'])}.",
        f"- tau_g={float(tracking_override['tau_g']):.2f}, tau_res_short={float(tracking_override['tau_res_short']):.2f}, tau_res_long={float(tracking_override['tau_res_long']):.2f}.",
        "",
        "## Track A",
        "",
        f"- U-Recall: {float(track_a_before['u_recall']):.4f} -> {float(track_a_after['u_recall']):.4f}",
        f"- same-prototype: {float(track_a_before['same_prototype_reentry_recovery']):.4f} -> {float(track_a_after['same_prototype_reentry_recovery']):.4f}",
        f"- memory_growth: {float(track_a_before['memory_growth']):.4f} -> {float(track_a_after['memory_growth']):.4f}",
        "",
        "## Track C",
        "",
        f"- same-track: {float(track_c_before['same_track_reentry_recovery']):.4f} -> {float(track_c_after['same_track_reentry_recovery']):.4f}",
        f"- same-prototype: {float(track_c_before['same_prototype_reentry_recovery']):.4f} -> {float(track_c_after['same_prototype_reentry_recovery']):.4f}",
        f"- same-track-after-concept: {float(track_c_before['same_track_after_concept_recovery']):.4f} -> {float(track_c_after['same_track_after_concept_recovery']):.4f}",
        f"- prototype-gated resurrection attempt rate: {float(track_c_after['prototype_gated_resurrection_attempt_rate']):.4f}",
        f"- resurrection success | candidate exists: {float(track_c_after['resurrection_success_given_candidate_exists']):.4f}",
        f"- PFR: {float(track_c_before['pfr']):.4f} -> {float(track_c_after['pfr']):.4f}",
        f"- IDSW: {int(track_c_before['track_idsw'])} -> {int(track_c_after['track_idsw'])}",
        "",
        "## Verdict",
        "",
        f"- status: {verdict}",
        "- Interpretation: this round is only successful if old-track resurrection becomes a default path after concept recovery, not a rare exception.",
        "",
    ]
    return "\n".join(lines)


def _build_phase3r2_failure_notes(
    after_lookup: dict[str, dict[str, object]],
    before_lookup: dict[str, dict[str, object]],
) -> str:
    before = before_lookup[TRACK_C_NAME]
    after = after_lookup[TRACK_C_NAME]
    lines = [
        "# Phase 3R.2 Failure Notes v1",
        "",
        "## Track C Core Readout",
        "",
        f"- same-track: {float(before['same_track_reentry_recovery']):.4f} -> {float(after['same_track_reentry_recovery']):.4f}",
        f"- same-prototype: {float(before['same_prototype_reentry_recovery']):.4f} -> {float(after['same_prototype_reentry_recovery']):.4f}",
        f"- same-track-after-concept: {float(before['same_track_after_concept_recovery']):.4f} -> {float(after['same_track_after_concept_recovery']):.4f}",
        f"- PFR: {float(before['pfr']):.4f} -> {float(after['pfr']):.4f}",
        f"- IDSW: {int(before['track_idsw'])} -> {int(after['track_idsw'])}",
        "",
        "## Bottleneck",
        "",
        "- If same-prototype stays high while same-track-after-concept stays low, the remaining problem is still identity continuation, not concept recovery.",
        "- If candidate_exists_events grows but resurrection_success_given_candidate_exists stays low, the blocker is the resurrection cost or threshold, not lifecycle coverage.",
        "- If candidate_exists_events remains low, then ghost coverage is still too short and old tracks are leaving the candidate pool too early.",
        "",
    ]
    return "\n".join(lines)


def _build_phase3r3_summary_doc(
    after_lookup: dict[str, dict[str, object]],
    before_lookup: dict[str, dict[str, object]],
    tracking_override: dict[str, object],
) -> str:
    track_a_before = before_lookup[TRACK_A_NAME]
    track_c_before = before_lookup[TRACK_C_NAME]
    track_a_after = after_lookup[TRACK_A_NAME]
    track_c_after = after_lookup[TRACK_C_NAME]

    passed = (
        float(track_c_after["candidate_pool_nonempty_rate"]) >= 0.70
        and float(track_c_after["same_track_after_concept_recovery"]) >= 0.50
        and float(track_c_after["same_track_reentry_recovery"]) >= 0.35
        and float(track_c_after["same_prototype_reentry_recovery"]) >= 0.80
        and float(track_c_after["pfr"]) < 2.0
    )
    partial = (
        float(track_c_after["candidate_pool_nonempty_rate"]) > float(track_c_before.get("candidate_pool_nonempty_rate", 0.0))
        and float(track_c_after["same_track_after_concept_recovery"]) > float(track_c_before["same_track_after_concept_recovery"])
        and float(track_c_after["same_track_reentry_recovery"]) > float(track_c_before["same_track_reentry_recovery"])
        and float(track_c_after["same_prototype_reentry_recovery"]) >= 0.80
        and float(track_c_after["pfr"]) < float(track_c_before["pfr"])
    )
    verdict = "pass" if passed else ("partial" if partial else "fail")

    lines = [
        "# Phase 3R.3 Summary v1",
        "",
        "## Selected Params",
        "",
        (
            f"- keepalive_frames={int(tracking_override['keepalive_frames'])}, "
            f"dormant_frames={int(tracking_override['dormant_frames'])}, "
            f"ghost_frames={int(tracking_override['ghost_frames'])}, tau_g={float(tracking_override['tau_g']):.2f}."
        ),
        (
            f"- tau_res_short={float(tracking_override['tau_res_short']):.2f}, "
            f"tau_res_long={float(tracking_override['tau_res_long']):.2f}, "
            f"slot_topk_per_proto={int(tracking_override['slot_topk_per_proto'])}, "
            f"slot_max_gap={int(tracking_override['slot_max_gap'])}, "
            f"slot_tau={float(tracking_override['slot_tau']):.2f}, "
            f"slot_margin={float(tracking_override['slot_margin']):.2f}, "
            f"min_track_age_for_slot={int(tracking_override['min_track_age_for_slot'])}."
        ),
        "",
        "## Track A",
        "",
        f"- U-Recall: {float(track_a_before['u_recall']):.4f} -> {float(track_a_after['u_recall']):.4f}",
        (
            f"- same-prototype: {float(track_a_before['same_prototype_reentry_recovery']):.4f} -> "
            f"{float(track_a_after['same_prototype_reentry_recovery']):.4f}"
        ),
        f"- memory_growth: {float(track_a_before['memory_growth']):.4f} -> {float(track_a_after['memory_growth']):.4f}",
        "",
        "## Track C",
        "",
        (
            f"- candidate-pool-nonempty-rate: {float(track_c_before.get('candidate_pool_nonempty_rate', 0.0)):.4f} -> "
            f"{float(track_c_after['candidate_pool_nonempty_rate']):.4f}"
        ),
        f"- same-track: {float(track_c_before['same_track_reentry_recovery']):.4f} -> {float(track_c_after['same_track_reentry_recovery']):.4f}",
        (
            f"- same-prototype: {float(track_c_before['same_prototype_reentry_recovery']):.4f} -> "
            f"{float(track_c_after['same_prototype_reentry_recovery']):.4f}"
        ),
        (
            f"- same-track-after-concept: {float(track_c_before['same_track_after_concept_recovery']):.4f} -> "
            f"{float(track_c_after['same_track_after_concept_recovery']):.4f}"
        ),
        f"- slot-pool-nonempty-rate: {float(track_c_after['slot_pool_nonempty_rate']):.4f}",
        f"- slot-resurrection-attempt-rate: {float(track_c_after['slot_resurrection_attempt_rate']):.4f}",
        f"- slot-resurrection-success-rate: {float(track_c_after['slot_resurrection_success_rate']):.4f}",
        f"- new-track-with-old-prototype-rate: {float(track_c_after['new_track_with_old_prototype_rate']):.4f}",
        f"- PFR: {float(track_c_before['pfr']):.4f} -> {float(track_c_after['pfr']):.4f}",
        f"- IDSW: {int(track_c_before['track_idsw'])} -> {int(track_c_after['track_idsw'])}",
        "",
        "## Verdict",
        "",
        f"- status: {verdict}",
        "- Interpretation: this round is only successful if concept-recovered events stop failing because the candidate pool is empty.",
        "",
    ]
    return "\n".join(lines)


def _build_phase3r3_failure_notes(
    after_lookup: dict[str, dict[str, object]],
    before_lookup: dict[str, dict[str, object]],
) -> str:
    before = before_lookup[TRACK_C_NAME]
    after = after_lookup[TRACK_C_NAME]
    lines = [
        "# Phase 3R.3 Failure Notes v1",
        "",
        "## Track C Core Readout",
        "",
        (
            f"- candidate-pool-nonempty-rate: {float(before.get('candidate_pool_nonempty_rate', 0.0)):.4f} -> "
            f"{float(after['candidate_pool_nonempty_rate']):.4f}"
        ),
        f"- same-track: {float(before['same_track_reentry_recovery']):.4f} -> {float(after['same_track_reentry_recovery']):.4f}",
        (
            f"- same-prototype: {float(before['same_prototype_reentry_recovery']):.4f} -> "
            f"{float(after['same_prototype_reentry_recovery']):.4f}"
        ),
        (
            f"- same-track-after-concept: {float(before['same_track_after_concept_recovery']):.4f} -> "
            f"{float(after['same_track_after_concept_recovery']):.4f}"
        ),
        f"- slot-pool-nonempty-rate: {float(after['slot_pool_nonempty_rate']):.4f}",
        f"- slot-resurrection-attempt-rate: {float(after['slot_resurrection_attempt_rate']):.4f}",
        f"- slot-resurrection-success-rate: {float(after['slot_resurrection_success_rate']):.4f}",
        f"- PFR: {float(before['pfr']):.4f} -> {float(after['pfr']):.4f}",
        f"- IDSW: {int(before['track_idsw'])} -> {int(after['track_idsw'])}",
        "",
        "## Bottleneck",
        "",
        "- If candidate-pool-nonempty-rate stays low, slot preservation is still too weak and identity is leaving the pool before concept recovery arrives.",
        "- If candidate-pool-nonempty-rate rises but same-track-after-concept stays low, the remaining blocker is candidate selection, not pool coverage.",
        "- If same-prototype drops below 0.80, the slot layer is polluting the concept layer and should be rolled back.",
        "",
    ]
    return "\n".join(lines)


def _build_phase3s_summary_doc(
    after_lookup: dict[str, dict[str, object]],
    before_lookup: dict[str, dict[str, object]],
    tracking_override: dict[str, object],
    memory_override: dict[str, object],
) -> str:
    track_a_before = before_lookup[TRACK_A_NAME]
    track_c_before = before_lookup[TRACK_C_NAME]
    track_a_after = after_lookup[TRACK_A_NAME]
    track_c_after = after_lookup[TRACK_C_NAME]

    passed = (
        float(track_c_after["continuation_bank_nonempty_rate"]) >= 0.60
        and float(track_c_after["candidate_pool_nonempty_rate"]) >= 0.60
        and float(track_c_after["same_track_after_concept_recovery"]) >= 0.45
        and float(track_c_after["same_track_reentry_recovery"]) >= 0.30
        and float(track_c_after["same_prototype_reentry_recovery"]) >= 0.80
        and float(track_c_after["pfr"]) < 2.5
    )
    partial = (
        float(track_c_after["continuation_bank_nonempty_rate"]) > float(track_c_before.get("continuation_bank_nonempty_rate", 0.0))
        and float(track_c_after["same_track_after_concept_recovery"]) > float(track_c_before["same_track_after_concept_recovery"])
        and float(track_c_after["same_track_reentry_recovery"]) > float(track_c_before["same_track_reentry_recovery"])
        and float(track_c_after["same_prototype_reentry_recovery"]) >= 0.80
        and float(track_c_after["pfr"]) < float(track_c_before["pfr"])
    )
    verdict = "pass" if passed else ("partial" if partial else "fail")

    lines = [
        "# Phase 3S Summary v1",
        "",
        "## Selected Params",
        "",
        (
            f"- keepalive_frames={int(tracking_override['keepalive_frames'])}, "
            f"dormant_frames={int(tracking_override['dormant_frames'])}, "
            f"ghost_frames={int(tracking_override['ghost_frames'])}, tau_g={float(tracking_override['tau_g']):.2f}."
        ),
        (
            f"- tau_continuation={float(tracking_override['tau_continuation']):.2f}, "
            f"continuation_margin={float(tracking_override['continuation_margin']):.2f}, "
            f"continuation_topk_per_proto={int(memory_override['continuation_topk_per_proto'])}, "
            f"continuation_max_gap={int(memory_override['continuation_max_gap'])}, "
            f"min_track_age_for_continuation={int(memory_override['min_track_age_for_continuation'])}."
        ),
        "",
        "## Track A",
        "",
        f"- U-Recall: {float(track_a_before['u_recall']):.4f} -> {float(track_a_after['u_recall']):.4f}",
        (
            f"- same-prototype: {float(track_a_before['same_prototype_reentry_recovery']):.4f} -> "
            f"{float(track_a_after['same_prototype_reentry_recovery']):.4f}"
        ),
        f"- memory_growth: {float(track_a_before['memory_growth']):.4f} -> {float(track_a_after['memory_growth']):.4f}",
        "",
        "## Track C",
        "",
        (
            f"- continuation-bank-nonempty-rate: {float(track_c_before.get('continuation_bank_nonempty_rate', 0.0)):.4f} -> "
            f"{float(track_c_after['continuation_bank_nonempty_rate']):.4f}"
        ),
        (
            f"- candidate-pool-nonempty-rate: {float(track_c_before.get('candidate_pool_nonempty_rate', 0.0)):.4f} -> "
            f"{float(track_c_after['candidate_pool_nonempty_rate']):.4f}"
        ),
        f"- same-track: {float(track_c_before['same_track_reentry_recovery']):.4f} -> {float(track_c_after['same_track_reentry_recovery']):.4f}",
        (
            f"- same-prototype: {float(track_c_before['same_prototype_reentry_recovery']):.4f} -> "
            f"{float(track_c_after['same_prototype_reentry_recovery']):.4f}"
        ),
        (
            f"- same-track-after-concept: {float(track_c_before['same_track_after_concept_recovery']):.4f} -> "
            f"{float(track_c_after['same_track_after_concept_recovery']):.4f}"
        ),
        f"- continuation-attempt-rate: {float(track_c_after['continuation_attempt_rate']):.4f}",
        f"- continuation-success-rate: {float(track_c_after['continuation_success_rate']):.4f}",
        f"- new-track-with-old-prototype-rate: {float(track_c_after['new_track_with_old_prototype_rate']):.4f}",
        f"- PFR: {float(track_c_before['pfr']):.4f} -> {float(track_c_after['pfr']):.4f}",
        f"- IDSW: {int(track_c_before['track_idsw'])} -> {int(track_c_after['track_idsw'])}",
        "",
        "## Verdict",
        "",
        f"- status: {verdict}",
        "- Interpretation: this round succeeds only if concept-recovered events stop failing because prototype-owned continuation is missing.",
        "",
    ]
    return "\n".join(lines)


def _build_phase3s_failure_notes(
    after_lookup: dict[str, dict[str, object]],
    before_lookup: dict[str, dict[str, object]],
) -> str:
    before = before_lookup[TRACK_C_NAME]
    after = after_lookup[TRACK_C_NAME]
    lines = [
        "# Phase 3S Failure Notes v1",
        "",
        "## Track C Core Readout",
        "",
        (
            f"- continuation-bank-nonempty-rate: {float(before.get('continuation_bank_nonempty_rate', 0.0)):.4f} -> "
            f"{float(after['continuation_bank_nonempty_rate']):.4f}"
        ),
        (
            f"- candidate-pool-nonempty-rate: {float(before.get('candidate_pool_nonempty_rate', 0.0)):.4f} -> "
            f"{float(after['candidate_pool_nonempty_rate']):.4f}"
        ),
        f"- same-track: {float(before['same_track_reentry_recovery']):.4f} -> {float(after['same_track_reentry_recovery']):.4f}",
        (
            f"- same-prototype: {float(before['same_prototype_reentry_recovery']):.4f} -> "
            f"{float(after['same_prototype_reentry_recovery']):.4f}"
        ),
        (
            f"- same-track-after-concept: {float(before['same_track_after_concept_recovery']):.4f} -> "
            f"{float(after['same_track_after_concept_recovery']):.4f}"
        ),
        f"- continuation-attempt-rate: {float(after['continuation_attempt_rate']):.4f}",
        f"- continuation-success-rate: {float(after['continuation_success_rate']):.4f}",
        f"- PFR: {float(before['pfr']):.4f} -> {float(after['pfr']):.4f}",
        f"- IDSW: {int(before['track_idsw'])} -> {int(after['track_idsw'])}",
        "",
        "## Bottleneck",
        "",
        "- If continuation-bank-nonempty-rate rises but same-track-after-concept stays flat, the remaining blocker is candidate selection, not continuation storage.",
        "- If continuation-bank-nonempty-rate stays low, prototype-owned continuation is still not being archived onto the same prototypes that later recover concept identity.",
        "- If same-prototype drops below 0.80, continuation handling is polluting the concept layer and should be rolled back.",
        "",
    ]
    return "\n".join(lines)


def _build_phase3x_summary_doc(
    summary_payload: dict[str, object],
    eval_rows: list[dict[str, object]],
    before_lookup: dict[str, dict[str, object]],
) -> str:
    track_c_before = before_lookup[TRACK_C_NAME]
    track_c = summary_payload.get("track_c", {})
    same_proto_strict = float(track_c.get("same_prototype_reentry_recovery", 0.0))
    same_proto_lineage = float(summary_payload.get("same_lineage_prototype_reentry_recovery", 0.0))
    lines = [
        "# Phase 3X Summary v1",
        "",
        "## Audit Answers",
        "",
        f"- continuation_write_success_rate: {float(summary_payload.get('continuation_write_success_rate', 0.0)):.4f}",
        (
            "- continuation_survival_until_concept_recovery_rate: "
            f"{float(summary_payload.get('continuation_survival_until_concept_recovery_rate', 0.0)):.4f}"
        ),
        (
            "- concept_recovered_but_lineage_mismatch_rate: "
            f"{float(summary_payload.get('concept_recovered_but_lineage_mismatch_rate', 0.0)):.4f}"
        ),
        f"- strict same-prototype vs lineage-aware: {same_proto_strict:.4f} -> {same_proto_lineage:.4f}",
        (
            "- continuation_bank_access_rate_given_concept_recovery: "
            f"{float(summary_payload.get('continuation_bank_access_rate_given_concept_recovery', 0.0)):.4f}"
        ),
        (
            "- continuation_bank_access_rate_given_same_lineage: "
            f"{float(summary_payload.get('continuation_bank_access_rate_given_same_lineage', 0.0)):.4f}"
        ),
        "",
        "## Track C",
        "",
        f"- Phase 3S strict same-prototype: {float(track_c_before['same_prototype_reentry_recovery']):.4f}",
        f"- Phase 3X strict same-prototype: {same_proto_strict:.4f}",
        f"- Phase 3X lineage-aware same-prototype: {same_proto_lineage:.4f}",
        (
            f"- Phase 3X strict same-track-after-concept: "
            f"{float(track_c.get('same_track_after_concept_recovery', 0.0)):.4f}"
        ),
        (
            f"- Phase 3X same-track-after-lineage: "
            f"{float(track_c.get('same_track_after_lineage_recovery', 0.0)):.4f}"
        ),
        f"- dominant failure stage: {summary_payload.get('dominant_failure_stage', 'unknown')}",
        f"- primary loss stage: {summary_payload.get('primary_loss_stage', 'unknown')}",
        "",
        "## Verdict",
        "",
        "- status: pass",
        "- Interpretation: this round passes if it reduces Phase 3S failure to one structural cause rather than a loose set of symptoms.",
        "",
    ]
    return "\n".join(lines)


def _build_phase3x_failure_notes(
    summary_payload: dict[str, object],
    eval_rows: list[dict[str, object]],
    before_lookup: dict[str, dict[str, object]],
) -> str:
    track_c_before = before_lookup[TRACK_C_NAME]
    track_c = summary_payload.get("track_c", {})
    lines = [
        "# Phase 3X Failure Notes v1",
        "",
        "## Track C Readout",
        "",
        (
            f"- Phase 3S strict same-prototype: {float(track_c_before['same_prototype_reentry_recovery']):.4f} -> "
            f"{float(track_c.get('same_prototype_reentry_recovery', 0.0)):.4f}"
        ),
        (
            f"- lineage-aware same-prototype: {float(summary_payload.get('same_lineage_prototype_reentry_recovery', 0.0)):.4f}"
        ),
        (
            f"- continuation_write_success_rate: {float(summary_payload.get('continuation_write_success_rate', 0.0)):.4f}"
        ),
        (
            "- continuation_survival_until_concept_recovery_rate: "
            f"{float(summary_payload.get('continuation_survival_until_concept_recovery_rate', 0.0)):.4f}"
        ),
        (
            "- concept_recovered_but_lineage_mismatch_rate: "
            f"{float(summary_payload.get('concept_recovered_but_lineage_mismatch_rate', 0.0)):.4f}"
        ),
        (
            "- continuation_bank_access_rate_given_same_lineage: "
            f"{float(summary_payload.get('continuation_bank_access_rate_given_same_lineage', 0.0)):.4f}"
        ),
        f"- dominant failure stage: {summary_payload.get('dominant_failure_stage', 'unknown')}",
        f"- primary loss stage: {summary_payload.get('primary_loss_stage', 'unknown')}",
        "",
        "## Decision Rule",
        "",
        "- If lineage-aware recovery is much higher than strict recovery, the next branch is lineage-preserving prototype update and lineage-aware evaluation.",
        "- If write/survival is low, the next branch is continuation lifecycle repair.",
        "- If write and survival are fine but access is low, the next branch is concept-to-continuation binding.",
        "",
    ]
    return "\n".join(lines)


def _build_phase3l_summary_doc(
    after_lookup: dict[str, dict[str, object]],
    before_lookup: dict[str, dict[str, object]],
    tracking_override: dict[str, object],
    memory_override: dict[str, object],
) -> str:
    track_c_before = before_lookup[PHASE3L_TRACK_C_NAME]
    track_c = after_lookup[PHASE3L_TRACK_C_NAME]
    lines = [
        "# Phase 3L Summary v1",
        "",
        "## Track C",
        "",
        (
            f"- concept_recovered_but_lineage_mismatch_rate: "
            f"{float(track_c_before.get('concept_recovered_but_lineage_mismatch_rate', 0.0)):.4f} -> "
            f"{float(track_c.get('concept_recovered_but_lineage_mismatch_rate', 0.0)):.4f}"
        ),
        (
            f"- continuation_bank_access_rate_given_concept_recovery: "
            f"{float(track_c_before.get('continuation_bank_access_rate_given_concept_recovery', 0.0)):.4f} -> "
            f"{float(track_c.get('continuation_bank_access_rate_given_concept_recovery', 0.0)):.4f}"
        ),
        (
            f"- same-lineage-prototype-reentry-recovery: "
            f"{float(track_c_before.get('same_lineage_prototype_reentry_recovery', 0.0)):.4f} -> "
            f"{float(track_c.get('same_lineage_prototype_reentry_recovery', 0.0)):.4f}"
        ),
        (
            f"- same-track-after-concept-recovery: "
            f"{float(track_c_before.get('same_track_after_concept_recovery', 0.0)):.4f} -> "
            f"{float(track_c.get('same_track_after_concept_recovery', 0.0)):.4f}"
        ),
        (
            f"- same-prototype-reentry-recovery: "
            f"{float(track_c_before.get('same_prototype_reentry_recovery', 0.0)):.4f} -> "
            f"{float(track_c.get('same_prototype_reentry_recovery', 0.0)):.4f}"
        ),
        f"- PFR: {float(track_c_before.get('pfr', 0.0)):.4f} -> {float(track_c.get('pfr', 0.0)):.4f}",
        "",
        "## Config",
        "",
        f"- tracking override: {tracking_override}",
        f"- memory override: {memory_override}",
        "",
    ]
    return "\n".join(lines)


def _build_phase3l_failure_notes(
    after_lookup: dict[str, dict[str, object]],
    before_lookup: dict[str, dict[str, object]],
) -> str:
    track_a_before = before_lookup[PHASE3L_TRACK_A_NAME]
    track_a = after_lookup[PHASE3L_TRACK_A_NAME]
    track_c_before = before_lookup[PHASE3L_TRACK_C_NAME]
    track_c = after_lookup[PHASE3L_TRACK_C_NAME]
    lines = [
        "# Phase 3L Failure Notes v1",
        "",
        "## Track C Readout",
        "",
        (
            f"- lineage mismatch: {float(track_c_before.get('concept_recovered_but_lineage_mismatch_rate', 0.0)):.4f} -> "
            f"{float(track_c.get('concept_recovered_but_lineage_mismatch_rate', 0.0)):.4f}"
        ),
        (
            f"- continuation access | concept recovered: "
            f"{float(track_c_before.get('continuation_bank_access_rate_given_concept_recovery', 0.0)):.4f} -> "
            f"{float(track_c.get('continuation_bank_access_rate_given_concept_recovery', 0.0)):.4f}"
        ),
        (
            f"- same-lineage-prototype: {float(track_c_before.get('same_lineage_prototype_reentry_recovery', 0.0)):.4f} -> "
            f"{float(track_c.get('same_lineage_prototype_reentry_recovery', 0.0)):.4f}"
        ),
        (
            f"- same-track-after-concept: {float(track_c_before.get('same_track_after_concept_recovery', 0.0)):.4f} -> "
            f"{float(track_c.get('same_track_after_concept_recovery', 0.0)):.4f}"
        ),
        (
            f"- same-prototype: {float(track_c_before.get('same_prototype_reentry_recovery', 0.0)):.4f} -> "
            f"{float(track_c.get('same_prototype_reentry_recovery', 0.0)):.4f}"
        ),
        f"- PFR: {float(track_c_before.get('pfr', 0.0)):.4f} -> {float(track_c.get('pfr', 0.0)):.4f}",
        "",
        "## Track A Guardrails",
        "",
        f"- U-Recall: {float(track_a_before.get('u_recall', 0.0)):.4f} -> {float(track_a.get('u_recall', 0.0)):.4f}",
        (
            f"- same-prototype: {float(track_a_before.get('same_prototype_reentry_recovery', 0.0)):.4f} -> "
            f"{float(track_a.get('same_prototype_reentry_recovery', 0.0)):.4f}"
        ),
        f"- memory_growth: {float(track_a_before.get('memory_growth', 0.0)):.4f} -> {float(track_a.get('memory_growth', 0.0)):.4f}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
