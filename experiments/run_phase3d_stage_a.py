"""Run Phase 3D Stage A: decoupled recovery attach wiring + preview artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3d_utils import (
    build_phase3d_event_audit_rows,
    default_phase3d_stagea_memory_override,
    default_phase3d_stagea_tracking_override,
    evaluate_phase3d_stagea_bundle,
    save_run_artifacts,
    summarize_phase3d_stagea,
    write_phase3d_audit_summary,
    write_phase3d_design_notes,
    write_visual_manifest,
)
from experiments.phase3r_utils import extract_reentry_events, write_csv
from experiments.phase3s_utils import TRACK_A_NAME, TRACK_C_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3D Stage A structure validation.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--output-dir", default="results/phase3d")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_override = default_phase3d_stagea_tracking_override()
    memory_override = default_phase3d_stagea_memory_override()
    bundle = evaluate_phase3d_stagea_bundle(
        args.config,
        seed=args.seed,
        scenario_names=[TRACK_A_NAME, TRACK_C_NAME],
    )
    summary_lookup = {str(row["scenario_name"]): row for row in bundle["rows"]}

    all_audit_rows: list[dict[str, object]] = []
    all_visuals: list[dict[str, str]] = []
    run_summary_rows: list[dict[str, object]] = []

    for run in bundle["runs"]:
        scenario_name = str(run["scenario_name"])
        summary_row = dict(summary_lookup[scenario_name])
        run_summary_rows.append(summary_row)
        event_rows, _ = extract_reentry_events(scenario_name, run["sequence"], run["result"])
        audit_rows = build_phase3d_event_audit_rows(event_rows)
        all_audit_rows.extend(audit_rows)
        run_output_dir = output_dir / f"stage_a_{scenario_name}"
        all_visuals.extend(
            save_run_artifacts(
                run=run,
                summary_row=summary_row,
                audit_rows=audit_rows,
                output_dir=run_output_dir,
                config_path=args.config,
                tracking_override=tracking_override,
                memory_override=memory_override,
            )
        )

    summary = summarize_phase3d_stagea(all_audit_rows, bundle)

    write_phase3d_design_notes(output_dir / "phase3d_design_notes.md")
    write_phase3d_audit_summary(output_dir / "phase3d_audit_summary.md", summary)
    write_csv(output_dir / "phase3d_event_audit.csv", all_audit_rows)
    write_csv(output_dir / "phase3d_stagea_bundle_summary.csv", run_summary_rows)
    write_visual_manifest(output_dir / "visual_manifest.md", all_visuals)
    (output_dir / "phase3d_stagea_metrics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"saved_design_notes={output_dir / 'phase3d_design_notes.md'}")
    print(f"saved_audit_summary={output_dir / 'phase3d_audit_summary.md'}")
    print(f"saved_event_audit={output_dir / 'phase3d_event_audit.csv'}")
    print(f"saved_bundle_summary={output_dir / 'phase3d_stagea_bundle_summary.csv'}")
    print(f"saved_visual_manifest={output_dir / 'visual_manifest.md'}")
    print(f"saved_metrics_json={output_dir / 'phase3d_stagea_metrics_summary.json'}")


if __name__ == "__main__":
    main()
