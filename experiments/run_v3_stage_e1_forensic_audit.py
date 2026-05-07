from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3r_utils import REENTRY_WINDOW, evaluate_phase3_scenarios, extract_reentry_events, write_csv
from experiments.v3_utils import (
    TRACK_A_NAME,
    TRACK_C_NAME,
    build_event_ledger_entries,
    build_stage_e1_event_rows,
    build_stage_e1_failure_rows,
    render_stage_e1_report,
    summarize_stage_e1,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 Stage E1 baseline forensic audit.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--output-dir", default="results/v3_e1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-version", default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_runs = evaluate_phase3_scenarios(
        args.config,
        scenario_names=[TRACK_A_NAME, TRACK_C_NAME],
        tracking_override=None,
        memory_override=None,
        collect_frames=True,
        frame_record_mode="lite",
        seed=args.seed,
    )

    event_ledger_entries: list[dict[str, Any]] = []
    event_audit_rows: list[dict[str, Any]] = []
    scenario_summary_rows: list[dict[str, Any]] = []

    for run in scenario_runs:
        scenario_name = str(run["scenario_name"])
        result = run["result"]
        events, _frame_logs = extract_reentry_events(
            scenario_name,
            run["sequence"],
            result,
            recovery_window=REENTRY_WINDOW,
        )
        event_ledger_entries.extend(
            build_event_ledger_entries(
                scenario_name,
                events,
                recovery_window=REENTRY_WINDOW,
            )
        )
        event_audit_rows.extend(
            build_stage_e1_event_rows(
                scenario_name,
                events,
                recovery_window=REENTRY_WINDOW,
            )
        )
        scenario_summary_rows.append(
            {
                "scenario_name": scenario_name,
                "u_recall": float(result.summary.u_recall),
                "pfr": float(result.summary.pfr),
                "track_idsw": int(result.primary_monitoring["track_idsw"]),
                "memory_growth": float(result.summary.memory_growth),
                "purity": float(result.summary.purity),
            }
        )

    failure_rows = build_stage_e1_failure_rows(event_audit_rows)
    summary_payload = summarize_stage_e1(scenario_summary_rows, event_audit_rows)
    meta_payload = {
        "stage_tag": "E1_baseline_forensic_audit",
        "config_path": str(args.config),
        "config_hash": _config_hash(args.config),
        "git_commit": _git_commit(),
        "seed": int(args.seed),
        "artifact_version": str(args.artifact_version),
        "recovery_window": int(REENTRY_WINDOW),
        "scenario_names": [TRACK_A_NAME, TRACK_C_NAME],
    }

    ledger_path = output_dir / "event_ledger.yaml"
    event_audit_path = output_dir / f"stage_E1_event_audit_{args.artifact_version}.csv"
    failure_path = output_dir / f"stage_E1_failure_slicing_{args.artifact_version}.csv"
    summary_csv_path = output_dir / f"stage_E1_summary_{args.artifact_version}.csv"
    summary_json_path = output_dir / f"stage_E1_summary_{args.artifact_version}.json"
    report_path = output_dir / f"stage_E1_report_{args.artifact_version}.md"

    write_yaml(
        ledger_path,
        {
            "meta": meta_payload,
            "events": event_ledger_entries,
        },
    )
    write_csv(event_audit_path, event_audit_rows)
    write_csv(failure_path, failure_rows)
    write_csv(summary_csv_path, scenario_summary_rows)
    summary_json_path.write_text(
        json.dumps({"meta": meta_payload, "summary": summary_payload}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path.write_text(render_stage_e1_report(summary_payload, event_audit_rows), encoding="utf-8")

    print(f"saved_ledger={ledger_path}")
    print(f"saved_event_audit={event_audit_path}")
    print(f"saved_failure_slicing={failure_path}")
    print(f"saved_summary_json={summary_json_path}")
    print(f"saved_report={report_path}")
    print(json.dumps(summary_payload["overall"], ensure_ascii=False))


def _config_hash(config_path: str | Path) -> str:
    content = Path(config_path).read_bytes()
    return hashlib.sha1(content).hexdigest()[:12]


def _git_commit() -> str:
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10)
        return output.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
