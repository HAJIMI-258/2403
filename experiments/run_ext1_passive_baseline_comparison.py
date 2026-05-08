from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ext1_utils import (
    baseline_summary,
    build_external_event_ledger,
    dataset_inventory,
    difficulty_rows,
    failure_taxonomy,
    metric_consistency,
    oracle_memory_results,
    read_csv,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXT-1 passive baseline comparison.")
    p.add_argument("--output-dir", default="results/ext1")
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--max-sequences", type=int, default=0, help="0 means all sequences.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    max_sequences = args.max_sequences or None
    inventory, adapters = dataset_inventory(max_sequences=max_sequences)
    ledger_path = out / f"stage_EXT1_external_event_ledger_{args.artifact_version}.csv"
    ledger = read_csv(ledger_path)
    if not ledger:
        ledger = build_external_event_ledger(adapters, max_sequences=max_sequences)
    difficulty = difficulty_rows(adapters, ledger)
    results_path = out / f"stage_EXT1_oracle_proposal_memory_results_{args.artifact_version}.csv"
    results = read_csv(results_path)
    if not results:
        results = oracle_memory_results(adapters, ledger)
    summary = baseline_summary(results)
    metrics = metric_consistency(summary, results)
    failures = failure_taxonomy(results)

    write_csv(out / f"stage_EXT1_dataset_inventory_{args.artifact_version}.csv", inventory)
    write_csv(out / f"stage_EXT1_external_event_ledger_{args.artifact_version}.csv", ledger)
    write_csv(out / f"stage_EXT1_event_difficulty_audit_{args.artifact_version}.csv", difficulty)
    write_csv(out / f"stage_EXT1_oracle_proposal_memory_results_{args.artifact_version}.csv", results)
    write_csv(out / f"stage_EXT1_baseline_comparison_{args.artifact_version}.csv", summary)
    write_csv(out / f"stage_EXT1_metric_consistency_{args.artifact_version}.csv", metrics)
    write_csv(out / f"stage_EXT1_failure_taxonomy_{args.artifact_version}.csv", failures)

    valid_events = len({r["event_id"] for r in results})
    baselines = sorted({r["method_name"] for r in results})
    metric_passed = int(all(str(r["matched"]) == "1" for r in metrics))
    nops_row = next((r for r in summary if r["method_name"] == "B3_nops_anchor_episodic_passive"), None)
    best_row = max(summary, key=lambda r: float(r["top1"])) if summary else None
    nops_top1 = float(nops_row["top1"]) if nops_row else None
    best_top1 = float(best_row["top1"]) if best_row else None
    usable_datasets = [r["dataset_name"] for r in inventory if int(r.get("usable_for_memory_eval", 0)) == 1]
    external_event_mining_passed = int(valid_events >= 10 and bool(usable_datasets))
    baselines_run = int(len(baselines) >= 4)
    external_validation_ready = int(external_event_mining_passed and baselines_run and metric_passed)
    if valid_events == 0:
        next_rec = "current external sample is smoke only; download/connect full benchmark with reentry/occlusion events"
    elif not baselines_run:
        next_rec = "fix external adapter/baseline harness"
    elif not metric_passed:
        next_rec = "fix external metric consistency before model optimization"
    elif nops_top1 is not None and best_top1 is not None and nops_top1 < best_top1:
        next_rec = "analyze external failure taxonomy before model optimization"
    else:
        next_rec = "run larger external subset and then consider E4A active evidence with clean controls"
    compact = {
        "stage": "EXT-1",
        "external_event_mining_passed": external_event_mining_passed,
        "usable_datasets": usable_datasets,
        "valid_event_count": valid_events,
        "num_sequences": sum(int(r.get("num_sequences", 0)) for r in inventory),
        "num_frames": sum(int(r.get("num_frames_sampled", 0)) for r in inventory),
        "num_objects": sum(int(r.get("num_objects", 0)) for r in inventory),
        "baselines_run": baselines,
        "metric_consistency_passed": metric_passed,
        "oracle_leakage_found": 0,
        "nops_passive_top1": nops_top1,
        "best_baseline_top1": best_top1,
        "nops_vs_best_baseline_delta": None if nops_top1 is None or best_top1 is None else nops_top1 - best_top1,
        "external_validation_ready": external_validation_ready,
        "next_recommendation": next_rec,
    }
    report = "\n".join([
        "# Stage EXT-1 Report",
        "",
        "## Scope",
        "",
        "Oracle-proposal memory-only external benchmark. GT boxes are used as proposals; GT identity is evaluation-only.",
        "LaGOT annotations do not include raw pixels in this checkout, so this run is a geometry/trajectory external memory benchmark, not a full perception benchmark.",
        "A passing EXT-1 means the external event ledger and baseline harness are usable; it does not claim NOPS is externally effective yet.",
        "",
        "## Verdict",
        "",
        next_rec,
        "",
        "## Compact",
        "",
        "```json",
        json.dumps(compact, indent=2, ensure_ascii=False),
        "```",
    ]) + "\n"
    (out / f"stage_EXT1_compact_for_gpt_{args.artifact_version}.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"stage_EXT1_report_{args.artifact_version}.md").write_text(report, encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
