from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ext1_utils import (
    build_external_event_ledger,
    dataset_inventory,
    oracle_memory_results,
    read_csv,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXT-1 oracle-proposal memory-only benchmark.")
    p.add_argument("--output-dir", default="results/ext1")
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--max-sequences", type=int, default=0, help="0 means all sequences.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    max_sequences = args.max_sequences or None
    _, adapters = dataset_inventory(max_sequences=max_sequences)
    ledger_path = out / f"stage_EXT1_external_event_ledger_{args.artifact_version}.csv"
    ledger = read_csv(ledger_path)
    if not ledger:
        ledger = build_external_event_ledger(adapters, max_sequences=max_sequences)
    results = oracle_memory_results(adapters, ledger)
    write_csv(out / f"stage_EXT1_oracle_proposal_memory_results_{args.artifact_version}.csv", results)
    compact = {
        "stage": "EXT-1-oracle-proposal-memory",
        "proposal_mode": "oracle_gt_box_memory_only",
        "num_results": len(results),
        "num_methods": len({r["method_name"] for r in results}),
        "num_events": len({r["event_id"] for r in results}),
    }
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()

