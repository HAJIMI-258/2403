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
    difficulty_rows,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXT-1 external dataset inventory and event mining.")
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
    ledger = build_external_event_ledger(adapters, max_sequences=max_sequences)
    difficulty = difficulty_rows(adapters, ledger)
    write_csv(out / f"stage_EXT1_dataset_inventory_{args.artifact_version}.csv", inventory)
    write_csv(out / f"stage_EXT1_external_event_ledger_{args.artifact_version}.csv", ledger)
    write_csv(out / f"stage_EXT1_event_difficulty_audit_{args.artifact_version}.csv", difficulty)
    valid_events = [r for r in ledger if str(r.get("event_usable")) == "1"]
    compact = {
        "stage": "EXT-1-event-mining",
        "usable_datasets": [r["dataset_name"] for r in inventory if int(r.get("usable_for_memory_eval", 0)) == 1],
        "valid_event_count": len(valid_events),
        "num_sequences": sum(int(r.get("num_sequences", 0)) for r in inventory),
        "num_frames_sampled": sum(int(r.get("num_frames_sampled", 0)) for r in inventory),
        "next_step": "run oracle-proposal memory benchmark" if valid_events else "download/connect full benchmark with reentry/occlusion events",
    }
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()

