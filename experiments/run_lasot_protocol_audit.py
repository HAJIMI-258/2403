"""Audit LaSOT visibility annotations for long-gap re-entry events."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.external.lasot_adapter import LaSOTAdapter  # noqa: E402
from nops_owr.evaluation.reentry_audit import gap_bucket  # noqa: E402


INVENTORY_FIELDS = [
    "sequence_id",
    "category",
    "sequence_dir",
    "frame_count",
    "has_img_dir",
    "has_groundtruth",
    "has_full_occlusion",
    "has_out_of_view",
    "visible_frame_count",
    "invisible_frame_count",
    "reentry_event_count",
    "long_gap_reentry_event_count",
    "benchmark_usable",
]

EVENT_FIELDS = [
    "dataset_name",
    "sequence_id",
    "category",
    "instance_id",
    "disappear_frame",
    "reappear_frame",
    "gap_length",
    "gap_bucket",
    "event_type",
    "usable",
    "not_usable_reason",
]


def run_audit(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_protocol_audit",
    max_sequences: int = 50,
    min_gap: int = 8,
    category_filter: str = "",
    sequence_filter: str = "",
) -> dict[str, Any]:
    root_path = Path(root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    adapter = LaSOTAdapter(root_path)
    sequences = _filter_sequences(
        list(adapter.iter_sequences()),
        category_filter=category_filter,
        sequence_filter=sequence_filter,
    )[: int(max_sequences)]

    inventory_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    category_event_counts: Counter[str] = Counter()
    gap_counts: Counter[str] = Counter()
    total_frames = 0
    usable_sequences = 0

    for sequence_id in sequences:
        category = _category(sequence_id)
        seq_dir = adapter._sequence_dir(sequence_id)  # noqa: SLF001 - audit script needs path diagnostics.
        frame_count = len(list((seq_dir / "img").glob("*.jpg"))) if seq_dir else 0
        visibility = adapter._visibility_flags(seq_dir, frame_count) if seq_dir else []  # noqa: SLF001
        events = list(adapter.derive_events(sequence_id))
        long_events = [event for event in events if int(event.gap_length) >= int(min_gap)]
        total_frames += frame_count
        usable = bool(long_events)
        usable_sequences += int(usable)
        inventory_rows.append(
            {
                "sequence_id": sequence_id,
                "category": category,
                "sequence_dir": "" if seq_dir is None else str(seq_dir),
                "frame_count": frame_count,
                "has_img_dir": int(seq_dir is not None and (seq_dir / "img").exists()),
                "has_groundtruth": int(seq_dir is not None and (seq_dir / "groundtruth.txt").exists()),
                "has_full_occlusion": int(seq_dir is not None and (seq_dir / "full_occlusion.txt").exists()),
                "has_out_of_view": int(seq_dir is not None and (seq_dir / "out_of_view.txt").exists()),
                "visible_frame_count": sum(int(v) for v in visibility),
                "invisible_frame_count": sum(int(not v) for v in visibility),
                "reentry_event_count": len(events),
                "long_gap_reentry_event_count": len(long_events),
                "benchmark_usable": int(usable),
            }
        )
        for event in events:
            event_gap_bucket = gap_bucket(int(event.gap_length))
            is_usable = int(event.gap_length) >= int(min_gap)
            if is_usable:
                category_event_counts[category] += 1
                gap_counts[event_gap_bucket] += 1
            event_rows.append(
                {
                    "dataset_name": event.dataset_name,
                    "sequence_id": event.sequence_id,
                    "category": category,
                    "instance_id": event.instance_id,
                    "disappear_frame": event.disappear_frame,
                    "reappear_frame": event.reappear_frame,
                    "gap_length": event.gap_length,
                    "gap_bucket": event_gap_bucket,
                    "event_type": event.event_type,
                    "usable": int(is_usable),
                    "not_usable_reason": "" if is_usable else f"gap<{min_gap}",
                }
            )

    summary = {
        "dataset_name": "lasot",
        "root": str(root_path),
        "sequence_count_scanned": len(sequences),
        "usable_sequence_count": usable_sequences,
        "total_frame_count": total_frames,
        "total_reentry_event_count": len(event_rows),
        "total_long_gap_reentry_event_count": sum(int(row["usable"]) for row in event_rows),
        "gap_bucket_counts": dict(gap_counts),
        "category_event_counts": dict(category_event_counts),
        "benchmark_valid": bool(sum(int(row["usable"]) for row in event_rows) > 0),
        "benchmark_invalid_reason": "" if event_rows else "no_sequences_or_events",
    }
    if not summary["benchmark_valid"]:
        summary["benchmark_invalid_reason"] = "no_long_gap_reentry_events"

    _write_csv(output_path / "lasot_sequence_inventory.csv", inventory_rows, INVENTORY_FIELDS)
    _write_csv(output_path / "lasot_reentry_events.csv", event_rows, EVENT_FIELDS)
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "report.md").write_text(_report(summary, inventory_rows), encoding="utf-8")
    return summary


def _filter_sequences(sequences: list[str], *, category_filter: str, sequence_filter: str) -> list[str]:
    categories = {item.strip() for item in category_filter.split(",") if item.strip()}
    seq_filter = sequence_filter.strip()
    output = []
    for sequence_id in sorted(sequences):
        if categories and _category(sequence_id) not in categories:
            continue
        if seq_filter and seq_filter not in sequence_id:
            continue
        output.append(sequence_id)
    return output


def _category(sequence_id: str) -> str:
    return sequence_id.split("-")[0]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _report(summary: dict[str, Any], inventory_rows: list[dict[str, Any]]) -> str:
    usable = [row for row in inventory_rows if int(row["benchmark_usable"])]
    recommendations = "\n".join(
        f"- {row['sequence_id']}: {row['long_gap_reentry_event_count']} long-gap events"
        for row in sorted(usable, key=lambda item: int(item["long_gap_reentry_event_count"]), reverse=True)[:10]
    ) or "- no usable long-gap sequences in scanned subset"
    return (
        "# LaSOT Re-entry Protocol Audit\n\n"
        f"- root: `{summary['root']}`\n"
        f"- sequence_count_scanned: {summary['sequence_count_scanned']}\n"
        f"- usable_sequence_count: {summary['usable_sequence_count']}\n"
        f"- total_long_gap_reentry_event_count: {summary['total_long_gap_reentry_event_count']}\n"
        f"- gap_bucket_counts: `{summary['gap_bucket_counts']}`\n"
        f"- category_event_counts: `{summary['category_event_counts']}`\n\n"
        "## Recommended Sequences\n\n"
        + recommendations
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_protocol_audit")
    parser.add_argument("--max-sequences", type=int, default=50)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--sequence-filter", default="")
    args = parser.parse_args()
    summary = run_audit(
        root=args.root,
        output_dir=args.output_dir,
        max_sequences=args.max_sequences,
        min_gap=args.min_gap,
        category_filter=args.category_filter,
        sequence_filter=args.sequence_filter,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
