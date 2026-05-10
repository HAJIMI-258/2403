from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.synth_stream import SynthDatasetConfig, SyntheticStreamGenerator
from experiments.run_v3_stage_e4a_active_evidence_acquisition import crop_descriptor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1K windowed render/cache descriptor smoke.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--ledger", default="results/core1f/stage_CORE1F_dense_event_ledger_v1.csv")
    p.add_argument("--window-plan", default="results/core1j/stage_CORE1J_window_plan_v1.csv")
    p.add_argument("--output-dir", default="results/core1k")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sequences", type=int, default=3)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def descriptor_to_str(desc: np.ndarray) -> str:
    return "|".join(f"{float(v):.6f}" for v in np.asarray(desc, dtype=np.float32).reshape(-1))


def parse_descriptor(text: str) -> np.ndarray:
    return np.asarray([float(x) for x in text.split("|") if x != ""], dtype=np.float32)


def cosine01(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    m = min(aa.size, bb.size)
    if m == 0:
        return 0.0
    aa = aa[:m]
    bb = bb[:m]
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(aa, bb) / (na * nb), -1.0, 1.0) * 0.5 + 0.5)


def box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def ensure_min_crop_box(box: tuple[int, int, int, int], frame_shape: tuple[int, ...], min_size: int = 3) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    if x2 - x1 >= min_size and y2 - y1 >= min_size:
        return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = max(1, min_size // 2)
    nx1 = max(0, cx - half)
    ny1 = max(0, cy - half)
    nx2 = min(w, nx1 + min_size)
    ny2 = min(h, ny1 + min_size)
    nx1 = max(0, nx2 - min_size)
    ny1 = max(0, ny2 - min_size)
    return (nx1, ny1, nx2, ny2)


def load_config(path: Path, seed: int) -> SynthDatasetConfig:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    cfg = SynthDatasetConfig.from_dict(payload)
    # Keep the CLI seed explicit; the generator itself receives it.
    _ = seed
    return cfg


def select_windows(rows: list[dict[str, str]], max_sequences: int) -> list[dict[str, str]]:
    selected_sequences: list[int] = []
    selected: list[dict[str, str]] = []
    for row in rows:
        seq_id = int(row["sequence_id"])
        if seq_id not in selected_sequences:
            if len(selected_sequences) >= max_sequences:
                continue
            selected_sequences.append(seq_id)
        if seq_id in selected_sequences:
            selected.append(row)
    return selected


def aggregate_event_descriptors(trace_rows: list[dict[str, Any]]) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for row in trace_rows:
        if not row.get("descriptor_available"):
            continue
        grouped[(row["event_id"], row["window_kind"])].append(parse_descriptor(row["descriptor"]))
    return {key: np.mean(np.stack(vals, axis=0), axis=0).astype(np.float32) for key, vals in grouped.items() if vals}


def compute_pair_margins(
    selected_window_rows: list[dict[str, str]],
    aggregated: dict[tuple[str, str], np.ndarray],
) -> list[dict[str, Any]]:
    event_to_sequence = {row["event_id"]: int(row["sequence_id"]) for row in selected_window_rows}
    events = sorted({row["event_id"] for row in selected_window_rows})
    disappear = {event_id: aggregated.get((event_id, "disappear")) for event_id in events}
    reappear = {event_id: aggregated.get((event_id, "reappear")) for event_id in events}
    rows: list[dict[str, Any]] = []
    for event_id in events:
        query = reappear.get(event_id)
        target = disappear.get(event_id)
        if query is None or target is None:
            rows.append(
                {
                    "event_id": event_id,
                    "sequence_id": event_to_sequence[event_id],
                    "target_score": "",
                    "wrong_top_score": "",
                    "same_event_margin": "",
                    "same_event_top1": 0,
                    "candidate_count": 0,
                    "failure_reason": "missing_disappear_or_reappear_descriptor",
                }
            )
            continue

        same_sequence = [
            other
            for other in events
            if other != event_id and event_to_sequence.get(other) == event_to_sequence[event_id] and disappear.get(other) is not None
        ]
        candidate_events = same_sequence or [other for other in events if other != event_id and disappear.get(other) is not None]
        scores = [(other, cosine01(query, disappear[other])) for other in candidate_events]
        target_score = cosine01(query, target)
        wrong_top_event = ""
        wrong_top_score = -1.0
        if scores:
            wrong_top_event, wrong_top_score = max(scores, key=lambda x: x[1])
        margin = target_score - wrong_top_score if scores else target_score
        rows.append(
            {
                "event_id": event_id,
                "sequence_id": event_to_sequence[event_id],
                "target_score": target_score,
                "wrong_top_event": wrong_top_event,
                "wrong_top_score": wrong_top_score if scores else "",
                "same_event_margin": margin,
                "same_event_top1": int(not scores or target_score > wrong_top_score),
                "candidate_count": len(scores) + 1,
                "failure_reason": "none" if (not scores or target_score > wrong_top_score) else "wrong_window_descriptor_closer",
            }
        )
    return rows


def extract_descriptor_rows(
    *,
    sequence_id: int,
    event_id: str,
    window_kind: str,
    instance_id: int,
    frames_by_idx: dict[int, Any],
    start_frame: int,
    end_frame: int,
    descriptor_source: str,
) -> tuple[list[dict[str, Any]], int, int, int]:
    rows: list[dict[str, Any]] = []
    available = 0
    visible = 0
    descriptor_count = 0
    for frame_idx in range(start_frame, end_frame + 1):
        frame_sample = frames_by_idx.get(frame_idx)
        if frame_sample is None:
            continue
        available += 1
        if instance_id not in frame_sample.instance_ids:
            continue
        inst_idx = frame_sample.instance_ids.index(instance_id)
        if frame_sample.visibility_flags and not frame_sample.visibility_flags[inst_idx]:
            continue
        box = tuple(int(v) for v in frame_sample.boxes[inst_idx])
        if box_area(box) <= 0:
            continue
        crop_box = ensure_min_crop_box(box, frame_sample.frame.shape)
        visible += 1
        desc_payload = crop_descriptor(frame_sample.frame, None, crop_box, box)
        desc = np.asarray(desc_payload["descriptor"], dtype=np.float32)
        descriptor_count += 1
        rows.append(
            {
                "sequence_id": sequence_id,
                "event_id": event_id,
                "window_kind": window_kind,
                "descriptor_source": descriptor_source,
                "frame_idx": frame_idx,
                "instance_id_eval_only": instance_id,
                "box": "|".join(str(v) for v in box),
                "crop_box": "|".join(str(v) for v in crop_box),
                "descriptor_available": 1,
                "descriptor_norm": float(np.linalg.norm(desc)),
                "descriptor_entropy_proxy": float(np.std(desc)),
                "edge_density": desc_payload["edge_density"],
                "objectness_crop_mean": desc_payload["objectness_crop_mean"],
                "descriptor": descriptor_to_str(desc),
            }
        )
    return rows, available, visible, descriptor_count


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.perf_counter()
    cfg = load_config(Path(args.config), args.seed)
    generator = SyntheticStreamGenerator(cfg, seed=args.seed)

    ledger_rows = read_csv(Path(args.ledger))
    event_to_instance = {row["event_id"]: int(row["instance_id_eval_only"]) for row in ledger_rows}
    window_rows = select_windows(read_csv(Path(args.window_plan)), args.max_sequences)
    sequence_to_windows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in window_rows:
        sequence_to_windows[int(row["sequence_id"])].append(row)

    render_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for sequence_id, seq_windows in sorted(sequence_to_windows.items()):
        seq_start = time.perf_counter()
        sequence = generator.generate_sequence(sequence_id)
        seq_elapsed = time.perf_counter() - seq_start
        frames_by_idx = {frame.frame_index: frame for frame in sequence.frames}
        runtime_rows.append(
            {
                "sequence_id": sequence_id,
                "window_count": len(seq_windows),
                "generation_time_sec": seq_elapsed,
                "sequence_length": len(sequence.frames),
                "runtime_mode": "full_sequence_generate_selected_windows_only",
            }
        )

        for window in seq_windows:
            event_id = window["event_id"]
            instance_id = event_to_instance.get(event_id)
            start_frame = int(window["start_frame"])
            end_frame = int(window["end_frame"])
            extracted, available, visible, descriptor_count = extract_descriptor_rows(
                sequence_id=sequence_id,
                event_id=event_id,
                window_kind=window["window_kind"],
                instance_id=instance_id,
                frames_by_idx=frames_by_idx,
                start_frame=start_frame,
                end_frame=end_frame,
                descriptor_source="planned_window",
            )
            trace_rows.extend(extracted)
            fallback_used = 0
            if descriptor_count == 0 and window["window_kind"] == "disappear":
                fallback_start = max(0, start_frame - 80)
                fallback_end = max(fallback_start, start_frame - 1)
                extracted, fb_available, fb_visible, fb_descriptor_count = extract_descriptor_rows(
                    sequence_id=sequence_id,
                    event_id=event_id,
                    window_kind=window["window_kind"],
                    instance_id=instance_id,
                    frames_by_idx=frames_by_idx,
                    start_frame=fallback_start,
                    end_frame=fallback_end,
                    descriptor_source="fallback_pre_disappear_visible_search",
                )
                if fb_descriptor_count > 0:
                    trace_rows.extend(extracted)
                    available += fb_available
                    visible += fb_visible
                    descriptor_count += fb_descriptor_count
                    fallback_used = 1

            render_rows.append(
                {
                    "sequence_id": sequence_id,
                    "event_id": event_id,
                    "window_kind": window["window_kind"],
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "planned_frame_count": int(window["frame_count"]),
                    "available_frame_count": available,
                    "target_visible_frame_count": visible,
                    "descriptor_frame_count": descriptor_count,
                    "descriptor_available": int(descriptor_count > 0),
                    "fallback_used": fallback_used,
                }
            )

    aggregated = aggregate_event_descriptors(trace_rows)
    margin_rows = compute_pair_margins(window_rows, aggregated)

    total_windows = len(window_rows)
    available_windows = sum(int(r["descriptor_available"]) for r in render_rows)
    margins = [float(r["same_event_margin"]) for r in margin_rows if r["same_event_margin"] != ""]
    top1_values = [int(r["same_event_top1"]) for r in margin_rows if int(r["candidate_count"]) > 0]
    processed_events = len([r for r in margin_rows if r["same_event_margin"] != ""])
    mean_margin = float(np.mean(margins)) if margins else 0.0
    top1_rate = float(np.mean(top1_values)) if top1_values else 0.0
    random_top1 = float(np.mean([1.0 / max(int(r["candidate_count"]), 1) for r in margin_rows if int(r["candidate_count"]) > 0])) if margin_rows else 0.0
    total_elapsed = time.perf_counter() - start_total

    compact = {
        "stage": "CORE-1K",
        "artifact_version": args.artifact_version,
        "config": args.config,
        "window_plan": args.window_plan,
        "selected_sequence_count": len(sequence_to_windows),
        "selected_window_count": total_windows,
        "processed_event_count": processed_events,
        "descriptor_available_rate": available_windows / max(total_windows, 1),
        "mean_same_event_margin": mean_margin,
        "same_event_top1_rate": top1_rate,
        "random_top1_rate": random_top1,
        "runtime_sec": total_elapsed,
        "fallback_window_count": sum(int(r.get("fallback_used", 0)) for r in render_rows),
        "window_cache_smoke_passed": int(total_windows > 0 and available_windows / max(total_windows, 1) >= 0.8 and mean_margin > 0 and top1_rate > random_top1),
        "tracker_executed": 0,
        "raw_frame_cache_written": 0,
        "next_recommendation": (
            "CORE-1L windowed tracker pair mining on selected rendered windows"
            if total_windows > 0 and available_windows / max(total_windows, 1) >= 0.8 and mean_margin > 0 and top1_rate > random_top1
            else "inspect window descriptor failures before tracker-derived pair mining"
        ),
    }

    report = f"""# CORE-1K Windowed Render Cache Smoke

This stage renders only the selected CORE-1J event windows and extracts same-space crop descriptors. It does not execute the full tracker and does not write raw frame caches.

## Result

- Selected sequences: {compact['selected_sequence_count']}
- Selected windows: {compact['selected_window_count']}
- Processed events with disappear/reappear descriptors: {compact['processed_event_count']}
- Descriptor available rate: {compact['descriptor_available_rate']:.4f}
- Mean same-event margin: {compact['mean_same_event_margin']:.6f}
- Same-event top1 rate: {compact['same_event_top1_rate']:.4f}
- Random top1 rate: {compact['random_top1_rate']:.4f}
- Runtime seconds: {compact['runtime_sec']:.2f}
- Fallback pre-disappear windows used: {compact['fallback_window_count']}
- Smoke passed: {compact['window_cache_smoke_passed']}

## Interpretation

CORE-1K is a runtime and descriptor separability gate. A pass means the selected rendered windows contain enough same-space descriptor signal to justify a windowed tracker-pair mining pass. A fail means the window descriptor path should be inspected before spending time on tracker execution.

Next recommendation: {compact['next_recommendation']}
"""

    prefix = f"stage_CORE1K_"
    write_csv(
        out_dir / f"{prefix}window_render_cache_summary_{args.artifact_version}.csv",
        render_rows,
        [
            "sequence_id",
            "event_id",
            "window_kind",
            "start_frame",
            "end_frame",
            "planned_frame_count",
            "available_frame_count",
            "target_visible_frame_count",
            "descriptor_frame_count",
            "descriptor_available",
            "fallback_used",
        ],
    )
    write_csv(
        out_dir / f"{prefix}window_descriptor_trace_{args.artifact_version}.csv",
        trace_rows,
        [
            "sequence_id",
            "event_id",
            "window_kind",
            "descriptor_source",
            "frame_idx",
            "instance_id_eval_only",
            "box",
            "crop_box",
            "descriptor_available",
            "descriptor_norm",
            "descriptor_entropy_proxy",
            "edge_density",
            "objectness_crop_mean",
            "descriptor",
        ],
    )
    write_csv(
        out_dir / f"{prefix}descriptor_pair_margin_{args.artifact_version}.csv",
        margin_rows,
        [
            "event_id",
            "sequence_id",
            "target_score",
            "wrong_top_event",
            "wrong_top_score",
            "same_event_margin",
            "same_event_top1",
            "candidate_count",
            "failure_reason",
        ],
    )
    write_csv(
        out_dir / f"{prefix}runtime_audit_{args.artifact_version}.csv",
        runtime_rows,
        ["sequence_id", "window_count", "generation_time_sec", "sequence_length", "runtime_mode"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
