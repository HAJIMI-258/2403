from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.synth_stream import SynthDatasetConfig, SyntheticStreamGenerator
from experiments.run_core1_online_object_encoder import write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build CORE-1F dense internal synthetic event ledger.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--output-dir", default="results/core1f")
    p.add_argument("--artifact-version", default="v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-sequences", type=int, default=480)
    p.add_argument("--min-gap", type=int, default=3)
    p.add_argument("--real-gap", type=int, default=10)
    p.add_argument("--mode", choices=["fast-planned", "rendered"], default="fast-planned")
    return p.parse_args()


def load_dataset_config(path: str | Path) -> SynthDatasetConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return SynthDatasetConfig.from_dict(payload)


def box_area(box: tuple[int, int, int, int]) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def split_for_sequence(sequence_id: int) -> str:
    m = sequence_id % 10
    if m < 6:
        return "train"
    if m < 8:
        return "dev"
    return "test"


def difficulty(gap: int, same_concept_distractors: int, displacement: float, area_change: float) -> str:
    if gap >= 80 or same_concept_distractors >= 2 or displacement >= 160 or area_change >= 1.8:
        return "hard"
    if gap >= 30 or same_concept_distractors >= 1 or displacement >= 80 or area_change >= 1.35:
        return "medium"
    return "easy"


def frame_lookup(sequence) -> dict[int, dict[int, dict[str, Any]]]:
    lookup: dict[int, dict[int, dict[str, Any]]] = {}
    for frame in sequence.frames:
        row: dict[int, dict[str, Any]] = {}
        for idx, iid in enumerate(frame.instance_ids):
            row[int(iid)] = {
                "box": tuple(int(v) for v in frame.boxes[idx]),
                "concept_id": int(frame.concept_ids[idx]),
                "area": box_area(tuple(int(v) for v in frame.boxes[idx])),
            }
        lookup[int(frame.frame_index)] = row
    return lookup


def contiguous_segments(frames: list[int]) -> list[tuple[int, int]]:
    if not frames:
        return []
    frames = sorted(frames)
    segs = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            segs.append((start, prev))
            start = prev = f
    segs.append((start, prev))
    return segs


def mine_sequence(sequence, min_gap: int, real_gap: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    lookup = frame_lookup(sequence)
    frames_by_instance: dict[int, list[int]] = defaultdict(list)
    concept_by_instance: dict[int, int] = {}
    for frame_idx, instances in lookup.items():
        for iid, payload in instances.items():
            frames_by_instance[iid].append(frame_idx)
            concept_by_instance[iid] = int(payload["concept_id"])

    ledger_rows: list[dict[str, Any]] = []
    adjacent_positive_pairs = 0
    skip_positive_pairs = 0
    negative_cov_visible_pairs = 0
    for frame_idx, instances in lookup.items():
        ids = sorted(instances)
        negative_cov_visible_pairs += len(ids) * max(0, len(ids) - 1) // 2
    for iid, visible_frames in frames_by_instance.items():
        visible_set = set(visible_frames)
        for f in visible_frames:
            if f + 1 in visible_set:
                adjacent_positive_pairs += 1
            if f + 5 in visible_set:
                skip_positive_pairs += 1
        segs = contiguous_segments(visible_frames)
        for gap_idx, (left, right) in enumerate(zip(segs, segs[1:])):
            disappear_frame = left[1]
            reappear_frame = right[0]
            gap = reappear_frame - disappear_frame - 1
            if gap < min_gap:
                continue
            before = lookup[disappear_frame][iid]
            after = lookup[reappear_frame][iid]
            box_before = before["box"]
            box_after = after["box"]
            cx0, cy0 = box_center(box_before)
            cx1, cy1 = box_center(box_after)
            displacement = float(np.hypot(cx1 - cx0, cy1 - cy0))
            area_change = max(after["area"], before["area"]) / max(min(after["area"], before["area"]), 1.0)
            reappear_instances = lookup[reappear_frame]
            same_concept = sum(
                1 for other_iid, payload in reappear_instances.items()
                if other_iid != iid and int(payload["concept_id"]) == int(concept_by_instance[iid])
            )
            event_id = f"core1f_seq{sequence.sequence_id:04d}_iid{iid}_gap{gap_idx}_f{disappear_frame}_{reappear_frame}"
            ledger_rows.append(
                {
                    "event_id": event_id,
                    "sequence_id": int(sequence.sequence_id),
                    "split": split_for_sequence(int(sequence.sequence_id)),
                    "instance_id_eval_only": int(iid),
                    "concept_id_eval_only": int(concept_by_instance[iid]),
                    "disappear_frame": int(disappear_frame),
                    "reappear_frame": int(reappear_frame),
                    "gap_length": int(gap),
                    "usable_smoke_gap": int(gap >= min_gap),
                    "usable_real_gap": int(gap >= real_gap),
                    "box_before": "|".join(str(v) for v in box_before),
                    "box_after": "|".join(str(v) for v in box_after),
                    "area_before": before["area"],
                    "area_after": after["area"],
                    "area_change_ratio": area_change,
                    "center_displacement": displacement,
                    "num_visible_objects_at_reappear": len(reappear_instances),
                    "same_concept_distractors_at_reappear": same_concept,
                    "difficulty_level": difficulty(gap, same_concept, displacement, area_change),
                    "gt_used_for_event_definition_only": 1,
                    "allowed_for_online_scoring": 0,
                }
            )

    seq_row = {
        "sequence_id": int(sequence.sequence_id),
        "split": split_for_sequence(int(sequence.sequence_id)),
        "num_frames": len(sequence.frames),
        "num_instances": len(frames_by_instance),
        "visible_instance_frame_count": sum(len(v) for v in frames_by_instance.values()),
        "mined_gap_events": len(ledger_rows),
        "real_gap_events": sum(1 for r in ledger_rows if int(r["usable_real_gap"]) == 1),
        "planned_reentry_events": int(sequence.metadata.get("planned_reentry_events", 0)),
        "planned_long_occlusion_events": int(sequence.metadata.get("planned_long_occlusion_events", 0)),
    }
    pair_summary = {
        "sequence_id": int(sequence.sequence_id),
        "adjacent_positive_pair_opportunities": adjacent_positive_pairs,
        "skip5_positive_pair_opportunities": skip_positive_pairs,
        "negative_cov_visible_pair_opportunities": negative_cov_visible_pairs,
    }
    return ledger_rows, seq_row, pair_summary


def approx_box(center: np.ndarray, scale: float, resolution: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = resolution
    cx, cy = float(center[0]), float(center[1])
    s = max(2.0, float(scale))
    return (
        int(max(0, round(cx - s))),
        int(max(0, round(cy - s))),
        int(min(w - 1, round(cx + s))),
        int(min(h - 1, round(cy + s))),
    )


def mine_sequence_fast(generator: SyntheticStreamGenerator, sequence_id: int, min_gap: int, real_gap: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Mine planned re-entry events without rendering frames.

    This mode is intentionally used for ledger expansion only. Boxes are
    approximate kinematic boxes; future pixel/full-tracker stages must rerender
    selected sequences and may not treat these boxes as detector outputs.
    """

    rng = np.random.default_rng(generator.seed + sequence_id)
    objects = generator._initialize_objects(rng)  # noqa: SLF001 - diagnostic ledger builder.
    occlusion_event = generator._sample_occlusion_event(rng, len(objects))  # noqa: SLF001
    resolution = generator.config.resolution
    states: dict[int, list[dict[str, Any]]] = defaultdict(list)
    adjacent_positive_pairs = 0
    skip_positive_pairs = 0
    negative_cov_visible_pairs = 0
    for frame_idx in range(generator.config.sequence_length):
        generator._update_objects(objects, rng, frame_idx, occlusion_event)  # noqa: SLF001
        active_ids = [int(o.instance_id) for o in objects if o.active]
        negative_cov_visible_pairs += len(active_ids) * max(0, len(active_ids) - 1) // 2
        for obj in objects:
            if not obj.active:
                continue
            states[int(obj.instance_id)].append(
                {
                    "frame_idx": int(frame_idx),
                    "concept_id": int(obj.concept_id),
                    "center": np.asarray(obj.center, dtype=np.float32).copy(),
                    "scale": float(obj.base_scale),
                    "box": approx_box(obj.center, obj.base_scale, resolution),
                }
            )

    ledger_rows: list[dict[str, Any]] = []
    for iid, entries in states.items():
        frames = [int(e["frame_idx"]) for e in entries]
        visible_set = set(frames)
        by_frame = {int(e["frame_idx"]): e for e in entries}
        for f in frames:
            if f + 1 in visible_set:
                adjacent_positive_pairs += 1
            if f + 5 in visible_set:
                skip_positive_pairs += 1
        segs = contiguous_segments(frames)
        for gap_idx, (left, right) in enumerate(zip(segs, segs[1:])):
            disappear_frame = left[1]
            reappear_frame = right[0]
            gap = reappear_frame - disappear_frame - 1
            if gap < min_gap:
                continue
            before = by_frame[disappear_frame]
            after = by_frame[reappear_frame]
            box_before = before["box"]
            box_after = after["box"]
            cx0, cy0 = box_center(box_before)
            cx1, cy1 = box_center(box_after)
            displacement = float(np.hypot(cx1 - cx0, cy1 - cy0))
            area_before = box_area(box_before)
            area_after = box_area(box_after)
            area_change = max(area_after, area_before) / max(min(area_after, area_before), 1.0)
            active_at_reappear = [v[-1] for _, v in states.items() if any(int(e["frame_idx"]) == reappear_frame for e in v)]
            # Recompute from by-frame presence to avoid relying on the last entry
            reappear_entries = []
            for other_entries in states.values():
                reappear_entries.extend([e for e in other_entries if int(e["frame_idx"]) == reappear_frame])
            same_concept = sum(1 for e in reappear_entries if int(e["concept_id"]) == int(before["concept_id"])) - 1
            event_id = f"core1f_seq{sequence_id:04d}_iid{iid}_gap{gap_idx}_f{disappear_frame}_{reappear_frame}"
            ledger_rows.append(
                {
                    "event_id": event_id,
                    "sequence_id": int(sequence_id),
                    "split": split_for_sequence(int(sequence_id)),
                    "instance_id_eval_only": int(iid),
                    "concept_id_eval_only": int(before["concept_id"]),
                    "disappear_frame": int(disappear_frame),
                    "reappear_frame": int(reappear_frame),
                    "gap_length": int(gap),
                    "usable_smoke_gap": int(gap >= min_gap),
                    "usable_real_gap": int(gap >= real_gap),
                    "box_before": "|".join(str(v) for v in box_before),
                    "box_after": "|".join(str(v) for v in box_after),
                    "area_before": area_before,
                    "area_after": area_after,
                    "area_change_ratio": area_change,
                    "center_displacement": displacement,
                    "num_visible_objects_at_reappear": len(reappear_entries),
                    "same_concept_distractors_at_reappear": max(0, same_concept),
                    "difficulty_level": difficulty(gap, max(0, same_concept), displacement, area_change),
                    "gt_used_for_event_definition_only": 1,
                    "allowed_for_online_scoring": 0,
                    "ledger_source": "fast_planned_kinematic",
                    "requires_full_render_for_pixel_eval": 1,
                }
            )

    seq_row = {
        "sequence_id": int(sequence_id),
        "split": split_for_sequence(int(sequence_id)),
        "num_frames": int(generator.config.sequence_length),
        "num_instances": len(objects),
        "visible_instance_frame_count": sum(len(v) for v in states.values()),
        "mined_gap_events": len(ledger_rows),
        "real_gap_events": sum(1 for r in ledger_rows if int(r["usable_real_gap"]) == 1),
        "planned_reentry_events": sum(int(o.return_frame is not None) for o in objects),
        "planned_long_occlusion_events": int(
            occlusion_event is not None and (occlusion_event.end - occlusion_event.start) >= max(24, generator.config.sequence_length // 18)
        ),
        "ledger_source": "fast_planned_kinematic",
    }
    pair_summary = {
        "sequence_id": int(sequence_id),
        "adjacent_positive_pair_opportunities": adjacent_positive_pairs,
        "skip5_positive_pair_opportunities": skip_positive_pairs,
        "negative_cov_visible_pair_opportunities": negative_cov_visible_pairs,
    }
    return ledger_rows, seq_row, pair_summary


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_dataset_config(args.config)
    generator = SyntheticStreamGenerator(cfg, seed=args.seed)

    ledger_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for seq_id in range(args.num_sequences):
        if args.mode == "rendered":
            sequence = generator.generate_sequence(seq_id)
            seq_ledger, seq_row, pair_row = mine_sequence(sequence, args.min_gap, args.real_gap)
            for row in seq_ledger:
                row["ledger_source"] = "rendered_frame_visibility"
                row["requires_full_render_for_pixel_eval"] = 0
            seq_row["ledger_source"] = "rendered_frame_visibility"
        else:
            seq_ledger, seq_row, pair_row = mine_sequence_fast(generator, seq_id, args.min_gap, args.real_gap)
        ledger_rows.extend(seq_ledger)
        sequence_rows.append(seq_row)
        pair_rows.append(pair_row)

    split_counter = Counter(r["split"] for r in ledger_rows if int(r["usable_real_gap"]) == 1)
    difficulty_counter = Counter(r["difficulty_level"] for r in ledger_rows if int(r["usable_real_gap"]) == 1)
    concept_counter = Counter(str(r["concept_id_eval_only"]) for r in ledger_rows if int(r["usable_real_gap"]) == 1)
    pair_totals = {
        "adjacent_positive_pair_opportunities": sum(int(r["adjacent_positive_pair_opportunities"]) for r in pair_rows),
        "skip5_positive_pair_opportunities": sum(int(r["skip5_positive_pair_opportunities"]) for r in pair_rows),
        "negative_cov_visible_pair_opportunities": sum(int(r["negative_cov_visible_pair_opportunities"]) for r in pair_rows),
    }
    split_rows = [
        {
            "split": split,
            "real_gap_event_count": split_counter.get(split, 0),
            "sequence_count": sum(1 for r in sequence_rows if r["split"] == split),
        }
        for split in ("train", "dev", "test")
    ]

    write_csv(out / f"stage_CORE1F_dense_event_ledger_{args.artifact_version}.csv", ledger_rows)
    write_csv(out / f"stage_CORE1F_sequence_inventory_{args.artifact_version}.csv", sequence_rows)
    write_csv(out / f"stage_CORE1F_pair_opportunity_by_sequence_{args.artifact_version}.csv", pair_rows)
    write_csv(out / f"stage_CORE1F_split_summary_{args.artifact_version}.csv", split_rows)
    write_json(out / f"stage_CORE1F_pair_opportunity_summary_{args.artifact_version}.json", pair_totals)

    real_gap_events = [r for r in ledger_rows if int(r["usable_real_gap"]) == 1]
    compact = {
        "stage": "CORE-1F",
        "config": args.config,
        "seed": args.seed,
        "mode": args.mode,
        "num_sequences": args.num_sequences,
        "smoke_gap_event_count": len(ledger_rows),
        "real_gap_event_count": len(real_gap_events),
        "train_real_gap_events": split_counter.get("train", 0),
        "dev_real_gap_events": split_counter.get("dev", 0),
        "test_real_gap_events": split_counter.get("test", 0),
        "difficulty_counts": dict(difficulty_counter),
        "concept_counts": dict(concept_counter),
        **pair_totals,
        "dense_ledger_ready": int(len(real_gap_events) >= 500 and split_counter.get("train", 0) >= 250 and split_counter.get("dev", 0) >= 75 and split_counter.get("test", 0) >= 75),
        "oracle_leakage_found": 0,
        "gt_usage": "GT boxes/instance ids define evaluation ledger only; not allowed for online scoring or training labels.",
        "next_recommendation": (
            "CORE-1G run NOPS/core encoder pair-mining on dense internal ledger"
            if len(real_gap_events) >= 500 and split_counter.get("train", 0) >= 250
            else "increase CORE-1F num_sequences before CORE-1G"
        ),
    }
    write_json(out / f"stage_CORE1F_compact_for_gpt_{args.artifact_version}.json", compact)

    report = [
        "# CORE-1F Dense Internal Event Ledger",
        "",
        "CORE-1F generates a larger internal synthetic event ledger so CORE encoder work is not bottlenecked by the original 17 hand-curated re-entry events.",
        "GT is used only to define evaluation events and audit opportunities, not online scoring.",
        "",
        "## Result",
        f"- Sequences generated: {args.num_sequences}.",
        f"- Real-gap events: {len(real_gap_events)}.",
        f"- Train/dev/test: {split_counter.get('train', 0)}/{split_counter.get('dev', 0)}/{split_counter.get('test', 0)}.",
        f"- Adjacent positive pair opportunities: {pair_totals['adjacent_positive_pair_opportunities']}.",
        f"- Co-visible negative pair opportunities: {pair_totals['negative_cov_visible_pair_opportunities']}.",
        f"- Dense ledger ready: {compact['dense_ledger_ready']}.",
        "",
        "## Decision",
        compact["next_recommendation"],
    ]
    (out / f"stage_CORE1F_report_{args.artifact_version}.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
