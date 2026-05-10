from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_core1j_rendered_tracker_pair_audit import box_iou
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1Q assignment fragmentation audit.")
    p.add_argument("--core1p-observations", default="results/core1p/stage_CORE1P_assignment_observation_trace_v1.csv")
    p.add_argument("--core1p-pairs", default="results/core1p/stage_CORE1P_gated_pair_trace_v1.csv")
    p.add_argument("--gate-name", default="A3_score050_cost_le_050")
    p.add_argument("--output-dir", default="results/core1q")
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def i(v: Any, default: int = 0) -> int:
    if v in (None, ""):
        return default
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default


def f(v: Any, default: float = 0.0) -> float:
    if v in (None, ""):
        return default
    try:
        return float(v)
    except Exception:
        return default


def box_from_text(text: str) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    try:
        vals = [int(float(x)) for x in str(text).split("|")]
        return vals[0], vals[1], vals[2], vals[3]
    except Exception:
        return None


def obs_key(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (str(row["sequence_id"]), str(row["event_id"]), str(row["window_kind"]), i(row["frame_idx"]), i(row["track_id"]))


def pair_failure_reason(pair: dict[str, str], obs_by_key: dict[tuple[str, str, str, int, int], dict[str, str]]) -> str:
    iid_i = str(pair.get("gt_instance_i_eval_only", ""))
    iid_j = str(pair.get("gt_instance_j_eval_only", ""))
    ptype = str(pair["pair_type"])
    if iid_i == "" or iid_j == "":
        return "unmatched_assignment_in_pair"
    if ptype.startswith("positive"):
        if iid_i != iid_j:
            return "track_identity_switched_between_frames"
        return "none"
    if iid_i == iid_j:
        return "same_gt_fragmented_across_tracks"
    key_i = (pair["sequence_id"], pair["event_id"], pair["window_kind"], i(pair["frame_i"]), i(pair["track_i"]))
    key_j = (pair["sequence_id"], pair["event_id"], pair["window_kind"], i(pair["frame_j"]), i(pair["track_j"]))
    box_i = box_from_text(obs_by_key.get(key_i, {}).get("box", ""))
    box_j = box_from_text(obs_by_key.get(key_j, {}).get("box", ""))
    if box_i is not None and box_j is not None and box_iou(box_i, box_j) > 0.20:
        return "overlapping_tracks_for_different_gt"
    return "none"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_rows = read_csv(Path(args.core1p_observations))
    all_pair_rows = read_csv(Path(args.core1p_pairs))
    pair_rows = [r for r in all_pair_rows if r.get("gate_name") == args.gate_name]
    obs_by_key = {obs_key(r): r for r in obs_rows}

    by_frame: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    by_track: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    by_gt: dict[tuple[str, str, str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in obs_rows:
        frame_key = (row["sequence_id"], row["event_id"], row["window_kind"], i(row["frame_idx"]))
        by_frame[frame_key].append(row)
        by_track[(row["sequence_id"], row["event_id"], row["window_kind"], i(row["track_id"]))].append(row)
        if row.get("gt_instance_eval_only", "") != "":
            by_gt[(row["sequence_id"], row["event_id"], row["window_kind"], i(row["frame_idx"]), str(row["gt_instance_eval_only"]))].append(row)

    frame_rows: list[dict[str, Any]] = []
    for key, rows in by_frame.items():
        matched = [r for r in rows if r.get("gt_instance_eval_only", "") != ""]
        unmatched = [r for r in rows if r.get("gt_instance_eval_only", "") == ""]
        gt_counts = Counter(str(r["gt_instance_eval_only"]) for r in matched)
        fragmented_gt_count = sum(1 for _gt, count in gt_counts.items() if count > 1)
        frame_rows.append(
            {
                "sequence_id": key[0],
                "event_id": key[1],
                "window_kind": key[2],
                "frame_idx": key[3],
                "assignment_count": len(rows),
                "matched_assignment_count": len(matched),
                "unmatched_assignment_count": len(unmatched),
                "matched_assignment_rate": len(matched) / max(len(rows), 1),
                "unique_gt_count": len(gt_counts),
                "fragmented_gt_count": fragmented_gt_count,
                "max_tracks_per_gt": max(gt_counts.values(), default=0),
            }
        )

    track_rows: list[dict[str, Any]] = []
    for key, rows in by_track.items():
        ordered = sorted(rows, key=lambda r: i(r["frame_idx"]))
        matched_iids = [str(r["gt_instance_eval_only"]) for r in ordered if r.get("gt_instance_eval_only", "") != ""]
        unique_iids = sorted(set(matched_iids))
        unmatched_count = len([r for r in ordered if r.get("gt_instance_eval_only", "") == ""])
        switches = 0
        prev = None
        for iid in matched_iids:
            if prev is not None and iid != prev:
                switches += 1
            prev = iid
        track_rows.append(
            {
                "sequence_id": key[0],
                "event_id": key[1],
                "window_kind": key[2],
                "track_id": key[3],
                "obs_count": len(ordered),
                "matched_obs_count": len(matched_iids),
                "unmatched_obs_count": unmatched_count,
                "unique_gt_count": len(unique_iids),
                "identity_switch_count_eval_only": switches,
                "dominant_gt_eval_only": Counter(matched_iids).most_common(1)[0][0] if matched_iids else "",
                "dominant_gt_fraction": Counter(matched_iids).most_common(1)[0][1] / max(len(matched_iids), 1) if matched_iids else 0.0,
            }
        )

    pair_audit_rows: list[dict[str, Any]] = []
    for row in pair_rows:
        reason = pair_failure_reason(row, obs_by_key)
        pair_audit_rows.append(
            {
                **row,
                "failure_reason": reason,
                "pair_correct_eval_only": i(row.get("pair_correct_eval_only")),
            }
        )

    failure_counts = Counter(r["failure_reason"] for r in pair_audit_rows if r["failure_reason"] != "none")
    pos_pairs = [r for r in pair_audit_rows if str(r["pair_type"]).startswith("positive")]
    neg_pairs = [r for r in pair_audit_rows if str(r["pair_type"]).startswith("negative")]
    pos_precision = float(np.mean([i(r["pair_correct_eval_only"]) for r in pos_pairs])) if pos_pairs else 0.0
    neg_precision = float(np.mean([i(r["pair_correct_eval_only"]) for r in neg_pairs])) if neg_pairs else 0.0
    unmatched_frame_rate = float(np.mean([1.0 - f(r["matched_assignment_rate"]) for r in frame_rows])) if frame_rows else 0.0
    fragmented_frame_rate = float(np.mean([int(i(r["fragmented_gt_count"]) > 0) for r in frame_rows])) if frame_rows else 0.0
    switched_track_rate = float(np.mean([int(i(r["identity_switch_count_eval_only"]) > 0) for r in track_rows])) if track_rows else 0.0
    main_failure = failure_counts.most_common(1)[0][0] if failure_counts else "none"
    next_recommendation = {
        "unmatched_assignment_in_pair": "add matched-observation confidence gate or improve objectness/proposal localization before training",
        "same_gt_fragmented_across_tracks": "add assignment dedup/NMS by overlap and GT-free duplicate proxies before pair mining",
        "track_identity_switched_between_frames": "repair short-window tracker continuity before encoder training",
        "overlapping_tracks_for_different_gt": "add overlap-aware negative filtering before pair mining",
        "none": "CORE-1Q found no dominant fragmentation issue",
    }.get(main_failure, "inspect pair failure taxonomy")

    compact = {
        "stage": "CORE-1Q",
        "artifact_version": args.artifact_version,
        "source_gate": args.gate_name,
        "positive_pair_count": len(pos_pairs),
        "negative_pair_count": len(neg_pairs),
        "positive_pair_precision_eval_only": pos_precision,
        "negative_pair_precision_eval_only": neg_precision,
        "unmatched_frame_rate": unmatched_frame_rate,
        "fragmented_frame_rate": fragmented_frame_rate,
        "switched_track_rate": switched_track_rate,
        "main_failure_counts": dict(failure_counts),
        "main_failure_type": main_failure,
        "oracle_leakage_found": 0,
        "ready_for_encoder_training": 0,
        "next_recommendation": next_recommendation,
    }

    report = f"""# CORE-1Q Assignment Fragmentation Audit

This stage audits why CORE-1P gated assignment pairs remain too noisy. It uses GT only for audit labels, not for online scoring or pair selection.

## Result

- Source gate: {args.gate_name}
- Positive precision eval-only: {pos_precision:.4f}
- Negative precision eval-only: {neg_precision:.4f}
- Unmatched frame rate: {unmatched_frame_rate:.4f}
- Fragmented frame rate: {fragmented_frame_rate:.4f}
- Switched track rate: {switched_track_rate:.4f}
- Main failure type: {main_failure}
- Ready for encoder training: 0

Failure counts: {dict(failure_counts)}

Next recommendation: {next_recommendation}
"""

    prefix = "stage_CORE1Q_"
    write_csv(
        out_dir / f"{prefix}frame_fragmentation_audit_{args.artifact_version}.csv",
        frame_rows,
        [
            "sequence_id",
            "event_id",
            "window_kind",
            "frame_idx",
            "assignment_count",
            "matched_assignment_count",
            "unmatched_assignment_count",
            "matched_assignment_rate",
            "unique_gt_count",
            "fragmented_gt_count",
            "max_tracks_per_gt",
        ],
    )
    write_csv(
        out_dir / f"{prefix}track_continuity_audit_{args.artifact_version}.csv",
        track_rows,
        [
            "sequence_id",
            "event_id",
            "window_kind",
            "track_id",
            "obs_count",
            "matched_obs_count",
            "unmatched_obs_count",
            "unique_gt_count",
            "identity_switch_count_eval_only",
            "dominant_gt_eval_only",
            "dominant_gt_fraction",
        ],
    )
    write_csv(
        out_dir / f"{prefix}pair_failure_taxonomy_{args.artifact_version}.csv",
        pair_audit_rows,
        [
            "pair_id",
            "gate_name",
            "sequence_id",
            "event_id",
            "window_kind",
            "frame_i",
            "frame_j",
            "track_i",
            "track_j",
            "pair_type",
            "gt_instance_i_eval_only",
            "gt_instance_j_eval_only",
            "pair_correct_eval_only",
            "failure_reason",
        ],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
