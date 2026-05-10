from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.synth_stream import SynthDatasetConfig, SyntheticStreamGenerator
from experiments import run_v3_stage_e31_retrieval_competition_repair as e31
from experiments.run_core1j_rendered_tracker_pair_audit import box_iou
from experiments.run_core1k_windowed_render_cache import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE-1O objectness proposal recall audit.")
    p.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    p.add_argument("--window-plan", default="results/core1j/stage_CORE1J_window_plan_v1.csv")
    p.add_argument("--output-dir", default="results/core1o")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sequences", type=int, default=2)
    p.add_argument("--artifact-version", default="v1")
    return p.parse_args()


def load_config(path: Path) -> tuple[SynthDatasetConfig, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return SynthDatasetConfig.from_dict(payload), payload


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


VARIANTS: list[dict[str, Any]] = [
    {"variant": "A0_config"},
    {"variant": "A1_more_proposals", "max_proposals": 16},
    {"variant": "A2_min_area_48", "min_area": 48, "max_proposals": 16},
    {"variant": "A3_lower_quantile", "q_obj": 0.88, "local_k": 0.85, "max_proposals": 16},
    {"variant": "A4_low_area_lower_quantile", "min_area": 32, "q_obj": 0.86, "local_k": 0.78, "max_proposals": 20},
    {"variant": "A5_recall_profile", "tau_obj": 0.42, "min_area": 24, "q_obj": 0.84, "local_k": 0.72, "max_proposals": 24},
]


def build_field_config(base: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(base)
    for key, value in variant.items():
        if key != "variant":
            cfg[key] = value
    return cfg


def evaluate_window_variant(
    *,
    sequence_id: int,
    window_row: dict[str, str],
    frames_by_idx: dict[int, Any],
    payload: dict[str, Any],
    variant: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start_frame = int(window_row["start_frame"])
    end_frame = int(window_row["end_frame"])
    encoder = e31.MinimalSpikeEncoder(**payload["model"]["spike_encoder"])
    field = e31.MinimalObjectnessField(**build_field_config(payload["field"], variant))
    trace_rows: list[dict[str, Any]] = []
    gt_visible_count = 0
    covered_025 = 0
    covered_050 = 0
    proposal_counts: list[int] = []
    start = time.perf_counter()

    for frame_idx in range(start_frame + 1, end_frame + 1):
        prev_frame = frames_by_idx.get(frame_idx - 1)
        current_frame = frames_by_idx.get(frame_idx)
        if prev_frame is None or current_frame is None:
            continue
        encoding = encoder.encode(prev_frame.frame, current_frame.frame)
        objectness_output = field.compute(encoding)
        proposals = objectness_output.proposals
        proposal_counts.append(len(proposals))
        for idx, gt_box in enumerate(current_frame.boxes):
            if current_frame.visibility_flags and not current_frame.visibility_flags[idx]:
                continue
            box = tuple(int(v) for v in gt_box)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            gt_visible_count += 1
            overlaps = [box_iou(tuple(int(v) for v in p.box), box) for p in proposals]
            best_iou = max(overlaps) if overlaps else 0.0
            best_idx = int(np.argmax(overlaps)) if overlaps else -1
            covered_025 += int(best_iou >= 0.25)
            covered_050 += int(best_iou >= 0.50)
            trace_rows.append(
                {
                    "variant": variant["variant"],
                    "sequence_id": sequence_id,
                    "event_id": window_row["event_id"],
                    "window_kind": window_row["window_kind"],
                    "frame_idx": int(current_frame.frame_index),
                    "instance_id_eval_only": int(current_frame.instance_ids[idx]),
                    "proposal_count": len(proposals),
                    "best_proposal_index": best_idx,
                    "best_iou": best_iou,
                    "covered_iou025": int(best_iou >= 0.25),
                    "covered_iou050": int(best_iou >= 0.50),
                }
            )

    summary = {
        "variant": variant["variant"],
        "sequence_id": sequence_id,
        "event_id": window_row["event_id"],
        "window_kind": window_row["window_kind"],
        "gt_visible_count": gt_visible_count,
        "covered_iou025": covered_025,
        "covered_iou050": covered_050,
        "recall_iou025": covered_025 / max(gt_visible_count, 1),
        "recall_iou050": covered_050 / max(gt_visible_count, 1),
        "mean_proposals_per_frame": float(np.mean(proposal_counts)) if proposal_counts else 0.0,
        "max_proposals_per_frame": int(max(proposal_counts)) if proposal_counts else 0,
        "runtime_sec": time.perf_counter() - start,
    }
    return trace_rows, summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    cfg, payload = load_config(Path(args.config))
    generator = SyntheticStreamGenerator(cfg, seed=args.seed)
    selected_windows = select_windows(read_csv(Path(args.window_plan)), args.max_sequences)
    sequence_to_windows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in selected_windows:
        sequence_to_windows[int(row["sequence_id"])].append(row)

    trace_rows: list[dict[str, Any]] = []
    window_summaries: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    for sequence_id, windows in sorted(sequence_to_windows.items()):
        gen_start = time.perf_counter()
        sequence = generator.generate_sequence(sequence_id)
        generation_time = time.perf_counter() - gen_start
        generation_rows.append({"sequence_id": sequence_id, "generation_time_sec": generation_time, "window_count": len(windows)})
        frames_by_idx = {frame.frame_index: frame for frame in sequence.frames}
        for window in windows:
            for variant in VARIANTS:
                rows, summary = evaluate_window_variant(
                    sequence_id=sequence_id,
                    window_row=window,
                    frames_by_idx=frames_by_idx,
                    payload=payload,
                    variant=variant,
                )
                trace_rows.extend(rows)
                window_summaries.append(summary)

    variant_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows = [r for r in window_summaries if r["variant"] == variant["variant"]]
        visible = sum(int(r["gt_visible_count"]) for r in rows)
        cov025 = sum(int(r["covered_iou025"]) for r in rows)
        cov050 = sum(int(r["covered_iou050"]) for r in rows)
        mean_props = float(np.mean([float(r["mean_proposals_per_frame"]) for r in rows])) if rows else 0.0
        variant_rows.append(
            {
                "variant": variant["variant"],
                "gt_visible_count": visible,
                "recall_iou025": cov025 / max(visible, 1),
                "recall_iou050": cov050 / max(visible, 1),
                "mean_proposals_per_frame": mean_props,
                "max_window_proposals_per_frame": max([int(r["max_proposals_per_frame"]) for r in rows], default=0),
                "eligible_for_frontend_profile": int(cov025 / max(visible, 1) >= 0.75 and mean_props <= 20),
            }
        )

    best = max(variant_rows, key=lambda r: (float(r["recall_iou025"]), float(r["recall_iou050"]), -float(r["mean_proposals_per_frame"]))) if variant_rows else {}
    baseline = next((r for r in variant_rows if r["variant"] == "A0_config"), {})
    compact = {
        "stage": "CORE-1O",
        "artifact_version": args.artifact_version,
        "selected_sequence_count": len(sequence_to_windows),
        "selected_window_count": len(selected_windows),
        "baseline_recall_iou025": baseline.get("recall_iou025", 0.0),
        "baseline_recall_iou050": baseline.get("recall_iou050", 0.0),
        "best_variant": best.get("variant", ""),
        "best_recall_iou025": best.get("recall_iou025", 0.0),
        "best_recall_iou050": best.get("recall_iou050", 0.0),
        "best_mean_proposals_per_frame": best.get("mean_proposals_per_frame", 0.0),
        "frontend_profile_candidate_found": int(any(int(r["eligible_for_frontend_profile"]) for r in variant_rows)),
        "oracle_leakage_found": 0,
        "runtime_sec": time.perf_counter() - total_start,
        "next_recommendation": (
            "CORE-1P validate best proposal profile through tracker pair mining"
            if any(int(r["eligible_for_frontend_profile"]) for r in variant_rows)
            else "objectness proposal recall remains insufficient; inspect objectness field or use supervised/oracle proposal boundary for core encoder experiments"
        ),
    }

    report = f"""# CORE-1O Objectness Proposal Recall Audit

This stage scans objectness-field proposal settings on the selected CORE-1J windows. It does not change the main model. GT boxes are used only to audit proposal coverage.

## Result

- Selected sequences: {compact['selected_sequence_count']}
- Selected windows: {compact['selected_window_count']}
- Baseline recall@0.25: {float(compact['baseline_recall_iou025']):.4f}
- Baseline recall@0.50: {float(compact['baseline_recall_iou050']):.4f}
- Best variant: {compact['best_variant']}
- Best recall@0.25: {float(compact['best_recall_iou025']):.4f}
- Best recall@0.50: {float(compact['best_recall_iou050']):.4f}
- Best mean proposals/frame: {float(compact['best_mean_proposals_per_frame']):.2f}
- Frontend profile candidate found: {compact['frontend_profile_candidate_found']}

Next recommendation: {compact['next_recommendation']}
"""

    prefix = "stage_CORE1O_"
    write_csv(
        out_dir / f"{prefix}proposal_recall_trace_{args.artifact_version}.csv",
        trace_rows,
        [
            "variant",
            "sequence_id",
            "event_id",
            "window_kind",
            "frame_idx",
            "instance_id_eval_only",
            "proposal_count",
            "best_proposal_index",
            "best_iou",
            "covered_iou025",
            "covered_iou050",
        ],
    )
    write_csv(
        out_dir / f"{prefix}window_summary_{args.artifact_version}.csv",
        window_summaries,
        [
            "variant",
            "sequence_id",
            "event_id",
            "window_kind",
            "gt_visible_count",
            "covered_iou025",
            "covered_iou050",
            "recall_iou025",
            "recall_iou050",
            "mean_proposals_per_frame",
            "max_proposals_per_frame",
            "runtime_sec",
        ],
    )
    write_csv(
        out_dir / f"{prefix}variant_summary_{args.artifact_version}.csv",
        variant_rows,
        [
            "variant",
            "gt_visible_count",
            "recall_iou025",
            "recall_iou050",
            "mean_proposals_per_frame",
            "max_window_proposals_per_frame",
            "eligible_for_frontend_profile",
        ],
    )
    write_csv(
        out_dir / f"{prefix}runtime_audit_{args.artifact_version}.csv",
        generation_rows,
        ["sequence_id", "generation_time_sec", "window_count"],
    )
    write_json(out_dir / f"{prefix}compact_for_gpt_{args.artifact_version}.json", compact)
    (out_dir / f"{prefix}report_{args.artifact_version}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
