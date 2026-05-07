"""Phase 2F-R Part A: false-positive attribution audit for proposal representation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase2fr_utils import (
    RegionDescriptor,
    box_iou,
    build_track_scenarios,
    copy_config,
    encode_objectness_sequence,
    load_config_payload,
    mask_iou,
    serialize_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2F-R false-positive attribution audit.")
    parser.add_argument("--config", default="configs/bridge_synth_generic_v1.yaml")
    parser.add_argument("--output-dir", default="results/phase2fr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reclassify-from-csv",
        default="",
        help="Reuse an existing proposal_fp_attribution.csv and only rerun classification + summary generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_frame_payloads: dict[str, dict[str, Any]] = {}
    if args.reclassify_from_csv:
        scenario_rows = _read_existing_rows(Path(args.reclassify_from_csv))
        _classify_rows(scenario_rows)
    else:
        config_payload = load_config_payload(args.config)
        scenarios = build_track_scenarios(args.config)
        target_names = {"track_a_bridge", "track_c_long_horizon"}
        scenarios = [scenario for scenario in scenarios if scenario["name"] in target_names]

        scenario_rows: list[dict[str, Any]] = []
        for index, scenario in enumerate(scenarios):
            seed = args.seed + index * 17
            run_payload = encode_objectness_sequence(
                config_payload=config_payload,
                scenario_name=scenario["name"],
                scenario_config=copy_config(scenario["config"]),
                seed=seed,
                sequence_id=0,
            )
            scenario_frame_payloads[scenario["name"]] = run_payload
            scenario_rows.extend(_collect_fp_candidates(run_payload))

        _annotate_temporal_persistence(scenario_rows)
        _annotate_fragmentation(scenario_rows)
        _classify_rows(scenario_rows)

    csv_path = output_dir / "proposal_fp_attribution.csv"
    summary_path = output_dir / "proposal_fp_attribution_summary.md"
    design_path = output_dir / "phase2fr_design_notes.md"
    final_summary_path = output_dir / "phase2fr_final_summary_v1.md"
    gallery_path = output_dir / "proposal_fp_gallery.png"

    _write_csv(csv_path, scenario_rows)
    summary_path.write_text(_build_summary_markdown(scenario_rows), encoding="utf-8")
    design_path.write_text(_build_design_notes(), encoding="utf-8")
    final_summary_path.write_text(_build_final_summary(scenario_rows), encoding="utf-8")
    if scenario_frame_payloads:
        _save_gallery(gallery_path, scenario_rows, scenario_frame_payloads)

    print(f"saved_csv={csv_path}")
    print(f"saved_summary={summary_path}")
    print(f"saved_design_notes={design_path}")
    print(f"saved_final_summary={final_summary_path}")
    print(f"saved_gallery={gallery_path}")


def _read_existing_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _collect_fp_candidates(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenario_name = str(run_payload["scenario_name"])
    sequence_id = int(run_payload["sequence_id"])

    for frame_record in run_payload["frame_records"]:
        gt_boxes = list(frame_record["gt_boxes"])
        gt_masks = [np.asarray(mask, dtype=bool) for mask in frame_record["gt_masks"]]
        max_gt_bbox_area = max(((box[2] - box[0]) * (box[3] - box[1]) for box in gt_boxes), default=0)
        max_gt_mask_area = max((int(mask.sum()) for mask in gt_masks), default=0)
        gt_union = np.zeros_like(frame_record["objectness_output"].binary_mask, dtype=bool)
        for gt_mask in gt_masks:
            gt_union |= gt_mask

        for descriptor in frame_record["components"]:
            max_iou_with_gt = max((box_iou(descriptor.box, gt_box) for gt_box in gt_boxes), default=0.0)
            local_gt_pixels = 0
            x1, y1, x2, y2 = descriptor.box
            gt_crop = gt_union[y1:y2, x1:x2]
            if gt_crop.shape == descriptor.local_mask.shape:
                local_gt_pixels = int(np.logical_and(gt_crop, descriptor.local_mask).sum())
            region_to_gt_cover = float(local_gt_pixels / max(descriptor.region_area, 1))
            overlaps_gt = int(max_iou_with_gt >= 0.10 or region_to_gt_cover >= 0.10)
            if overlaps_gt:
                continue

            rows.append(
                {
                    "scenario_name": scenario_name,
                    "sequence_id": sequence_id,
                    "frame_id": int(frame_record["frame_index"]),
                    "proposal_id": int(descriptor.component_id),
                    "fp_type": "",
                    "raw_objectness_score": float(descriptor.raw_objectness_score),
                    "proposal_score": float(descriptor.proposal_score),
                    "region_area": int(descriptor.region_area),
                    "bbox_area": int(descriptor.bbox_area),
                    "bbox_aspect_ratio": float(descriptor.bbox_aspect_ratio),
                    "bbox_tightness": float(descriptor.region_fill_ratio),
                    "persistence_length": 1,
                    "chain_id": -1,
                    "center_stability": 0.0,
                    "near_boundary": int(descriptor.near_boundary),
                    "region_compactness": float(descriptor.region_compactness),
                    "region_fill_ratio": float(descriptor.region_fill_ratio),
                    "boundary_smoothness": float(descriptor.boundary_smoothness),
                    "overlaps_gt": 0,
                    "max_iou_with_gt": float(max_iou_with_gt),
                    "region_to_gt_cover": float(region_to_gt_cover),
                    "source_stage": "",
                    "dominant_term": descriptor.dominant_term,
                    "boundary_mean": float(descriptor.boundary_mean),
                    "persistence_mean": float(descriptor.persistence_mean),
                    "surprise_mean": float(descriptor.surprise_mean),
                    "habituation_mean": float(descriptor.habituation_mean),
                    "background_gradient_mean": float(descriptor.gradient_mean),
                    "drift_strength": float(frame_record["drift_strength"]),
                    "blur_level": float(frame_record["blur_level"]),
                    "noise_level": float(frame_record["noise_level"]),
                    "reentry_event": int(frame_record["reentry_event"]),
                    "max_gt_bbox_area": int(max_gt_bbox_area),
                    "max_gt_mask_area": int(max_gt_mask_area),
                    "support_box": descriptor.box,
                    "centroid_x": float(descriptor.centroid_x),
                    "centroid_y": float(descriptor.centroid_y),
                    "fragment_cluster_size": 1,
                }
            )
    return rows


def _annotate_temporal_persistence(rows: list[dict[str, Any]]) -> None:
    rows_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_scenario[str(row["scenario_name"])].append(row)

    next_chain_id = 0
    for scenario_rows in rows_by_scenario.values():
        scenario_rows.sort(key=lambda item: (int(item["frame_id"]), int(item["proposal_id"])))
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in scenario_rows:
            by_frame[int(row["frame_id"])].append(row)

        chain_members: dict[int, list[dict[str, Any]]] = {}
        last_by_chain: dict[int, dict[str, Any]] = {}
        for frame_id in sorted(by_frame):
            current_rows = by_frame[frame_id]
            available_chain_ids = [chain_id for chain_id, last_row in last_by_chain.items() if int(last_row["frame_id"]) == frame_id - 1]
            used_chains: set[int] = set()

            for row in current_rows:
                best_chain_id = None
                best_score = -1.0
                for chain_id in available_chain_ids:
                    if chain_id in used_chains:
                        continue
                    candidate = last_by_chain[chain_id]
                    score = _temporal_match_score(candidate, row)
                    if score > best_score:
                        best_score = score
                        best_chain_id = chain_id
                if best_chain_id is None or best_score < 0.18:
                    row["chain_id"] = next_chain_id
                    chain_members[next_chain_id] = [row]
                    last_by_chain[next_chain_id] = row
                    next_chain_id += 1
                    continue
                row["chain_id"] = best_chain_id
                chain_members[best_chain_id].append(row)
                last_by_chain[best_chain_id] = row
                used_chains.add(best_chain_id)

        for chain_id, members in chain_members.items():
            xs = np.array([float(member["centroid_x"]) for member in members], dtype=np.float32)
            ys = np.array([float(member["centroid_y"]) for member in members], dtype=np.float32)
            center_stability = float(np.mean(np.sqrt((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)))
            persistence = len(members)
            for member in members:
                member["chain_id"] = int(chain_id)
                member["persistence_length"] = int(persistence)
                member["center_stability"] = center_stability


def _temporal_match_score(previous: dict[str, Any], current: dict[str, Any]) -> float:
    iou = box_iou(tuple(previous["support_box"]), tuple(current["support_box"]))
    dx = float(previous["centroid_x"]) - float(current["centroid_x"])
    dy = float(previous["centroid_y"]) - float(current["centroid_y"])
    distance = math.sqrt(dx * dx + dy * dy)
    prev_box = tuple(previous["support_box"])
    curr_box = tuple(current["support_box"])
    diagonal = 0.5 * (
        math.hypot(prev_box[2] - prev_box[0], prev_box[3] - prev_box[1])
        + math.hypot(curr_box[2] - curr_box[0], curr_box[3] - curr_box[1])
    )
    distance_score = 1.0 - min(1.0, distance / max(diagonal * 1.25, 12.0))
    score_similarity = 1.0 - min(1.0, abs(float(previous["proposal_score"]) - float(current["proposal_score"])))
    return 0.55 * iou + 0.30 * distance_score + 0.15 * score_similarity


def _annotate_fragmentation(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scenario_name"]), int(row["frame_id"]))].append(row)

    for frame_rows in groups.values():
        for row in frame_rows:
            cluster_size = 1
            row_box = tuple(row["support_box"])
            row_diag = math.hypot(row_box[2] - row_box[0], row_box[3] - row_box[1])
            for other in frame_rows:
                if other is row:
                    continue
                other_box = tuple(other["support_box"])
                center_distance = math.hypot(
                    float(row["centroid_x"]) - float(other["centroid_x"]),
                    float(row["centroid_y"]) - float(other["centroid_y"]),
                )
                if box_iou(row_box, other_box) > 0.0 or center_distance <= max(18.0, 0.75 * row_diag):
                    cluster_size += 1
            row["fragment_cluster_size"] = int(cluster_size)


def _classify_rows(rows: list[dict[str, Any]]) -> None:
    rows_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_scenario[str(row["scenario_name"])].append(row)

    for scenario_name, scenario_rows in rows_by_scenario.items():
        stats = _scenario_stats(scenario_rows)
        for row in scenario_rows:
            fp_type, source_stage = _classify_row(row, stats)
            row["fp_type"] = fp_type
            row["source_stage"] = source_stage


def _scenario_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    def q(key: str, q_value: float, default: float) -> float:
        values = np.array([float(row[key]) for row in rows], dtype=np.float32)
        return float(np.quantile(values, q_value)) if values.size else default

    return {
        "grad_q75": q("background_gradient_mean", 0.75, 0.12),
        "grad_q90": q("background_gradient_mean", 0.90, 0.16),
        "grad_q55": q("background_gradient_mean", 0.55, 0.08),
        "fill_q15": q("region_fill_ratio", 0.15, 0.18),
        "fill_q35": q("region_fill_ratio", 0.35, 0.32),
        "bbox_q90": q("bbox_area", 0.90, 12000.0),
        "region_q35": q("region_area", 0.35, 320.0),
        "persist_q70": q("persistence_length", 0.70, 3.0),
        "persist_q80": q("persistence_length", 0.80, 3.0),
        "stability_q35": q("center_stability", 0.35, 8.0),
        "stability_q50": q("center_stability", 0.50, 8.0),
        "drift_q70": q("drift_strength", 0.70, 0.18),
        "drift_q85": q("drift_strength", 0.85, 0.24),
    }


def _classify_row(row: dict[str, Any], stats: dict[str, float]) -> tuple[str, str]:
    bbox_area = float(row["bbox_area"])
    region_area = float(row["region_area"])
    fill_ratio = float(row["region_fill_ratio"])
    compactness = float(row["region_compactness"])
    aspect_ratio = float(row["bbox_aspect_ratio"])
    persistence = int(row["persistence_length"])
    drift = float(row["drift_strength"])
    gradient = float(row["background_gradient_mean"])
    fragment_cluster = int(row["fragment_cluster_size"])
    max_gt_bbox_area = float(row["max_gt_bbox_area"])
    dominance = str(row["dominant_term"])

    oversized_against_gt = max_gt_bbox_area > 0 and bbox_area >= 1.75 * max_gt_bbox_area
    oversized_absolute = bbox_area >= max(stats["bbox_q90"], 9000.0)
    if (oversized_against_gt or oversized_absolute) and fill_ratio <= max(stats["fill_q35"], 0.50):
        return "overgrown_component_fp", "component"

    if fill_ratio <= min(stats["fill_q15"], 0.18) and compactness >= 0.08:
        return "box_fitting_artifact_fp", "box_fit"

    if fragment_cluster >= 2 and region_area <= max(stats["region_q35"], 900.0):
        return "fragmented_component_fp", "component"

    # Boundary-driven false positives should be assigned before long-lived field residue.
    # This keeps block-edge / seam responses out of the generic upstream-noise bucket.
    if (
        gradient >= max(stats["grad_q90"], 0.055)
        or int(row["near_boundary"]) == 1
        or dominance == "boundary"
    ) and persistence <= max(5, int(round(stats["persist_q70"]))):
        if (
            aspect_ratio >= 1.4
            or int(row["near_boundary"]) == 1
            or fill_ratio <= max(stats["fill_q35"], 0.50)
            or gradient >= max(stats["grad_q90"], 0.065)
        ):
            return "background_block_boundary_fp", "field"

    # Static texture is persistent, spatially stable, and repeatedly wins in place.
    if (
        persistence >= max(6, int(round(stats["persist_q70"])) + 1)
        and float(row["center_stability"]) <= max(stats["stability_q50"], 6.5)
        and dominance != "surprise"
    ):
        return "static_texture_fp", "field"

    # Drift-driven background residue tends to persist while frame-level drift is elevated.
    if (
        persistence >= max(5, int(round(stats["persist_q70"])))
        and drift >= max(stats["drift_q85"], 0.22)
        and dominance in {"persistence", "habituation", "boundary"}
    ):
        return "drift_region_fp", "field"

    if (
        persistence >= 4
        and drift >= max(stats["drift_q70"], 0.18)
        and float(row["center_stability"]) > max(stats["stability_q50"], 5.5)
        and dominance in {"persistence", "habituation"}
    ):
        return "drift_region_fp", "field"

    if persistence >= 5 and float(row["center_stability"]) <= max(stats["stability_q35"], 6.0):
        return "static_texture_fp", "field"

    return "upstream_noise_fp", "field"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No false-positive rows collected.")
    fieldnames = [
        "scenario_name",
        "sequence_id",
        "frame_id",
        "proposal_id",
        "fp_type",
        "raw_objectness_score",
        "proposal_score",
        "region_area",
        "bbox_area",
        "bbox_aspect_ratio",
        "bbox_tightness",
        "persistence_length",
        "chain_id",
        "center_stability",
        "near_boundary",
        "region_compactness",
        "region_fill_ratio",
        "boundary_smoothness",
        "overlaps_gt",
        "max_iou_with_gt",
        "region_to_gt_cover",
        "source_stage",
        "dominant_term",
        "boundary_mean",
        "persistence_mean",
        "surprise_mean",
        "habituation_mean",
        "background_gradient_mean",
        "drift_strength",
        "blur_level",
        "noise_level",
        "reentry_event",
        "max_gt_bbox_area",
        "max_gt_mask_area",
        "fragment_cluster_size",
        "support_box",
        "centroid_x",
        "centroid_y",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(serialize_row({key: row[key] for key in fieldnames}))


def _build_summary_markdown(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row["fp_type"]) for row in rows)
    source_counts = Counter(str(row["source_stage"]) for row in rows)
    by_scenario: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_scenario[str(row["scenario_name"])][str(row["fp_type"])] += 1

    lines = [
        "# Proposal FP Attribution Summary",
        "",
        "## Overall Distribution",
        "",
        "| fp_type | count | ratio |",
        "| --- | ---: | ---: |",
    ]
    total = max(1, len(rows))
    for fp_type, count in counts.most_common():
        lines.append(f"| {fp_type} | {count} | {count / total:.4f} |")

    lines.extend(
        [
            "",
            "## Source Stage Distribution",
            "",
            "| source_stage | count | ratio |",
            "| --- | ---: | ---: |",
        ]
    )
    for source_stage, count in source_counts.most_common():
        lines.append(f"| {source_stage} | {count} | {count / total:.4f} |")

    lines.extend(["", "## Scenario Breakdown", ""])
    for scenario_name, counter in sorted(by_scenario.items()):
        scenario_total = max(1, sum(counter.values()))
        lines.append(f"### {scenario_name}")
        lines.append("")
        lines.append("| fp_type | count | ratio |")
        lines.append("| --- | ---: | ---: |")
        for fp_type, count in counter.most_common():
            lines.append(f"| {fp_type} | {count} | {count / scenario_total:.4f} |")
        lines.append("")

    geometry_tail = sum(
        1
        for row in rows
        if str(row["fp_type"]) in {"fragmented_component_fp", "box_fitting_artifact_fp", "overgrown_component_fp"}
    )
    lines.extend(
        [
            "## Readout",
            "",
            _build_readout_sentence(rows),
            "",
            f"Geometry-stage tail (`fragmented_component_fp` + `box_fitting_artifact_fp` + `overgrown_component_fp`) accounts for {geometry_tail}/{len(rows)} = {geometry_tail / total:.4f}.",
            "",
            _build_scenario_readout(rows),
            "",
            "The main purpose of this audit is attribution, not filtering. The next Phase 2F-R step should target the dominant source stage first: field suppression if field-driven FP dominates, connected-component repair if overgrown or fragmented components dominate, or box fitting if low-fill artifacts dominate.",
        ]
    )
    return "\n".join(lines)


def _build_readout_sentence(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No false-positive proposals were collected."
    counts = Counter(str(row["fp_type"]) for row in rows)
    source_counts = Counter(str(row["source_stage"]) for row in rows)
    top_fp, top_fp_count = counts.most_common(1)[0]
    top_source, top_source_count = source_counts.most_common(1)[0]
    top_mix = _format_top_mix(counts, len(rows), limit=3)
    return (
        f"Dominant FP class is `{top_fp}` ({top_fp_count}/{len(rows)} = {top_fp_count / max(1, len(rows)):.4f}); "
        f"dominant source stage is `{top_source}` ({top_source_count}/{len(rows)} = {top_source_count / max(1, len(rows)):.4f}). "
        f"Top FP mix: {top_mix}."
    )


def _build_design_notes() -> str:
    return "\n".join(
        [
            "# Phase 2F-R Design Notes",
            "",
            "## Scope",
            "",
            "This pass only completes Part A: false-positive attribution on Track A and Track C bridge-synthetic scenarios. Proposal generation is unchanged.",
            "",
            "## Audit Primitive",
            "",
            "- objectness heatmap is kept as-is",
            "- proposals are reconstructed as connected support regions from the existing binary mask",
            "- each region is described by area, bbox area, fill ratio, compactness, aspect ratio, background gradient, and dominant objectness term",
            "- unmatched proposals are chained across adjacent frames to estimate persistence and center stability",
            "",
            "## Attribution Rule",
            "",
            "FP attribution is treated as a structural source diagnosis, not as a deletion policy. Each false-positive proposal is mapped to one mutually exclusive category and one source stage: field, component, or box_fit.",
            "",
            "## Expected Next Step",
            "",
            "Once the dominant failure source is confirmed, Part B should replace raw bbox extraction with region refinement and refined box fitting, instead of adding more threshold-based filters.",
        ]
    )


def _build_final_summary(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row["fp_type"]) for row in rows)
    source_counts = Counter(str(row["source_stage"]) for row in rows)
    by_scenario: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_scenario[str(row["scenario_name"])][str(row["fp_type"])] += 1
    total = max(1, len(rows))
    top_fp, top_fp_count = counts.most_common(1)[0]
    top_source, top_source_count = source_counts.most_common(1)[0]
    geometry_tail = sum(
        1
        for row in rows
        if str(row["fp_type"]) in {"fragmented_component_fp", "box_fitting_artifact_fp", "overgrown_component_fp"}
    )
    return "\n".join(
        [
            "# Phase 2F-R Final Summary v1",
            "",
            "This run completes only Stage 1 / Part A: false-positive attribution audit.",
            "",
            f"- total false-positive proposals audited: {len(rows)}",
            f"- dominant FP type: `{top_fp}` ({top_fp_count / total:.4f})",
            f"- dominant source stage: `{top_source}` ({top_source_count / total:.4f})",
            f"- top FP mix: {_format_top_mix(counts, total, limit=4)}",
            f"- geometry-stage tail share: {geometry_tail / total:.4f}",
            "",
            "Scenario split:",
            "",
            _build_scenario_bullet("track_a_bridge", by_scenario.get("track_a_bridge", Counter())),
            _build_scenario_bullet("track_c_long_horizon", by_scenario.get("track_c_long_horizon", Counter())),
            "",
            "Conclusion for the next step:",
            "",
            "Do not add more heuristic proposal filters first. Part B should prioritize field-side background suppression and object-aligned region extraction, then repair the smaller but real geometry tail with refined connected-component shaping and tighter box fitting.",
        ]
    )


def _build_scenario_bullet(scenario_name: str, counter: Counter[str]) -> str:
    total = max(1, sum(counter.values()))
    if not counter:
        return f"- {scenario_name}: no false-positive proposals were collected."
    top_type, top_count = counter.most_common(1)[0]
    geometry_tail = (
        counter.get("fragmented_component_fp", 0)
        + counter.get("box_fitting_artifact_fp", 0)
        + counter.get("overgrown_component_fp", 0)
    )
    return (
        f"- {scenario_name}: dominant FP is `{top_type}` ({top_count / total:.4f}); "
        f"top mix is {_format_top_mix(counter, total, limit=3)}; "
        f"geometry tail is {geometry_tail / total:.4f}."
    )


def _format_top_mix(counter: Counter[str], total: int, *, limit: int = 3) -> str:
    if not counter:
        return "none"
    pieces = []
    for fp_type, count in counter.most_common(limit):
        pieces.append(f"`{fp_type}` {count / max(1, total):.4f}")
    return ", ".join(pieces)


def _build_scenario_readout(rows: list[dict[str, Any]]) -> str:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[str(row["scenario_name"])].append(row)

    snippets: list[str] = []
    for scenario_name, scenario_rows in sorted(by_scenario.items()):
        counter = Counter(str(row["fp_type"]) for row in scenario_rows)
        total = max(1, len(scenario_rows))
        top_type, top_count = counter.most_common(1)[0]
        geometry_tail = counter.get("fragmented_component_fp", 0) + counter.get("box_fitting_artifact_fp", 0) + counter.get("overgrown_component_fp", 0)
        snippets.append(
            f"`{scenario_name}` is led by `{top_type}` ({top_count / total:.4f}); geometry-driven tail is {geometry_tail / total:.4f}."
        )
    return " ".join(snippets)


def _save_gallery(path: Path, rows: list[dict[str, Any]], scenario_payloads: dict[str, dict[str, Any]]) -> None:
    selected = _select_gallery_rows(rows)
    if not selected:
        return

    ncols = min(3, len(selected))
    nrows = int(math.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.2 * nrows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    for axis in axes_array.ravel():
        axis.axis("off")

    frame_cache = _build_frame_cache_for_gallery(selected, scenario_payloads)
    for axis, row in zip(axes_array.ravel(), selected):
        frame_payload = frame_cache[(str(row["scenario_name"]), int(row["frame_id"]))]
        frame = frame_payload["frame"]
        gt_boxes = frame_payload["gt_boxes"]
        descriptor = frame_payload["descriptor"]
        axis.imshow(frame)
        axis.axis("off")
        for gt_box in gt_boxes:
            x1, y1, x2, y2 = gt_box
            axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.5, ec="#22c55e"))
        x1, y1, x2, y2 = descriptor.box
        overlay = np.zeros((*frame.shape[:2], 4), dtype=np.float32)
        overlay[y1:y2, x1:x2, 0] = descriptor.local_mask.astype(np.float32)
        overlay[y1:y2, x1:x2, 2] = descriptor.local_mask.astype(np.float32)
        overlay[y1:y2, x1:x2, 3] = descriptor.local_mask.astype(np.float32) * 0.30
        axis.imshow(overlay)
        axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.8, ec="#f8fafc"))
        axis.set_title(
            f"{row['scenario_name']} f={row['frame_id']}\n"
            f"{row['fp_type']} | src={row['source_stage']}\n"
            f"raw={row['raw_objectness_score']:.3f} fill={row['region_fill_ratio']:.3f}",
            fontsize=9,
        )

    fig.savefig(path, dpi=180)
    plt.close(fig)


def _select_gallery_rows(rows: list[dict[str, Any]], max_items: int = 9) -> list[dict[str, Any]]:
    candidates = [row for row in rows if float(row["max_iou_with_gt"]) <= 0.01 and float(row["region_to_gt_cover"]) <= 0.01]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_type[str(row["fp_type"])].append(row)
    selected: list[dict[str, Any]] = []
    for fp_type, group in sorted(by_type.items()):
        group.sort(key=lambda item: (float(item["raw_objectness_score"]), float(item["bbox_area"])), reverse=True)
        selected.append(group[0])
        if len(selected) >= max_items:
            break
    if len(selected) < max_items:
        remaining = [row for row in candidates if row not in selected]
        remaining.sort(key=lambda item: (float(item["raw_objectness_score"]), float(item["bbox_area"])), reverse=True)
        selected.extend(remaining[: max_items - len(selected)])
    return selected[:max_items]


def _build_frame_cache_for_gallery(
    selected_rows: list[dict[str, Any]],
    scenario_payloads: dict[str, dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    wanted = {(str(row["scenario_name"]), int(row["frame_id"]), int(row["proposal_id"])) for row in selected_rows}
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    for scenario_name, run_payload in scenario_payloads.items():
        for frame_record in run_payload["frame_records"]:
            frame_id = int(frame_record["frame_index"])
            matching_rows = [item for item in wanted if item[0] == scenario_name and item[1] == frame_id]
            if not matching_rows:
                continue
            descriptors = {int(descriptor.component_id): descriptor for descriptor in frame_record["components"]}
            for _, _, proposal_id in matching_rows:
                descriptor = descriptors.get(proposal_id)
                if descriptor is None:
                    continue
                cache[(scenario_name, frame_id)] = {
                    "frame": frame_record["frame"],
                    "gt_boxes": frame_record["gt_boxes"],
                    "descriptor": descriptor,
                }
    return cache


if __name__ == "__main__":
    main()
