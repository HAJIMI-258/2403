"""Run the visual cognitive loop on LaSOT re-entry events.

This is a real-pixel, oracle-ledger evaluation. LaSOT annotations are used to
derive visibility re-entry events and to audit results, not to make online
memory/retrieval decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.external.base_video_memory_dataset import FrameSampleExternal  # noqa: E402
from datasets.external.lasot_adapter import LaSOTAdapter  # noqa: E402
from nops_owr.cognition import PredictiveRecognizer  # noqa: E402
from nops_owr.cognition.object_file import ObjectFile, ObjectFileBuilder, SupportMaskSummary  # noqa: E402
from nops_owr.cognition.visual_cognitive_loop import CognitiveFrameResult, VisualCognitiveLoop  # noqa: E402
from nops_owr.encoder.spike_encoder import MinimalSpikeEncoder, SpikeEncoding  # noqa: E402
from nops_owr.evaluation.attention_profiles import build_attention_from_profile  # noqa: E402
from nops_owr.evaluation.reentry_audit import (  # noqa: E402
    bbox_iou,
    failure_bucket,
    find_target_episode_from_bundles,
    gap_bucket,
    summarize_reentry_rows,
    write_reentry_report,
)
from nops_owr.evaluation.objectness_profiles import build_objectness_from_profile  # noqa: E402
from nops_owr.memory import EpisodicMemory, MinimalPrototypeMemory, RetrievalContext  # noqa: E402
from nops_owr.tracking.temporal_identity import MinimalTemporalIdentityTracker  # noqa: E402


REENTRY_FIELDNAMES = [
    "dataset_name",
    "sequence_id",
    "category",
    "instance_id",
    "disappear_frame",
    "reappear_frame",
    "gap_length",
    "gap_bucket",
    "gt_box_present",
    "matched_object_iou",
    "proposal_or_object_missing",
    "matched_object_file_id",
    "object_attended",
    "attention_failure",
    "target_episode_exists",
    "target_episode_id",
    "target_episode_closed",
    "target_episode_rank",
    "target_episode_score",
    "target_content_score",
    "target_support_score",
    "target_context_score",
    "target_motion_score",
    "target_temporal_score",
    "target_accessibility_score",
    "target_base_score",
    "target_adjusted_score",
    "target_closed_bonus",
    "target_reentry_gap_bonus",
    "target_active_conflict_penalty",
    "target_status_penalty",
    "top1_episode_id",
    "top1_score",
    "top1_margin",
    "top1_gt_instance_id",
    "top1_active_conflict",
    "top1_bundle_closed",
    "top1_reentry_gap",
    "top1_content_score",
    "top1_support_score",
    "top1_context_score",
    "top1_motion_score",
    "top1_temporal_score",
    "top1_accessibility_score",
    "top1_base_score",
    "top1_adjusted_score",
    "top1_closed_bonus",
    "top1_reentry_gap_bonus",
    "top1_active_conflict_penalty",
    "top1_status_penalty",
    "score_gap_top1_minus_target",
    "topk_contains_target",
    "decision_type",
    "rejection_reason",
    "success_same_instance",
    "false_resurrection",
    "unresolved_but_target_in_topk",
    "prediction_error",
    "familiarity_score",
    "novelty_score",
    "failure_bucket",
]

FRAME_FIELDNAMES = [
    "dataset_name",
    "sequence_id",
    "category",
    "frame_index",
    "object_file_count",
    "attended_object_count",
    "decision_count",
    "memory_context_used",
    "active_episode_count",
]


def run_eval(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_cognitive_reentry_eval",
    max_sequences: int = 10,
    max_frames: int = 300,
    min_gap: int = 8,
    category_filter: str = "",
    sequence_filter: str = "",
    strict_min_iou: float = 0.25,
    frame_stride: int = 1,
    start_index: int = 0,
    image_backend: str = "pil",
    max_image_side: int = 160,
    oracle_gt_box_eval_only: bool = False,
    objectness_profile: str = "A0_current_fixed_tau035_area16_props8",
    attention_profile: str = "A0_current_max4",
) -> dict[str, Any]:
    if image_backend != "pil":
        raise ValueError("Only --image-backend pil is currently supported.")
    _require_pil()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    adapter = LaSOTAdapter(root)
    sequences = _filter_sequences(
        list(adapter.iter_sequences()),
        category_filter=category_filter,
        sequence_filter=sequence_filter,
    )[: int(max_sequences)]

    frame_rows: list[dict[str, Any]] = []
    reentry_rows: list[dict[str, Any]] = []
    evaluated_sequences = 0
    for sequence_id in sequences:
        category = _category(sequence_id)
        all_frames = list(adapter.iter_frames(sequence_id))
        selected_frames = [
            frame
            for frame in all_frames
            if int(frame.frame_idx) >= int(start_index)
            and (int(frame.frame_idx) - int(start_index)) % max(1, int(frame_stride)) == 0
        ][: int(max_frames)]
        if len(selected_frames) < 2:
            continue
        frame_index_set = {int(frame.frame_idx) for frame in selected_frames}
        events = [
            event
            for event in adapter.derive_events(sequence_id)
            if int(event.gap_length) >= int(min_gap)
            and int(event.reappear_frame) in frame_index_set
        ]
        if not events:
            continue
        evaluated_sequences += 1
        loop = _build_loop(objectness_profile=objectness_profile, attention_profile=attention_profile)
        event_by_reappear: dict[int, list[Any]] = {}
        for event in events:
            event_by_reappear.setdefault(int(event.reappear_frame), []).append(event)

        prev_image = _load_image(selected_frames[0].frame_path, max_image_side=max_image_side)[0]
        for frame in selected_frames[1:]:
            current_image, scale_x, scale_y = _load_image(frame.frame_path, max_image_side=max_image_side)
            scaled_boxes = [_scale_box(box, scale_x, scale_y) for box in frame.boxes]
            result = loop.step(
                prev_image,
                current_image,
                int(frame.frame_idx),
                ground_truth={
                    "boxes": scaled_boxes,
                    # LaSOT is single-target; numeric id 1 is an eval-local target id.
                    "instance_ids": [1 for _ in scaled_boxes],
                    "concept_ids": [1 for _ in scaled_boxes],
                },
            )
            frame_rows.append(
                {
                    "dataset_name": "lasot",
                    "sequence_id": sequence_id,
                    "category": category,
                    "frame_index": frame.frame_idx,
                    "object_file_count": len(result.object_files),
                    "attended_object_count": len(result.attended_object_files),
                    "decision_count": len(result.recognition_decisions),
                    "memory_context_used": int(result.memory_context_used),
                    "active_episode_count": result.active_episode_count,
                }
            )
            for event in event_by_reappear.get(int(frame.frame_idx), []):
                gt_box = scaled_boxes[0] if scaled_boxes else None
                reentry_rows.append(
                    _reentry_row(
                        event=event,
                        result=result,
                        loop=loop,
                        gt_box=gt_box,
                        strict_min_iou=strict_min_iou,
                        oracle_gt_box_eval_only=oracle_gt_box_eval_only,
                    )
                )
            prev_image = current_image

    _write_csv(output_path / "frame_metrics.csv", frame_rows, FRAME_FIELDNAMES)
    _write_csv(output_path / "reentry_events.csv", reentry_rows, REENTRY_FIELDNAMES)
    summary = summarize_reentry_rows(
        reentry_rows,
        dataset_name="lasot",
        sequence_count=len(sequences),
        evaluated_sequence_count=evaluated_sequences,
        extra={
            "oracle_gt_box_eval_only": int(oracle_gt_box_eval_only),
            "category_filter": category_filter,
            "sequence_filter": sequence_filter,
            "max_image_side": int(max_image_side),
            "objectness_profile": objectness_profile,
            "attention_profile": attention_profile,
        },
    )
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_reentry_report(output_path / "report.md", summary, reentry_rows, title="LaSOT Cognitive Re-entry Eval")
    return summary


def _build_loop(
    objectness_profile: str = "A0_current_fixed_tau035_area16_props8",
    attention_profile: str = "A0_current_max4",
) -> VisualCognitiveLoop:
    return VisualCognitiveLoop(
        encoder=MinimalSpikeEncoder(),
        objectness_field=build_objectness_from_profile(objectness_profile),
        tracker=MinimalTemporalIdentityTracker(),
        prototype_memory=MinimalPrototypeMemory(memory_budget=96),
        attention_gate=build_attention_from_profile(attention_profile),
        episodic_memory=EpisodicMemory(memory_budget=256),
        recognizer=PredictiveRecognizer(),
    )


def _reentry_row(
    *,
    event: Any,
    result: CognitiveFrameResult,
    loop: VisualCognitiveLoop,
    gt_box: tuple[float, float, float, float] | None,
    strict_min_iou: float,
    oracle_gt_box_eval_only: bool,
    oracle_force_query: bool = False,
) -> dict[str, Any]:
    instance_numeric_id = 1
    matched_object, matched_iou = _matched_object(result, instance_numeric_id, gt_box, strict_min_iou)
    retrievals = []
    decision = None
    object_attended = 0
    if oracle_force_query and gt_box is not None:
        matched_object = _oracle_object_file(result.encoding, gt_box, int(event.reappear_frame), str(event.sequence_id))
        matched_iou = 1.0
        object_attended = 1
        retrieval_context = RetrievalContext(
            frame_index=int(event.reappear_frame),
            mode="reentry",
            min_reentry_gap=8,
            prefer_closed_episodes=True,
            suppress_active_conflicts=True,
        )
        retrievals = loop.episodic_memory.retrieve(matched_object, top_k=5, context=retrieval_context)
        decision = loop.recognizer.recognize(
            matched_object,
            episodic_candidates=retrievals,
            retrieval_mode="reentry",
        )
    elif matched_object is not None:
        object_attended = int(any(row.object_file_id == matched_object.object_file_id for row in result.attended_object_files))
        retrievals = result.episodic_retrievals.get(matched_object.object_file_id, [])
        decision = next(
            (row for row in result.recognition_decisions if row.object_file_id == matched_object.object_file_id),
            None,
        )
    elif oracle_gt_box_eval_only and gt_box is not None:
        matched_object = _oracle_object_file(result.encoding, gt_box, int(event.reappear_frame), str(event.sequence_id))
        matched_iou = 1.0
        object_attended = 1
        retrieval_context = RetrievalContext(
            frame_index=int(event.reappear_frame),
            mode="reentry",
            min_reentry_gap=8,
            prefer_closed_episodes=True,
            suppress_active_conflicts=True,
        )
        retrievals = loop.episodic_memory.retrieve(matched_object, top_k=5, context=retrieval_context)
        decision = loop.recognizer.recognize(
            matched_object,
            episodic_candidates=retrievals,
            retrieval_mode="reentry",
        )

    top1 = retrievals[0] if retrievals else None
    top_episode = None if top1 is None else top1.bundle
    top_gt = None if top_episode is None else top_episode.metadata.get("gt_instance_id")
    linked_episode_id = None if decision is None else decision.linked_episode_id
    linked_episode = loop.episodic_memory.get_episode(linked_episode_id)
    linked_gt = None if linked_episode is None else linked_episode.metadata.get("gt_instance_id")
    target_episode = find_target_episode_from_bundles(
        loop.episodic_memory.bundles,
        instance_numeric_id,
        int(event.reappear_frame),
    )
    target_episode_id = None if target_episode is None else target_episode.episode_id
    target_candidate = None
    target_rank = 0
    target_score = 0.0
    for candidate in retrievals:
        if target_episode_id is not None and candidate.bundle.episode_id == target_episode_id:
            target_candidate = candidate
            target_rank = int(candidate.rank)
            target_score = float(candidate.score)
            break

    decision_type = "" if decision is None else decision.decision_type
    rejection_reason = "" if decision is None else str(decision.metadata.get("rejection_reason", ""))
    success = int(decision_type == "same_instance" and ((linked_gt == instance_numeric_id) or (top_gt == instance_numeric_id)))
    false_resurrection = int(
        decision_type == "same_instance"
        and ((linked_gt is not None and linked_gt != instance_numeric_id) or (linked_gt is None and top_gt is not None and top_gt != instance_numeric_id))
    )
    topk_contains_target = int(target_rank > 0)
    bucket = failure_bucket(
        gt_box_present=gt_box is not None,
        matched_object=matched_object is not None,
        object_attended=bool(object_attended),
        target_episode=target_episode is not None,
        target_rank=target_rank,
        decision_type=decision_type,
        rejection_reason=rejection_reason,
        success=bool(success),
        false_resurrection=bool(false_resurrection),
    )
    category = event.metadata.get("category_id", _category(str(event.sequence_id)))
    return {
        "dataset_name": "lasot",
        "sequence_id": event.sequence_id,
        "category": category,
        "instance_id": event.instance_id,
        "disappear_frame": event.disappear_frame,
        "reappear_frame": event.reappear_frame,
        "gap_length": event.gap_length,
        "gap_bucket": gap_bucket(int(event.gap_length)),
        "gt_box_present": int(gt_box is not None),
        "matched_object_iou": float(matched_iou),
        "proposal_or_object_missing": int(matched_object is None),
        "matched_object_file_id": "" if matched_object is None else matched_object.object_file_id,
        "object_attended": int(object_attended),
        "attention_failure": int(matched_object is not None and not object_attended),
        "target_episode_exists": int(target_episode is not None),
        "target_episode_id": "" if target_episode is None else target_episode.episode_id,
        "target_episode_closed": 0 if target_episode is None else int(target_episode.closed),
        "target_episode_rank": target_rank,
        "target_episode_score": target_score,
        "target_content_score": _evidence(target_candidate, "content"),
        "target_support_score": _evidence(target_candidate, "support"),
        "target_context_score": _evidence(target_candidate, "context"),
        "target_motion_score": _evidence(target_candidate, "motion"),
        "target_temporal_score": _evidence(target_candidate, "temporal"),
        "target_accessibility_score": _evidence(target_candidate, "accessibility"),
        "target_base_score": _evidence(target_candidate, "base_score"),
        "target_adjusted_score": _evidence(target_candidate, "adjusted_score"),
        "target_closed_bonus": _evidence(target_candidate, "closed_bonus"),
        "target_reentry_gap_bonus": _evidence(target_candidate, "reentry_gap_bonus"),
        "target_active_conflict_penalty": _evidence(target_candidate, "active_conflict_penalty"),
        "target_status_penalty": _evidence(target_candidate, "status_penalty"),
        "top1_episode_id": "" if top_episode is None else top_episode.episode_id,
        "top1_score": 0.0 if top1 is None else top1.score,
        "top1_margin": 0.0 if top1 is None else top1.margin_to_next,
        "top1_gt_instance_id": "" if top_gt is None else top_gt,
        "top1_active_conflict": 0 if top1 is None else int(top1.active_conflict),
        "top1_bundle_closed": 0 if top_episode is None else int(top_episode.closed),
        "top1_reentry_gap": 0 if top1 is None else top1.reentry_gap,
        "top1_content_score": _evidence(top1, "content"),
        "top1_support_score": _evidence(top1, "support"),
        "top1_context_score": _evidence(top1, "context"),
        "top1_motion_score": _evidence(top1, "motion"),
        "top1_temporal_score": _evidence(top1, "temporal"),
        "top1_accessibility_score": _evidence(top1, "accessibility"),
        "top1_base_score": _evidence(top1, "base_score"),
        "top1_adjusted_score": _evidence(top1, "adjusted_score"),
        "top1_closed_bonus": _evidence(top1, "closed_bonus"),
        "top1_reentry_gap_bonus": _evidence(top1, "reentry_gap_bonus"),
        "top1_active_conflict_penalty": _evidence(top1, "active_conflict_penalty"),
        "top1_status_penalty": _evidence(top1, "status_penalty"),
        "score_gap_top1_minus_target": (0.0 if top1 is None or target_candidate is None else float(top1.score - target_candidate.score)),
        "topk_contains_target": topk_contains_target,
        "decision_type": decision_type,
        "rejection_reason": rejection_reason,
        "success_same_instance": success,
        "false_resurrection": false_resurrection,
        "unresolved_but_target_in_topk": int(topk_contains_target and not success),
        "prediction_error": 0.0 if decision is None else decision.prediction_error,
        "familiarity_score": 0.0 if decision is None else decision.familiarity_score,
        "novelty_score": 0.0 if decision is None else decision.novelty_score,
        "failure_bucket": bucket,
    }


def _evidence(candidate: Any, key: str) -> float:
    if candidate is None:
        return 0.0
    breakdown = getattr(candidate, "evidence_breakdown", {}) or {}
    return float(breakdown.get(key, 0.0))


def _matched_object(
    result: CognitiveFrameResult,
    instance_numeric_id: int,
    gt_box: tuple[float, float, float, float] | None,
    strict_min_iou: float,
) -> tuple[ObjectFile | None, float]:
    best = None
    best_iou = 0.0
    for object_file in result.object_files:
        iou = float(object_file.metadata.get("gt_iou_eval_only", 0.0))
        if object_file.metadata.get("gt_instance_id") == instance_numeric_id:
            iou = max(iou, bbox_iou(object_file.box, gt_box))
        if iou > best_iou:
            best = object_file
            best_iou = iou
    if best is None or best_iou < float(strict_min_iou):
        return None, best_iou
    return best, best_iou


def _oracle_object_file(
    encoding: SpikeEncoding,
    gt_box: tuple[float, float, float, float],
    frame_index: int,
    sequence_id: str,
) -> ObjectFile:
    height, width = encoding.current_gray.shape
    x1, y1, x2, y2 = [int(round(v)) for v in gt_box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    box = (x1, y1, x2, y2)
    area = float((x2 - x1) * (y2 - y1))
    appearance = _appearance_signature(box, encoding)
    shape = np.asarray(
        [
            (x2 - x1) / max(1.0, float(width)),
            (y2 - y1) / max(1.0, float(height)),
            (x2 - x1) / max(1.0, float(y2 - y1)),
            area / max(1.0, float(width * height)),
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )
    context = np.asarray(
        [
            ((x1 + x2) * 0.5) / max(1.0, float(width)),
            ((y1 + y2) * 0.5) / max(1.0, float(height)),
            1.0,
            float(x1 <= 4 or y1 <= 4 or x2 >= width - 4 or y2 >= height - 4),
            min(x1, max(0, width - x2)) / max(1.0, float(width)),
            min(y1, max(0, height - y2)) / max(1.0, float(height)),
        ],
        dtype=np.float32,
    )
    return ObjectFile(
        object_file_id=f"oracle:{sequence_id}:{frame_index}",
        frame_index=int(frame_index),
        proposal_index=-1,
        box=box,
        raw_box=box,
        support_box=box,
        centroid=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
        area=area,
        score=1.0,
        quality_score=1.0,
        support_mask_summary=SupportMaskSummary(area=area, bbox=box, fill_ratio=1.0, compactness=1.0, boundary_smoothness=1.0),
        appearance_signature=appearance,
        shape_signature=shape,
        context_signature=context,
        motion_signature=np.zeros(0, dtype=np.float32),
        novelty_score=0.0,
        familiarity_score=1.0,
        prediction_error=0.0,
        confidence=1.0,
        metadata={"gt_instance_id": 1, "gt_iou_eval_only": 1.0, "oracle_eval_only": 1},
    )


def _appearance_signature(box: tuple[int, int, int, int], encoding: SpikeEncoding) -> np.ndarray:
    patches = [_crop(encoding.current_gray, box), _crop(encoding.edge_map, box), _crop(encoding.spike_response, box)]
    stats: list[float] = []
    for patch in patches:
        if patch.size == 0:
            stats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            values = patch.reshape(-1).astype(np.float32)
            stats.extend(
                [
                    float(np.mean(values)),
                    float(np.std(values)),
                    float(np.quantile(values, 0.25)),
                    float(np.quantile(values, 0.50)),
                    float(np.quantile(values, 0.75)),
                ]
            )
    return np.asarray(stats, dtype=np.float32)


def _crop(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return array[y1:y2, x1:x2].astype(np.float32, copy=False)


def _load_image(path: Path | None, *, max_image_side: int) -> tuple[np.ndarray, float, float]:
    if path is None:
        raise FileNotFoundError("Frame path is missing.")
    from PIL import Image

    image = Image.open(path).convert("RGB")
    original_w, original_h = image.size
    max_side = max(original_w, original_h)
    if int(max_image_side) > 0 and max_side > int(max_image_side):
        scale = float(max_image_side) / float(max_side)
        new_size = (max(1, int(round(original_w * scale))), max(1, int(round(original_h * scale))))
        image = image.resize(new_size)
    array = np.asarray(image, dtype=np.uint8)
    scaled_h, scaled_w = array.shape[:2]
    return array, scaled_w / max(1.0, float(original_w)), scaled_h / max(1.0, float(original_h))


def _scale_box(box: tuple[float, float, float, float], scale_x: float, scale_y: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


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


def _require_pil() -> None:
    try:
        import PIL  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on local optional package.
        raise RuntimeError("PIL/Pillow required for pixel eval") from exc


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_cognitive_reentry_eval")
    parser.add_argument("--max-sequences", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--strict-min-iou", type=float, default=0.25)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--image-backend", default="pil")
    parser.add_argument("--max-image-side", type=int, default=160)
    parser.add_argument("--oracle-gt-box-eval-only", type=int, default=0)
    parser.add_argument("--objectness-profile", default="A0_current_fixed_tau035_area16_props8")
    parser.add_argument("--attention-profile", default="A0_current_max4")
    args = parser.parse_args()
    summary = run_eval(
        root=args.root,
        output_dir=args.output_dir,
        max_sequences=args.max_sequences,
        max_frames=args.max_frames,
        min_gap=args.min_gap,
        category_filter=args.category_filter,
        sequence_filter=args.sequence_filter,
        strict_min_iou=args.strict_min_iou,
        frame_stride=args.frame_stride,
        start_index=args.start_index,
        image_backend=args.image_backend,
        max_image_side=args.max_image_side,
        oracle_gt_box_eval_only=bool(args.oracle_gt_box_eval_only),
        objectness_profile=args.objectness_profile,
        attention_profile=args.attention_profile,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
