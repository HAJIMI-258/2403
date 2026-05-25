"""LaSOT event-window evaluation for spiking object permanence.

Normal mode keeps spiking capsule writes GT-free. Oracle modes are offline
diagnostics that use GT boxes to separate objectness failure from capsule
retrieval failure.
"""

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
from experiments.run_lasot_cognitive_reentry_eval import _build_loop, _matched_object, _oracle_object_file  # noqa: E402
from nops_owr.cognition.permanence_recognizer import PermanenceRecognizer  # noqa: E402
from nops_owr.descriptor.spiking_invariant_descriptor import SpikingInvariantDescriptorBuilder  # noqa: E402
from nops_owr.evaluation.external_event_windows import (  # noqa: E402
    collect_lasot_reentry_events,
    frame_gt_box,
    frame_is_visible,
    load_rgb_frame,
    make_event_window,
    scale_box,
    sequence_category,
)
from nops_owr.evaluation.permanence_audit import (  # noqa: E402
    find_target_capsule_from_bank,
    permanence_failure_bucket,
    summarize_permanence_rows,
    write_permanence_report,
)
from nops_owr.evaluation.reentry_audit import gap_bucket  # noqa: E402
from nops_owr.memory.spiking_object_memory import SpikingObjectMemoryBank  # noqa: E402


REENTRY_FIELDS = [
    "dataset_name",
    "sequence_id",
    "category",
    "event_id",
    "instance_id",
    "disappear_frame",
    "reappear_frame",
    "gap_length",
    "gap_bucket",
    "mode",
    "gt_box_present",
    "matched_object_iou",
    "object_attended",
    "capsule_count_before_reentry",
    "capsule_count_after_reentry",
    "spiking_memory_bytes",
    "mean_spike_density",
    "target_capsule_exists",
    "target_capsule_id",
    "top1_capsule_id",
    "top1_score",
    "top1_margin",
    "top1_is_target_capsule",
    "top1_gt_instance_id",
    "target_capsule_rank",
    "target_capsule_score",
    "permanence_decision_type",
    "permanence_score",
    "permanence_false_resurrection_risk",
    "same_instance_success",
    "false_resurrection",
    "failure_bucket",
]

FRAME_FIELDS = [
    "dataset_name",
    "sequence_id",
    "category",
    "event_id",
    "frame_idx",
    "object_file_count",
    "attended_object_count",
    "capsule_count",
    "spiking_memory_bytes",
    "mean_spike_density",
    "permanence_decision_counts",
]


def run_eval(
    root: str | Path = "data/external/lasot",
    output_dir: str | Path = "results/lasot_spiking_permanence_eval",
    max_events: int = 50,
    min_gap: int = 8,
    pre_context: int = 80,
    post_context: int = 20,
    category_filter: str = "",
    sequence_filter: str = "",
    max_image_side: int = 160,
    strict_min_iou: float = 0.25,
    frame_stride: int = 1,
    objectness_profile: str = "A8_quantile_q050_component_props48",
    attention_profile: str = "A10_source_spatial_diverse_max16",
    spike_dim: int = 128,
    max_capsules: int = 128,
    match_profile: str = "hash_chroma_deform",
    same_object_threshold: float = 0.90,
    same_object_margin_threshold: float = 0.14,
    false_resurrection_risk_threshold: float = 0.25,
    mode: str = "normal",
    component_ranking_profile: str = "R0_current_quality",
    support_box_profile: str = "B0_refined_box_current",
) -> dict[str, Any]:
    if mode not in {"normal", "oracle_reappear_only", "oracle_history_and_reappear"}:
        raise ValueError(f"Unsupported mode: {mode}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapter = LaSOTAdapter(root)
    events = collect_lasot_reentry_events(
        adapter,
        min_gap=min_gap,
        category_filter=category_filter,
        sequence_filter=sequence_filter,
        max_events=max_events,
    )
    frames_by_sequence: dict[str, list[Any]] = {}
    reentry_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for event in events:
        frames = frames_by_sequence.setdefault(event.sequence_id, list(adapter.iter_frames(event.sequence_id)))
        window = make_event_window(
            frames,
            event,
            pre_context=pre_context,
            post_context=post_context,
            frame_stride=frame_stride,
        )
        if window is None:
            skipped["invalid_window"] += 1
            continue
        bank = SpikingObjectMemoryBank(
            max_capsules=max_capsules,
            spike_dim=spike_dim,
            min_match_score=same_object_threshold,
            min_same_object_margin=same_object_margin_threshold,
            match_profile=match_profile,
        )
        descriptor_builder = SpikingInvariantDescriptorBuilder(spike_dim=spike_dim, hash_bits=spike_dim)
        permanence = PermanenceRecognizer(
            same_object_threshold=same_object_threshold,
            same_object_margin_threshold=same_object_margin_threshold,
            false_resurrection_risk_threshold=false_resurrection_risk_threshold,
        )
        loop = _build_loop(
            objectness_profile=objectness_profile,
            attention_profile=attention_profile,
            component_ranking_profile=component_ranking_profile,
            support_box_profile=support_box_profile,
        )
        loop.spiking_memory_bank = bank
        loop.spiking_descriptor_builder = descriptor_builder
        loop.permanence_recognizer = permanence
        loop.use_spiking_long_term_memory = True
        event_id = _event_id(event)
        prev_image = None
        oracle_target_capsule_id: int | None = None
        for frame in window.frames:
            current_image, scale_x, scale_y = load_rgb_frame(frame.frame_path, max_image_side=max_image_side)
            if prev_image is None:
                prev_image = current_image
            gt_box = scale_box(frame_gt_box(frame), scale_x, scale_y)
            scaled_boxes = [] if gt_box is None else [gt_box]
            is_reappear = int(frame.frame_idx) == int(event.reappear_frame)
            capsule_count_before_reentry = len(bank) if is_reappear else 0
            original_spiking_enabled = loop.use_spiking_long_term_memory
            if is_reappear and mode != "normal":
                loop.use_spiking_long_term_memory = False
            result = loop.step(
                prev_image,
                current_image,
                int(frame.frame_idx),
                ground_truth={
                    "boxes": scaled_boxes,
                    "instance_ids": [1 for _ in scaled_boxes],
                    "concept_ids": [1 for _ in scaled_boxes],
                },
            )
            loop.use_spiking_long_term_memory = original_spiking_enabled
            _tag_eval_capsules(bank, result, frame_index=int(frame.frame_idx), reappear_frame=int(event.reappear_frame))
            if (
                mode == "oracle_history_and_reappear"
                and gt_box is not None
                and frame_is_visible(frame)
                and int(frame.frame_idx) < int(event.reappear_frame)
            ):
                oracle_object = _oracle_object_file(result.encoding, gt_box, int(frame.frame_idx), str(event.sequence_id))
                descriptor = descriptor_builder.build(oracle_object, result.encoding)
                if oracle_target_capsule_id is None:
                    oracle_target_capsule_id = bank.create_capsule(
                        descriptor,
                        frame_index=int(frame.frame_idx),
                        metadata={"object_id_eval_only": 1, "gt_instance_id": 1, "oracle_history_eval_only": 1},
                    )
                else:
                    bank.update_capsule(oracle_target_capsule_id, descriptor, int(frame.frame_idx), confidence=1.0)
                    bank.capsules[oracle_target_capsule_id].metadata.update(
                        {"object_id_eval_only": 1, "gt_instance_id": 1, "oracle_history_eval_only": 1}
                    )

            frame_rows.append(
                {
                    "dataset_name": "lasot",
                    "sequence_id": event.sequence_id,
                    "category": sequence_category(event.sequence_id),
                    "event_id": event_id,
                    "frame_idx": int(frame.frame_idx),
                    "object_file_count": len(result.object_files),
                    "attended_object_count": len(result.attended_object_files),
                    "capsule_count": len(bank),
                    "spiking_memory_bytes": bank.memory_bytes(),
                    "mean_spike_density": bank.mean_spike_density(),
                    "permanence_decision_counts": json.dumps(_decision_counts(result.attended_object_files), sort_keys=True),
                }
            )
            if is_reappear:
                row = _spiking_reentry_row(
                    event=event,
                    event_id=event_id,
                    mode=mode,
                    result=result,
                    gt_box=gt_box,
                    strict_min_iou=strict_min_iou,
                    bank=bank,
                    descriptor_builder=descriptor_builder,
                    permanence=permanence,
                    capsule_count_before_reentry=capsule_count_before_reentry,
                )
                reentry_rows.append(row)
            prev_image = current_image

    _write_csv(out / "spiking_reentry_events.csv", reentry_rows, REENTRY_FIELDS)
    _write_csv(out / "spiking_frame_metrics.csv", frame_rows, FRAME_FIELDS)
    summary = summarize_permanence_rows(
        reentry_rows,
        dataset_name="lasot",
        sequence_count=len({event.sequence_id for event in events}),
        extra={
            "mode": mode,
            "total_candidate_events": len(events),
            "skipped_event_count": int(sum(skipped.values())),
            "skip_reasons": dict(skipped),
            "objectness_profile": objectness_profile,
            "attention_profile": attention_profile,
            "component_ranking_profile": component_ranking_profile,
            "support_box_profile": support_box_profile,
            "match_profile": match_profile,
            "same_object_threshold": float(same_object_threshold),
            "same_object_margin_threshold": float(same_object_margin_threshold),
            "false_resurrection_risk_threshold": float(false_resurrection_risk_threshold),
            "gt_used_for_offline_diagnostic_only": int(mode != "normal"),
        },
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_permanence_report(out / "report.md", summary, reentry_rows, title="LaSOT Spiking Permanence Eval")
    return summary


def _spiking_reentry_row(
    *,
    event: Any,
    event_id: str,
    mode: str,
    result: Any,
    gt_box: tuple[float, float, float, float] | None,
    strict_min_iou: float,
    bank: SpikingObjectMemoryBank,
    descriptor_builder: SpikingInvariantDescriptorBuilder,
    permanence: PermanenceRecognizer,
    capsule_count_before_reentry: int,
) -> dict[str, Any]:
    matched_object = None
    matched_iou = 0.0
    object_attended = 0
    if mode == "normal":
        matched_object, matched_iou = _matched_object(result, 1, gt_box, strict_min_iou)
        object_attended = int(
            matched_object is not None
            and any(row.object_file_id == matched_object.object_file_id for row in result.attended_object_files)
        )
        query_object = matched_object if object_attended else None
    else:
        query_object = None if gt_box is None else _oracle_object_file(result.encoding, gt_box, int(event.reappear_frame), str(event.sequence_id))
        matched_object = query_object
        matched_iou = 1.0 if query_object is not None else 0.0
        object_attended = int(query_object is not None)

    matches = []
    decision = None
    if query_object is not None:
        descriptor = descriptor_builder.build(query_object, result.encoding)
        matches = bank.match(descriptor, frame_index=int(event.reappear_frame), top_k=5)
        decision = permanence.decide(query_object, matches)
    target_capsule = find_target_capsule_from_bank(bank, 1, int(event.reappear_frame))
    target_capsule_id = None if target_capsule is None else int(target_capsule.capsule_id)
    target_rank = 0
    target_score = 0.0
    for match in matches:
        if target_capsule_id is not None and int(match.capsule_id) == target_capsule_id:
            target_rank = int(match.rank)
            target_score = float(match.score)
            break
    top1 = matches[0] if matches else None
    top1_capsule_id = None if top1 is None else int(top1.capsule_id)
    top1_capsule = None if top1_capsule_id is None else bank.capsules.get(top1_capsule_id)
    top1_gt = "" if top1_capsule is None else top1_capsule.metadata.get("object_id_eval_only", top1_capsule.metadata.get("gt_instance_id", ""))
    decision_type = "" if decision is None else decision.decision_type
    decision_capsule_id = None if decision is None else decision.capsule_id
    success = int(
        decision_type in {"same_object", "familiar_but_deformed"}
        and target_capsule_id is not None
        and int(decision_capsule_id if decision_capsule_id is not None else -1) == int(target_capsule_id)
    )
    false_resurrection = int(
        decision_type in {"same_object", "familiar_but_deformed"}
        and decision_capsule_id is not None
        and target_capsule_id is not None
        and int(decision_capsule_id) != int(target_capsule_id)
    )
    if decision_type in {"same_object", "familiar_but_deformed"} and target_capsule_id is None and top1_gt not in {"", 1, "1"}:
        false_resurrection = 1
    bucket = permanence_failure_bucket(
        gt_box_present=gt_box is not None,
        matched_object=matched_object is not None,
        object_attended=bool(object_attended),
        target_capsule=target_capsule is not None,
        target_rank=target_rank,
        decision_type=decision_type,
        success=bool(success),
        false_resurrection=bool(false_resurrection),
    )
    return {
        "dataset_name": "lasot",
        "sequence_id": event.sequence_id,
        "category": sequence_category(str(event.sequence_id)),
        "event_id": event_id,
        "instance_id": event.instance_id,
        "disappear_frame": event.disappear_frame,
        "reappear_frame": event.reappear_frame,
        "gap_length": event.gap_length,
        "gap_bucket": gap_bucket(int(event.gap_length)),
        "mode": mode,
        "gt_box_present": int(gt_box is not None),
        "matched_object_iou": float(matched_iou),
        "object_attended": int(object_attended),
        "capsule_count_before_reentry": int(capsule_count_before_reentry),
        "capsule_count_after_reentry": len(bank),
        "spiking_memory_bytes": bank.memory_bytes(),
        "mean_spike_density": bank.mean_spike_density(),
        "target_capsule_exists": int(target_capsule is not None),
        "target_capsule_id": "" if target_capsule_id is None else target_capsule_id,
        "top1_capsule_id": "" if top1_capsule_id is None else top1_capsule_id,
        "top1_score": 0.0 if top1 is None else float(top1.score),
        "top1_margin": 0.0 if top1 is None else float(top1.metadata.get("top1_margin", 0.0)),
        "top1_is_target_capsule": int(target_capsule_id is not None and top1_capsule_id == target_capsule_id),
        "top1_gt_instance_id": top1_gt,
        "target_capsule_rank": target_rank,
        "target_capsule_score": target_score,
        "permanence_decision_type": decision_type,
        "permanence_score": 0.0 if decision is None else float(decision.score),
        "permanence_false_resurrection_risk": 0.0 if decision is None else float(decision.false_resurrection_risk),
        "same_instance_success": success,
        "false_resurrection": false_resurrection,
        "failure_bucket": bucket,
    }


def _tag_eval_capsules(bank: SpikingObjectMemoryBank, result: Any, *, frame_index: int, reappear_frame: int) -> None:
    del reappear_frame
    for object_file in result.attended_object_files:
        if object_file.metadata.get("gt_instance_id") is None:
            continue
        capsule_id = object_file.metadata.get("spiking_capsule_id")
        if capsule_id in {"", None}:
            continue
        capsule = bank.capsules.get(int(capsule_id))
        if capsule is None:
            continue
        capsule.metadata.setdefault("object_id_eval_only", int(object_file.metadata["gt_instance_id"]))
        capsule.metadata.setdefault("gt_instance_id", int(object_file.metadata["gt_instance_id"]))
        capsule.metadata["last_eval_tag_frame"] = int(frame_index)


def _decision_counts(object_files: list[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for object_file in object_files:
        decision = str(object_file.metadata.get("permanence_decision_type", "none"))
        counts[decision] += 1
    return dict(counts)


def _event_id(event: Any) -> str:
    return f"{event.sequence_id}:{event.disappear_frame}:{event.reappear_frame}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/lasot")
    parser.add_argument("--output-dir", default="results/lasot_spiking_permanence_eval")
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--pre-context", type=int, default=80)
    parser.add_argument("--post-context", type=int, default=20)
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--sequence-filter", default="")
    parser.add_argument("--max-image-side", type=int, default=160)
    parser.add_argument("--strict-min-iou", type=float, default=0.25)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--objectness-profile", default="A8_quantile_q050_component_props48")
    parser.add_argument("--attention-profile", default="A10_source_spatial_diverse_max16")
    parser.add_argument("--spike-dim", type=int, default=128)
    parser.add_argument("--max-capsules", type=int, default=128)
    parser.add_argument("--match-profile", default="hash_chroma_deform")
    parser.add_argument("--same-object-threshold", type=float, default=0.90)
    parser.add_argument("--same-object-margin-threshold", type=float, default=0.14)
    parser.add_argument("--false-resurrection-risk-threshold", type=float, default=0.25)
    parser.add_argument("--mode", default="normal", choices=["normal", "oracle_reappear_only", "oracle_history_and_reappear"])
    parser.add_argument("--component-ranking-profile", default="R0_current_quality")
    parser.add_argument("--support-box-profile", default="B0_refined_box_current")
    args = parser.parse_args()
    summary = run_eval(**vars(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
