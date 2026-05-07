"""Shared evaluation helpers for Phase 2B analysis and reruns."""

from __future__ import annotations

from typing import Any

from datasets import load_synth_dataset_config
from experiments.scenario_presets import build_phase1_scenarios
from nops_owr.evaluation import StreamingEpisodeEvaluator


def get_field_config(config_payload: dict) -> dict:
    if "field" in config_payload:
        return dict(config_payload["field"])
    return dict(config_payload["model"]["objectness"])


def build_scenarios(config_path: str) -> list[dict[str, Any]]:
    base_config = load_synth_dataset_config(config_path)
    return build_phase1_scenarios(base_config)


def evaluate_main_pipeline(
    sequence,
    config_payload: dict,
    *,
    field_override: dict | None = None,
    tracking_override: dict | None = None,
    memory_override: dict | None = None,
    collect_frames: bool = False,
) -> dict[str, Any]:
    evaluator = StreamingEpisodeEvaluator(
        config_payload,
        field_override=field_override,
        tracking_override=tracking_override,
        memory_override=memory_override,
    )
    result = evaluator.evaluate(sequence, collect_frames=collect_frames)

    return {
        "summary": result.summary,
        "audit": result.audit,
        "field_config": evaluator.field_config,
        "tracking_config": evaluator.tracking_config,
        "memory_config": evaluator.memory_config,
        "action_counter": result.action_counter,
        "decay_count": result.decay_count,
        "frame_records": [
            {
                "frame_index": frame.frame_index,
                "frame": frame.frame,
                "gt_boxes": frame.gt_boxes,
                "masks": frame.masks,
                "instance_ids": frame.instance_ids,
                "concept_ids": frame.concept_ids,
                "objectness_output": frame.objectness_output,
                "tracking_output": frame.tracking_output,
                "memory_output": frame.memory_output,
                "predicted_boxes": frame.predicted_boxes,
                "predicted_ids": frame.predicted_ids,
                "matches": frame.matches,
                "recall_hit": frame.recall_hit,
                "frame_purity": frame.frame_purity,
                "objectness_recall": frame.objectness_recall,
                "false_hot_area": frame.false_hot_area,
            }
            for frame in result.frame_records
        ],
        "frame_metrics": result.frame_metrics,
        "budget_report": result.budget_report,
        "primary_monitoring": result.primary_monitoring,
        "secondary_monitoring": result.secondary_monitoring,
    }
