"""Named attention profiles for external evaluation.

These profiles are diagnostic configurations for sparse object-file selection.
They do not change the default VisualCognitiveLoop unless explicitly selected
by an evaluation script.
"""

from __future__ import annotations

from typing import Any

from nops_owr.attention.attention_gate import AttentionGate, AttentionGateConfig


def attention_profile_specs() -> list[dict[str, Any]]:
    return [
        {
            "profile_name": "A0_current_max4",
            "max_attended_objects": 4,
            "min_quality": 0.05,
            "quality_weight": 0.30,
            "novelty_weight": 0.20,
            "surprise_weight": 0.15,
            "prediction_error_weight": 0.20,
            "motion_weight": 0.10,
            "low_familiarity_weight": 0.05,
            "task_salience_weight": 0.15,
        },
        {
            "profile_name": "A1_capacity_max8",
            "max_attended_objects": 8,
            "min_quality": 0.03,
            "quality_weight": 0.30,
            "novelty_weight": 0.20,
            "surprise_weight": 0.15,
            "prediction_error_weight": 0.20,
            "motion_weight": 0.10,
            "low_familiarity_weight": 0.05,
            "task_salience_weight": 0.18,
        },
        {
            "profile_name": "A2_capacity_max12",
            "max_attended_objects": 12,
            "min_quality": 0.02,
            "quality_weight": 0.30,
            "novelty_weight": 0.20,
            "surprise_weight": 0.15,
            "prediction_error_weight": 0.20,
            "motion_weight": 0.10,
            "low_familiarity_weight": 0.05,
            "task_salience_weight": 0.18,
        },
        {
            "profile_name": "A3_quality_surprise_max12",
            "max_attended_objects": 12,
            "min_quality": 0.02,
            "quality_weight": 0.42,
            "novelty_weight": 0.08,
            "surprise_weight": 0.25,
            "prediction_error_weight": 0.15,
            "motion_weight": 0.05,
            "low_familiarity_weight": 0.05,
        },
        {
            "profile_name": "A4_recall_max16",
            "max_attended_objects": 16,
            "min_quality": 0.00,
            "quality_weight": 0.25,
            "novelty_weight": 0.20,
            "surprise_weight": 0.20,
            "prediction_error_weight": 0.20,
            "motion_weight": 0.10,
            "low_familiarity_weight": 0.05,
            "task_salience_weight": 0.18,
        },
        {
            "profile_name": "A5_recall_max24",
            "max_attended_objects": 24,
            "min_quality": 0.00,
            "quality_weight": 0.22,
            "novelty_weight": 0.20,
            "surprise_weight": 0.22,
            "prediction_error_weight": 0.20,
            "motion_weight": 0.11,
            "low_familiarity_weight": 0.05,
            "task_salience_weight": 0.20,
        },
        {
            "profile_name": "A6_diagnostic_max32",
            "max_attended_objects": 32,
            "min_quality": 0.00,
            "quality_weight": 0.18,
            "novelty_weight": 0.20,
            "surprise_weight": 0.24,
            "prediction_error_weight": 0.22,
            "motion_weight": 0.11,
            "low_familiarity_weight": 0.05,
            "task_salience_weight": 0.25,
        },
        {
            "profile_name": "A7_source_aware_component_max16",
            "max_attended_objects": 16,
            "min_quality": 0.00,
            "quality_weight": 0.22,
            "novelty_weight": 0.14,
            "surprise_weight": 0.20,
            "prediction_error_weight": 0.18,
            "motion_weight": 0.08,
            "low_familiarity_weight": 0.03,
            "task_salience_weight": 0.22,
            "proposal_source_score_weight": 0.10,
            "component_source_bonus": 0.08,
            "saliency_window_source_bonus": 0.02,
            "memory_guided_window_source_bonus": 0.00,
            "memory_template_window_source_bonus": -0.10,
        },
        {
            "profile_name": "A8_memory_salience_source_guard_max16",
            "max_attended_objects": 16,
            "min_quality": 0.00,
            "quality_weight": 0.18,
            "novelty_weight": 0.12,
            "surprise_weight": 0.18,
            "prediction_error_weight": 0.18,
            "motion_weight": 0.07,
            "low_familiarity_weight": 0.02,
            "task_salience_weight": 0.35,
            "proposal_source_score_weight": 0.08,
            "component_source_bonus": 0.05,
            "saliency_window_source_bonus": 0.01,
            "memory_guided_window_source_bonus": 0.00,
            "memory_template_window_source_bonus": -0.12,
        },
        {
            "profile_name": "A9_source_aware_component_max32_diagnostic",
            "max_attended_objects": 32,
            "min_quality": 0.00,
            "quality_weight": 0.22,
            "novelty_weight": 0.14,
            "surprise_weight": 0.20,
            "prediction_error_weight": 0.18,
            "motion_weight": 0.08,
            "low_familiarity_weight": 0.03,
            "task_salience_weight": 0.22,
            "proposal_source_score_weight": 0.10,
            "component_source_bonus": 0.08,
            "saliency_window_source_bonus": 0.02,
            "memory_guided_window_source_bonus": 0.00,
            "memory_template_window_source_bonus": -0.10,
        },
        {
            "profile_name": "A10_source_spatial_diverse_max16",
            "max_attended_objects": 16,
            "min_quality": 0.00,
            "quality_weight": 0.20,
            "novelty_weight": 0.14,
            "surprise_weight": 0.20,
            "prediction_error_weight": 0.18,
            "motion_weight": 0.08,
            "low_familiarity_weight": 0.03,
            "task_salience_weight": 0.22,
            "proposal_source_score_weight": 0.08,
            "component_source_bonus": 0.08,
            "saliency_window_source_bonus": 0.00,
            "memory_guided_window_source_bonus": 0.00,
            "memory_template_window_source_bonus": -0.12,
            "diversity_iou_penalty": 0.18,
            "diversity_center_penalty": 0.10,
            "diversity_center_scale": 52.0,
        },
        {
            "profile_name": "A11_source_spatial_diverse_max24",
            "max_attended_objects": 24,
            "min_quality": 0.00,
            "quality_weight": 0.20,
            "novelty_weight": 0.14,
            "surprise_weight": 0.20,
            "prediction_error_weight": 0.18,
            "motion_weight": 0.08,
            "low_familiarity_weight": 0.03,
            "task_salience_weight": 0.22,
            "proposal_source_score_weight": 0.08,
            "component_source_bonus": 0.08,
            "saliency_window_source_bonus": 0.00,
            "memory_guided_window_source_bonus": 0.00,
            "memory_template_window_source_bonus": -0.12,
            "diversity_iou_penalty": 0.18,
            "diversity_center_penalty": 0.10,
            "diversity_center_scale": 52.0,
        },
    ]


def get_attention_profile(profile_name: str) -> dict[str, Any]:
    if profile_name in {"", "default", "current"}:
        profile_name = "A0_current_max4"
    for profile in attention_profile_specs():
        if profile["profile_name"] == profile_name:
            return dict(profile)
    known = ", ".join(profile["profile_name"] for profile in attention_profile_specs())
    raise ValueError(f"Unknown attention profile {profile_name!r}. Known profiles: {known}")


def build_attention_from_profile(profile_name: str) -> AttentionGate:
    kwargs = get_attention_profile(profile_name)
    kwargs.pop("profile_name", None)
    max_attended = int(kwargs.pop("max_attended_objects"))
    min_quality = float(kwargs.pop("min_quality"))
    config = AttentionGateConfig(
        max_attended_objects=max_attended,
        min_quality=min_quality,
        **kwargs,
    )
    return AttentionGate(config=config)
