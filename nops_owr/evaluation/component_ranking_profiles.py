"""GT-free proposal ranking profiles for external diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from nops_owr.objectness.field import Proposal


def component_ranking_profile_specs() -> list[dict[str, Any]]:
    return [
        {"profile_name": "R0_current_quality"},
        {"profile_name": "R1_score_first"},
        {"profile_name": "R2_area_score"},
        {"profile_name": "R3_raw_area_support"},
        {"profile_name": "R4_less_shape_penalty"},
        {"profile_name": "R5_boundary_tolerant"},
    ]


def known_component_ranking_profiles() -> set[str]:
    return {str(profile["profile_name"]) for profile in component_ranking_profile_specs()}


def proposal_rank_score(proposal: Proposal, profile_name: str, frame_shape: tuple[int, int]) -> float:
    height, width = frame_shape
    frame_area = max(1.0, float(height * width))
    area_norm = float(proposal.area) / frame_area
    raw_area_norm = float(proposal.raw_area) / frame_area
    score = float(proposal.score)
    quality = float(proposal.quality_score)
    fill = float(np.clip(proposal.fill_ratio, 0.0, 1.0))
    near_boundary = float(int(proposal.near_boundary))

    if profile_name in {"", "R0_current_quality"}:
        return float(score + 0.35 * (quality - score))
    if profile_name == "R1_score_first":
        return score + 1e-3 * quality
    if profile_name == "R2_area_score":
        return score + 0.10 * np.clip(area_norm * 20.0, 0.0, 1.0) + 1e-3 * quality
    if profile_name == "R3_raw_area_support":
        return score + 0.08 * np.clip(raw_area_norm * 20.0, 0.0, 1.0) + 0.05 * fill
    if profile_name == "R4_less_shape_penalty":
        return score + 0.05 * fill + 0.04 * np.clip(area_norm * 20.0, 0.0, 1.0) - 0.02 * near_boundary
    if profile_name == "R5_boundary_tolerant":
        return score + 0.08 * np.clip(raw_area_norm * 20.0, 0.0, 1.0) + 0.04 * fill

    known = ", ".join(sorted(known_component_ranking_profiles()))
    raise ValueError(f"Unknown component ranking profile {profile_name!r}. Known profiles: {known}")


def sort_proposals_by_profile(
    proposals: list[Proposal],
    profile_name: str = "R0_current_quality",
    frame_shape: tuple[int, int] | None = None,
) -> list[Proposal]:
    if profile_name in {"", "default"}:
        profile_name = "R0_current_quality"
    if frame_shape is None:
        frame_shape = _infer_frame_shape(proposals)
    return sorted(
        proposals,
        key=lambda proposal: (
            proposal_rank_score(proposal, profile_name, frame_shape),
            proposal.quality_score,
            proposal.score,
            proposal.area,
        ),
        reverse=True,
    )


def _infer_frame_shape(proposals: list[Proposal]) -> tuple[int, int]:
    max_x = max((proposal.box[2] for proposal in proposals), default=1)
    max_y = max((proposal.box[3] for proposal in proposals), default=1)
    return (max_y, max_x)
