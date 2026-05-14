"""Evaluation-only adapters around objectness outputs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from nops_owr.encoder.spike_encoder import SpikeEncoding
from nops_owr.evaluation.component_ranking_profiles import sort_proposals_by_profile
from nops_owr.evaluation.support_refinement_profiles import apply_box_profile_to_proposals
from nops_owr.objectness.field import ObjectnessOutput


class ProfiledObjectnessField:
    """Wrap an objectness field with GT-free eval ranking/box profiles."""

    def __init__(
        self,
        base_objectness: Any,
        *,
        component_ranking_profile: str = "R0_current_quality",
        support_box_profile: str = "B0_refined_box_current",
    ) -> None:
        self.base_objectness = base_objectness
        self.component_ranking_profile = component_ranking_profile or "R0_current_quality"
        self.support_box_profile = support_box_profile or "B0_refined_box_current"

    def reset(self) -> None:
        reset = getattr(self.base_objectness, "reset", None)
        if callable(reset):
            reset()

    def compute(self, encoding: SpikeEncoding) -> ObjectnessOutput:
        output = self.base_objectness.compute(encoding)
        frame_shape = output.heatmap.shape
        proposals = apply_box_profile_to_proposals(
            list(output.proposals),
            profile_name=self.support_box_profile,
            frame_shape=frame_shape,
        )
        proposals = sort_proposals_by_profile(
            proposals,
            profile_name=self.component_ranking_profile,
            frame_shape=frame_shape,
        )
        return replace(output, proposals=proposals)
