from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nops_owr.evaluation.support_refinement_profiles import apply_box_profile_to_proposals, box_for_proposal
from nops_owr.objectness.field import Proposal


def _proposal() -> Proposal:
    return Proposal(
        box=(20, 20, 40, 40),
        raw_box=(15, 15, 45, 45),
        support_box=(18, 18, 42, 42),
        area=400,
        raw_area=900,
        score=0.5,
        quality_score=0.6,
        centroid=(30.0, 30.0),
        support_mask=np.ones((20, 20), dtype=bool),
        fill_ratio=0.6,
        compactness=0.5,
        boundary_smoothness=0.5,
        near_boundary=0,
        source="component",
        source_score=0.5,
        metadata={},
    )


class SupportRefinementProfileTest(unittest.TestCase):
    def test_box_profiles_are_distinct_and_clipped(self) -> None:
        proposal = _proposal()
        shape = (50, 50)

        self.assertEqual(box_for_proposal(proposal, "B0_refined_box_current", shape), (20, 20, 40, 40))
        self.assertEqual(box_for_proposal(proposal, "B1_raw_box_eval_profile", shape), (15, 15, 45, 45))
        self.assertEqual(box_for_proposal(proposal, "B2_support_box_eval_profile", shape), (18, 18, 42, 42))
        self.assertEqual(box_for_proposal(proposal, "B3_blend_raw_refined_50", shape), (18, 18, 42, 42))
        self.assertEqual(box_for_proposal(proposal, "B4_expand_refined_10", shape), (18, 18, 42, 42))

    def test_apply_box_profile_updates_centroid_and_metadata(self) -> None:
        updated = apply_box_profile_to_proposals([_proposal()], "B1_raw_box_eval_profile", (50, 50))[0]

        self.assertEqual(updated.box, (15, 15, 45, 45))
        self.assertEqual(updated.centroid, (30.0, 30.0))
        self.assertEqual(updated.metadata["support_box_profile"], "B1_raw_box_eval_profile")


if __name__ == "__main__":
    unittest.main()
