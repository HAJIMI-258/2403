from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nops_owr.evaluation.component_ranking_profiles import sort_proposals_by_profile
from nops_owr.objectness.field import Proposal


def _proposal(
    *,
    box: tuple[int, int, int, int],
    score: float,
    quality: float,
    area: int,
    fill: float,
    compactness: float,
    near_boundary: int = 0,
) -> Proposal:
    return Proposal(
        box=box,
        raw_box=box,
        support_box=box,
        area=area,
        raw_area=area,
        score=score,
        quality_score=quality,
        centroid=((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5),
        support_mask=np.ones((max(1, box[3] - box[1]), max(1, box[2] - box[0])), dtype=bool),
        fill_ratio=fill,
        compactness=compactness,
        boundary_smoothness=compactness,
        near_boundary=near_boundary,
        source="component",
        source_score=score,
        metadata={},
    )


class ComponentRankingProfileTest(unittest.TestCase):
    def test_boundary_tolerant_profile_can_reorder_large_low_compactness_component(self) -> None:
        compact_small = _proposal(box=(10, 10, 14, 14), score=0.55, quality=0.80, area=16, fill=1.0, compactness=1.0)
        real_like_large = _proposal(
            box=(5, 20, 55, 30),
            score=0.54,
            quality=0.52,
            area=500,
            fill=0.10,
            compactness=0.15,
            near_boundary=1,
        )
        proposals = [compact_small, real_like_large]

        current = sort_proposals_by_profile(proposals, "R0_current_quality", (100, 100))
        tolerant = sort_proposals_by_profile(proposals, "R5_boundary_tolerant", (100, 100))

        self.assertIs(current[0], compact_small)
        self.assertIs(tolerant[0], real_like_large)


if __name__ == "__main__":
    unittest.main()
