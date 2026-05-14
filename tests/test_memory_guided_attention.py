from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nops_owr.attention.attention_gate import AttentionGate, AttentionGateConfig
from nops_owr.cognition.object_file import ObjectFile, SupportMaskSummary


class MemoryGuidedAttentionTest(unittest.TestCase):
    def test_task_salience_can_promote_memory_relevant_object(self) -> None:
        gate = AttentionGate(
            config=AttentionGateConfig(
                max_attended_objects=1,
                min_quality=0.0,
                quality_weight=0.2,
                novelty_weight=0.0,
                surprise_weight=0.0,
                prediction_error_weight=0.0,
                motion_weight=0.0,
                low_familiarity_weight=0.0,
                task_salience_weight=0.8,
            )
        )
        weak_memory = _object_file("memory", quality=0.1)
        strong_bottom_up = _object_file("bottom", quality=0.4)
        selected = gate.select(
            [strong_bottom_up, weak_memory],
            task_salience={weak_memory.object_file_id: 1.0, strong_bottom_up.object_file_id: 0.0},
        )
        self.assertEqual(selected[0].object_file_id, weak_memory.object_file_id)

    def test_source_aware_gate_prefers_component_over_template_noise(self) -> None:
        gate = AttentionGate(
            config=AttentionGateConfig(
                max_attended_objects=1,
                min_quality=0.0,
                quality_weight=0.5,
                novelty_weight=0.0,
                surprise_weight=0.0,
                prediction_error_weight=0.0,
                motion_weight=0.0,
                low_familiarity_weight=0.0,
                task_salience_weight=0.0,
                proposal_source_score_weight=0.1,
                component_source_bonus=0.12,
                memory_template_window_source_bonus=-0.12,
            )
        )
        component_target = _object_file("component", quality=0.32, proposal_source="component", source_score=0.4)
        template_noise = _object_file(
            "template",
            quality=0.40,
            proposal_source="memory_template_window",
            source_score=0.9,
        )
        selected = gate.select([template_noise, component_target])
        self.assertEqual(selected[0].object_file_id, component_target.object_file_id)

    def test_diverse_gate_spreads_attention_across_spatial_clusters(self) -> None:
        gate = AttentionGate(
            config=AttentionGateConfig(
                max_attended_objects=2,
                min_quality=0.0,
                quality_weight=1.0,
                novelty_weight=0.0,
                surprise_weight=0.0,
                prediction_error_weight=0.0,
                motion_weight=0.0,
                low_familiarity_weight=0.0,
                diversity_iou_penalty=0.4,
                diversity_center_penalty=0.2,
                diversity_center_scale=16.0,
            )
        )
        cluster_top = _object_file("cluster_top", quality=0.90, box=(0, 0, 20, 20))
        cluster_second = _object_file("cluster_second", quality=0.88, box=(2, 2, 22, 22))
        separate = _object_file("separate", quality=0.70, box=(60, 60, 80, 80))
        selected = gate.select([cluster_top, cluster_second, separate])
        selected_ids = {item.object_file_id for item in selected}
        self.assertIn("cluster_top", selected_ids)
        self.assertIn("separate", selected_ids)


def _object_file(
    name: str,
    quality: float,
    proposal_source: str = "component",
    source_score: float = 0.0,
    box: tuple[int, int, int, int] = (0, 0, 8, 8),
) -> ObjectFile:
    x1, y1, x2, y2 = box
    return ObjectFile(
        object_file_id=name,
        frame_index=1,
        proposal_index=0,
        box=box,
        raw_box=box,
        support_box=box,
        centroid=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
        area=float(max(1, (x2 - x1) * (y2 - y1))),
        score=quality,
        quality_score=quality,
        support_mask_summary=SupportMaskSummary(area=64.0, bbox=box, fill_ratio=1.0, compactness=1.0, boundary_smoothness=1.0),
        appearance_signature=np.ones(15, dtype=np.float32),
        shape_signature=np.ones(7, dtype=np.float32),
        context_signature=np.ones(6, dtype=np.float32),
        motion_signature=np.zeros(0, dtype=np.float32),
        confidence=quality,
        proposal_source=proposal_source,
        proposal_source_score=source_score,
    )


if __name__ == "__main__":
    unittest.main()
