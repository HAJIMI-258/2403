from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nops_owr.cognition.object_file import ObjectFile, SupportMaskSummary  # noqa: E402
from nops_owr.memory import EpisodicMemory, RetrievalContext  # noqa: E402


def make_object_file(object_id: str, frame_index: int, track_id: int | None) -> ObjectFile:
    obj = ObjectFile(
        object_file_id=object_id,
        frame_index=frame_index,
        proposal_index=0,
        box=(10, 10, 20, 20),
        raw_box=(10, 10, 20, 20),
        support_box=(10, 10, 20, 20),
        centroid=(15.0, 15.0),
        area=100.0,
        score=0.9,
        quality_score=0.9,
        support_mask_summary=SupportMaskSummary(100.0, (10, 10, 20, 20), 1.0, 1.0, 1.0),
        appearance_signature=np.ones(8, dtype=np.float32),
        shape_signature=np.ones(6, dtype=np.float32),
        context_signature=np.ones(5, dtype=np.float32),
        motion_signature=np.ones(4, dtype=np.float32),
        confidence=0.9,
    )
    obj.linked_track_id = track_id
    return obj


class RetrievalContextTest(unittest.TestCase):
    def test_reentry_context_prioritizes_closed_and_flags_active_conflict(self) -> None:
        memory = EpisodicMemory(memory_budget=8)
        cue = make_object_file("cue", frame_index=50, track_id=99)

        target = make_object_file("closed_target", frame_index=10, track_id=1)
        target_episode_id = memory.begin_episode(target, frame_index=10, track_id=1)
        memory.close_episode(target_episode_id, frame_index=12, close_reason="disappeared")

        conflict = make_object_file("active_conflict", frame_index=45, track_id=2)
        conflict_episode_id = memory.begin_episode(conflict, frame_index=45, track_id=2)
        self.assertIsNotNone(conflict_episode_id)

        same_track = make_object_file("same_track", frame_index=48, track_id=99)
        same_track_episode_id = memory.begin_episode(same_track, frame_index=48, track_id=99)
        self.assertIsNotNone(same_track_episode_id)

        rows = memory.retrieve(
            cue,
            top_k=3,
            context=RetrievalContext(
                frame_index=50,
                query_track_id=99,
                active_track_ids={2, 99},
                mode="reentry",
                min_reentry_gap=8,
                prefer_closed_episodes=True,
                suppress_active_conflicts=True,
            ),
        )

        self.assertEqual(rows[0].bundle.episode_id, target_episode_id)
        self.assertGreater(rows[0].closed_bonus, 0.0)
        self.assertGreaterEqual(rows[0].reentry_gap, 8)
        self.assertEqual(rows[0].rank, 1)
        self.assertGreaterEqual(rows[0].margin_to_next, 0.0)
        conflict_row = next(row for row in rows if row.bundle.episode_id == conflict_episode_id)
        self.assertTrue(conflict_row.active_conflict)
        self.assertLess(conflict_row.evidence_breakdown["active_conflict_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
