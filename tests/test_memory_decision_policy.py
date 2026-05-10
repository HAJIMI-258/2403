from __future__ import annotations

import unittest

from nops_owr.memory import (
    MemoryDecision,
    MemoryDecisionConfig,
    RetrievalState,
    assert_safe_side_effects,
    can_release_after_wait,
    decide_memory_retrieval,
)


class MemoryDecisionPolicyTest(unittest.TestCase):
    def test_low_margin_routes_to_uncertain_without_side_effects(self) -> None:
        decision = decide_memory_retrieval(0.001, MemoryDecisionConfig(uncertainty_margin_threshold=0.0194))

        self.assertEqual(decision.retrieval_state, RetrievalState.UNCERTAIN_NEED_MORE_EVIDENCE)
        self.assertFalse(decision.memory_update_allowed)
        self.assertTrue(decision.evidence_queue_enqueued)
        self.assertFalse(decision.attach_allowed)
        self.assertFalse(decision.promotion_allowed)
        self.assertFalse(decision.head_update_allowed)
        assert_safe_side_effects(decision)

    def test_high_margin_routes_to_old_recall_candidate_only(self) -> None:
        decision = decide_memory_retrieval(0.12, MemoryDecisionConfig(uncertainty_margin_threshold=0.0194))

        self.assertEqual(decision.retrieval_state, RetrievalState.OLD_RECALL_CANDIDATE)
        self.assertTrue(decision.memory_update_allowed)
        self.assertFalse(decision.evidence_queue_enqueued)
        self.assertFalse(decision.attach_allowed)
        self.assertFalse(decision.promotion_allowed)
        self.assertFalse(decision.head_update_allowed)
        assert_safe_side_effects(decision)

    def test_uncertain_side_effect_violation_is_rejected(self) -> None:
        bad = MemoryDecision(
            retrieval_state=RetrievalState.UNCERTAIN_NEED_MORE_EVIDENCE,
            top1_margin=0.0,
            threshold=0.0194,
            memory_update_allowed=True,
            evidence_queue_enqueued=True,
        )

        with self.assertRaises(ValueError):
            assert_safe_side_effects(bad)

    def test_bounded_wait_release_uses_margin_and_horizon_only(self) -> None:
        cfg = MemoryDecisionConfig(uncertainty_margin_threshold=0.0194, bounded_wait_horizon_frames=10)

        self.assertTrue(can_release_after_wait(wait_frames=2, release_margin=0.04, config=cfg))
        self.assertFalse(can_release_after_wait(wait_frames=0, release_margin=0.04, config=cfg))
        self.assertFalse(can_release_after_wait(wait_frames=11, release_margin=0.04, config=cfg))
        self.assertFalse(can_release_after_wait(wait_frames=2, release_margin=0.001, config=cfg))


if __name__ == "__main__":
    unittest.main()
