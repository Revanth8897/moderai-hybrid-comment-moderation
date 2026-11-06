import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database.base import Base
from backend.app.database.models import Review
from moderation.review.manager import ReviewManager


class TestReviewManager(unittest.TestCase):

    def setUp(self):
        self.test_engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        
        Base.metadata.create_all(self.test_engine)
        
        TestSessionLocal = sessionmaker(
            bind=self.test_engine,
            autoflush=False,
            autocommit=False,
        )
        
        self.manager = ReviewManager(
            session_factory=TestSessionLocal
        )
        
    def tearDown(self):
        Base.metadata.drop_all(bind=self.test_engine)
        self.test_engine.dispose()
        
        
    def create_review(
        self,
        safety_signals=None,
        intent="QUESTION"
    ):

        if safety_signals is None:
            safety_signals = []

        pipeline_result = {
            "comment": f"Test comment for {intent}",
            "analysis": {
                "requires_human_review": True,
                "engagement_analysis": {
                    "predicted_intent": intent
                },
                "safety_analysis": {
                    "qualified_review_signals": safety_signals
                }
            },
            "routing": {
                "route": "HUMAN_REVIEW",
                "priority": "NORMAL"
            }
        }

        return self.manager.add_to_queue(pipeline_result)

    def move_to_review(self, review_id):

        return self.manager.update_status(
            review_id,
            "IN_REVIEW",
            "admin_001"
        )

    # --------------------------------------------------
    # Queue tests
    # --------------------------------------------------

    def test_add_review_to_queue(self):

        review = self.create_review()

        self.assertEqual(review["review_status"], "PENDING")
        self.assertIsNotNone(review["review_id"])

    def test_pending_reviews_are_retrieved(self):

        self.create_review()

        pending = self.manager.get_pending_reviews()

        self.assertEqual(len(pending), 1)

    def test_get_individual_review(self):

        review = self.create_review()

        retrieved = self.manager.get_review(
            review["review_id"]
        )

        self.assertEqual(
            retrieved["review_id"],
            review["review_id"]
        )

    def test_review_requires_human_review(self):

        pipeline_result = {
            "comment": "Hello",
            "analysis": {
                "requires_human_review": False
            },
            "routing": {}
        }

        with self.assertRaises(ValueError):
            self.manager.add_to_queue(pipeline_result)

    # --------------------------------------------------
    # Priority tests
    # --------------------------------------------------

    def test_review_priority_is_stored(self):

        review = self.create_review(
            safety_signals=["toxicity"]
        )

        self.assertEqual(
            review["review_priority"],
            "CRITICAL"
        )

        self.assertTrue(
            review["priority_reason"]
        )

    def test_critical_reviews_appear_first(self):

        normal_review = self.create_review(
            intent="QUESTION"
        )

        critical_review = self.create_review(
            safety_signals=["toxicity"]
        )

        pending = self.manager.get_pending_reviews()

        self.assertEqual(
            pending[0]["review_id"],
            critical_review["review_id"]
        )

    def test_high_reviews_appear_before_normal(self):

        normal_review = self.create_review(
            intent="QUESTION"
        )

        high_review = self.create_review(
            intent="UNCLEAR"
        )

        pending = self.manager.get_pending_reviews()

        self.assertEqual(
            pending[0]["review_id"],
            high_review["review_id"]
        )

    def test_fifo_order_for_same_priority(self):

        first_review = self.create_review(
            intent="POSITIVE"
        )

        second_review = self.create_review(
            intent="NEGATIVE"
        )

        third_review = self.create_review(
            intent="QUESTION"
        )

        pending = self.manager.get_pending_reviews()

        self.assertEqual(
            [review["review_id"] for review in pending],
            [
                first_review["review_id"],
                second_review["review_id"],
                third_review["review_id"]
            ]
        )

    # --------------------------------------------------
    # Status tests
    # --------------------------------------------------

    def test_update_status_to_in_review(self):

        review = self.create_review()

        updated = self.move_to_review(
            review["review_id"]
        )

        self.assertEqual(
            updated["review_status"],
            "IN_REVIEW"
        )

    def test_direct_approval_is_blocked(self):

        review = self.create_review()

        self.move_to_review(review["review_id"])

        with self.assertRaises(ValueError):
            self.manager.update_status(
                review["review_id"],
                "APPROVED",
                "admin_001"
            )

    def test_invalid_status_transition_is_rejected(self):

        review = self.create_review()

        with self.assertRaises(ValueError):
            self.manager.update_status(
                review["review_id"],
                "COMPLETED",
                "admin_001"
            )

    # --------------------------------------------------
    # Decision tests
    # --------------------------------------------------

    def test_approve_for_draft(self):

        review = self.create_review()

        self.move_to_review(review["review_id"])

        result = self.manager.decide_review(
            review["review_id"],
            "APPROVE_FOR_DRAFT",
            "admin_001",
            "Safe question."
        )

        self.assertEqual(
            result["review_status"],
            "APPROVED"
        )

    def test_no_response_decision(self):

        review = self.create_review()

        self.move_to_review(review["review_id"])

        result = self.manager.decide_review(
            review["review_id"],
            "NO_RESPONSE",
            "admin_001"
        )

        self.assertEqual(
            result["review_status"],
            "COMPLETED"
        )

    def test_reject_decision(self):

        review = self.create_review()

        self.move_to_review(review["review_id"])

        result = self.manager.decide_review(
            review["review_id"],
            "REJECT",
            "admin_001"
        )

        self.assertEqual(
            result["review_status"],
            "REJECTED"
        )

    def test_escalate_decision(self):

        review = self.create_review()

        self.move_to_review(review["review_id"])

        result = self.manager.decide_review(
            review["review_id"],
            "ESCALATE",
            "admin_001"
        )

        self.assertEqual(
            result["review_status"],
            "ESCALATED"
        )

    def test_decision_reason_is_saved(self):

        review = self.create_review()

        self.move_to_review(review["review_id"])

        result = self.manager.decide_review(
            review["review_id"],
            "APPROVE_FOR_DRAFT",
            "admin_001",
            "Reviewed and approved."
        )

        self.assertEqual(
            result["decision_reason"],
            "Reviewed and approved."
        )

    def test_invalid_decision_is_rejected(self):

        review = self.create_review()

        self.move_to_review(review["review_id"])

        with self.assertRaises(ValueError):
            self.manager.decide_review(
                review["review_id"],
                "UNKNOWN",
                "admin_001"
            )

    def test_decision_requires_in_review_status(self):

        review = self.create_review()

        with self.assertRaises(ValueError):
            self.manager.decide_review(
                review["review_id"],
                "APPROVE_FOR_DRAFT",
                "admin_001"
            )

    # --------------------------------------------------
    # Safety tests
    # --------------------------------------------------

    def test_safety_signals_block_approval(self):

        review = self.create_review(["toxicity"])

        self.move_to_review(review["review_id"])

        with self.assertRaises(ValueError):
            self.manager.decide_review(
                review["review_id"],
                "APPROVE_FOR_DRAFT",
                "admin_001"
            )

    def test_safety_signals_allow_escalation(self):

        review = self.create_review(["toxicity"])

        self.move_to_review(review["review_id"])

        result = self.manager.decide_review(
            review["review_id"],
            "ESCALATE",
            "admin_001"
        )

        self.assertEqual(
            result["review_status"],
            "ESCALATED"
        )

    # --------------------------------------------------
    # Data and action safety tests
    # --------------------------------------------------

    def test_automatic_action_remains_disabled(self):

        review = self.create_review()

        self.assertFalse(
            review["automatic_action_allowed"]
        )

    def test_returned_record_does_not_modify_storage(self):

        review = self.create_review()

        review["comment"] = "Modified externally"

        stored = self.manager.get_review(
            review["review_id"]
        )

        self.assertEqual(
            stored["comment"],
            "Test comment for QUESTION"
        )

    def test_invalid_review_id_is_rejected(self):

        with self.assertRaises(ValueError):
            self.manager.get_review("unknown-id")


if __name__ == "__main__":
    unittest.main()
