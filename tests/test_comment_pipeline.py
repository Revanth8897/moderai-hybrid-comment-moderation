import unittest
from unittest.mock import Mock

from moderation.comment_pipeline import CommentModerationPipeline


class TestCommentModerationPipeline(unittest.TestCase):

    def setUp(self):
        self.mock_analyzer = Mock()
        self.mock_routing_engine = Mock()
        self.mock_response_controller = Mock()
        self.mock_review_manager = Mock()

        self.sample_comment = "Can you explain database indexing?"

        self.analysis_result = {
            "comment": self.sample_comment,
            "safety_analysis": {
                "qualified_review_signals": []
            },
            "engagement_analysis": {
                "predicted_intent": "QUESTION"
            },
            "requires_human_review": True
        }

        self.routing_result = {
            "route": "HUMAN_REVIEW",
            "priority": "NORMAL",
            "reason": "Analysis requires human confirmation.",
            "requires_human_approval": True,
            "automatic_action_allowed": False,
            "routing_version": "moderai_routing_v1"
        }

        self.response_result = {
            "status": "NO_DRAFT",
            "reason": "Human review is required.",
            "automatic_action_allowed": False
        }

        self.review_record = {
            "review_id": "test-review-001",
            "status": "PENDING",
            "reviewer_id": None,
            "review_decision": None,
            "automatic_action_allowed": False
        }

        self.mock_analyzer.analyze.return_value = (
            self.analysis_result
        )

        self.mock_routing_engine.route.return_value = (
            self.routing_result
        )

        self.mock_response_controller.generate.return_value = (
            self.response_result
        )

        self.mock_review_manager.add_to_queue.return_value = (
            self.review_record
        )

        self.pipeline = CommentModerationPipeline(
            analyzer=self.mock_analyzer,
            routing_engine=self.mock_routing_engine,
            response_controller=self.mock_response_controller,
            review_manager=self.mock_review_manager
        )

    def test_analyzer_receives_comment(self):
        self.pipeline.process(self.sample_comment)

        self.mock_analyzer.analyze.assert_called_once_with(
            self.sample_comment
        )

    def test_routing_engine_receives_analysis(self):
        self.pipeline.process(self.sample_comment)

        self.mock_routing_engine.route.assert_called_once_with(
            self.analysis_result
        )

    def test_pipeline_returns_complete_result(self):
        result = self.pipeline.process(self.sample_comment)

        self.assertEqual(result["comment"], self.sample_comment)
        self.assertEqual(result["analysis"], self.analysis_result)
        self.assertEqual(result["routing"], self.routing_result)
        self.assertEqual(
            result["response_generation"],
            self.response_result
        )

    def test_pipeline_version_is_present(self):
        result = self.pipeline.process(self.sample_comment)

        self.assertEqual(
            result["pipeline_version"],
            "moderai_pipeline_v3"
        )

    def test_automatic_action_remains_disabled(self):
        result = self.pipeline.process(self.sample_comment)

        self.assertFalse(
            result["routing"]["automatic_action_allowed"]
        )

        self.assertFalse(
            result["response_generation"]["automatic_action_allowed"]
        )

    def test_human_review_is_added_to_queue(self):
        result = self.pipeline.process(self.sample_comment)

        self.mock_review_manager.add_to_queue.assert_called_once()

        self.assertEqual(
            result["review"],
            self.review_record
        )

    def test_review_manager_receives_complete_pipeline_result(self):
        self.pipeline.process(self.sample_comment)

        queued_result = (
            self.mock_review_manager.add_to_queue.call_args[0][0]
        )

        self.assertEqual(
            queued_result["comment"],
            self.sample_comment
        )

        self.assertIn("analysis", queued_result)
        self.assertIn("routing", queued_result)
        self.assertIn("response_generation", queued_result)

    def test_review_is_not_added_when_not_required(self):
        self.analysis_result["requires_human_review"] = False

        result = self.pipeline.process(self.sample_comment)

        self.mock_review_manager.add_to_queue.assert_not_called()

        self.assertIsNone(result["review"])

    def test_pipeline_uses_default_review_manager(self):
        pipeline = CommentModerationPipeline(
            analyzer=self.mock_analyzer,
            routing_engine=self.mock_routing_engine,
            response_controller=self.mock_response_controller
        )

        self.assertIsNotNone(pipeline.review_manager)

        self.assertEqual(
            pipeline.review_manager.__class__.__name__,
            "ReviewManager"
        )


if __name__ == "__main__":
    unittest.main()