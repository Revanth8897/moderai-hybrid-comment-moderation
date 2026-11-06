
import unittest
from unittest.mock import Mock

from moderation.routing.engine import RoutingEngine


class TestRoutingEngine(unittest.TestCase):

    def setUp(self):
        self.mock_policy = Mock()

        self.expected_result = {
            "route": "HUMAN_REVIEW",
            "priority": "NORMAL",
            "reason": "Human confirmation required.",
            "requires_human_approval": True,
            "automatic_action_allowed": False,
            "routing_version": "moderai_routing_v1"
        }

        self.mock_policy.decide.return_value = self.expected_result

        self.engine = RoutingEngine(policy=self.mock_policy)

        self.sample_analysis = {
            "safety_analysis": {
                "qualified_review_signals": []
            },
            "engagement_analysis": {
                "predicted_intent": "QUESTION"
            },
            "requires_human_review": True
        }

    def test_engine_calls_policy(self):
        self.engine.route(self.sample_analysis)

        self.mock_policy.decide.assert_called_once_with(
            self.sample_analysis
        )

    def test_engine_returns_policy_result(self):
        result = self.engine.route(self.sample_analysis)

        self.assertEqual(result, self.expected_result)

    def test_invalid_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.engine.route("invalid analysis")

    def test_none_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.engine.route(None)

    def test_default_policy_is_used(self):
        engine = RoutingEngine()

        result = engine.route(self.sample_analysis)

        self.assertEqual(result["route"], "HUMAN_REVIEW")
        self.assertFalse(result["automatic_action_allowed"])


if __name__ == "__main__":
    unittest.main()
