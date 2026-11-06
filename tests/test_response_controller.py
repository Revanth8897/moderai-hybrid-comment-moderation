
import unittest
from unittest.mock import Mock

from moderation.response.controller import (
    ResponseGenerationController
)


class TestResponseGenerationController(unittest.TestCase):

    def setUp(self):
        self.template_generator = Mock()
        self.llm_generator = Mock()

        self.template_generator.generate.return_value = {
            "status": "DRAFT",
            "reply": "Thank you! Glad you enjoyed it.",
            "generator": "template"
        }

        self.llm_generator.generate.return_value = {
            "status": "SIMULATED_DRAFT",
            "reply": "PRODUCED BY LLM",
            "generator": "llm_simulation"
        }

        self.controller = ResponseGenerationController(
            template_generator=self.template_generator,
            llm_generator=self.llm_generator
        )

    def create_pipeline_result(
        self,
        route="TEMPLATE_REPLY",
        intent="POSITIVE",
        requires_review=False,
        safety_signals=None
    ):
        return {
            "comment": "Amazing explanation!",
            "analysis": {
                "comment": "Amazing explanation!",
                "requires_human_review": requires_review,
                "safety_analysis": {
                    "qualified_review_signals": (
                        safety_signals if safety_signals is not None else []
                    )
                },
                "engagement_analysis": {
                    "predicted_intent": intent
                }
            },
            "routing": {
                "route": route
            }
        }

    def test_template_route_calls_template_generator(self):
        result = self.controller.generate(
            self.create_pipeline_result()
        )

        self.template_generator.generate.assert_called_once()
        self.llm_generator.generate.assert_not_called()

        self.assertEqual(result["status"], "DRAFT")
        self.assertEqual(result["route"], "TEMPLATE_REPLY")

    def test_llm_route_calls_llm_generator(self):
        result = self.controller.generate(
            self.create_pipeline_result(
                route="LLM_REQUIRED",
                intent="QUESTION"
            )
        )

        self.llm_generator.generate.assert_called_once()
        self.template_generator.generate.assert_not_called()

        self.assertEqual(result["status"], "SIMULATED_DRAFT")
        self.assertEqual(
            result["response"]["reply"],
            "PRODUCED BY LLM"
        )

    def test_no_response_route_generates_nothing(self):
        result = self.controller.generate(
            self.create_pipeline_result(
                route="NO_RESPONSE",
                intent="NEGATIVE"
            )
        )

        self.assertEqual(result["status"], "NO_DRAFT")
        self.assertIsNone(result["response"])

        self.template_generator.generate.assert_not_called()
        self.llm_generator.generate.assert_not_called()

    def test_human_review_route_generates_nothing(self):
        result = self.controller.generate(
            self.create_pipeline_result(
                route="HUMAN_REVIEW",
                requires_review=True
            )
        )

        self.assertEqual(result["status"], "NO_DRAFT")
        self.assertEqual(result["route"], "HUMAN_REVIEW")

    def test_human_review_blocks_template_generation(self):
        result = self.controller.generate(
            self.create_pipeline_result(
                route="TEMPLATE_REPLY",
                intent="POSITIVE",
                requires_review=True
            )
        )

        self.assertEqual(result["route"], "HUMAN_REVIEW")
        self.assertIsNone(result["response"])
        self.template_generator.generate.assert_not_called()

    def test_safety_escalation_blocks_generation(self):
        result = self.controller.generate(
            self.create_pipeline_result(
                route="SAFETY_ESCALATION",
                intent="CRITICAL",
                safety_signals=["toxicity"]
            )
        )

        self.assertEqual(result["route"], "SAFETY_ESCALATION")
        self.assertIsNone(result["response"])
        self.template_generator.generate.assert_not_called()
        self.llm_generator.generate.assert_not_called()

    def test_safety_signal_blocks_generation(self):
        result = self.controller.generate(
            self.create_pipeline_result(
                route="TEMPLATE_REPLY",
                intent="POSITIVE",
                safety_signals=["insult"]
            )
        )

        self.assertEqual(result["route"], "SAFETY_ESCALATION")
        self.assertIsNone(result["response"])

    def test_intent_route_mismatch_is_rejected(self):
        result = self.controller.generate(
            self.create_pipeline_result(
                route="TEMPLATE_REPLY",
                intent="NEGATIVE"
            )
        )

        self.assertEqual(result["route"], "HUMAN_REVIEW")
        self.assertIsNone(result["response"])
        self.template_generator.generate.assert_not_called()

    def test_automatic_action_is_never_allowed(self):
        result = self.controller.generate(
            self.create_pipeline_result()
        )

        self.assertFalse(result["automatic_action_allowed"])
        self.assertTrue(result["requires_human_approval"])

    def test_invalid_pipeline_input_is_rejected(self):
        with self.assertRaises(TypeError):
            self.controller.generate("invalid input")

    def test_missing_analysis_is_rejected(self):
        with self.assertRaises(ValueError):
            self.controller.generate({
                "routing": {"route": "TEMPLATE_REPLY"}
            })

    def test_missing_routing_is_rejected(self):
        with self.assertRaises(ValueError):
            self.controller.generate({
                "analysis": {}
            })


if __name__ == "__main__":
    unittest.main()
