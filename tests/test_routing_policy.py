
import unittest

from moderation.routing.policy import RoutingPolicy


def make_analysis(
    intent="POSITIVE",
    safety_signals=None,
    requires_human_review=False
):
    return {
        "safety_analysis": {
            "qualified_review_signals": (
                safety_signals if safety_signals is not None else []
            )
        },
        "engagement_analysis": {
            "predicted_intent": intent,
            "requires_review": requires_human_review
        },
        "requires_human_review": requires_human_review
    }


class TestRoutingPolicy(unittest.TestCase):

    def test_safety_signal_routes_to_escalation(self):
        analysis = make_analysis(
            intent="POSITIVE",
            safety_signals=["toxicity"],
            requires_human_review=True
        )

        result = RoutingPolicy.decide(analysis)

        self.assertEqual(result["route"], "SAFETY_ESCALATION")
        self.assertEqual(result["priority"], "CRITICAL")

    def test_human_review_takes_priority(self):
        analysis = make_analysis(
            intent="POSITIVE",
            requires_human_review=True
        )

        result = RoutingPolicy.decide(analysis)

        self.assertEqual(result["route"], "HUMAN_REVIEW")

    def test_positive_routes_to_template(self):
        result = RoutingPolicy.decide(
            make_analysis("POSITIVE")
        )

        self.assertEqual(result["route"], "TEMPLATE_REPLY")

    def test_question_routes_to_llm(self):
        result = RoutingPolicy.decide(
            make_analysis("QUESTION")
        )

        self.assertEqual(result["route"], "LLM_REQUIRED")

    def test_critical_routes_to_llm(self):
        result = RoutingPolicy.decide(
            make_analysis("CRITICAL")
        )

        self.assertEqual(result["route"], "LLM_REQUIRED")

    def test_negative_routes_to_no_response(self):
        result = RoutingPolicy.decide(
            make_analysis("NEGATIVE")
        )

        self.assertEqual(result["route"], "NO_RESPONSE")

    def test_neutral_routes_to_no_response(self):
        result = RoutingPolicy.decide(
            make_analysis("NEUTRAL")
        )

        self.assertEqual(result["route"], "NO_RESPONSE")

    def test_unknown_intent_routes_to_human_review(self):
        result = RoutingPolicy.decide(
            make_analysis("UNKNOWN")
        )

        self.assertEqual(result["route"], "HUMAN_REVIEW")

    def test_missing_human_review_status_routes_to_review(self):
        analysis = make_analysis("POSITIVE")
        del analysis["requires_human_review"]

        result = RoutingPolicy.decide(analysis)

        self.assertEqual(result["route"], "HUMAN_REVIEW")

    def test_automatic_actions_are_never_allowed(self):
        for intent in [
            "POSITIVE",
            "QUESTION",
            "CRITICAL",
            "NEGATIVE",
            "NEUTRAL"
        ]:
            result = RoutingPolicy.decide(
                make_analysis(intent)
            )

            self.assertFalse(result["automatic_action_allowed"])
            self.assertTrue(result["requires_human_approval"])


if __name__ == "__main__":
    unittest.main()
