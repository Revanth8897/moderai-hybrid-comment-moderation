
import unittest
from unittest.mock import Mock

from moderation.unified_analyzer import UnifiedCommentAnalyzer


class TestUnifiedCommentAnalyzer(unittest.TestCase):

    def setUp(self):
        self.safety_predictor = Mock()
        self.engagement_predictor = Mock()

        self.safety_predictor.predict.return_value = {
            "safety_scores": {
                "toxicity": 0.10,
                "obscene": 0.02,
                "threat": 0.01,
                "insult": 0.08,
                "identity_attack": 0.01,
                "sexual_explicit": 0.01
            },
            "qualified_review_signals": [],
            "qualified_review_signal_present": False,
            "moderation_action": "no_automatic_action"
        }

        self.engagement_predictor.classify_comment.return_value = {
            "predicted_intent": "QUESTION",
            "max_probability": 0.75,
            "second_probability": 0.15,
            "confidence_margin": 0.60,
            "class_probabilities": {},
            "requires_review": True,
            "review_reason": "Human confirmation required.",
            "model_version": "engagement_tfidf_logreg_v1"
        }

        self.analyzer = UnifiedCommentAnalyzer(
            safety_predictor=self.safety_predictor,
            engagement_predictor=self.engagement_predictor
        )

    def test_both_engines_are_called(self):
        self.analyzer.analyze("Can you explain this?")

        self.safety_predictor.predict.assert_called_once()
        self.engagement_predictor.classify_comment.assert_called_once()

    def test_both_engines_receive_same_normalized_comment(self):
        self.analyzer.analyze("  Can you explain this?  ")

        self.safety_predictor.predict.assert_called_once_with(
            "Can you explain this?"
        )

        self.engagement_predictor.classify_comment.assert_called_once_with(
            "Can you explain this?"
        )

    def test_output_contains_both_analyses(self):
        result = self.analyzer.analyze("Can you explain this?")

        self.assertIn("safety_analysis", result)
        self.assertIn("engagement_analysis", result)

    def test_human_review_required_for_engagement(self):
        result = self.analyzer.analyze("Can you explain this?")

        self.assertTrue(result["requires_human_review"])
        self.assertIn(
            "Human confirmation required.",
            result["review_reasons"]
        )

    def test_safety_signal_requires_review(self):
        self.safety_predictor.predict.return_value[
            "qualified_review_signals"
        ] = ["toxicity"]

        self.safety_predictor.predict.return_value[
            "qualified_review_signal_present"
        ] = True

        # Simulate an engagement engine that does not require review.
        self.engagement_predictor.classify_comment.return_value[
            "requires_review"
        ] = False

        result = self.analyzer.analyze("Test comment")

        self.assertTrue(result["requires_human_review"])
        self.assertTrue(
            any("toxicity" in reason for reason in result["review_reasons"])
        )

    def test_no_review_when_neither_engine_requires_it(self):
        self.engagement_predictor.classify_comment.return_value[
            "requires_review"
        ] = False

        result = self.analyzer.analyze("A normal comment")

        self.assertFalse(result["requires_human_review"])
        self.assertEqual(result["review_reasons"], [])

    def test_no_automatic_moderation_action(self):
        result = self.analyzer.analyze("A normal comment")

        self.assertEqual(
            result["moderation_action"],
            "no_automatic_action"
        )

    def test_empty_comment_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.analyzer.analyze("   ")

    def test_non_string_comment_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.analyzer.analyze(123)


if __name__ == "__main__":
    unittest.main()
