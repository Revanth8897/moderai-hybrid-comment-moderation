
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moderation.engagement.predictor import EngagementPredictor


class TestEngagementPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the model once for the entire test class."""
        cls.predictor = EngagementPredictor()

    def test_model_has_five_intent_classes(self):
        expected_classes = {
            "POSITIVE",
            "NEGATIVE",
            "CRITICAL",
            "QUESTION",
            "NEUTRAL",
        }

        self.assertEqual(
            set(self.predictor.classes),
            expected_classes
        )

    def test_prediction_returns_required_fields(self):
        result = self.predictor.classify_comment(
            "Amazing explanation!"
        )

        required_fields = {
            "predicted_intent",
            "max_probability",
            "second_probability",
            "confidence_margin",
            "class_probabilities",
            "requires_review",
            "review_reason",
            "model_version",
        }

        self.assertTrue(
            required_fields.issubset(result.keys())
        )

    def test_prediction_has_valid_intent(self):
        result = self.predictor.classify_comment(
            "How can I learn more?"
        )

        self.assertIn(
            result["predicted_intent"],
            self.predictor.classes
        )

    def test_probabilities_are_valid(self):
        result = self.predictor.classify_comment(
            "This is a useful tutorial."
        )

        probabilities = result["class_probabilities"]

        self.assertEqual(len(probabilities), 5)

        for probability in probabilities.values():
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

        self.assertAlmostEqual(
            sum(probabilities.values()),
            1.0,
            places=5
        )

    def test_confidence_and_margin_are_valid(self):
        result = self.predictor.classify_comment(
            "I have a question about this."
        )

        self.assertGreaterEqual(
            result["max_probability"], 0.0
        )

        self.assertLessEqual(
            result["max_probability"], 1.0
        )

        self.assertGreaterEqual(
            result["confidence_margin"], 0.0
        )

        self.assertLessEqual(
            result["confidence_margin"], 1.0
        )

        self.assertLessEqual(
            result["second_probability"],
            result["max_probability"]
        )

    def test_review_is_required_in_v1(self):
        result = self.predictor.classify_comment(
            "I completely disagree."
        )

        self.assertTrue(result["requires_review"])

    def test_empty_comment_raises_error(self):
        with self.assertRaises(ValueError):
            self.predictor.classify_comment("   ")

    def test_non_string_input_raises_error(self):
        with self.assertRaises(TypeError):
            self.predictor.classify_comment(123)


if __name__ == "__main__":
    unittest.main(verbosity=2)
