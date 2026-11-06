
import unittest

from moderation.response.llm_generator import LLMResponseGenerator


class TestLLMResponseGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = LLMResponseGenerator()

    def test_generates_question_response(self):
        result = self.generator.generate(
            "Can you explain indexing?",
            "QUESTION"
        )

        self.assertEqual(result["reply"], "PRODUCED BY LLM")
        self.assertEqual(result["status"], "SIMULATED_DRAFT")

    def test_generates_critical_response(self):
        result = self.generator.generate(
            "You missed explaining B-tree indexing.",
            "CRITICAL"
        )

        self.assertEqual(result["reply"], "PRODUCED BY LLM")
        self.assertEqual(result["predicted_intent"], "CRITICAL")

    def test_marks_response_as_simulated(self):
        result = self.generator.generate(
            "Can you explain indexing?",
            "QUESTION"
        )

        self.assertTrue(result["is_simulated"])
        self.assertEqual(result["generator"], "llm_simulation")

    def test_human_approval_is_required(self):
        result = self.generator.generate(
            "Can you explain indexing?",
            "QUESTION"
        )

        self.assertTrue(result["requires_human_approval"])
        self.assertFalse(result["automatic_action_allowed"])

    def test_negative_intent_is_rejected(self):
        with self.assertRaises(ValueError):
            self.generator.generate(
                "This is bad.",
                "NEGATIVE"
            )

    def test_empty_comment_is_rejected(self):
        with self.assertRaises(ValueError):
            self.generator.generate("", "QUESTION")

    def test_non_string_comment_is_rejected(self):
        with self.assertRaises(TypeError):
            self.generator.generate(123, "QUESTION")

    def test_non_string_intent_is_rejected(self):
        with self.assertRaises(TypeError):
            self.generator.generate("Explain indexing", 123)


if __name__ == "__main__":
    unittest.main()
