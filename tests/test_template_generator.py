
import unittest

from moderation.response.template_generator import (
    TemplateResponseGenerator
)


class TestTemplateResponseGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = TemplateResponseGenerator()

    def test_generates_positive_reply(self):
        result = self.generator.generate(
            "Amazing video!",
            "POSITIVE"
        )

        self.assertEqual(result["status"], "DRAFT")
        self.assertEqual(result["generator"], "template")
        self.assertEqual(
            result["reply"],
            "Thank you! Glad you enjoyed it."
        )

    def test_generates_thank_you_reply(self):
        result = self.generator.generate(
            "Thanks for explaining!",
            "POSITIVE"
        )

        self.assertEqual(
            result["reply"],
            "You're very welcome! Glad I could help."
        )

    def test_generates_helpful_reply(self):
        result = self.generator.generate(
            "This tutorial was very useful.",
            "POSITIVE"
        )

        self.assertEqual(
            result["reply"],
            "Thank you! I'm glad you found it helpful."
        )

    def test_human_approval_is_required(self):
        result = self.generator.generate(
            "Amazing video!",
            "POSITIVE"
        )

        self.assertTrue(result["requires_human_approval"])
        self.assertFalse(result["automatic_action_allowed"])

    def test_non_positive_intent_is_rejected(self):
        with self.assertRaises(ValueError):
            self.generator.generate(
                "This video is bad.",
                "NEGATIVE"
            )

    def test_empty_comment_is_rejected(self):
        with self.assertRaises(ValueError):
            self.generator.generate("", "POSITIVE")

    def test_non_string_comment_is_rejected(self):
        with self.assertRaises(TypeError):
            self.generator.generate(123, "POSITIVE")

    def test_non_string_intent_is_rejected(self):
        with self.assertRaises(TypeError):
            self.generator.generate("Amazing video!", 123)


if __name__ == "__main__":
    unittest.main()
