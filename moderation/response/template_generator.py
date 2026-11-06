
from typing import Any


class TemplateResponseGenerator:
    """
    Generates predefined friendly response drafts.

    This generator does not call an LLM or publish replies.
    """

    GENERATOR_VERSION = "moderai_template_v1"
    DEFAULT_TONE = "friendly"

    def generate(self, comment_text: str, intent: str) -> dict[str, Any]:

        if not isinstance(comment_text, str):
            raise TypeError("Comment must be a string.")

        normalized_comment = comment_text.strip()

        if not normalized_comment:
            raise ValueError("Comment cannot be empty.")

        if not isinstance(intent, str):
            raise TypeError("Intent must be a string.")

        if intent.upper() != "POSITIVE":
            raise ValueError(
                "Template generator only supports POSITIVE comments."
            )

        lower_comment = normalized_comment.lower()

        if any(word in lower_comment for word in ["thank you", "thanks"]):
            reply = "You're very welcome! Glad I could help."

        elif any(word in lower_comment for word in ["helpful", "useful"]):
            reply = "Thank you! I'm glad you found it helpful."

        else:
            reply = "Thank you! Glad you enjoyed it."

        return {
            "original_comment": normalized_comment,
            "status": "DRAFT",
            "generator": "template",
            "reply": reply,
            "tone": self.DEFAULT_TONE,
            "requires_human_approval": True,
            "automatic_action_allowed": False,
            "generator_version": self.GENERATOR_VERSION
        }


if __name__ == "__main__":
    generator = TemplateResponseGenerator()

    result = generator.generate(
        comment_text="Amazing video!",
        intent="POSITIVE"
    )

    print("\nMODERAI TEMPLATE RESPONSE")
    print("=" * 45)

    for key, value in result.items():
        print(f"{key}: {value}")
