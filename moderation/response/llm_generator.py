
from typing import Any


class LLMResponseGenerator:
    """
    Simulates the LLM response-generation stage.

    No actual LLM or external API is called.
    """

    GENERATOR_VERSION = "moderai_llm_simulation_v1"

    def generate(self, comment_text: str, intent: str) -> dict[str, Any]:

        if not isinstance(comment_text, str):
            raise TypeError("Comment must be a string.")

        normalized_comment = comment_text.strip()

        if not normalized_comment:
            raise ValueError("Comment cannot be empty.")

        if not isinstance(intent, str):
            raise TypeError("Intent must be a string.")

        normalized_intent = intent.upper()

        if normalized_intent not in {"QUESTION", "CRITICAL"}:
            raise ValueError(
                "Simulated LLM only supports QUESTION and CRITICAL intents."
            )

        return {
            "original_comment": normalized_comment,
            "status": "SIMULATED_DRAFT",
            "generator": "llm_simulation",
            "reply": "PRODUCED BY LLM",
            "tone": "friendly",
            "predicted_intent": normalized_intent,
            "is_simulated": True,
            "requires_human_approval": True,
            "automatic_action_allowed": False,
            "generator_version": self.GENERATOR_VERSION
        }


if __name__ == "__main__":
    generator = LLMResponseGenerator()

    result = generator.generate(
        comment_text="Can you explain database indexing?",
        intent="QUESTION"
    )

    print("\nMODERAI SIMULATED LLM RESPONSE")
    print("=" * 45)

    for key, value in result.items():
        print(f"{key}: {value}")
