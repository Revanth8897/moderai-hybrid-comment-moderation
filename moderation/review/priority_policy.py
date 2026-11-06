
from typing import Any


class ReviewPriorityPolicy:
    """
    Determines the priority of a comment in the
    MODERAI human-review queue.

    Priority does not determine whether review is required.
    All V1 comments continue to require human review.
    """

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"

    VALID_INTENTS = {
        "POSITIVE",
        "NEGATIVE",
        "CRITICAL",
        "QUESTION",
        "NEUTRAL",
        "UNCLEAR",
    }

    @classmethod
    def evaluate(cls, analysis: dict[str, Any]) -> dict[str, str]:
        """
        Evaluate analysis and return review priority
        with an explanation.
        """

        if not isinstance(analysis, dict):
            return {
                "priority": cls.HIGH,
                "reason": "Invalid analysis structure."
            }

        safety = analysis.get("safety_analysis")
        engagement = analysis.get("engagement_analysis")

        if not isinstance(safety, dict):
            return {
                "priority": cls.HIGH,
                "reason": "Safety analysis is missing or invalid."
            }

        if not isinstance(engagement, dict):
            return {
                "priority": cls.HIGH,
                "reason": "Engagement analysis is missing or invalid."
            }

        signals = safety.get("qualified_review_signals")

        if not isinstance(signals, list):
            return {
                "priority": cls.HIGH,
                "reason": "Qualified safety signals are invalid."
            }

        # Safety always receives the highest review priority.
        if signals:
            return {
                "priority": cls.CRITICAL,
                "reason": (
                    "Qualified safety signals detected: "
                    + ", ".join(str(signal) for signal in signals)
                )
            }

        intent = engagement.get("predicted_intent")

        if not isinstance(intent, str) or intent not in cls.VALID_INTENTS:
            return {
                "priority": cls.HIGH,
                "reason": "Engagement intent is missing or invalid."
            }

        if intent == "UNCLEAR":
            return {
                "priority": cls.HIGH,
                "reason": "Comment intent is unclear."
            }

        return {
            "priority": cls.NORMAL,
            "reason": "Standard human review required."
        }
