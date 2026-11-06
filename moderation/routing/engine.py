
from typing import Any

from moderation.routing.policy import RoutingPolicy


class RoutingEngine:
    """
    Coordinates MODERAI's response-routing decisions.

    Receives unified analysis and delegates the decision
    to the configured routing policy.

    Does not generate replies or perform platform actions.
    """

    def __init__(self, policy=None):
        self.policy = (
            policy if policy is not None else RoutingPolicy
        )

        print("MODERAI Routing Engine initialized.")

    def route(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """
        Determine the appropriate route for an analyzed comment.
        """

        if not isinstance(analysis, dict):
            raise TypeError("Analysis must be a dictionary.")

        routing_result = self.policy.decide(analysis)

        return routing_result


if __name__ == "__main__":

    engine = RoutingEngine()

    sample_analysis = {
        "safety_analysis": {
            "qualified_review_signals": []
        },
        "engagement_analysis": {
            "predicted_intent": "QUESTION",
            "requires_review": True
        },
        "requires_human_review": True
    }

    result = engine.route(sample_analysis)

    print("\nMODERAI ROUTING RESULT")
    print("=" * 50)

    for key, value in result.items():
        print(f"{key}: {value}")
