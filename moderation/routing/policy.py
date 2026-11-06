
from enum import Enum


class Route(str, Enum):
    TEMPLATE_REPLY = "TEMPLATE_REPLY"
    LLM_REQUIRED = "LLM_REQUIRED"
    NO_RESPONSE = "NO_RESPONSE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    SAFETY_ESCALATION = "SAFETY_ESCALATION"


class Priority(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RoutingPolicy:
    """
    MODERAI V1 deterministic routing policy.

    The policy defines route decisions.
    It does not publish, delete, hide, or generate replies.
    """

    ROUTING_VERSION = "moderai_routing_v1"

    @staticmethod
    def decide(analysis: dict) -> dict:

        # Rule 1: Validate the analysis structure.
        if not isinstance(analysis, dict):
            raise TypeError("Analysis must be a dictionary.")

        safety = analysis.get("safety_analysis")
        engagement = analysis.get("engagement_analysis")

        if not isinstance(safety, dict) or not isinstance(engagement, dict):
            return RoutingPolicy._result(
                route=Route.HUMAN_REVIEW,
                priority=Priority.HIGH,
                reason="Safety or engagement analysis is missing or invalid."
            )

        # Rule 2: Qualified safety signals receive priority review.
        safety_signals = safety.get("qualified_review_signals")

        if not isinstance(safety_signals, list):
            return RoutingPolicy._result(
                route=Route.HUMAN_REVIEW,
                priority=Priority.HIGH,
                reason="Safety review signals are missing or invalid."
            )

        if safety_signals:
            return RoutingPolicy._result(
                route=Route.SAFETY_ESCALATION,
                priority=Priority.CRITICAL,
                reason=(
                    "Qualified safety review signals detected: "
                    + ", ".join(str(signal) for signal in safety_signals)
                )
            )

        # Rule 3: Human review takes priority over engagement routing.
        human_review = analysis.get("requires_human_review")
        
        if not isinstance(human_review, bool):
            return RoutingPolicy._result(
                route=Route.HUMAN_REVIEW,
                priority=Priority.HIGH,
                reason="Human review status is missing or invalid."
            )
            
        if human_review:
            return RoutingPolicy._result(
                route=Route.HUMAN_REVIEW,
                priority=Priority.NORMAL,
                reason="Analysis requires human confirmation."
            )

        # Rule 4: Validate engagement intent.
        intent = engagement.get("predicted_intent")

        if not isinstance(intent, str):
            return RoutingPolicy._result(
                route=Route.HUMAN_REVIEW,
                priority=Priority.HIGH,
                reason="Engagement intent is missing or invalid."
            )

        intent = intent.upper()

        # Rule 5: Map engagement intent to a route.
        if intent == "POSITIVE":
            return RoutingPolicy._result(
                route=Route.TEMPLATE_REPLY,
                priority=Priority.LOW,
                reason="Positive comment is suitable for a predefined reply."
            )

        if intent in ("QUESTION", "CRITICAL"):
            return RoutingPolicy._result(
                route=Route.LLM_REQUIRED,
                priority=Priority.NORMAL,
                reason=f"{intent} comment may require a contextual response."
            )

        if intent in ("NEGATIVE", "NEUTRAL"):
            return RoutingPolicy._result(
                route=Route.NO_RESPONSE,
                priority=Priority.LOW,
                reason=f"{intent} comment does not require an automatic reply."
            )

        # Rule 6: Unknown intent must never be automatically routed.
        return RoutingPolicy._result(
            route=Route.HUMAN_REVIEW,
            priority=Priority.HIGH,
            reason=f"Unsupported engagement intent: {intent}"
        )

    @staticmethod
    def _result(route, priority, reason):
        return {
            "route": route.value,
            "priority": priority.value,
            "reason": reason,
            "requires_human_approval": True,
            "automatic_action_allowed": False,
            "routing_version": RoutingPolicy.ROUTING_VERSION
        }
