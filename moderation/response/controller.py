
from typing import Any

from moderation.response.template_generator import (
    TemplateResponseGenerator
)
from moderation.response.llm_generator import (
    LLMResponseGenerator
)


class ResponseGenerationController:
    """
    Selects the appropriate response generator based on
    the MODERAI routing result.

    All generated replies remain drafts.
    No automatic publishing is permitted.
    """

    CONTROLLER_VERSION = "moderai_response_controller_v1"

    def __init__(
        self,
        template_generator=None,
        llm_generator=None
    ):
        self.template_generator = (
            template_generator
            if template_generator is not None
            else TemplateResponseGenerator()
        )

        self.llm_generator = (
            llm_generator
            if llm_generator is not None
            else LLMResponseGenerator()
        )

        print("MODERAI Response Generation Controller initialized.")

    def generate(self, pipeline_result: dict[str, Any]) -> dict[str, Any]:

        if not isinstance(pipeline_result, dict):
            raise TypeError("Pipeline result must be a dictionary.")

        analysis = pipeline_result.get("analysis")
        routing = pipeline_result.get("routing")

        if not isinstance(analysis, dict):
            raise ValueError("Pipeline analysis is missing or invalid.")

        if not isinstance(routing, dict):
            raise ValueError("Pipeline routing is missing or invalid.")

        safety = analysis.get("safety_analysis")
        engagement = analysis.get("engagement_analysis")

        if not isinstance(safety, dict) or not isinstance(engagement, dict):
            raise ValueError("Safety or engagement analysis is missing.")

        comment = analysis.get("comment")
        intent = engagement.get("predicted_intent")
        route = routing.get("route")

        if not isinstance(comment, str) or not comment.strip():
            raise ValueError("Original comment is missing or invalid.")

        if not isinstance(intent, str):
            raise ValueError("Predicted intent is missing or invalid.")

        if not isinstance(route, str):
            raise ValueError("Routing decision is missing or invalid.")

        # Safety escalation always takes priority.
        if route == "SAFETY_ESCALATION":
            return self._no_draft(
                route,
                "Priority safety review is required."
            )

        # Never bypass the unified analyzer's review requirement.
        if analysis.get("requires_human_review") is not False:
            return self._no_draft(
                "HUMAN_REVIEW",
                "Human confirmation is required before response generation."
            )

        # Do not generate replies for safety review signals.
        qualified_signals = safety.get("qualified_review_signals")

        if qualified_signals:
            return self._no_draft(
                "SAFETY_ESCALATION",
                "Qualified safety signals require priority human review."
            )

        # Route-specific generator selection.
        if route == "TEMPLATE_REPLY":

            if intent.upper() != "POSITIVE":
                return self._no_draft(
                    "HUMAN_REVIEW",
                    "Template route does not match the predicted intent."
                )

            response = self.template_generator.generate(
                comment_text=comment,
                intent=intent
            )

            return self._draft_result(route, response)

        if route == "LLM_REQUIRED":

            if intent.upper() not in {"QUESTION", "CRITICAL"}:
                return self._no_draft(
                    "HUMAN_REVIEW",
                    "LLM route does not match the predicted intent."
                )

            response = self.llm_generator.generate(
                comment_text=comment,
                intent=intent
            )

            return self._draft_result(route, response)

        if route == "NO_RESPONSE":
            return self._no_draft(
                route,
                "No response is required for this comment."
            )

        if route == "HUMAN_REVIEW":
            return self._no_draft(
                route,
                "Comment requires human review."
            )

        return self._no_draft(
            "HUMAN_REVIEW",
            "Unsupported routing decision. Manual review required."
        )

    def generate_after_human_approval(
        self,
        pipeline_result: dict[str, Any],
        approved: bool = False
    ) -> dict[str, Any]:
        """
        Generate a response draft only after explicit human approval.

        This method does not publish or send the generated reply.
        """

        if not isinstance(pipeline_result, dict):
            raise TypeError("Pipeline result must be a dictionary.")

        if approved is not True:
            raise ValueError(
                "Explicit human approval is required for draft generation."
            )

        analysis = pipeline_result.get("analysis")
        routing = pipeline_result.get("routing")

        if not isinstance(analysis, dict):
            raise ValueError("Pipeline analysis is missing or invalid.")

        if not isinstance(routing, dict):
            raise ValueError("Pipeline routing is missing or invalid.")

        safety = analysis.get("safety_analysis")
        engagement = analysis.get("engagement_analysis")

        if not isinstance(safety, dict) or not isinstance(engagement, dict):
            raise ValueError("Safety or engagement analysis is missing.")

        comment = analysis.get("comment")
        intent = engagement.get("predicted_intent")
        route = routing.get("route")

        if not isinstance(comment, str) or not comment.strip():
            raise ValueError("Original comment is missing or invalid.")

        if not isinstance(intent, str) or not isinstance(route, str):
            raise ValueError("Intent or routing decision is invalid.")

        # Safety escalation always blocks draft generation.
        if route == "SAFETY_ESCALATION":
            return self._no_draft(
                route,
                "Priority safety review prevents draft generation."
            )

        # Qualified safety signals must never be bypassed.
        qualified_signals = safety.get("qualified_review_signals")

        if not isinstance(qualified_signals, list):
            raise ValueError("Qualified safety signals are invalid.")

        if qualified_signals:
            return self._no_draft(
                "SAFETY_ESCALATION",
                "Qualified safety signals prevent draft generation."
            )

        # After human approval, positive comments can use
        # the template generator even when their route is HUMAN_REVIEW.
        if route in {"TEMPLATE_REPLY", "HUMAN_REVIEW"}:

            if intent.upper() == "POSITIVE":
                response = self.template_generator.generate(
                    comment_text=comment,
                    intent=intent
                )

                return self._draft_result(route, response)

            if route == "TEMPLATE_REPLY":
                return self._no_draft(
                    "HUMAN_REVIEW",
                    "Template route does not match the predicted intent."
                )

        # Generate an LLM draft after approval for eligible comments.
        if route in {"LLM_REQUIRED", "HUMAN_REVIEW"}:

            if intent.upper() not in {"QUESTION", "CRITICAL"}:
                return self._no_draft(
                    "HUMAN_REVIEW",
                    "LLM route does not match the predicted intent."
                )

            response = self.llm_generator.generate(
                comment_text=comment,
                intent=intent
            )

            return self._draft_result("LLM_REQUIRED", response)

        # Other routes do not support response generation.
        return self._no_draft(
            route,
            "This routing decision does not support draft generation."
        )

    def _draft_result(self, route: str, response: dict) -> dict[str, Any]:
        return {
            "status": response["status"],
            "route": route,
            "response": response,
            "requires_human_approval": True,
            "automatic_action_allowed": False,
            "controller_version": self.CONTROLLER_VERSION
        }

    def _no_draft(self, route: str, reason: str) -> dict[str, Any]:
        return {
            "status": "NO_DRAFT",
            "route": route,
            "response": None,
            "reason": reason,
            "requires_human_approval": True,
            "automatic_action_allowed": False,
            "controller_version": self.CONTROLLER_VERSION
        }


if __name__ == "__main__":
    from moderation.comment_pipeline import CommentModerationPipeline

    pipeline = CommentModerationPipeline()

    pipeline_result = pipeline.process(
        "The tutorial is useful, but can you explain database indexing?"
    )

    controller = ResponseGenerationController()
    result = controller.generate(pipeline_result)

    print("\nMODERAI RESPONSE GENERATION RESULT")
    print("=" * 50)

    for key, value in result.items():
        print(f"{key}: {value}")
