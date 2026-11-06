from typing import Any

from moderation.unified_analyzer import UnifiedCommentAnalyzer
from moderation.routing.engine import RoutingEngine
from moderation.response.controller import ResponseGenerationController
from moderation.review.manager import ReviewManager


class CommentModerationPipeline:
    """
    Coordinates comment analysis, routing,
    response generation, and human review queueing.

    It does not publish, hide, delete,
    or send replies.
    """

    PIPELINE_VERSION = "moderai_pipeline_v3"

    def __init__(
        self,
        analyzer=None,
        routing_engine=None,
        response_controller=None,
        review_manager=None
    ):
        self.analyzer = (
            analyzer
            if analyzer is not None
            else UnifiedCommentAnalyzer()
        )

        self.routing_engine = (
            routing_engine
            if routing_engine is not None
            else RoutingEngine()
        )

        self.response_controller = (
            response_controller
            if response_controller is not None
            else ResponseGenerationController()
        )

        self.review_manager = (
            review_manager
            if review_manager is not None
            else ReviewManager(
		response_controller=self.response_controller
	    )
        )

        print("MODERAI Comment Moderation Pipeline initialized.")

    def process(self, comment_text: str) -> dict[str, Any]:

        # Stage 1: Analyze the comment.
        analysis_result = self.analyzer.analyze(comment_text)

        # Stage 2: Select a routing decision.
        routing_result = self.routing_engine.route(analysis_result)

        # Stage 3: Build the pipeline result.
        result = {
            "comment": analysis_result["comment"],
            "analysis": analysis_result,
            "routing": routing_result,
            "pipeline_version": self.PIPELINE_VERSION
        }

        # Stage 4: Generate a draft or return a no-draft result.
        response_result = self.response_controller.generate(result)

        result["response_generation"] = response_result

        # Stage 5: Add the result to the human review queue
        # only when the analyzer explicitly requires review.
        if analysis_result.get("requires_human_review") is True:

            review_record = self.review_manager.add_to_queue(result)

            result["review"] = review_record

        else:
            result["review"] = None

        return result


if __name__ == "__main__":

    pipeline = CommentModerationPipeline()

    sample_comment = (
        "The tutorial is useful, but you missed explaining "
        "database indexing. Can you cover that?"
    )

    result = pipeline.process(sample_comment)

    print("\nMODERAI COMPLETE PIPELINE RESULT")
    print("=" * 60)

    print("\nCOMMENT:")
    print(result["comment"])

    print("\nENGAGEMENT INTENT:")
    print(
        result["analysis"]["engagement_analysis"]["predicted_intent"]
    )

    print("\nSAFETY SIGNALS:")
    print(
        result["analysis"]["safety_analysis"]["qualified_review_signals"]
    )

    print("\nSELECTED ROUTE:")
    print(result["routing"]["route"])

    print("\nRESPONSE GENERATION:")
    print(result["response_generation"])

    print("\nHUMAN REVIEW:")
    print(result["review"])

    print("\nAUTOMATIC ACTION ALLOWED:")
    print(
        result["response_generation"]["automatic_action_allowed"]
    )