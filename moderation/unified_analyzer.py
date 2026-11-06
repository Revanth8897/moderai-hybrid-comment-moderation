
from typing import Any

from moderation.toxicity.predictor import SafetyPredictor
from moderation.engagement.predictor import EngagementPredictor


class UnifiedCommentAnalyzer:
    """
    Coordinates MODERAI's independent safety and engagement
    prediction engines.

    This component combines model evidence only.
    It does not make moderation or response-routing decisions.
    """

    def __init__(
        self,
        safety_predictor=None,
        engagement_predictor=None
    ):
        # Load each model only once when the analyzer is created.
        self.safety_predictor = (
            safety_predictor
            if safety_predictor is not None
            else SafetyPredictor()
        )

        self.engagement_predictor = (
            engagement_predictor
            if engagement_predictor is not None
            else EngagementPredictor()
        )

        print("MODERAI Unified Comment Analyzer initialized.")

    def analyze(self, comment_text: str) -> dict[str, Any]:
        """
        Run safety and engagement analysis on one comment.

        Returns the independent outputs from both engines.
        No moderation action is performed.
        """

        # Step 1: Validate input.
        if not isinstance(comment_text, str):
            raise TypeError("Comment text must be a string.")

        normalized_comment = comment_text.strip()

        if not normalized_comment:
            raise ValueError("Comment text cannot be empty.")

        # Step 2: Run the safety engine.
        safety_result = self.safety_predictor.predict(
            normalized_comment
        )

        # Step 3: Run the engagement engine.
        engagement_result = (
            self.engagement_predictor.classify_comment(
                normalized_comment
            )
        )

        # Step 4: Collect review reasons.
        review_reasons = []

        # Engagement V1 always requires human confirmation.
        if engagement_result["requires_review"]:
            review_reasons.append(
                engagement_result["review_reason"]
            )

        # Preserve qualified safety signals separately.
        qualified_safety_signals = (
            safety_result["qualified_review_signals"]
        )

        if qualified_safety_signals:
            review_reasons.append(
                "Qualified safety review signals detected: "
                + ", ".join(qualified_safety_signals)
            )

        # Step 5: Combine the independent results.
        return {
            "comment": comment_text,
            "normalized_comment": normalized_comment,

            "safety_analysis": safety_result,

            "engagement_analysis": engagement_result,

            "requires_human_review": (
                engagement_result["requires_review"]
                or safety_result[
                    "qualified_review_signal_present"
                ]
            ),

            "review_reasons": review_reasons,

            # This service only analyzes.
            "moderation_action": "no_automatic_action",

            "analysis_version": "moderai_unified_v1"
        }


if __name__ == "__main__":

    analyzer = UnifiedCommentAnalyzer()

    sample_comment = (
        "The tutorial is useful, but you missed explaining "
        "database indexing. Can you cover that?"
    )

    result = analyzer.analyze(sample_comment)

    print("\nMODERAI UNIFIED ANALYSIS")
    print("=" * 60)

    print("\nCOMMENT:")
    print(result["comment"])

    print("\nSAFETY ANALYSIS:")
    for label, score in result["safety_analysis"][
        "safety_scores"
    ].items():
        print(f"{label}: {score}")

    print("\nQUALIFIED SAFETY SIGNALS:")
    print(
        result["safety_analysis"][
            "qualified_review_signals"
        ]
    )

    print("\nENGAGEMENT ANALYSIS:")
    print(
        "Predicted intent:",
        result["engagement_analysis"]["predicted_intent"]
    )

    print(
        "Maximum probability:",
        result["engagement_analysis"]["max_probability"]
    )

    print(
        "Confidence margin:",
        result["engagement_analysis"]["confidence_margin"]
    )

    print("\nHUMAN REVIEW REQUIRED:")
    print(result["requires_human_review"])

    print("\nREVIEW REASONS:")
    for reason in result["review_reasons"]:
        print("-", reason)

    print("\nMODERATION ACTION:")
    print(result["moderation_action"])
