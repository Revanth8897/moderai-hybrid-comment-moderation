
from typing import Any

from moderation.comment_pipeline import CommentModerationPipeline


class CommentService:
    """
    Connects the FastAPI layer with the
    existing comment moderation pipeline.
    """

    def __init__(
        self,
        pipeline: CommentModerationPipeline | None = None,
    ):
        self.pipeline = (
            pipeline
            if pipeline is not None
            else CommentModerationPipeline()
        )

    def analyze_comment(self, comment_text: str) -> dict[str, Any]:
        return self.pipeline.process(comment_text)


# Shared production service instance
comment_service = CommentService()
