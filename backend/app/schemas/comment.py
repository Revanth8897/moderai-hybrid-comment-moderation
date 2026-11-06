
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CommentAnalysisRequest(BaseModel):
    """Incoming comment analysis request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    comment_text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Comment text to analyze",
    )


class CommentAnalysisResponse(BaseModel):
    """Complete comment analysis response."""

    comment: str
    analysis: dict[str, Any]
    routing: dict[str, Any]
    pipeline_version: str
    response_generation: dict[str, Any]
    review: dict[str, Any] | None = None
