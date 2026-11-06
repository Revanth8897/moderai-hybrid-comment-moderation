
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ReviewStartRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )


class ReviewDecisionRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    decision: Literal[
        "APPROVE_FOR_DRAFT",
        "NO_RESPONSE",
        "REJECT",
        "ESCALATE",
    ] = Field(
        ...,
        description="The reviewer's decision",
    )

    reason: str = Field(
        default="",
        max_length=1000,
        description="Reason for the review decision",
    )
