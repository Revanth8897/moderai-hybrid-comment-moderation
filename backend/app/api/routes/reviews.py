
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from backend.app.database.models import User
from backend.app.dependencies.roles import require_roles

from backend.app.schemas.common import APIResponse
from backend.app.schemas.review import (
    ReviewStartRequest,
    ReviewDecisionRequest,
)
from backend.app.services.comment_service import comment_service
from backend.app.utils.response import success_response


router = APIRouter(
    prefix="/reviews",
    tags=["Human Review"],
)


# Both ADMIN and REVIEWER can access review operations.
reviewer_access = require_roles("ADMIN", "REVIEWER")


@router.get(
    "/pending",
    response_model=APIResponse,
)
def get_pending_reviews(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=10, ge=1, le=100),
    priority: Literal["CRITICAL", "HIGH", "NORMAL"] | None = Query(
        default=None
    ),
    current_user: User = Depends(reviewer_access),
):
    """
    Retrieve pending reviews with priority filtering
    and pagination.
    """

    review_manager = (
        comment_service.pipeline.review_manager
    )

    all_reviews = review_manager.get_pending_reviews()

    if priority is not None:
        all_reviews = [
            review
            for review in all_reviews
            if review.get("review_priority") == priority
        ]

    total = len(all_reviews)

    reviews = all_reviews[skip:skip + limit]

    has_more = skip + len(reviews) < total

    return success_response(
        message="Pending reviews retrieved successfully.",
        data={
            "total": total,
            "skip": skip,
            "limit": limit,
            "priority": priority,
            "has_more": has_more,
            "reviews": reviews,
        },
    )


@router.get(
    "/{review_id}",
    response_model=APIResponse,
)
def get_review(
    review_id: str,
    current_user: User = Depends(reviewer_access),
):
    """
    Retrieve an individual review by its ID.
    """

    review_manager = (
        comment_service.pipeline.review_manager
    )

    try:
        review = review_manager.get_review(review_id)

        return success_response(
            message="Review retrieved successfully.",
            data=review,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@router.post(
    "/{review_id}/start",
    response_model=APIResponse,
)
def start_review(
    review_id: str,
    request: ReviewStartRequest,
    current_user: User = Depends(reviewer_access),
):
    """
    Move a pending review into IN_REVIEW status.
    Reviewer identity comes from the authenticated user.
    """

    review_manager = (
        comment_service.pipeline.review_manager
    )

    try:
        review = review_manager.update_status(
            review_id=review_id,
            new_status="IN_REVIEW",
            reviewer_id=str(current_user.id),
        )

        return success_response(
            message="Review started successfully.",
            data=review,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@router.post(
    "/{review_id}/decision",
    response_model=APIResponse,
)
def submit_review_decision(
    review_id: str,
    request: ReviewDecisionRequest,
    current_user: User = Depends(reviewer_access),
):
    """
    Submit a human review decision.
    Reviewer identity comes from the authenticated user.
    """

    review_manager = (
        comment_service.pipeline.review_manager
    )

    try:
        review = review_manager.decide_review(
            review_id=review_id,
            decision=request.decision,
            reviewer_id=str(current_user.id),
            reason=request.reason,
        )

        return success_response(
            message="Review decision submitted successfully.",
            data=review,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
