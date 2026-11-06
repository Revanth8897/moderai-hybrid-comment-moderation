
from fastapi import APIRouter, HTTPException

from backend.app.schemas.comment import (
    CommentAnalysisRequest,
)
from backend.app.schemas.common import APIResponse
from backend.app.services.comment_service import comment_service
from backend.app.utils.response import success_response


router = APIRouter(
    prefix="/comments",
    tags=["Comment Analysis"],
)


@router.post(
    "/analyze",
    response_model=APIResponse,
)
def analyze_comment(
    request: CommentAnalysisRequest,
):
    """
    Analyze a comment and return a standardized API response.
    """

    try:
        result = comment_service.analyze_comment(
            request.comment_text
        )

        return success_response(
            message="Comment analyzed successfully.",
            data=result,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Comment analysis failed.",
        ) from exc
