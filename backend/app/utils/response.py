
from typing import Any

from backend.app.schemas.common import APIResponse, APIError


def success_response(
    message: str,
    data: Any = None,
) -> dict:
    """
    Create a standardized successful API response.
    """

    response = APIResponse(
        success=True,
        message=message,
        data=data,
        error=None,
    )

    return response.model_dump()


def error_response(
    message: str,
    code: str,
    details: Any = None,
) -> dict:
    """
    Create a standardized error API response.
    """

    response = APIResponse(
        success=False,
        message=message,
        data=None,
        error=APIError(
            code=code,
            details=details,
        ),
    )

    return response.model_dump()
