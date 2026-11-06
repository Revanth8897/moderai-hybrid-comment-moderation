
from typing import Any

from pydantic import BaseModel


class APIError(BaseModel):
    """
    Standard structure for API errors.
    """

    code: str
    details: Any | None = None


class APIResponse(BaseModel):
    """
    Common response structure for all API endpoints.
    """

    success: bool
    message: str
    data: Any | None = None
    error: APIError | None = None
