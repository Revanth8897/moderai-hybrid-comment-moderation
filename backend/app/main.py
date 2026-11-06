
import logging
import time
import uuid

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from backend.app.api.routes.auth import router as auth_router
from backend.app.core.logging_config import configure_logging


from backend.app.api.routes.comments import (
    router as comments_router,
)
from backend.app.api.routes.reviews import (
    router as reviews_router,
)


from backend.app.utils.response import error_response

configure_logging()

app = FastAPI(
    title="HYBRID COMMENT MODERATION SYSTEM",
    description=(
        "An AI-powered hybrid comment moderation "
        "and engagement management system."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins={
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    },
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = logging.getLogger("moderation.api")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,  
            2,
        )

        logger.exception(
            "Request failed",
            extra={"request_id": request_id},
        )
        raise

    duration_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    return response

# Handle HTTP errors.
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
):
    error_codes = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
    }

    code = error_codes.get(
        exc.status_code,
        "HTTP_ERROR",
    )

    message = (
        exc.detail
        if isinstance(exc.detail, str)
        else "The request could not be processed."
    )

    logger.warning(
        "HTTP request error",
        extra={
            "request_id": getattr(
                request.state,
                "request_id",
                None,
            ),
            "method": request.method,
            "path": request.url.path,
            "status_code": exc.status_code,
            "error_code": code,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            message=message,
            code=code,
        ),
        headers=exc.headers,
    )


# Handle request validation errors.

# Handle request validation errors.
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    # Keep only JSON-safe validation information.
    validation_details = [
        {
            "type": error.get("type"),
            "location": list(error.get("loc", ())),
            "message": error.get("msg"),
        }
        for error in exc.errors()
    ]

    logger.warning(
        "Request validation failed",
        extra={
            "request_id": getattr(
                request.state,
                "request_id",
                None,
            ),
            "method": request.method,
            "path": request.url.path,
            "status_code": 422,
            "error_count": len(validation_details),
        },
    )

    return JSONResponse(
        status_code=422,
        content=error_response(
            message="Request validation failed.",
            code="VALIDATION_ERROR",
            details=validation_details,
        ),
    )

# Handle unexpected application errors.
@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "Unhandled application exception",
        extra={
            "request_id": getattr(
                request.state,
                "request_id",
                None,
            ),
            "method": request.method,
            "path": request.url.path,
        },
    )

    return JSONResponse(
        status_code=500,
        content=error_response(
            message="An unexpected server error occurred.",
            code="INTERNAL_SERVER_ERROR",
        ),
    )

# Register API routers.
app.include_router(comments_router)
app.include_router(reviews_router)
app.include_router(auth_router)

@app.get("/")
def root():
    return {
        "application": "HYBRID COMMENT MODERATION SYSTEM",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "HYBRID COMMENT MODERATION SYSTEM",
    }
