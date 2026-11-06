
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.database.models import User
from backend.app.database.session import get_db
from backend.app.schemas.auth import (
    RegisterRequest,
    UserResponse,
    TokenResponse,
)
from backend.app.utils.response import success_response
from backend.app.security.password import (
    hash_password,
    verify_password,
)
from backend.app.security.jwt import create_access_token
from backend.app.dependencies.auth import get_current_user


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
)
def register_user(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):
    """Register a new reviewer account."""

    user = User(
        email=request.email,
        hashed_password=hash_password(request.password),
        role="REVIEWER",
        is_active=True,
    )

    try:
        db.add(user)
        db.commit()
        db.refresh(user)

    except IntegrityError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    except Exception:
        db.rollback()
        raise

    user_data = UserResponse.model_validate(user)

    return success_response(
        message="User registered successfully.",
        data=user_data.model_dump(),
    )



@router.post(
    "/login",
    response_model=TokenResponse,
)
def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Authenticate a user and return an OAuth2-compatible access token."""

    email = form_data.username.strip().lower()

    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if user is None:
        raise unauthorized

    if not verify_password(
        form_data.password,
        user.hashed_password,
    ):
        raise unauthorized

    if not user.is_active:
        raise unauthorized

    access_token = create_access_token(
        user_id=str(user.id),
        role=user.role,
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
    )


@router.get("/me")
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    """Return the authenticated user's profile."""

    user_data = UserResponse.model_validate(current_user)

    return success_response(
        message="User profile retrieved successfully.",
        data=user_data.model_dump(),
    )
